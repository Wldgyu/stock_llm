# -*- coding: utf-8 -*-
"""
ml_stock.py
───────────────────────────────────────────────────────────────────
[역할] 머신러닝 학습 및 분석 엔진
  - LSTMModel       : PyTorch 기반 LSTM 가격 예측 모델
  - StockAIAgentV3  : 데이터 수집 / 뉴스 수집 / 백테스트 / 예측 / 시각화
  - run_and_return(): llm_stock.py 에서 import 하여 결과를 딕셔너리로 받음

[단독 실행 시] python ml_stock.py
"""

from datetime import datetime, time as clock_time, timedelta
from zoneinfo import ZoneInfo

import warnings
import traceback
import urllib.parse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import feedparser

try:
    import trafilatura
    from googlenewsdecoder import gnewsdecoder
    ARTICLE_EXTRACTION_AVAILABLE = True
except ImportError:
    trafilatura = None
    gnewsdecoder = None
    ARTICLE_EXTRACTION_AVAILABLE = False

import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

from database import DB_PATH, connect_db
from backtesting import compare_strategies
from trading_dates import prediction_dates, completed_bars
from news_selection import COMPANIES, select_news

# ── TabPFN / HuggingFace 인증 자동 로드 ──────────────────────
load_dotenv(Path(__file__).with_name(".env"))


def _load_tabpfn_token() -> bool:
    """`.env`의 TABPFN_TOKEN 설정 여부를 확인합니다."""
    token = os.environ.get("TABPFN_TOKEN", "").strip()
    if not token:
        print("⚠️ .env에 TABPFN_TOKEN이 설정되지 않았습니다.")
        return False
    print("✅ TabPFN 토큰 환경변수 로드 완료")
    return True


def _hf_login_from_env() -> bool:
    """`.env`의 HF_TOKEN으로 Hugging Face 세션을 인증합니다."""
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        print("⚠️ .env에 HF_TOKEN이 설정되지 않았습니다.")
        return False
    try:
        from huggingface_hub import login as hf_login

        hf_login(token=hf_token, add_to_git_credential=False)
        print("✅ HuggingFace 로그인 완료")
        return True
    except Exception as e:
        print(f"⚠️ HuggingFace 로그인 실패: {e}")
        return False

_AUTH_INITIALIZED = False


def _ensure_model_auth():
    """TabPFN을 실제 사용할 때 `.env` 인증정보를 적용합니다."""
    global _AUTH_INITIALIZED
    if _AUTH_INITIALIZED:
        return
    tabpfn_ready = _load_tabpfn_token()
    huggingface_ready = _hf_login_from_env()
    # 둘 다 실패하면 다음 종목에서 다시 시도할 수 있도록 False를 유지합니다.
    _AUTH_INITIALIZED = tabpfn_ready or huggingface_ready

# TabPFN (Google PriorLabs) - 테이블형 데이터 Transformer 기반 회귀 모델
try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("⚠️ tabpfn 미설치 → pip install tabpfn")

warnings.filterwarnings('ignore')



# ──────────────────────────────────────────────
# 공통 유틸: 다음 거래일 계산
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# 1. LSTM 모델 정의
# ──────────────────────────────────────────────
class LSTMModel(nn.Module):
    """
    3-step 다중 출력 LSTM 모델.
    입력: (batch, seq_len, input_dim)
    출력: (batch, output_dim=3)  → [T+1, T+4, T+7] 예측
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        output_dim=3,
        dropout=0.2,
    ):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(self.dropout(out[:, -1, :]))


# ──────────────────────────────────────────────
# 2. 통합 Stock AI Agent V3
# ──────────────────────────────────────────────
class StockAIAgentV3:
    """
    머신러닝 기반 주식 분석 에이전트 (V3).

    기능:
      - 주가 + 매크로(환율·SOX) 데이터 수집
      - LLM용 뉴스 제목 및 본문 수집
      - RSI 기반 백테스트
      - Sharpe Ratio / MDD 성과 지표
      - LSTM 3-step 가격 예측 (T+1, T+4, T+7)
    """

    # 기본 학습 범위와 LSTM 크기를 한곳에서 조정합니다.
    HISTORY_PERIOD = "3y"
    LSTM_SEQUENCE_LENGTH = 30
    LSTM_HIDDEN_SIZE = 32
    LSTM_NUM_LAYERS = 1
    LSTM_MAX_EPOCHS = 100
    WALK_FORWARD_FOLDS = 3

    def __init__(self):
        self.tickers = {
            "삼성전자": "005930.KS",
            # "SK하이닉스": "000660.KS",
            # "한미반도체": "042700.KS",
            "엔비디아": "NVDA",
            "인텔":    "INTC",
            # "알파벳": "GOOGL",
            # "브로드컴": "AVGO",
        }
        self.db_path = DB_PATH
        self._init_analysis_db()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _init_analysis_db(self):
        """웹 대시보드가 읽는 ML 분석 이력 테이블을 준비합니다."""
        conn = connect_db(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT,
                    name        TEXT,
                    symbol      TEXT,
                    last_close  REAL,
                    rsi         REAL,
                    trend       TEXT,
                    support     REAL,
                    resistance  REAL,
                    usd_krw     REAL,
                    sox         REAL,
                    sentiment   REAL,
                    sent_label  TEXT,
                    pred_low    REAL,
                    pred_high   REAL,
                    lstm_t1     REAL,
                    lstm_t4     REAL,
                    lstm_t7     REAL,
                    tabpfn_t1   REAL,
                    tabpfn_t4   REAL,
                    tabpfn_t7   REAL,
                    lstm_metrics_json   TEXT,
                    tabpfn_metrics_json TEXT,
                    risk_score  INTEGER
                )
            """)
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(analysis_log)")
            }
            # 기존 DB에도 모델별 예측과 검증 결과 컬럼을 자동으로 추가합니다.
            new_columns = {
                "data_date": "TEXT",
                "target_t1": "TEXT",
                "target_t4": "TEXT",
                "target_t7": "TEXT",
                "lstm_t1": "REAL",
                "lstm_t4": "REAL",
                "lstm_t7": "REAL",
                "tabpfn_t1": "REAL",
                "tabpfn_t4": "REAL",
                "tabpfn_t7": "REAL",
                "lstm_metrics_json": "TEXT",
                "tabpfn_metrics_json": "TEXT",
            }
            for column, column_type in new_columns.items():
                if column not in columns:
                    conn.execute(
                        f"ALTER TABLE analysis_log "
                        f"ADD COLUMN {column} {column_type}"
                    )
            conn.commit()
        finally:
            conn.close()

    def _save_analysis_result(self, name: str, result: dict):
        """한 번 계산한 분석 결과를 중복 계산 없이 DB에 저장합니다."""
        df = result["df"]
        rsi = float(df["RSI"].dropna().iloc[-1])
        trend = "과매수" if rsi > 70 else ("과매도" if rsi < 30 else "중립")
        predictions = [float(value) for value in result["preds"]]
        tabpfn_predictions = result.get("tabpfn_preds") or [None, None, None]
        tabpfn_predictions = [
            float(value) if value is not None else None
            for value in tabpfn_predictions
        ]
        recent = df.tail(20)
        volatility = float(df["Close"].pct_change().std() * np.sqrt(252) * 100)
        risk_score = min(10, max(1, int(round(volatility / 5))))

        values = (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            name,
            result["symbol"],
            float(df["Close"].iloc[-1]),
            rsi,
            trend,
            float(recent["Low"].min()),
            float(recent["High"].max()),
            float(df["USD_KRW"].iloc[-1]),
            float(df["SOX_Index"].iloc[-1]),
            None,
            None,
            min(predictions),
            max(predictions),
            predictions[0],
            predictions[1],
            predictions[2],
            tabpfn_predictions[0],
            tabpfn_predictions[1],
            tabpfn_predictions[2],
            json.dumps(result["lstm_metrics"], ensure_ascii=False),
            json.dumps(result.get("tabpfn_metrics"), ensure_ascii=False)
            if result.get("tabpfn_metrics") is not None else None,
            risk_score,
            result["data_date"], result["target_t1"], result["target_t4"], result["target_t7"],
        )
        conn = connect_db(self.db_path)
        try:
            cursor = conn.execute("""
                INSERT INTO analysis_log (
                    timestamp, name, symbol, last_close, rsi, trend,
                    support, resistance, usd_krw, sox, sentiment,
                    sent_label, pred_low, pred_high,
                    lstm_t1, lstm_t4, lstm_t7,
                    tabpfn_t1, tabpfn_t4, tabpfn_t7,
                    lstm_metrics_json, tabpfn_metrics_json, risk_score,
                    data_date, target_t1, target_t4, target_t7
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, values)
            conn.commit()
            result["analysis_id"] = cursor.lastrowid
        finally:
            conn.close()

    def _analyze_stock(self, name: str, symbol: str):
        """단일 종목의 모든 ML 계산을 한 번 수행하고 저장합니다."""
        df = self.fetch_data(symbol)
        if df is None or len(df) < 30:
            print(f"⚠️ {name}: 데이터 부족, 건너뜁니다.")
            return None

        df["RSI"] = ta.rsi(df["Close"], length=14)
        print(f"📅 [{name}] 최신 확정 일봉: {df.index[-1].date()} | 수집 {len(df)}거래일")
        bt_ret, equity = self.backtest(df)
        sharpe, mdd    = self.calculate_performance_metrics(equity)
        titles = self.get_news_titles(name)
        news_articles = self.get_financial_news_articles(name, symbol)
        preds, model, last_seq, baseline_seq, lstm_metrics = self.predict_multi_step(df)
        tabpfn_preds, tabpfn_metrics = self.predict_tabpfn(df)
        model_metrics = {"lstm": lstm_metrics}
        if tabpfn_metrics:
            model_metrics["tabpfn"] = tabpfn_metrics
        comparison = compare_strategies(
            df, {key: value["oos_t1_predictions"] for key, value in model_metrics.items()}
        )
        comparison["data_date"] = str(df.index[-1].date())
        for value in model_metrics.values():
            value["strategy_backtest"] = comparison
        print(f"📊 OOS 전략 비교 {comparison['start']} ~ {comparison['end']} (비용 차감)")
        print("   ℹ️ T+1·T+4·T+7 정답이 모두 있는 공통 OOS 표본 기준: 최근 6거래일은 제외됩니다. 최신 예측 기준일과 다릅니다.")
        for key, value in comparison["strategies"].items():
            print(f"  {key}: 수익률 {value['return_pct']:.2f}% | Sharpe {value['sharpe']:.2f} | MDD {value['mdd_pct']:.2f}% | 주문 {value['orders']}회")

        result = {
            **prediction_dates(df.index[-1], symbol),
            "df": df,
            "symbol": symbol,
            "bt_ret": bt_ret,
            "sharpe": sharpe,
            "mdd": mdd,
            "titles": titles,
            "news_articles": news_articles,
            "preds": preds,
            "tabpfn_preds": tabpfn_preds,
            "lstm_metrics": lstm_metrics,
            "tabpfn_metrics": tabpfn_metrics,
            "model": model,
            "l_seq": last_seq,
            "baseline_seq": baseline_seq,
        }
        self._save_analysis_result(name, result)
        print(f"✅ [{name}] ML 분석 및 DB 저장 완료.")
        return result

    # ── 데이터 수집 ──────────────────────────
    @staticmethod
    def _drop_incomplete_daily_bar(df: pd.DataFrame, symbol: str):
        """거래소 정규장이 끝나기 전 생성된 당일 미완성 일봉을 제거합니다."""
        return completed_bars(df, symbol)

    @staticmethod
    def _close_times_utc(index, timezone_name: str, close_time: clock_time):
        """일봉 날짜와 시장 마감 시각을 결합해 UTC 확정 시각으로 변환합니다."""
        timezone = ZoneInfo(timezone_name)
        utc = ZoneInfo("UTC")
        close_timestamps = []
        for value in index:
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is not None:
                timestamp = timestamp.tz_convert(timezone)
            local_close = datetime.combine(
                timestamp.date(), close_time, tzinfo=timezone
            )
            close_timestamps.append(local_close.astimezone(utc))
        return pd.to_datetime(close_timestamps, utc=True)

    @classmethod
    def _merge_confirmed_macro(
        cls,
        stock_df: pd.DataFrame,
        macro_series: pd.Series,
        stock_symbol: str,
        macro_name: str,
    ) -> pd.Series:
        """주가 마감 전에 확정된 가장 최신 거시지표만 결합합니다."""
        is_korean = stock_symbol.endswith((".KS", ".KQ"))
        stock_timezone = "Asia/Seoul" if is_korean else "America/New_York"
        stock_close = clock_time(15, 30) if is_korean else clock_time(16, 0)
        macro_rules = {
            "USD_KRW": ("America/New_York", clock_time(17, 0)),
            "SOX_Index": ("America/New_York", clock_time(16, 0)),
        }
        macro_timezone, macro_close = macro_rules[macro_name]

        stock_times = cls._close_times_utc(
            stock_df.index, stock_timezone, stock_close
        )
        macro_times = cls._close_times_utc(
            macro_series.index, macro_timezone, macro_close
        )
        left = pd.DataFrame({
            "_stock_row": np.arange(len(stock_df)),
            "_confirmed_at": stock_times,
        }).sort_values("_confirmed_at")
        right = pd.DataFrame({
            "_confirmed_at": macro_times,
            macro_name: pd.to_numeric(macro_series, errors="coerce").to_numpy(),
        })
        right = (
            right.dropna(subset=[macro_name])
            .drop_duplicates("_confirmed_at", keep="last")
            .sort_values("_confirmed_at")
        )
        if right.empty:
            return pd.Series(np.nan, index=stock_df.index, name=macro_name)

        merged = pd.merge_asof(
            left,
            right,
            on="_confirmed_at",
            direction="backward",
            allow_exact_matches=True,
        )
        values = (
            merged.sort_values("_stock_row")[macro_name]
            .to_numpy(dtype=np.float64)
        )
        return pd.Series(values, index=stock_df.index, name=macro_name)

    def fetch_data(self, symbol: str):
        """yfinance로 3년치 일봉 + 환율(KRW=X) + SOX 지수를 다운로드합니다."""
        df = yf.download(
            symbol,
            period=self.HISTORY_PERIOD,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return None

        # MultiIndex 컬럼 처리 (최신 yfinance 대응)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = self._drop_incomplete_daily_bar(df, symbol)
        if df.empty:
            return None

        # 매크로 데이터 병합
        macros = {"USD_KRW": "KRW=X", "SOX_Index": "^SOX"}
        for col_name, m_sym in macros.items():
            m_data = yf.download(
                m_sym,
                period=self.HISTORY_PERIOD,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )['Close']
            if isinstance(m_data, pd.DataFrame):
                m_data = m_data.iloc[:, 0]
            # UTC 확정 시각을 기준으로 직전 사용 가능 값만 가져와 미래 누수를 막습니다.
            df[col_name] = self._merge_confirmed_macro(
                df, m_data, symbol, col_name
            )

        # 과거 값으로만 채워 미래 데이터가 과거 행에 들어가는 누수를 막습니다.
        df = df.ffill().dropna()
        return df

    # ── 뉴스 제목 수집 ────────────────────────
    def get_news_titles(self, name: str):
        """기사 본문을 얻지 못했을 때 LLM에 전달할 뉴스 제목을 수집합니다."""
        company = COMPANIES.get(name, (name, ()))[0]
        encoded_query = urllib.parse.quote(f'{company} when:14d')
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        rss = feedparser.parse(url)
        return [entry.title for entry, _, _ in select_news(rss.entries, name, limit=3)]

    def get_financial_news_articles(self, name: str, symbol: str):
        """지정 금융 언론사별 최근 14일의 주가 관련 기사를 선별해 본문을 수집합니다."""
        sources = {
            "Bloomberg": ("Bloomberg",),
            "Wall Street Journal": ("WSJ", "The Wall Street Journal", "Wall Street Journal"),
            "Financial Times": ("Financial Times",),
            "Reuters": ("Reuters",),
        }
        company_query = COMPANIES.get(name, (symbol.split(".")[0], ()))[0]
        articles = []

        for display_source, accepted_names in sources.items():
            source_query = accepted_names[0]
            encoded_query = urllib.parse.quote(
                f'{company_query} source:"{source_query}" when:14d'
            )
            url = (
                "https://news.google.com/rss/search"
                f"?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            )
            rss = feedparser.parse(url)
            selected = select_news(rss.entries, name, accepted_names)
            if not selected:
                continue
            entry, relevance_score, reasons = selected[0]
            print(f"      📰 {display_source}: {entry.title} (선정: {', '.join(reasons)})")

            article_url = entry.link
            body = ""
            content_type = "제목"
            if ARTICLE_EXTRACTION_AVAILABLE:
                try:
                    decoded = gnewsdecoder(entry.link, interval=0.2)
                    if decoded.get("status"):
                        article_url = decoded["decoded_url"]
                        downloaded = trafilatura.fetch_url(article_url)
                        if downloaded:
                            body = trafilatura.extract(
                                downloaded,
                                include_comments=False,
                                include_tables=False,
                            ) or ""
                            if len(body) >= 200:
                                content_type = "본문"
                            else:
                                metadata = trafilatura.extract_metadata(downloaded)
                                description = (
                                    metadata.description if metadata else ""
                                )
                                if description:
                                    body = description
                                    content_type = "기사 요약"
                except Exception as exc:
                    print(f"      ⚠️ {display_source} 본문 수집 실패: {exc}")

            # LLM 입력이 너무 커지지 않도록 기사당 본문 길이를 제한합니다.
            body = " ".join(body.split())[:2500]
            articles.append(
                {
                    "source": display_source,
                    "selection_reason": reasons,
                    "relevance_score": relevance_score,
                    "title": entry.title,
                    "published": getattr(entry, "published", ""),
                    "url": article_url,
                    "body": body,
                    "content_type": content_type,
                }
            )

        return articles

    # ── 성과 지표 ─────────────────────────────────────────────────
    def calculate_performance_metrics(self, equity: pd.Series):
        """
        전략 잔고 곡선(equity)의 일간 수익률로
        Sharpe Ratio 와 MDD(최대 낙폭)를 계산합니다.

        equity: backtest()가 반환한 날짜별 잔고 금액 Series
        """
        returns = equity.pct_change().dropna()
        if len(returns) == 0:
            return 0.0, 0.0
        sharpe = (
            np.sqrt(252) * (returns.mean() - (0.03 / 252)) / returns.std()
            if returns.std() != 0 else 0.0
        )
        # 잔고 곡선에서 직접 drawdown 계산 (첫날 고점 누락 방지)
        running_max = equity.cummax()
        mdd         = ((equity - running_max) / running_max).min() * 100
        return float(sharpe), float(mdd)

    # ── 백테스트 ─────────────────────────────────────────────────
    def backtest(self, df: pd.DataFrame):
        """
        RSI 기반 단순 전략의 과거 수익률과 일일 잔고 곡선을 반환합니다.

        체결 규칙:
            전날 RSI 신호로 다음날 종가에 체결 (lookahead bias 방지).
            i=0 은 신호 전날이므로 체결 없이 잔고만 기록합니다.

        반환 값:
            bt_ret (float)    : 전략 총수익률 (%)
            equity (pd.Series): 날짜별 잔고 금액 — Sharpe · MDD 계산에 사용
        """
        df = df.copy()
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df = df.dropna(subset=['RSI'])

        init_bal = bal = 10_000_000
        pos = 0
        equity_vals = []   # 날짜별 잔고 = 현금 + 보유 주식 평가액

        for i in range(len(df)):
            p = float(df['Close'].iloc[i])
            # 전날 RSI 신호로 오늘 종가에 체결 (i=0 은 이전 신호가 없으므로 건너뜀)
            if i > 0:
                prev_r = float(df['RSI'].iloc[i - 1])
                if prev_r < 35 and pos == 0:
                    pos = bal // p
                    bal -= pos * p
                elif prev_r > 65 and pos > 0:
                    bal += pos * p
                    pos = 0
            equity_vals.append(bal + pos * p)

        equity = pd.Series(equity_vals, index=df.index, name="equity")
        bt_ret = ((equity.iloc[-1] - init_bal) / init_bal) * 100
        return bt_ret, equity

    # ── 모델 검증 공통 함수 ────────────────────
    @staticmethod
    def _regression_metrics(y_true, y_pred, reference_values=None):
        """가격 오차와 기준 가격 대비 상승·하락 방향 정확도를 계산합니다."""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        reference = None
        if reference_values is not None:
            reference = np.asarray(reference_values, dtype=np.float64)
            if reference.ndim == 1 and y_true.ndim == 2:
                reference = np.repeat(reference[:, None], y_true.shape[1], axis=1)
            reference = np.broadcast_to(reference, y_true.shape)

        def calculate(actual, predicted, base=None):
            error = predicted - actual
            safe_actual = np.where(np.abs(actual) < 1e-8, 1e-8, np.abs(actual))
            result = {
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "mape": float(np.mean(np.abs(error) / safe_actual) * 100),
            }
            if base is not None:
                actual_direction = np.sign(actual - base)
                predicted_direction = np.sign(predicted - base)
                result["direction_accuracy"] = float(
                    np.mean(actual_direction == predicted_direction) * 100
                )
            return result

        result = calculate(y_true, y_pred, reference)
        if y_true.ndim == 2 and y_true.shape[1] == 3:
            # 각 예측 시점의 성능도 따로 확인할 수 있게 보관합니다.
            result["by_horizon"] = {
                label: calculate(
                    y_true[:, index],
                    y_pred[:, index],
                    reference[:, index] if reference is not None else None,
                )
                for index, label in enumerate(("T+1", "T+4", "T+7"))
            }
        return result

    @staticmethod
    def _walk_forward_boundaries(total_rows: int, fold_count: int = 3):
        """앞 70%를 시작 학습 구간으로 두고 이후 기간을 순서대로 나눕니다."""
        initial_train_end = int(total_rows * 0.70)
        boundaries = np.linspace(
            initial_train_end, total_rows, fold_count + 1, dtype=int
        )
        folds = [
            (int(boundaries[index]), int(boundaries[index + 1]))
            for index in range(fold_count)
            if boundaries[index + 1] > boundaries[index]
        ]
        if len(folds) != fold_count:
            raise ValueError("walk-forward fold를 만들기에 데이터가 부족합니다.")
        return folds

    @staticmethod
    def _compare_with_naive(model_metrics: dict, naive_metrics: dict):
        """모델 오차가 직전 종가 유지 기준보다 얼마나 개선됐는지 비교합니다."""
        def compare(model_result, naive_result):
            naive_mae = naive_result["mae"]
            improvement = (
                (naive_mae - model_result["mae"]) / naive_mae * 100
                if naive_mae > 1e-8 else 0.0
            )
            return {
                "beats_naive": model_result["mae"] < naive_mae,
                "mae_improvement_pct": float(improvement),
            }

        result = compare(model_metrics, naive_metrics)
        if "by_horizon" in model_metrics and "by_horizon" in naive_metrics:
            result["by_horizon"] = {
                horizon: compare(
                    model_metrics["by_horizon"][horizon],
                    naive_metrics["by_horizon"][horizon],
                )
                for horizon in ("T+1", "T+4", "T+7")
            }
        return result

    @staticmethod
    def _summarize_fold_metrics(fold_results: list):
        """fold별 성능의 평균과 표준편차를 모델·naive 기준별로 요약합니다."""
        metric_names = ("mae", "rmse", "mape", "direction_accuracy")

        def summarize(section_name):
            summary = {}
            for metric in metric_names:
                values = [
                    fold[section_name][metric]
                    for fold in fold_results
                    if metric in fold[section_name]
                ]
                if values:
                    summary[metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                    }

            summary["by_horizon"] = {}
            for horizon in ("T+1", "T+4", "T+7"):
                summary["by_horizon"][horizon] = {}
                for metric in metric_names:
                    values = [
                        fold[section_name]["by_horizon"][horizon][metric]
                        for fold in fold_results
                        if metric in fold[section_name]["by_horizon"][horizon]
                    ]
                    if values:
                        summary["by_horizon"][horizon][metric] = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                        }
            return summary

        return {
            "model": summarize("test"),
            "naive_baseline": summarize("naive_baseline"),
        }

    @staticmethod
    def _baseline_warning(metrics: dict, model_name: str):
        """전체 또는 개별 fold가 naive MAE를 넘지 못하면 경고를 반환합니다."""
        failed_sections = []
        if not metrics["comparison"]["beats_naive"]:
            failed_sections.append("전체")
        failed_sections.extend(
            f"fold {fold['fold']}"
            for fold in metrics["folds"]
            if not fold["comparison"]["beats_naive"]
        )
        if not failed_sections:
            return None
        return (
            f"{model_name} naive baseline 미달: "
            + ", ".join(failed_sections)
        )

    # 최대 100 epoch 안에서 검증 손실이 개선되지 않으면 조기 종료합니다.
    def _train_lstm(self, X_train, y_train, X_val=None, y_val=None, max_epochs=None):
        """LSTM을 학습하고 검증 손실이 개선되지 않으면 일찍 종료합니다."""
        max_epochs = max_epochs or self.LSTM_MAX_EPOCHS
        # ── 직접 조정하기 쉬운 LSTM 학습 설정 ──
        learning_rate = 0.005  # 학습률: 불안정하면 낮추고, 너무 느리면 조금 높입니다.
        batch_size = 16        # 배치 크기: 작을수록 세밀하지만 학습 시간이 늘어납니다.
        dropout_rate = 0.2     # 과적합 조절: 과적합이 크면 값을 조금 높입니다.
        huber_delta = 1.0      # 이상치 민감도: 작을수록 급등락의 영향을 덜 받습니다.
        gradient_clip = 1.0    # 기울기 제한: 학습 중 값이 폭주하는 것을 막습니다.
        patience = 12          # 조기 종료: 검증 손실 개선을 기다리는 epoch 수입니다.

        model = LSTMModel(
            input_dim=3,
            hidden_dim=self.LSTM_HIDDEN_SIZE,
            num_layers=self.LSTM_NUM_LAYERS,
            output_dim=3,
            dropout=dropout_rate,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Huber 손실은 급등락 같은 이상치가 학습을 과도하게 흔드는 것을 줄입니다.
        criterion = nn.HuberLoss(delta=huber_delta)
        best_state = None
        best_loss = float("inf")
        best_epoch = max_epochs
        wait = 0

        for epoch in range(max_epochs):
            model.train()
            indices = torch.randperm(X_train.size(0), device=X_train.device)
            for start in range(0, X_train.size(0), batch_size):
                batch_indices = indices[start:start + batch_size]
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]

                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                # LSTM의 기울기가 갑자기 커지는 현상을 제한합니다.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_clip
                )
                optimizer.step()

            # 검증 데이터가 있을 때만 조기 종료 여부를 확인합니다.
            if X_val is None:
                continue
            model.eval()
            with torch.no_grad():
                val_loss = float(criterion(model(X_val), y_val).item())
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model, best_epoch

    # ── LSTM 예측 ─────────────────────────────
    def predict_multi_step(self, df: pd.DataFrame):
        """
        시간순 3-fold walk-forward로 LSTM의 T+1, T+4, T+7을 검증합니다.
        각 fold는 직전 종가를 그대로 예측하는 naive baseline과 비교합니다.
        검증이 끝나면 전체 데이터로 최종 모델을 다시 학습해 미래를 예측합니다.
        """
        np.random.seed(42)
        torch.manual_seed(42)

        data = df[['Close', 'USD_KRW', 'SOX_Index']].values.astype(np.float32)
        seq_len = self.LSTM_SEQUENCE_LENGTH
        fold_boundaries = self._walk_forward_boundaries(
            len(data), self.WALK_FORWARD_FOLDS
        )
        fold_results = []
        best_epochs = []
        oos_predictions = {}
        all_val_true, all_val_pred = [], []
        all_test_true, all_test_pred, all_naive_pred = [], [], []

        for fold_number, (test_start, test_end) in enumerate(fold_boundaries, 1):
            # 각 fold 학습 구간의 마지막 15%는 조기 종료 검증에만 사용합니다.
            val_start = int(test_start * 0.85)
            if val_start <= seq_len + 7 or test_end - test_start < 8:
                raise ValueError("LSTM walk-forward 분리에 필요한 데이터가 부족합니다.")

            # fold마다 학습 구간 데이터로만 스케일러를 다시 맞춰 누수를 막습니다.
            scaler = StandardScaler()
            scaler.fit(data[:val_start])
            scaled = scaler.transform(data)
            split_X = {"train": [], "val": []}
            split_y = {"train": [], "val": []}

            for start in range(len(scaled) - seq_len - 6):
                target_first = start + seq_len
                target_last = target_first + 6
                if target_last < val_start:
                    split = "train"
                elif target_first >= val_start and target_last < test_start:
                    split = "val"
                else:
                    # fold 경계에 걸친 정답은 다른 구간 정보가 섞이지 않도록 제외합니다.
                    continue

                split_X[split].append(scaled[start:start + seq_len])
                split_y[split].append([
                    scaled[target_first, 0],
                    scaled[target_first + 3, 0],
                    scaled[target_first + 6, 0],
                ])

            if any(not split_X[name] for name in ("train", "val")):
                raise ValueError(
                    f"LSTM walk-forward fold {fold_number}에 빈 구간이 있습니다."
                )

            tensors = {}
            for split in ("train", "val"):
                tensors[f"X_{split}"] = torch.tensor(
                    np.asarray(split_X[split]),
                    dtype=torch.float32,
                    device=self.device,
                )
                tensors[f"y_{split}"] = torch.tensor(
                    np.asarray(split_y[split]),
                    dtype=torch.float32,
                    device=self.device,
                )

            eval_model, fold_best_epoch = self._train_lstm(
                tensors["X_train"],
                tensors["y_train"],
                tensors["X_val"],
                tensors["y_val"],
            )
            best_epochs.append(fold_best_epoch)
            eval_model.eval()
            with torch.no_grad():
                val_pred_scaled = eval_model(tensors["X_val"]).cpu().numpy()

            close_scale = scaler.scale_[0]
            close_mean = scaler.mean_[0]
            to_price = lambda values: (values * close_scale) + close_mean
            val_true = to_price(tensors["y_val"].cpu().numpy())
            val_pred = to_price(val_pred_scaled)

            # 선택한 epoch로 테스트 직전까지의 모든 과거 데이터를 다시 학습합니다.
            full_scaler = StandardScaler()
            full_scaler.fit(data[:test_start])
            full_scaled = full_scaler.transform(data)
            full_train_X, full_train_y = [], []
            full_test_X, full_test_y, full_test_naive = [], [], []
            test_dates = []
            for start in range(len(full_scaled) - seq_len - 6):
                target_first = start + seq_len
                target_last = target_first + 6
                features = full_scaled[start:start + seq_len]
                targets = [
                    full_scaled[target_first, 0],
                    full_scaled[target_first + 3, 0],
                    full_scaled[target_first + 6, 0],
                ]
                if target_last < test_start:
                    full_train_X.append(features)
                    full_train_y.append(targets)
                elif target_first >= test_start and target_last < test_end:
                    test_dates.append(str(df.index[target_first].date()))
                    full_test_X.append(features)
                    full_test_y.append(targets)
                    last_close = float(data[target_first - 1, 0])
                    full_test_naive.append([last_close, last_close, last_close])

            if not full_train_X or not full_test_X:
                raise ValueError(
                    f"LSTM walk-forward fold {fold_number} 재학습 구간이 비었습니다."
                )

            full_train_X = torch.tensor(
                np.asarray(full_train_X), dtype=torch.float32, device=self.device
            )
            full_train_y = torch.tensor(
                np.asarray(full_train_y), dtype=torch.float32, device=self.device
            )
            full_test_X = torch.tensor(
                np.asarray(full_test_X), dtype=torch.float32, device=self.device
            )
            full_test_y = torch.tensor(
                np.asarray(full_test_y), dtype=torch.float32, device=self.device
            )
            fold_model, _ = self._train_lstm(
                full_train_X,
                full_train_y,
                max_epochs=max(1, fold_best_epoch),
            )
            fold_model.eval()
            with torch.no_grad():
                test_pred_scaled = fold_model(full_test_X).cpu().numpy()

            full_close_scale = full_scaler.scale_[0]
            full_close_mean = full_scaler.mean_[0]
            test_true = (
                full_test_y.cpu().numpy() * full_close_scale
            ) + full_close_mean
            test_pred = (
                test_pred_scaled * full_close_scale
            ) + full_close_mean
            oos_predictions.update(zip(test_dates, map(float, test_pred[:, 0])))
            naive_pred = np.asarray(full_test_naive, dtype=np.float64)

            model_metrics = self._regression_metrics(
                test_true, test_pred, naive_pred
            )
            naive_metrics = self._regression_metrics(
                test_true, naive_pred, naive_pred
            )
            fold_results.append({
                "fold": fold_number,
                "train_end": test_start,
                "test_range": [test_start, test_end],
                "best_epoch": fold_best_epoch,
                "validation": self._regression_metrics(val_true, val_pred),
                "test": model_metrics,
                "naive_baseline": naive_metrics,
                "comparison": self._compare_with_naive(
                    model_metrics, naive_metrics
                ),
                "samples": {
                    "train": len(full_train_X),
                    "val": len(split_X["val"]),
                    "test": len(full_test_X),
                },
            })
            all_val_true.append(val_true)
            all_val_pred.append(val_pred)
            all_test_true.append(test_true)
            all_test_pred.append(test_pred)
            all_naive_pred.append(naive_pred)

        val_true_all = np.concatenate(all_val_true)
        val_pred_all = np.concatenate(all_val_pred)
        test_true_all = np.concatenate(all_test_true)
        test_pred_all = np.concatenate(all_test_pred)
        naive_pred_all = np.concatenate(all_naive_pred)
        test_metrics = self._regression_metrics(
            test_true_all, test_pred_all, naive_pred_all
        )
        naive_metrics = self._regression_metrics(
            test_true_all, naive_pred_all, naive_pred_all
        )
        best_epoch = max(1, int(round(float(np.median(best_epochs)))))
        metrics = {
            "walk_forward_folds": self.WALK_FORWARD_FOLDS,
            "folds": fold_results,
            "validation": self._regression_metrics(val_true_all, val_pred_all),
            "test": test_metrics,
            "naive_baseline": naive_metrics,
            "comparison": self._compare_with_naive(
                test_metrics, naive_metrics
            ),
            "fold_summary": self._summarize_fold_metrics(fold_results),
            "best_epoch": best_epoch,
        }
        metrics["oos_t1_predictions"] = oos_predictions
        metrics["baseline_warning"] = self._baseline_warning(metrics, "LSTM")
        metrics["meets_baseline"] = metrics["baseline_warning"] is None

        # 실제 미래 예측은 최신 정보까지 활용하도록 전체 데이터로 다시 학습합니다.
        final_scaler = StandardScaler()
        final_scaled = final_scaler.fit_transform(data)
        final_X, final_y = [], []
        for start in range(len(final_scaled) - seq_len - 6):
            target = start + seq_len
            final_X.append(final_scaled[start:start + seq_len])
            final_y.append([
                final_scaled[target, 0],
                final_scaled[target + 3, 0],
                final_scaled[target + 6, 0],
            ])
        X_all = torch.tensor(np.asarray(final_X), dtype=torch.float32, device=self.device)
        y_all = torch.tensor(np.asarray(final_y), dtype=torch.float32, device=self.device)
        final_model, _ = self._train_lstm(
            X_all, y_all, max_epochs=max(1, best_epoch)
        )

        final_model.eval()
        last_seq = torch.tensor(
            final_scaled[-seq_len:], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            preds_scaled = final_model(last_seq).cpu().numpy()[0]
        final_preds = (
            preds_scaled * final_scaler.scale_[0] + final_scaler.mean_[0]
        ).astype(float).tolist()
        # Integrated Gradients 배경(baseline): 학습에 사용된 시퀀스들의 평균
        baseline_seq = X_all.mean(dim=0, keepdim=True).detach()
        return final_preds, final_model, last_seq, baseline_seq, metrics

    # ── TabPFN 예측 ───────────────────────────
    def predict_tabpfn(self, df: pd.DataFrame):
        """
        시간순 3-fold walk-forward로 TabPFN의 성능을 검증합니다.
        각 fold는 직전 종가를 유지하는 naive baseline과 비교합니다.
        T+1, T+4, T+7은 각각 별도 회귀 모델로 직접 예측합니다.
        """
        if not TABPFN_AVAILABLE:
            return None, None

        try:
            _ensure_model_auth()
            # ── 피처 준비 ──────────────────────
            tdf = df.copy()
            tdf['EMA20']  = ta.ema(tdf['Close'], length=20)
            tdf['RSI']    = ta.rsi(tdf['Close'], length=14)
            tdf = tdf[['Close', 'USD_KRW', 'SOX_Index', 'RSI', 'EMA20', 'Volume']]
            tdf = tdf.dropna()

            feat_cols = ['Close', 'USD_KRW', 'SOX_Index', 'RSI', 'EMA20', 'Volume']
            look_back = 10   # 과거 N일을 피처로 사용

            all_X, all_y = [], []
            target_first_indices, target_last_indices = [], []
            all_naive = []

            # 과거 10일을 입력으로 만들고 세 미래 시점의 종가를 정답으로 둡니다.
            for i in range(look_back, len(tdf) - 6):
                window = tdf[feat_cols].iloc[i - look_back:i].values.flatten()
                targets = [
                    float(tdf['Close'].iloc[i]),
                    float(tdf['Close'].iloc[i + 3]),
                    float(tdf['Close'].iloc[i + 6]),
                ]
                all_X.append(window)
                all_y.append(targets)
                target_first_indices.append(i)
                target_last_indices.append(i + 6)
                last_close = float(tdf['Close'].iloc[i - 1])
                all_naive.append([last_close, last_close, last_close])

            all_X = np.asarray(all_X, dtype=np.float32)
            all_y = np.asarray(all_y, dtype=np.float32)
            all_naive = np.asarray(all_naive, dtype=np.float32)
            target_first_indices = np.asarray(target_first_indices)
            target_last_indices = np.asarray(target_last_indices)

            # TabPFN 권장 한도보다 많으면 각 fold의 최신 학습 샘플을 사용합니다.
            max_samples = 1000
            fold_boundaries = self._walk_forward_boundaries(
                len(tdf), self.WALK_FORWARD_FOLDS
            )
            fold_results = []
            all_test_true, all_test_pred, all_naive_pred = [], [], []
            oos_predictions = {}

            for fold_number, (test_start, test_end) in enumerate(
                fold_boundaries, 1
            ):
                train_indices = np.flatnonzero(target_last_indices < test_start)
                test_indices = np.flatnonzero(
                    (target_first_indices >= test_start)
                    & (target_last_indices < test_end)
                )
                if not len(train_indices) or not len(test_indices):
                    raise ValueError(
                        f"TabPFN walk-forward fold {fold_number}에 빈 구간이 있습니다."
                    )

                train_indices = train_indices[-max_samples:]
                X_train = all_X[train_indices]
                y_train = all_y[train_indices]
                X_test = all_X[test_indices]
                y_test = all_y[test_indices]
                naive_pred = all_naive[test_indices]
                test_predictions = np.zeros_like(y_test)

                # fold 시작 전 데이터만 학습하고 이후 구간을 순서대로 평가합니다.
                for horizon in range(3):
                    reg = TabPFNRegressor()
                    reg.fit(X_train, y_train[:, horizon])
                    test_predictions[:, horizon] = reg.predict(X_test)

                oos_predictions.update({
                    str(tdf.index[target_first_indices[idx]].date()): float(pred)
                    for idx, pred in zip(test_indices, test_predictions[:, 0])
                })
                model_metrics = self._regression_metrics(
                    y_test, test_predictions, naive_pred
                )
                naive_metrics = self._regression_metrics(
                    y_test, naive_pred, naive_pred
                )
                fold_results.append({
                    "fold": fold_number,
                    "train_end": test_start,
                    "test_range": [test_start, test_end],
                    "test": model_metrics,
                    "naive_baseline": naive_metrics,
                    "comparison": self._compare_with_naive(
                        model_metrics, naive_metrics
                    ),
                    "samples": {
                        "train": len(X_train),
                        "test": len(X_test),
                    },
                })
                all_test_true.append(y_test)
                all_test_pred.append(test_predictions)
                all_naive_pred.append(naive_pred)

            test_true_all = np.concatenate(all_test_true)
            test_pred_all = np.concatenate(all_test_pred)
            naive_pred_all = np.concatenate(all_naive_pred)
            test_metrics = self._regression_metrics(
                test_true_all, test_pred_all, naive_pred_all
            )
            naive_metrics = self._regression_metrics(
                test_true_all, naive_pred_all, naive_pred_all
            )
            metrics = {
                "walk_forward_folds": self.WALK_FORWARD_FOLDS,
                "folds": fold_results,
                "test": test_metrics,
                "naive_baseline": naive_metrics,
                "comparison": self._compare_with_naive(
                    test_metrics, naive_metrics
                ),
                "fold_summary": self._summarize_fold_metrics(fold_results),
            }
            metrics["oos_t1_predictions"] = oos_predictions
            metrics["baseline_warning"] = self._baseline_warning(
                metrics, "TabPFN"
            )
            metrics["meets_baseline"] = metrics["baseline_warning"] is None

            # 최종 예측 모델은 검증이 끝난 뒤 전체 과거 데이터를 사용합니다.
            latest_window = (
                tdf[feat_cols].iloc[-look_back:].values
                .astype(np.float32)
                .reshape(1, -1)
            )
            X_arr = all_X[-max_samples:]
            y_arr = all_y[-max_samples:]
            final_predictions = []
            for horizon in range(3):
                reg = TabPFNRegressor()
                reg.fit(X_arr, y_arr[:, horizon])
                final_predictions.append(float(reg.predict(latest_window)[0]))

            return final_predictions, metrics

        except Exception as e:
            print(f"      ⚠️ TabPFN 예측 실패: {e}")
            return None, None

    # ── 피처 기여도 (Integrated Gradients) ──
    def get_feature_importance(
        self, model, last_seq, baseline_seq=None
    ) -> dict:
        """
        PyTorch Integrated Gradients로 T+1 예측에 대한
        피처(Close·USD_KRW·SOX_Index) 기여도를 계산합니다.

        last_seq   : (1, seq_len, n_feat) 텐서
        baseline_seq: (1, seq_len, n_feat) 배경 시퀀스 평균
                      None이면 영 텐서(zero baseline) 사용
        반환       : {"Price": float, "USD_KRW": float, "SOX Index": float}
        """
        feature_names = ["Price", "USD_KRW", "SOX Index"]
        try:
            model.eval()
            inp = last_seq.clone().float()
            if baseline_seq is None:
                baseline = torch.zeros_like(inp)
            else:
                baseline = baseline_seq.clone().float().to(inp.device)

            n_steps = 50
            # 배경 → 입력 사이의 직선 보간 경로를 n_steps개 샘플
            alphas = torch.linspace(0.0, 1.0, n_steps, device=inp.device)
            interp = baseline + alphas.view(-1, 1, 1) * (inp - baseline)  # (n_steps, seq, feat)
            interp = interp.requires_grad_(True)

            # T+1 예측(출력 인덱스 0)의 합으로 스칼라 생성
            outputs = model(interp)[:, 0]  # (n_steps,)

            # IG = (inp - baseline) * mean_gradient  → (seq_len, n_feat)
            grads = torch.autograd.grad(outputs.sum(), interp)[0]
            mean_grads = grads.mean(dim=0)  # (seq_len, n_feat)
            ig = (inp.squeeze(0) - baseline.squeeze(0)) * mean_grads  # (seq_len, n_feat)

            # 전체 시퀀스 축 합산 → 피처별 기여도 (n_feat,)
            feat_contrib = ig.abs().sum(dim=0).detach().cpu().numpy()  # (n_feat,)

            total = feat_contrib.sum()
            if total > 0:
                feat_contrib = feat_contrib / total
            else:
                feat_contrib = np.zeros(len(feature_names))

            return dict(zip(feature_names, feat_contrib.tolist()))

        except Exception as e:
            print(f"      ⚠️ Integrated Gradients 계산 실패: {e}")
            # 실패 시 가짜 고정 중요도를 만들지 않습니다.
            return {}

    # ── V3 단독 실행 리포트 ───────────────────
    def run(self):
        """모든 종목에 대해 ML 분석 리포트를 콘솔에 출력합니다."""
        for name, symbol in self.tickers.items():
            try:
                result = self._analyze_stock(name, symbol)
                if not result:
                    continue

                df = result["df"]
                bt_ret = result["bt_ret"]
                sharpe = result["sharpe"]
                mdd = result["mdd"]
                titles = result["titles"]
                news_articles = result["news_articles"]
                preds = result["preds"]
                tabpfn_preds = result["tabpfn_preds"]
                lstm_metrics = result["lstm_metrics"]
                tabpfn_metrics = result["tabpfn_metrics"]

                print(f"{'='*65}")
                print(f"🚀 [AI Agent V3.2 리포트: {name} ({symbol})]")
                print(f"{'='*65}")

                last_p = float(df['Close'].iloc[-1])
                rsi_v  = float(df['RSI'].dropna().iloc[-1])
                trend  = "🔥 과매수" if rsi_v > 70 else ("❄️ 과매도" if rsi_v < 30 else "⚖️ 중립 유지")

                print(f"📊 현재가: {last_p:,.0f} | 추세: {trend} (RSI: {rsi_v:.2f})")
                print(f"📈 전략 수익률: {bt_ret:.2f}% | Sharpe: {sharpe:.2f} | MDD: {mdd:.2f}%")
                if news_articles:
                    for article in news_articles:
                        print(
                            f"      [{article['source']}] "
                            f"{article['content_type']} 분석 | {article['title'][:60]}..."
                        )
                else:
                    for i, title in enumerate(titles[:2]):
                        print(f"      {i+1}. {title[:50]}...")

                # ── LSTM 예측 출력 ──
                print(f"🔮 [LSTM]   예보: [T+1 {result['target_t1']}] {preds[0]:,.2f} | [T+4 {result['target_t4']}] {preds[1]:,.2f} | [T+7 {result['target_t7']}] {preds[2]:,.2f}")
                lstm_test = lstm_metrics["test"]
                print(
                    f"   ✅ 평가 모델 테스트 오차: MAE {lstm_test['mae']:,.2f} | "
                    f"RMSE {lstm_test['rmse']:,.2f} | MAPE {lstm_test['mape']:.2f}% | "
                    f"방향 정확도 {lstm_test['direction_accuracy']:.2f}%"
                )
                lstm_naive = lstm_metrics["naive_baseline"]
                lstm_compare = lstm_metrics["comparison"]
                print(
                    f"   📏 3-fold naive 기준: MAE {lstm_naive['mae']:,.2f} | "
                    f"개선율 {lstm_compare['mae_improvement_pct']:+.2f}% | "
                    f"{'LSTM 우수' if lstm_compare['beats_naive'] else 'naive 우수'}"
                )
                lstm_summary = lstm_metrics["fold_summary"]["model"]
                print(
                    f"   📊 fold 평균±표준편차: "
                    f"MAE {lstm_summary['mae']['mean']:,.2f}±{lstm_summary['mae']['std']:,.2f} | "
                    f"방향 {lstm_summary['direction_accuracy']['mean']:.2f}"
                    f"±{lstm_summary['direction_accuracy']['std']:.2f}%"
                )
                for fold in lstm_metrics["folds"]:
                    print(
                        f"      fold {fold['fold']}: "
                        f"LSTM MAE {fold['test']['mae']:,.2f} / "
                        f"naive MAE {fold['naive_baseline']['mae']:,.2f} | "
                        f"방향 {fold['test']['direction_accuracy']:.2f}%"
                    )

                # ── TabPFN 예측 출력 ──
                if tabpfn_preds:
                    print(f"🔮 [TabPFN] 예보: [T+1 {result['target_t1']}] {tabpfn_preds[0]:,.2f} | [T+4 {result['target_t4']}] {tabpfn_preds[1]:,.2f} | [T+7 {result['target_t7']}] {tabpfn_preds[2]:,.2f}")
                    tabpfn_test = tabpfn_metrics["test"]
                    print(
                        f"   ✅ 평가 모델 테스트 오차: MAE {tabpfn_test['mae']:,.2f} | "
                        f"RMSE {tabpfn_test['rmse']:,.2f} | MAPE {tabpfn_test['mape']:.2f}% | "
                        f"방향 정확도 {tabpfn_test['direction_accuracy']:.2f}%"
                    )
                    tabpfn_naive = tabpfn_metrics["naive_baseline"]
                    tabpfn_compare = tabpfn_metrics["comparison"]
                    print(
                        f"   📏 3-fold naive 기준: MAE {tabpfn_naive['mae']:,.2f} | "
                        f"개선율 {tabpfn_compare['mae_improvement_pct']:+.2f}% | "
                        f"{'TabPFN 우수' if tabpfn_compare['beats_naive'] else 'naive 우수'}"
                    )
                    tabpfn_summary = tabpfn_metrics["fold_summary"]["model"]
                    print(
                        f"   📊 fold 평균±표준편차: "
                        f"MAE {tabpfn_summary['mae']['mean']:,.2f}"
                        f"±{tabpfn_summary['mae']['std']:,.2f} | "
                        f"방향 {tabpfn_summary['direction_accuracy']['mean']:.2f}"
                        f"±{tabpfn_summary['direction_accuracy']['std']:.2f}%"
                    )
                    for fold in tabpfn_metrics["folds"]:
                        print(
                            f"      fold {fold['fold']}: "
                            f"TabPFN MAE {fold['test']['mae']:,.2f} / "
                            f"naive MAE {fold['naive_baseline']['mae']:,.2f} | "
                            f"방향 {fold['test']['direction_accuracy']:.2f}%"
                        )
                    diff1 = tabpfn_preds[0] - preds[0]
                    print(f"   📐 모델 차이(T+1): {diff1:+,.2f} ({'TabPFN 높음' if diff1 > 0 else 'LSTM 높음'})")
                else:
                    print(f"🔮 [TabPFN] 예보: 사용 불가")

                print(f"{'='*65}\n")

            except Exception as e:
                print(f"❌ {name} 분석 실패: {e}")
                traceback.print_exc()

    # ── llm_stock.py 연동용 반환 메서드 ──────
    def run_and_return(self) -> dict:
        """
        분석 결과를 딕셔너리로 반환합니다.
        llm_stock.py 의 StockAIAgentV4 에서 호출됩니다.

        Returns:
            {
                ticker_name: {
                    "df": DataFrame,
                    "symbol": str,
                    "bt_ret": float,
                    "sharpe": float,
                    "mdd": float,
                    "analysis_id": int,
                    "titles": list[str],
                    "preds": list[float],
                    "tabpfn_preds": list[float] | None,
                    "lstm_metrics": dict,
                    "tabpfn_metrics": dict | None,
                    "model": LSTMModel,
                    "l_seq": Tensor,
                    "baseline_seq": Tensor,  # IG 배경 시퀀스 (학습 구간 평균)
                }
            }
        """
        results = {}
        for name, symbol in self.tickers.items():
            try:
                result = self._analyze_stock(name, symbol)
                if result:
                    results[name] = result
            except Exception as e:
                print(f"❌ {name} ML 분석 실패: {e}")
                traceback.print_exc()

        return results


# ──────────────────────────────────────────────
# 단독 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    agent = StockAIAgentV3()
    agent.run()
    from cleanup_history import maintain_history
    maintain_history(agent.db_path)
