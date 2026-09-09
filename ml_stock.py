# -*- coding: utf-8 -*-
"""
ml_stock.py
───────────────────────────────────────────────────────────────────
[역할] 머신러닝 학습 및 분석 엔진
  - LSTMModel       : PyTorch 기반 LSTM 가격 예측 모델
  - StockAIAgentV3  : 데이터 수집 / 감성분석 / 백테스트 / 예측 / 시각화
  - run_and_return(): llm_stock.py 에서 import 하여 결과를 딕셔너리로 받음

[단독 실행 시] python ml_stock.py
"""

from datetime import datetime, time as clock_time, timedelta
from zoneinfo import ZoneInfo

import warnings
import traceback
import urllib.parse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import feedparser
import matplotlib.pyplot as plt

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
from transformers import pipeline

from database import DB_PATH, connect_db

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
def get_next_trading_date():
    """오늘 요일을 기준으로 다음 거래일(날짜, 상태 메시지)을 반환합니다."""
    today = datetime.now()
    weekday = today.weekday()  # 0:월 ~ 6:일

    if weekday == 4:   # 금요일 → 다음 월요일
        next_date = today + timedelta(days=3)
        status = "💤 주말 휴장 전 (월요일 예측)"
    elif weekday == 5: # 토요일 → 다음 월요일
        next_date = today + timedelta(days=2)
        status = "🏖️ 주말 휴장 중 (월요일 예측)"
    elif weekday == 6: # 일요일 → 다음 월요일
        next_date = today + timedelta(days=1)
        status = "🌙 휴장 마지막 날 (내일 예측)"
    else:              # 평일 → 내일
        next_date = today + timedelta(days=1)
        status = "🔔 시장 가동 중 (내일 예측)"

    return next_date.strftime('%Y-%m-%d'), status


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
      - FinBERT / KoELECTRA 뉴스 감성 분석
      - RSI 기반 백테스트
      - Sharpe Ratio / MDD 성과 지표
      - LSTM 3-step 가격 예측 (T+1, T+4, T+7)
      - Integrated Gradients 피처 기여도 시각화
    """

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

        # 영문 감성 분석 모델 (FinBERT)
        self.en_pipe = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=self.device
        )
        # 한국어 감성 분석 모델 (KoELECTRA)
        self.ko_pipe = pipeline(
            "sentiment-analysis",
            model="daekeun-ml/koelectra-small-v3-nsmc",
            device=self.device
        )

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
                    risk_score  INTEGER
                )
            """)
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(analysis_log)")
            }
            # 기존 DB에도 모델별 개별 예측 컬럼을 자동으로 추가합니다.
            for column in (
                "lstm_t1", "lstm_t4", "lstm_t7",
                "tabpfn_t1", "tabpfn_t4", "tabpfn_t7",
            ):
                if column not in columns:
                    conn.execute(f"ALTER TABLE analysis_log ADD COLUMN {column} REAL")
            conn.commit()
        finally:
            conn.close()

    def _save_analysis_result(self, name: str, result: dict):
        """한 번 계산한 분석 결과를 중복 계산 없이 DB에 저장합니다."""
        df = result["df"]
        rsi = float(df["RSI"].dropna().iloc[-1])
        trend = "과매수" if rsi > 70 else ("과매도" if rsi < 30 else "중립")
        sentiment = float(result["sentiment"])
        sent_label = "긍정" if sentiment > 0.05 else ("부정" if sentiment < -0.05 else "중립")
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
            sentiment,
            sent_label,
            min(predictions),
            max(predictions),
            predictions[0],
            predictions[1],
            predictions[2],
            tabpfn_predictions[0],
            tabpfn_predictions[1],
            tabpfn_predictions[2],
            risk_score,
        )
        conn = connect_db(self.db_path)
        try:
            conn.execute("""
                INSERT INTO analysis_log (
                    timestamp, name, symbol, last_close, rsi, trend,
                    support, resistance, usd_krw, sox, sentiment,
                    sent_label, pred_low, pred_high,
                    lstm_t1, lstm_t4, lstm_t7,
                    tabpfn_t1, tabpfn_t4, tabpfn_t7, risk_score
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )
            """, values)
            conn.commit()
        finally:
            conn.close()

    def _analyze_stock(self, name: str, symbol: str):
        """단일 종목의 모든 ML 계산을 한 번 수행하고 저장합니다."""
        df = self.fetch_data(symbol)
        if df is None or len(df) < 30:
            print(f"⚠️ {name}: 데이터 부족, 건너뜁니다.")
            return None

        df["RSI"] = ta.rsi(df["Close"], length=14)
        bt_ret, equity = self.backtest(df)
        sharpe, mdd    = self.calculate_performance_metrics(equity)
        sentiment, titles = self.get_realtime_sentiment(name)
        news_articles = self.get_financial_news_articles(name, symbol)
        preds, model, last_seq, baseline_seq, lstm_metrics = self.predict_multi_step(df)
        tabpfn_preds, tabpfn_metrics = self.predict_tabpfn(df)

        result = {
            "df": df,
            "symbol": symbol,
            "bt_ret": bt_ret,
            "sharpe": sharpe,
            "mdd": mdd,
            "sentiment": sentiment,
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
        if df.empty:
            return df

        is_korean = symbol.endswith((".KS", ".KQ"))
        timezone = ZoneInfo("Asia/Seoul" if is_korean else "America/New_York")
        market_close = clock_time(15, 30) if is_korean else clock_time(16, 0)
        market_now = datetime.now(timezone)
        latest_date = pd.Timestamp(df.index[-1]).date()

        # 오늘 날짜의 행은 정규장 종료 전에는 종가가 확정되지 않았으므로 제외합니다.
        local_time = market_now.time().replace(tzinfo=None)
        if latest_date == market_now.date() and local_time < market_close:
            print(
                f"      ℹ️ {symbol} 미완성 당일 일봉({latest_date})을 제외합니다."
            )
            return df.iloc[:-1].copy()
        return df

    @staticmethod
    def _macro_lags_for_symbol(symbol: str) -> dict:
        """각 시장 마감 시점에 실제로 확정된 거시지표의 지연일을 반환합니다."""
        is_korean = symbol.endswith((".KS", ".KQ"))
        return {
            "USD_KRW": 1,
            "SOX_Index": 1 if is_korean else 0,
        }

    def fetch_data(self, symbol: str):
        """yfinance로 1년치 일봉 + 환율(KRW=X) + SOX 지수를 다운로드합니다."""
        df = yf.download(
            symbol,
            period="1y",
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
        # 한국장은 미국장·FX 일봉보다 먼저 닫히므로 둘 다 전 거래일 값을 사용합니다.
        # 미국장은 SOX와 동시에 닫혀 SOX 당일값은 사용하고, 늦게 마감되는 FX만 지연합니다.
        macro_lags = self._macro_lags_for_symbol(symbol)
        for col_name, m_sym in macros.items():
            m_data = yf.download(
                m_sym,
                period="1y",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )['Close']
            if isinstance(m_data, pd.DataFrame):
                m_data = m_data.iloc[:, 0]
            lag = macro_lags[col_name]
            if lag:
                # 값의 행을 미는 대신 사용 가능 날짜를 옮겨 서로 다른 시장 휴장일도 처리합니다.
                m_data.index = m_data.index + pd.Timedelta(days=lag)
            df[col_name] = m_data

        # 과거 값으로만 채워 미래 데이터가 과거 행에 들어가는 누수를 막습니다.
        df = df.ffill().dropna()
        return df

    # ── 뉴스 감성 분석 ────────────────────────
    @staticmethod
    def _normalize_sentiment_score(label: str, confidence: float) -> float:
        """모델 라벨을 부정(-), 중립(0), 긍정(+) 점수로 변환합니다."""
        normalized_label = str(label).upper()
        if normalized_label in {"NEGATIVE", "LABEL_0"}:
            return -float(confidence)
        if normalized_label == "NEUTRAL":
            return 0.0
        if normalized_label in {"POSITIVE", "LABEL_1"}:
            return float(confidence)
        return 0.0

    def get_realtime_sentiment(self, name: str):
        """Google News RSS에서 헤드라인을 수집하고 감성 점수를 반환합니다."""
        encoded_query = urllib.parse.quote(name)
        is_ko = any(ord(c) > 127 for c in name)
        locale = 'ko&gl=KR&ceid=KR:ko' if is_ko else 'en-US&gl=US&ceid=US:en'
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl={locale}"
        rss = feedparser.parse(url)
        titles = [e.title for e in rss.entries[:5]]
        pipe = self.ko_pipe if is_ko else self.en_pipe

        scores = []
        for t in titles:
            try:
                res = pipe(t[:512])[0]
                scores.append(
                    self._normalize_sentiment_score(
                        res.get("label", ""), res.get("score", 0.0)
                    )
                )
            except Exception:
                continue

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score, titles

    def get_financial_news_articles(self, name: str, symbol: str):
        """지정 금융 언론사별 최신 기사 한 건의 접근 가능한 본문을 수집합니다."""
        sources = {
            "Bloomberg": ("Bloomberg",),
            "Wall Street Journal": ("WSJ", "The Wall Street Journal", "Wall Street Journal"),
            "Financial Times": ("Financial Times",),
            "Reuters": ("Reuters",),
        }
        query_names = {
            "삼성전자": "Samsung Electronics",
            "엔비디아": "NVIDIA",
            "인텔": "Intel",
        }
        company_query = query_names.get(name, symbol.split(".")[0])
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
            entry = next(
                (
                    item
                    for item in rss.entries
                    if getattr(getattr(item, "source", {}), "title", "")
                    in accepted_names
                ),
                None,
            )
            if entry is None:
                continue

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
    def _regression_metrics(y_true, y_pred):
        """실제값과 예측값의 평균 오차를 계산합니다."""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        def calculate(actual, predicted):
            error = predicted - actual
            safe_actual = np.where(np.abs(actual) < 1e-8, 1e-8, np.abs(actual))
            return {
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "mape": float(np.mean(np.abs(error) / safe_actual) * 100),
            }

        result = calculate(y_true, y_pred)
        if y_true.ndim == 2 and y_true.shape[1] == 3:
            # 각 예측 시점의 성능도 따로 확인할 수 있게 보관합니다.
            result["by_horizon"] = {
                label: calculate(y_true[:, index], y_pred[:, index])
                for index, label in enumerate(("T+1", "T+4", "T+7"))
            }
        return result

    # max_epochs=200: 조기 종료가 없을 때 실행할 최대 학습 횟수입니다.
    def _train_lstm(self, X_train, y_train, X_val=None, y_val=None, max_epochs=200):
        """LSTM을 학습하고 검증 손실이 개선되지 않으면 일찍 종료합니다."""
        # ── 직접 조정하기 쉬운 LSTM 학습 설정 ──
        learning_rate = 0.005  # 학습률: 불안정하면 낮추고, 너무 느리면 조금 높입니다.
        batch_size = 16        # 배치 크기: 작을수록 세밀하지만 학습 시간이 늘어납니다.
        dropout_rate = 0.2     # 과적합 조절: 과적합이 크면 값을 조금 높입니다.
        huber_delta = 1.0      # 이상치 민감도: 작을수록 급등락의 영향을 덜 받습니다.
        gradient_clip = 1.0    # 기울기 제한: 학습 중 값이 폭주하는 것을 막습니다.
        patience = 12          # 조기 종료: 검증 손실 개선을 기다리는 epoch 수입니다.

        model = LSTMModel(3, 64, 2, 3, dropout=dropout_rate).to(self.device)
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
        시간순 70/15/15 분리 후 LSTM의 T+1, T+4, T+7 성능을 검증합니다.
        검증이 끝나면 전체 데이터로 최종 모델을 다시 학습해 미래를 예측합니다.
        """
        np.random.seed(42)
        torch.manual_seed(42)

        data = df[['Close', 'USD_KRW', 'SOX_Index']].values.astype(np.float32)
        seq_len = 20
        train_end = int(len(data) * 0.70)
        val_end = int(len(data) * 0.85)
        if train_end <= seq_len + 7 or len(data) - val_end < 8:
            raise ValueError("LSTM 학습/검증/테스트 분리에 필요한 데이터가 부족합니다.")

        # 데이터 누수를 막기 위해 평가용 스케일러는 학습 구간에만 맞춥니다.
        eval_scaler = StandardScaler()
        eval_scaler.fit(data[:train_end])
        scaled = eval_scaler.transform(data)

        split_X = {"train": [], "val": [], "test": []}
        split_y = {"train": [], "val": [], "test": []}
        for start in range(len(scaled) - seq_len - 6):
            target_first = start + seq_len
            target_last = target_first + 6
            if target_last < train_end:
                split = "train"
            elif target_first >= train_end and target_last < val_end:
                split = "val"
            elif target_first >= val_end:
                split = "test"
            else:
                # 두 구간의 경계에 걸친 정답은 평가가 섞이지 않도록 제외합니다.
                continue
            split_X[split].append(scaled[start:start + seq_len])
            split_y[split].append([
                scaled[target_first, 0],
                scaled[target_first + 3, 0],
                scaled[target_first + 6, 0],
            ])

        if any(not split_X[name] for name in ("train", "val", "test")):
            raise ValueError("LSTM 분할 후 비어 있는 데이터 구간이 있습니다.")

        tensors = {}
        for split in ("train", "val", "test"):
            tensors[f"X_{split}"] = torch.tensor(
                np.asarray(split_X[split]), dtype=torch.float32, device=self.device
            )
            tensors[f"y_{split}"] = torch.tensor(
                np.asarray(split_y[split]), dtype=torch.float32, device=self.device
            )

        # 학습 70%와 검증 15%로 적절한 epoch 수를 선택합니다.
        eval_model, best_epoch = self._train_lstm(
            tensors["X_train"],
            tensors["y_train"],
            tensors["X_val"],
            tensors["y_val"],
        )
        eval_model.eval()
        with torch.no_grad():
            val_pred_scaled = eval_model(tensors["X_val"]).cpu().numpy()
            test_pred_scaled = eval_model(tensors["X_test"]).cpu().numpy()
        val_true_scaled = tensors["y_val"].cpu().numpy()
        test_true_scaled = tensors["y_test"].cpu().numpy()

        # 정규화된 종가를 실제 가격 단위로 되돌려 오차를 계산합니다.
        close_scale = eval_scaler.scale_[0]
        close_mean = eval_scaler.mean_[0]
        to_price = lambda values: (values * close_scale) + close_mean
        metrics = {
            "validation": self._regression_metrics(
                to_price(val_true_scaled), to_price(val_pred_scaled)
            ),
            "test": self._regression_metrics(
                to_price(test_true_scaled), to_price(test_pred_scaled)
            ),
            "best_epoch": best_epoch,
            "split_samples": {
                split: len(split_X[split])
                for split in ("train", "val", "test")
            },
        }

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
        시간순 70/15/15 분리로 TabPFN의 성능을 검증합니다.
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

            train_end = int(len(tdf) * 0.70)
            val_end = int(len(tdf) * 0.85)
            split_X = {"train": [], "val": [], "test": []}
            split_y = {"train": [], "val": [], "test": []}
            all_X, all_y = [], []

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

                target_last = i + 6
                if target_last < train_end:
                    split = "train"
                elif i >= train_end and target_last < val_end:
                    split = "val"
                elif i >= val_end:
                    split = "test"
                else:
                    # T+7 정답이 다음 구간에 걸치면 누수를 막기 위해 제외합니다.
                    continue
                split_X[split].append(window)
                split_y[split].append(targets)

            if any(not split_X[name] for name in ("train", "val", "test")):
                raise ValueError("TabPFN 분할 후 비어 있는 데이터 구간이 있습니다.")

            # TabPFN 권장 한도보다 많으면 각 구간의 최신 샘플을 사용합니다.
            max_samples = 1000
            arrays = {}
            for split in ("train", "val", "test"):
                arrays[f"X_{split}"] = np.asarray(
                    split_X[split][-max_samples:], dtype=np.float32
                )
                arrays[f"y_{split}"] = np.asarray(
                    split_y[split][-max_samples:], dtype=np.float32
                )
            X_train, y_train = arrays["X_train"], arrays["y_train"]
            X_val, y_val = arrays["X_val"], arrays["y_val"]
            X_test, y_test = arrays["X_test"], arrays["y_test"]

            # 평가 모델은 학습 구간만 보고 검증·테스트 구간을 예측합니다.
            val_predictions = np.zeros_like(y_val)
            test_predictions = np.zeros_like(y_test)
            for horizon in range(3):
                reg = TabPFNRegressor()
                reg.fit(X_train, y_train[:, horizon])
                val_predictions[:, horizon] = reg.predict(X_val)
                test_predictions[:, horizon] = reg.predict(X_test)

            metrics = {
                "validation": self._regression_metrics(y_val, val_predictions),
                "test": self._regression_metrics(y_test, test_predictions),
                "split_samples": {
                    "train": len(X_train),
                    "val": len(X_val),
                    "test": len(X_test),
                },
            }

            # 최종 예측 모델은 검증이 끝난 뒤 전체 과거 데이터를 사용합니다.
            latest_window = (
                tdf[feat_cols].iloc[-look_back:].values
                .astype(np.float32)
                .reshape(1, -1)
            )
            X_arr = np.asarray(all_X[-max_samples:], dtype=np.float32)
            y_arr = np.asarray(all_y[-max_samples:], dtype=np.float32)
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

            # 20일 축 합산 → 피처별 기여도 (n_feat,)
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

    # ── 기여도 시각화 ─────────────────────────
    def visualize_feature_importance(
        self, name: str, importance_dict: dict
    ) -> str | None:
        """피처 기여도를 수평 막대 차트로 저장하고 파일명을 반환합니다."""
        if not importance_dict:
            return None
        features   = list(importance_dict.keys())
        values     = list(importance_dict.values())
        sorted_idx = np.argsort(values)
        features   = [features[i] for i in sorted_idx]
        values     = [values[i] for i in sorted_idx]

        plt.figure(figsize=(10, 6))
        colors = plt.cm.GnBu(np.linspace(0.4, 0.8, len(values)))
        plt.barh(features, values, color=colors)
        plt.title(f"이번 예측 기여도 - {name}")
        plt.xlabel("정규화된 기여도 (Integrated Gradients)")
        plt.tight_layout()

        filename = f"contribution_{name}.png"
        plt.savefig(filename)
        plt.close()
        return filename

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
                sentiment = result["sentiment"]
                titles = result["titles"]
                news_articles = result["news_articles"]
                preds = result["preds"]
                tabpfn_preds = result["tabpfn_preds"]
                lstm_metrics = result["lstm_metrics"]
                tabpfn_metrics = result["tabpfn_metrics"]
                imp = self.get_feature_importance(
                    result["model"], result["l_seq"], result.get("baseline_seq")
                )
                chart_file = self.visualize_feature_importance(name, imp)

                print(f"{'='*65}")
                print(f"🚀 [AI Agent V3.2 리포트: {name} ({symbol})]")
                print(f"{'='*65}")

                last_p = float(df['Close'].iloc[-1])
                rsi_v  = float(df['RSI'].dropna().iloc[-1])
                trend  = "🔥 과매수" if rsi_v > 70 else ("❄️ 과매도" if rsi_v < 30 else "⚖️ 중립 유지")

                print(f"📊 현재가: {last_p:,.0f} | 추세: {trend} (RSI: {rsi_v:.2f})")
                print(f"📈 전략 수익률: {bt_ret:.2f}% | Sharpe: {sharpe:.2f} | MDD: {mdd:.2f}%")
                sentiment_label = (
                    "긍정" if sentiment > 0.05
                    else "부정" if sentiment < -0.05
                    else "중립"
                )
                print(f"📰 뉴스 심리: {sentiment_label} (Score: {sentiment:.2f})")
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
                print(f"🔮 [LSTM]   예보: [내일] {preds[0]:,.2f} | [4일뒤] {preds[1]:,.2f} | [7일뒤] {preds[2]:,.2f}")
                lstm_test = lstm_metrics["test"]
                print(
                    f"   ✅ 평가 모델 테스트 오차: MAE {lstm_test['mae']:,.2f} | "
                    f"RMSE {lstm_test['rmse']:,.2f} | MAPE {lstm_test['mape']:.2f}%"
                )

                # ── TabPFN 예측 출력 ──
                if tabpfn_preds:
                    print(f"🔮 [TabPFN] 예보: [내일] {tabpfn_preds[0]:,.2f} | [4일뒤] {tabpfn_preds[1]:,.2f} | [7일뒤] {tabpfn_preds[2]:,.2f}")
                    tabpfn_test = tabpfn_metrics["test"]
                    print(
                        f"   ✅ 평가 모델 테스트 오차: MAE {tabpfn_test['mae']:,.2f} | "
                        f"RMSE {tabpfn_test['rmse']:,.2f} | MAPE {tabpfn_test['mape']:.2f}%"
                    )
                    diff1 = tabpfn_preds[0] - preds[0]
                    print(f"   📐 모델 차이(T+1): {diff1:+,.2f} ({'TabPFN 높음' if diff1 > 0 else 'LSTM 높음'})")
                else:
                    print(f"🔮 [TabPFN] 예보: 사용 불가")

                if chart_file:
                    print(f"🔍 실제 피처 기여도 시각화 완료: {chart_file}")
                else:
                    print("⚠️ 피처 기여도 계산 실패로 시각화를 건너뜁니다.")
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
                    "sentiment": float,
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
