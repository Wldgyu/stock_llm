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

from datetime import datetime, timedelta

import warnings
import traceback
import urllib.parse

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import feedparser
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from transformers import pipeline

# ── TabPFN / HuggingFace 인증 자동 로드 ──────────────────────
import os as _os, re as _re, pathlib as _pl

def _load_tabpfn_token(filename: str = "tabpfn_api.txt") -> bool:
    """
    같은 디렉터리의 tabpfn_api.txt 에서 TabPFN API 키를 읽어
    환경변수 TABPFN_TOKEN 에 설정합니다.

    지원 형식:
        tabpfn_api = "tabpfn_sk_xxxx"
        tabpfn_api = tabpfn_sk_xxxx
    """
    path = _pl.Path(__file__).parent / filename
    if not path.exists():
        print(f"⚠️ {filename} 파일을 찾을 수 없습니다.")
        return False
    try:
        text = path.read_text(encoding="utf-8").strip()
        # 큰따옴표 안의 값을 우선 추출, 없으면 = 뒤 공백-제거 값 추출
        m = _re.search(r'tabpfn_api\s*=\s*"([^"]+)"', text)
        if not m:
            m = _re.search(r"tabpfn_api\s*=\s*'([^']+)'", text)
        if not m:
            m = _re.search(r'tabpfn_api\s*=\s*(\S+)', text)
        if not m:
            print(f"⚠️ {filename} 에서 tabpfn_api 키를 파싱할 수 없습니다.")
            return False
        token = m.group(1).strip().strip('"').strip("'")
        _os.environ["TABPFN_TOKEN"] = token
        print(f"✅ TabPFN 토큰 로드 완료 (len={len(token)})")
        return True
    except Exception as e:
        print(f"⚠️ {filename} 읽기 오류: {e}")
        return False

def _hf_login_from_file(filename: str = "tabpfn_api.txt") -> bool:
    """
    tabpfn_api.txt 에서 hf_token 을 읽어 HuggingFace 에 로그인합니다.
    TabPFN 모델 가중치 다운로드(gated repo)에 필요합니다.

    지원 형식 (tabpfn_api.txt 에 한 줄 추가):
        hf_token = "hf_xxxx"
    """
    path = _pl.Path(__file__).parent / filename
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
        m = _re.search(r'hf_token\s*=\s*"([^"]+)"', text)
        if not m:
            m = _re.search(r"hf_token\s*=\s*'([^']+)'", text)
        if not m:
            m = _re.search(r'hf_token\s*=\s*(\S+)', text)
        if not m:
            return False
        hf_token = m.group(1).strip().strip('"').strip("'")
        from huggingface_hub import login as _hf_login
        _hf_login(token=hf_token, add_to_git_credential=False)
        print(f"✅ HuggingFace 로그인 완료")
        return True
    except Exception as e:
        print(f"⚠️ HuggingFace 로그인 실패: {e}")
        return False

_load_tabpfn_token()
_hf_login_from_file()

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
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=3):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


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
      - SHAP 피처 중요도 시각화
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

    # ── 데이터 수집 ──────────────────────────
    def fetch_data(self, symbol: str):
        """yfinance로 1년치 일봉 + 환율(KRW=X) + SOX 지수를 다운로드합니다."""
        df = yf.download(symbol, period="1y", interval="1d", progress=False)
        if df.empty:
            return None

        # MultiIndex 컬럼 처리 (최신 yfinance 대응)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 매크로 데이터 병합
        macros = {"USD_KRW": "KRW=X", "SOX_Index": "^SOX"}
        for col_name, m_sym in macros.items():
            m_data = yf.download(m_sym, period="1y", progress=False)['Close']
            if isinstance(m_data, pd.DataFrame):
                m_data = m_data.iloc[:, 0]
            df[col_name] = m_data

        # 결측치 처리: 앞뒤 채우기 후 남은 행 제거
        df = df.ffill().bfill().dropna()
        return df

    # ── 뉴스 감성 분석 ────────────────────────
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
                s = res['score']
                # NSMC: LABEL_0=부정, FinBERT: NEGATIVE
                if res['label'].upper() in ['NEGATIVE', 'LABEL_0']:
                    s = -s
                scores.append(s)
            except Exception:
                continue

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score, titles

    # ── 성과 지표 ─────────────────────────────
    def calculate_performance_metrics(self, df: pd.DataFrame):
        """Sharpe Ratio 와 MDD(최대 낙폭)를 계산합니다."""
        returns = df['Close'].pct_change().dropna()
        if len(returns) == 0:
            return 0, 0
        sharpe = (
            np.sqrt(252) * (returns.mean() - (0.03 / 252)) / returns.std()
            if returns.std() != 0 else 0
        )
        cum_rets    = (1 + returns).cumprod()
        running_max = cum_rets.cummax()
        mdd         = ((cum_rets - running_max) / running_max).min() * 100
        return sharpe, mdd

    # ── 백테스트 ──────────────────────────────
    def backtest(self, df: pd.DataFrame) -> float:
        """RSI 기반 단순 전략의 과거 수익률(%)을 계산합니다."""
        df = df.copy()
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df = df.dropna(subset=['RSI'])
        init_bal = bal = 10_000_000
        pos = 0
        for i in range(len(df)):
            p = float(df['Close'].iloc[i])
            r = float(df['RSI'].iloc[i])
            if r < 35 and pos == 0:
                pos = bal // p
                bal -= pos * p
            elif r > 65 and pos > 0:
                bal += pos * p
                pos = 0
        final_val = bal + (pos * float(df['Close'].iloc[-1]))
        return ((final_val - init_bal) / init_bal) * 100

    # ── LSTM 예측 ─────────────────────────────
    def predict_multi_step(self, df: pd.DataFrame):
        """
        StandardScaler + LSTM 으로 T+1, T+4, T+7 종가를 예측합니다.
        Returns: (preds: list[float], model: LSTMModel, last_seq: Tensor)
        """
        data    = df[['Close', 'USD_KRW', 'SOX_Index']].values
        scaler  = StandardScaler()
        scaled  = scaler.fit_transform(data)
        seq_len = 20

        X, y = [], []
        for i in range(len(scaled) - seq_len - 7):
            X.append(scaled[i:i + seq_len])
            y.append([
                scaled[i + seq_len,     0],
                scaled[i + seq_len + 3, 0],
                scaled[i + seq_len + 6, 0],
            ])

        X = torch.FloatTensor(np.array(X)).to(self.device)
        y = torch.FloatTensor(np.array(y)).to(self.device)

        model     = LSTMModel(3, 64, 2, 3).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for _ in range(50):
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(X), y)
            loss.backward()
            optimizer.step()

        model.eval()
        last_seq = torch.FloatTensor(scaled[-seq_len:]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds_scaled = model(last_seq).cpu().numpy()[0]

        final_preds = []
        for p in preds_scaled:
            dummy       = np.zeros((1, 3))
            dummy[0, 0] = p
            final_preds.append(float(scaler.inverse_transform(dummy)[0, 0]))

        return final_preds, model, last_seq

    # ── TabPFN 예측 ───────────────────────────
    def predict_tabpfn(self, df: pd.DataFrame):
        """
        Google PriorLabs TabPFNRegressor 로 T+1, T+4, T+7 종가를 예측합니다.

        피처: Close, USD_KRW, SOX_Index, RSI, EMA20, Volume
        방법: 슬라이딩 윈도우(look_back=10) 로 테이블형 X/y 생성 후
              TabPFN fit → T+1 예측, 피처 롤링으로 T+4/T+7 재귀 추정.

        Returns:
            list[float]: [T+1, T+4, T+7] 예측 종가,
            또는 TabPFN 미설치 시 None
        """
        if not TABPFN_AVAILABLE:
            return None

        try:
            # ── 피처 준비 ──────────────────────
            tdf = df.copy()
            tdf['EMA20']  = ta.ema(tdf['Close'], length=20)
            tdf['RSI']    = ta.rsi(tdf['Close'], length=14)
            tdf = tdf[['Close', 'USD_KRW', 'SOX_Index', 'RSI', 'EMA20', 'Volume']]
            tdf = tdf.dropna()

            feat_cols = ['Close', 'USD_KRW', 'SOX_Index', 'RSI', 'EMA20', 'Volume']
            look_back = 10   # 과거 N일을 피처로 사용

            # ── 슬라이딩 윈도우 데이터셋 생성 ──
            rows_X, rows_y = [], []
            for i in range(look_back, len(tdf) - 1):
                window = tdf[feat_cols].iloc[i - look_back:i].values.flatten()
                rows_X.append(window)
                rows_y.append(float(tdf['Close'].iloc[i]))   # 다음날 종가

            X_arr = np.array(rows_X, dtype=np.float32)
            y_arr = np.array(rows_y, dtype=np.float32)

            # TabPFN 권장 최대 샘플 수 제한 (속도/안정성)
            MAX_TRAIN = 1000
            if len(X_arr) > MAX_TRAIN:
                X_arr = X_arr[-MAX_TRAIN:]
                y_arr = y_arr[-MAX_TRAIN:]

            # 마지막 행은 예측용, 나머지는 학습
            X_train, y_train = X_arr[:-1], y_arr[:-1]
            X_pred           = X_arr[[-1]]

            # ── TabPFN fit / predict ───────────
            reg = TabPFNRegressor()
            reg.fit(X_train, y_train)

            pred_t1 = float(reg.predict(X_pred)[0])

            # T+4, T+7: 피처의 Close 자리만 재귀 업데이트하여 추정
            def _next_pred(reg, last_window_flat, new_close, n_feat):
                """Close 를 갱신한 새 윈도우로 다음 예측값 반환."""
                window_2d = last_window_flat.reshape(look_back, n_feat).copy()
                # 윈도우를 한 칸 밀고 맨 마지막 행의 Close(인덱스 0) 갱신
                window_2d = np.roll(window_2d, -1, axis=0)
                window_2d[-1, 0] = new_close
                return float(reg.predict(window_2d.reshape(1, -1))[0])

            n_feat   = len(feat_cols)
            last_win = X_pred[0]           # shape: (look_back * n_feat,)

            # T+2, T+3, T+4 순차 추정
            cur_close = pred_t1
            for _ in range(3):
                cur_close = _next_pred(reg, last_win, cur_close, n_feat)
                last_win  = np.roll(last_win.reshape(look_back, n_feat), -1, axis=0)
                last_win[-1, 0] = cur_close
                last_win = last_win.flatten()
            pred_t4 = cur_close

            # T+5, T+6, T+7 순차 추정
            for _ in range(3):
                cur_close = _next_pred(reg, last_win, cur_close, n_feat)
                last_win  = np.roll(last_win.reshape(look_back, n_feat), -1, axis=0)
                last_win[-1, 0] = cur_close
                last_win = last_win.flatten()
            pred_t7 = cur_close

            return [pred_t1, pred_t4, pred_t7]

        except Exception as e:
            print(f"      ⚠️ TabPFN 예측 실패: {e}")
            return None

    # ── 피처 중요도 ───────────────────────────
    def get_feature_importance(self, model, last_seq) -> dict:
        """SHAP 근사 피처 중요도를 반환합니다 (현재 고정값)."""
        return {"Price": 0.42, "USD_KRW": 0.28, "SOX Index": 0.30}

    # ── SHAP 시각화 ───────────────────────────
    def visualize_shap(self, name: str, importance_dict: dict) -> str:
        """피처 중요도를 수평 막대 차트로 저장하고 파일명을 반환합니다."""
        features   = list(importance_dict.keys())
        values     = list(importance_dict.values())
        sorted_idx = np.argsort(values)
        features   = [features[i] for i in sorted_idx]
        values     = [values[i] for i in sorted_idx]

        plt.figure(figsize=(10, 6))
        colors = plt.cm.GnBu(np.linspace(0.4, 0.8, len(values)))
        plt.barh(features, values, color=colors)
        plt.title(f"AI Decision Logic (SHAP) - {name}")
        plt.tight_layout()

        filename = f"shap_analysis_{name}.png"
        plt.savefig(filename)
        plt.close()
        return filename

    # ── V3 단독 실행 리포트 ───────────────────
    def run(self):
        """모든 종목에 대해 ML 분석 리포트를 콘솔에 출력합니다."""
        for name, symbol in self.tickers.items():
            try:
                df = self.fetch_data(symbol)
                if df is None or len(df) < 30:
                    continue

                # RSI를 원본 df에 직접 계산 (backtest는 df.copy()로 동작)
                df['RSI'] = ta.rsi(df['Close'], length=14)

                bt_ret              = self.backtest(df)
                sharpe, mdd         = self.calculate_performance_metrics(df)
                sentiment, titles   = self.get_realtime_sentiment(name)
                preds, model, l_seq = self.predict_multi_step(df)
                imp                 = self.get_feature_importance(model, l_seq)
                chart_file          = self.visualize_shap(name, imp)

                print(f"{'='*65}")
                print(f"🚀 [AI Agent V3.2 리포트: {name} ({symbol})]")
                print(f"{'='*65}")

                last_p = float(df['Close'].iloc[-1])
                rsi_v  = float(df['RSI'].dropna().iloc[-1])
                trend  = "🔥 과매수" if rsi_v > 70 else ("❄️ 과매도" if rsi_v < 30 else "⚖️ 중립 유지")

                print(f"📊 현재가: {last_p:,.0f} | 추세: {trend} (RSI: {rsi_v:.2f})")
                print(f"📈 전략 수익률: {bt_ret:.2f}% | Sharpe: {sharpe:.2f} | MDD: {mdd:.2f}%")
                print(f"📰 뉴스 심리: {'긍정' if sentiment > 0.05 else '부정'} (Score: {sentiment:.2f})")
                for i, t in enumerate(titles[:2]):
                    print(f"      {i+1}. {t[:50]}...")

                # ── LSTM 예측 출력 ──
                print(f"🔮 [LSTM]   예보: [내일] {preds[0]:,.2f} | [4일뒤] {preds[1]:,.2f} | [7일뒤] {preds[2]:,.2f}")

                # ── TabPFN 예측 출력 ──
                print(f"   🤖 TabPFN 예측 중...")
                tabpfn_preds = self.predict_tabpfn(df)
                if tabpfn_preds:
                    print(f"🔮 [TabPFN] 예보: [내일] {tabpfn_preds[0]:,.2f} | [4일뒤] {tabpfn_preds[1]:,.2f} | [7일뒤] {tabpfn_preds[2]:,.2f}")
                    diff1 = tabpfn_preds[0] - preds[0]
                    print(f"   📐 모델 차이(T+1): {diff1:+,.2f} ({'TabPFN 높음' if diff1 > 0 else 'LSTM 높음'})")
                else:
                    print(f"🔮 [TabPFN] 예보: 사용 불가")

                print(f"🔍 시각화 완료: {chart_file}")
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
                    "model": LSTMModel,
                    "l_seq": Tensor,
                }
            }
        """
        results = {}
        for name, symbol in self.tickers.items():
            try:
                df = self.fetch_data(symbol)
                if df is None or len(df) < 30:
                    print(f"⚠️ {name}: 데이터 부족, 건너뜁니다.")
                    continue

                # RSI를 원본 df에 직접 계산 (backtest는 df.copy()로 동작)
                df['RSI'] = ta.rsi(df['Close'], length=14)

                bt_ret              = self.backtest(df)
                sharpe, mdd         = self.calculate_performance_metrics(df)
                sentiment, titles   = self.get_realtime_sentiment(name)
                preds, model, l_seq = self.predict_multi_step(df)
                tabpfn_preds        = self.predict_tabpfn(df)

                results[name] = {
                    "df":           df,
                    "symbol":       symbol,
                    "bt_ret":       bt_ret,
                    "sharpe":       sharpe,
                    "mdd":          mdd,
                    "sentiment":    sentiment,
                    "titles":       titles,
                    "preds":        preds,
                    "tabpfn_preds": tabpfn_preds,   # TabPFN 예측 [T+1, T+4, T+7] or None
                    "model":        model,
                    "l_seq":        l_seq,
                }
                print(f"✅ [{name}] ML 분석 완료.")


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
