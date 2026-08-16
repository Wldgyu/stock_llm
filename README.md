# 📈 Stock AI Portfolio — 주식 AI 분석 시스템

> LSTM 딥러닝 + LLM 페르소나 토론을 결합한 주식 분석 자동화 파이프라인

---

## 🗂️ 프로젝트 구조

```
stock포폴/
│
├── ml_stock.py       ← [1단계] 머신러닝 학습 & 분석 엔진
├── llm_stock.py      ← [2단계] LLM 페르소나 토론 엔진
├── final_stock.py    ← [3단계] 최종 결과 조회 & 리포트 출력
│
├── stock_analysis.db ← SQLite DB (자동 생성)
├── shap_analysis_*.png ← SHAP 시각화 이미지 (자동 생성)
└── ver1_1.py         ← 원본 통합 파일 (참조용)
```

---

## ⚙️ 실행 순서

```bash
# 1단계: ML 분석 (LSTM 학습 + 백테스트 + 감성분석)
python ml_stock.py

# 2단계: LLM 페르소나 토론 (Ollama 서버 필요)
python llm_stock.py

# 3단계: 최종 결과 확인
python final_stock.py
```

> **주의**: `llm_stock.py` 실행 전에 Ollama 서버(`gemma4:12b`)가 구동 중이어야 합니다.

---

## 📁 파일별 상세 설명

---

### 1️⃣ `ml_stock.py` — 머신러닝 분석 엔진

**역할**: 주가 데이터 수집부터 LSTM 가격 예측까지 ML 파이프라인 전체를 담당합니다.

#### 주요 클래스 & 함수

| 이름 | 유형 | 설명 |
|------|------|------|
| `get_next_trading_date()` | 함수 | 오늘 요일 기준으로 다음 거래일과 시장 상태를 반환 |
| `LSTMModel` | 클래스 | T+1 / T+4 / T+7 종가 예측용 PyTorch LSTM 모델 |
| `StockAIAgentV3` | 클래스 | 전체 ML 분석 파이프라인 |
| `.fetch_data()` | 메서드 | yfinance로 1년치 일봉 + 환율(KRW=X) + SOX 지수 수집 |
| `.get_realtime_sentiment()` | 메서드 | Google News RSS 수집 → FinBERT / KoELECTRA 감성 분석 |
| `.backtest()` | 메서드 | RSI 기반 단순 전략 과거 수익률 계산 |
| `.calculate_performance_metrics()` | 메서드 | Sharpe Ratio, MDD(최대 낙폭) 계산 |
| `.predict_multi_step()` | 메서드 | StandardScaler + LSTM으로 3-step 가격 예측 |
| `.visualize_shap()` | 메서드 | 피처 중요도 수평 막대 차트 저장 |
| `.run()` | 메서드 | 단독 실행용 — V3 분석 리포트 콘솔 출력 |
| `.run_and_return()` | 메서드 | llm_stock.py 연동용 — 분석 결과를 딕셔너리로 반환 |

#### 분석 대상 종목 (기본값)

```python
self.tickers = {
    "삼성전자": "005930.KS",
    "엔비디아": "NVDA",
    "인텔":    "INTC",
}
```

> `ml_stock.py` 내부의 `self.tickers` 딕셔너리를 수정하여 종목을 추가/변경할 수 있습니다.

#### 단독 실행 출력 예시

```
=================================================================
🚀 [AI Agent V3.2 리포트: 엔비디아 (NVDA)]
=================================================================
📊 현재가: 131,520 | 추세: ⚖️ 중립 유지 (RSI: 54.32)
📈 전략 수익률: 18.42% | Sharpe: 1.23 | MDD: -22.51%
📰 뉴스 심리: 긍정 (Score: 0.41)
      1. Nvidia's AI chip dominance continues in Q2...
🔮 AI 가격 예보: [내일] 132,100 | [4일뒤] 133,800 | [7일뒤] 135,200
🔍 시각화 완료: shap_analysis_엔비디아.png
=================================================================
```

---

### 2️⃣ `llm_stock.py` — LLM 페르소나 토론 엔진

**역할**: `ml_stock.py`의 분석 결과를 입력으로 받아, Ollama 서버에 3인의 AI 페르소나를 순차 호출하여 릴레이 토론 방식으로 최종 투자 의사결정 점수를 생성합니다.

#### 주요 클래스

| 이름 | 유형 | 설명 |
|------|------|------|
| `MarketAgentSimulator` | 클래스 | Ollama API 통신 및 3인 페르소나 관리 |
| `StockAIAgentV4` | 클래스 | `StockAIAgentV3` 상속 + LLM 릴레이 토론 통합 |

#### 3인 페르소나 구성

```
┌──────────────────────────────────────────────────────────┐
│  Step 1: 주식전문가                                       │
│    → 현재가·RSI·LSTM 예측가 기반 가격 분석               │
│    → 점수 출력: -1.0 (매도) ~ +1.0 (매수)                │
├──────────────────────────────────────────────────────────┤
│  Step 2: 뉴스기업전문가                                   │
│    → 최신 뉴스 3개 + Sharpe·MDD·백테스트 기반 질적 분석  │
│    → 점수 출력: -1.0 (비관) ~ +1.0 (낙관)                │
├──────────────────────────────────────────────────────────┤
│  Step 3: 최종결정자                                       │
│    → 두 전문가 의견 수신 → 최종 투자 방향 결정            │
│    → 최종 합의 점수: -1.0 ~ +1.0                          │
└──────────────────────────────────────────────────────────┘
```

#### Ollama 서버 설정

`llm_stock.py` 하단 `__main__` 블록에서 IP를 설정합니다:

```python
MY_MAIN_SERVER_IP = 본체 ip 서버 주소 ipv4 # ← 본체 서버 IP 입력
# 같은 PC라면: "127.0.0.1"
```

#### DB 저장 테이블

| 테이블명 | 저장 내용 |
|----------|----------|
| `persona_discussion_log` | 페르소나별 의견 전문(opinion) + 점수(score) |

#### 단독 실행 출력 예시

```
─────────────────────────────────────────────────────
🗨️ [엔비디아] 전문 에이전트 릴레이 의사결정 시작
─────────────────────────────────────────────────────
   ▶️ [호출] 주식전문가 에이전트 요청 송신 중...
      💾 [주식전문가] 분석 스냅샷 SQLite 저장 완료.
   ▶️ [호출] 뉴스기업전문가 에이전트 요청 송신 중...
      💾 [뉴스기업전문가] 분석 스냅샷 SQLite 저장 완료.
   ▶️ [호출] 최종결정자 에이전트 요청 송신 중...

=================================================================
🚀 [AI Agent V4.5 리포트: 엔비디아]
=================================================================
📊 현재가: $131,520.00 | LSTM 예측: $132,100.00

🎭 [AI 에이전트 토론 및 최종 결정] (최종 의사결정 점수: +0.45)
  🟢 주식전문가: +0.52점 | LSTM 예측이 상승세를 시사...
  🟢 뉴스기업전문가: +0.38점 | AI 반도체 수요 확대 뉴스...
  🟢 최종결정자: +0.45점 | 두 전문가 의견 종합 시 단기 매수...

🔮 예측 결과 및 백테스트 데이터
  - 에이전트 최종합의 예측가: $133,290.60
  - 전략 과거 수익률: 18.42%
=================================================================
```

---

### 3️⃣ `final_stock.py` — 최종 결과 조회 & 리포트

**역할**: SQLite DB에 저장된 모든 분석 이력을 조회하고 요약 리포트를 출력합니다.

#### 주요 함수

| 함수 | 설명 |
|------|------|
| `show_analysis_log(top_n)` | ML 분석 이력 최신 N건 출력 (`analysis_log`) |
| `show_persona_log(top_n)` | LLM 페르소나 토론 이력 최신 N건 출력 |
| `show_final_summary()` | 종목별 최종결정자 판단 요약 (최신 1건씩) |
| `clean_failed_logs()` | `opinion = '분석 실패'` 데이터 일괄 삭제 |
| `drop_all_tables(confirm=True)` | ⚠️ DB 전체 초기화 (데이터 완전 삭제) |

#### 단독 실행 출력 예시

```
###################################################################
  📊 Stock AI 최종 결과 리포트
  실행 시각: 2026-07-19 19:38:00
  DB 경로: stock_analysis.db
###################################################################

🔮 [종목별 최종 투자 판단 요약]
=================================================================

📌 [삼성전자]  |  2026-07-19 18:50:00
   최종 점수: +0.32  →  🟢 매수 우세
   의견 요약: 최근 HBM 공급 계약 확대와 RSI 중립 구간 유지...

📌 [엔비디아]  |  2026-07-19 18:55:00
   최종 점수: +0.45  →  🟢 매수 우세
   의견 요약: AI 인프라 투자 지속 확대로 실적 개선 기대...
```

---

## 🧱 기술 스택

| 분류 | 라이브러리 |
|------|-----------|
| **데이터 수집** | `yfinance`, `feedparser`, `requests` |
| **데이터 처리** | `pandas`, `numpy`, `pandas_ta` |
| **ML 모델** | `torch` (PyTorch LSTM), `sklearn` (StandardScaler) |
| **NLP 감성분석** | `transformers` (FinBERT, KoELECTRA) |
| **LLM 연동** | `requests` → Ollama API (`gemma4:12b`) |
| **시각화** | `matplotlib` |
| **데이터베이스** | `sqlite3` |

---

## 📦 설치

```bash
pip install yfinance pandas pandas_ta numpy feedparser \
            transformers torch scikit-learn matplotlib requests
```

> **GPU 사용 시**: PyTorch CUDA 버전을 별도 설치하세요.
> [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## 🔄 데이터 흐름

```
yfinance / Google News RSS
         │
         ▼
  ┌─────────────┐
  │  ml_stock   │  LSTM 학습 · 백테스트 · 감성분석
  │  (V3 Agent) │──────────────────────────────────────────┐
  └─────────────┘                                          │
         │ run_and_return()                                 │
         ▼                                                  │
  ┌─────────────┐                                          │
  │  llm_stock  │  3인 페르소나 릴레이 토론 (Ollama)        │
  │  (V4 Agent) │──────────────────────────────────────────┤
  └─────────────┘                                          │
         │ SQLite 저장                                      │
         ▼                                                  ▼
  ┌─────────────────────────────────────────────────────────┐
  │              stock_analysis.db                          │
  │   - analysis_log (ML 분석 이력)                          │
  │   - persona_discussion_log (LLM 토론 이력)               │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
                    ┌──────────────┐
                    │ final_stock  │  결과 조회 & 출력
                    └──────────────┘
```

---

## 📊 SQLite DB 스키마

### `analysis_log`
| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | PK |
| timestamp | TEXT | 분석 시각 |
| name / symbol | TEXT | 종목명 / 티커 |
| last_close | REAL | 최종 종가 |
| rsi | REAL | RSI 값 |
| trend | TEXT | 과매수/과매도/중립 |
| support / resistance | REAL | 지지선 / 저항선 |
| usd_krw / sox | REAL | 환율 / SOX 지수 |
| sentiment | REAL | 감성 점수 |
| pred_low / pred_high | REAL | 예측 하단/상단 |
| risk_score | INTEGER | 리스크 점수 (1~10) |

### `persona_discussion_log`
| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | PK |
| timestamp | TEXT | 저장 시각 |
| ticker_name | TEXT | 종목명 |
| persona | TEXT | 주식전문가 / 뉴스기업전문가 / 최종결정자 |
| opinion | TEXT | 페르소나 의견 전문 |
| score | REAL | 투자 점수 (-1.0 ~ +1.0) |

---

## ⚠️ 주의사항

- `llm_stock.py`는 각 페르소나 호출 후 **15초 대기** (VRAM 안정화)하므로 종목당 약 45~60초 소요됩니다.
- Ollama 서버가 응답하지 않을 경우 `opinion = '분석 실패'`로 저장되며, `final_stock.py`의 `clean_failed_logs()`로 정리할 수 있습니다.
- `drop_all_tables(confirm=True)` 는 **모든 이력 데이터를 삭제**합니다. 신중하게 사용하세요.

---

## 📜 버전 이력

| 버전 | 파일 | 주요 변경 |
|------|------|----------|
| V1 | `ver1_1.py` (초기 섹션) | 기본 지표 분석 + SQLite 저장 |
| V3 | `ml_stock.py` | LSTM 예측 + 감성분석 + 백테스트 통합 |
| V4 | `llm_stock.py` | 3인 LLM 페르소나 릴레이 토론 추가 |
| V4.5 | `llm_stock.py` + `final_stock.py` | 파일 분리 + 결과 조회 모듈화 |
