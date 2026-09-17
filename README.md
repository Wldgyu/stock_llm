# Stock AI 분석 시스템

주가·거시지표·뉴스를 수집하여 LSTM과 TabPFN으로 가격을 예측하고,
Ollama 기반 전문가 페르소나가 투자 의견을 생성하는 로컬 분석 프로젝트입니다.

## 주요 파일

LLM 응답 대기 제한은 요청당 기본 10분입니다(연결 제한 15초).
더 오래 기다려야 하면 `.env`에 `OLLAMA_TIMEOUT_SECONDS=3600`을 설정하면
응답 대기 제한이 1시간으로 늘어납니다. 변경 후 `llm_stock.py`를 다시 실행하세요.
호출마다 입력 문자 수, 대기 제한, 성공 시 응답 소요 시간을 출력합니다.
3년치 일봉은 ML 학습에 사용합니다. 주식전문가에는 최근 종가·RSI·ML 예측과
모델별 간단한 검증 요약만 전달하고, 뉴스전문가에는 수집 뉴스만 전달합니다.
최종결정자는 두 전문가의 의견과 점수(분석 실패 여부 포함)만 받습니다.
전체 OOS 전략표와 실제 성과·비중 보고서는 LLM 입력에 반복해서 넣지 않습니다.
역할별 입력 경계 검증: `python -m unittest test_llm_roles`.

`최신 확정 일봉`은 현재 예측에 사용하는 데이터 기준일입니다.
`OOS 전략 비교`는 T+1·T+4·T+7 실제 정답을 모두 확보한 공통 검증 표본의
T+1 날짜를 기준으로 하므로 최근 6거래일이 제외됩니다.
OOS 종료일을 오늘로 맞추는 것은 평가 방식 변경이며, 데이터 갱신과 별개입니다.

```text
ml_stock.py         데이터 수집, 뉴스 수집, LSTM·TabPFN 학습 및 결과 저장
llm_stock.py        Ollama 전문가 토론과 최종 투자 판단 저장
database.py         공통 SQLite 경로, 마이그레이션, WAL·잠금 설정
final_stock.py      저장된 분석 결과 조회
delete_sqlite.py    실패한 LLM 분석 기록 정리
requirements.txt   Python 라이브러리 목록
CODE_ANALYSIS.txt   코드 분석 및 추가 개선 과제
ver1_1.py           구버전 참고용 코드
```

## 2026-08-27 적용 내용

### 머신러닝 검증

- 주가와 거시지표 데이터는 `yfinance`의 최근 3년 일봉을 사용합니다.
- 초기 70%를 학습 시작 구간으로 두고 이후 30%를 3개 walk-forward fold로 평가합니다.
- 각 fold는 이전 fold까지의 데이터를 누적 학습하고, 미래 구간만 테스트합니다.
- LSTM은 최근 30거래일을 입력으로 사용하여 T+1, T+4, T+7 가격을 예측합니다.
- LSTM은 hidden size 32, 1개 계층으로 구성합니다.
- LSTM은 배치 크기 16, 학습률 0.005, dropout 0.2와 Huber 손실을 사용합니다.
- 최대 100 epoch 동안 학습하며 검증 손실이 개선되지 않으면 12회 대기 후 조기 종료합니다.
- TabPFN은 최근 10거래일의 가격·환율·SOX·RSI·EMA20·거래량을 사용합니다.
- TabPFN은 T+1, T+4, T+7별로 별도 회귀 모델을 학습합니다.
- 두 모델 모두 각 fold와 전체 테스트 구간의 MAE, RMSE, MAPE, 방향 정확도를 계산합니다.
- fold별 지표의 평균과 표준편차를 함께 표시해 시장 국면별 편차를 확인합니다.
- 모든 fold에서 직전 종가를 그대로 유지하는 naive baseline과 반드시 비교합니다.
- naive보다 MAE가 나쁜 전체 결과 또는 개별 fold가 있으면 기준 미달 경고를 생성합니다.
- 검증 결과는 DB의 JSON 컬럼에 저장되어 API·웹 화면·LLM 전문가 입력에 전달됩니다.
- 검증이 끝난 뒤 최종 예측 모델은 전체 과거 데이터로 다시 학습합니다.
- 거래소 정규장 종료 전에는 당일 미완성 일봉을 학습 데이터에서 제외합니다.
- 거시지표 결측치는 과거 값으로만 채워 미래 데이터가 섞이는 `bfill` 누수를 제거했습니다.
- 한국·미국 주가와 환율·SOX의 실제 마감 시각을 UTC로 변환합니다.
- `merge_asof`로 주가 마감 전에 확정된 최신 거시지표만 결합하여 주말·휴장일의 과도한 지연을 방지합니다.
- PNG 생성과 리포트용 기여도 계산 호출은 비활성화했습니다.

### 뉴스 분석

- Google News 최신 제목 최대 5개는 본문 수집 실패 시 LLM 대체 입력으로 사용합니다.
- FinBERT·KoELECTRA 감성 모델은 사용하지 않습니다. 뉴스 판단은 Ollama 전문가가 수행합니다.
- LLM 뉴스 전문가는 다음 금융 언론사의 최신 기사 한 건씩을 구분하여 분석합니다.
  - Bloomberg
  - Wall Street Journal(WSJ)
  - Financial Times
  - Reuters
- Google News 주소를 원본 언론사 주소로 변환한 뒤 접근 가능한 기사 본문을 추출합니다.
- LLM 입력에는 언론사, 제목, 게시일, 본문 또는 기사 요약이 함께 전달됩니다.
- 유료 구독 또는 접근 차단으로 본문을 읽을 수 없으면 기사 요약이나 제목으로 대체합니다.
- `yfinance` 뉴스는 언론사와 요약을 제공하지만 특정 언론사를 보장하지 않아 보조 자료로만 사용할 수 있습니다.

### LLM 결과 안정성

- 주식전문가, 뉴스기업전문가, 최종결정자의 3단계 릴레이 분석을 사용합니다.
- 각 실행에 `run_id`를 부여하여 서로 다른 실행의 페르소나 결과가 섞이지 않도록 했습니다.
- LLM 응답 파싱 실패는 중립 점수 `0.0`이 아니라 `NULL`로 저장합니다.
- 점수 파싱은 마지막 JSON의 `score`만 읽고 `-1.0~1.0` 범위를 검사합니다.
- 최종결정자는 일부 전문가 분석이 실패해도 실패 상태를 명시적으로 전달받습니다.

### SQLite 관리

- 모든 실행 파일이 `database.py`의 공통 DB 설정을 사용합니다.
- 기본 저장 경로는 `%LOCALAPPDATA%\StockAIDashboard\stock_analysis.db`입니다.
- 프로젝트 폴더의 기존 DB는 새 기본 경로가 없을 때 한 번 복사됩니다.
- `WAL`, 30초 연결 timeout, `busy_timeout=30000`을 적용했습니다.
- 실패 기록은 `score IS NULL` 또는 명시적인 분석 실패일 때만 삭제 대상으로 판단합니다.
- 정상적인 중립 점수 `0.0`은 삭제하지 않습니다.
- 삭제 명령은 기본적으로 미리보기만 수행하며 `--yes`가 있어야 실제로 삭제합니다.

## 가상환경 및 설치

권장 환경은 Miniconda의 `stock_ai_312`입니다.

PowerShell에서 환경을 활성화하려면:

```powershell
& "C:\Users\dark0\miniconda3\shell\condabin\conda-hook.ps1"
conda activate stock_ai_312
```

환경 활성화 없이 정확한 Python으로 전체 라이브러리를 설치하려면:

```powershell
& "C:\Users\dark0\miniconda3\envs\stock_ai_312\python.exe" -m pip install -r requirements.txt
```

뉴스 원문 처리에는 `feedparser`, `googlenewsdecoder`, `trafilatura`,
`requests`, `yfinance`가 사용됩니다.

## 모델 인증

LSTM만 사용할 때는 Hugging Face 토큰이 필수는 아닙니다.
TabPFN 모델 또는 제한된 모델 저장소를 사용할 때는 유효한 토큰이 필요할 수 있습니다.

`.env.example`을 `.env`로 복사한 뒤 실제 토큰을 입력합니다.

```text
TABPFN_TOKEN=tabpfn_token_here
HF_TOKEN=hf_token_here
```

Hugging Face 토큰을 로컬 환경에 등록하려면:

```powershell
hf auth login
```

`.env`는 Git에서 제외됩니다. 토큰을 Python 코드에 직접 작성하거나 Git에 커밋하지 마세요.

## 실행 순서

### 1. ML 분석

```powershell
& "C:\Users\dark0\miniconda3\envs\stock_ai_312\python.exe" ml_stock.py
```

삼성전자, 엔비디아, 인텔을 분석하고 결과를 `analysis_log`에 저장합니다.
TabPFN 실행이 불가능하면 해당 예측만 건너뛰고 LSTM 분석을 계속합니다.

### 2. LLM 페르소나 분석

Ollama 서버와 기본 `gemma4:12b` 모델을 준비한 뒤 실행합니다.

```powershell
& "C:\Users\dark0\miniconda3\envs\stock_ai_312\python.exe" llm_stock.py
```

다른 Ollama 서버를 사용할 때는 `.env`의 `OLLAMA_HOST`를 변경합니다.
설정 가능한 환경변수:

- `OLLAMA_HOST`: 기본값 `127.0.0.1`
- `OLLAMA_EXPERT_MODEL`: 기본값 `gemma4:12b`
- `OLLAMA_DECISION_MODEL`: 기본값 `gemma4:12b`
- `OLLAMA_PAUSE_SECONDS`: 페르소나 호출 사이 대기, 기본값 `15`
- `DB_PATH`: 기본 경로 대신 사용할 SQLite 파일

### 3. 결과 확인 및 DB 정리

```powershell
python final_stock.py
python delete_sqlite.py
python delete_sqlite.py --yes
```

첫 번째 삭제 명령은 대상만 보여주고, `--yes`를 사용한 명령만 실제 삭제를 수행합니다.

## 제한사항

- 뉴스 본문 수집은 각 언론사의 접근 정책과 유료 구독 상태에 영향을 받습니다.
- 제목만 확보된 뉴스는 기사 전체 문맥을 반영하지 못할 수 있습니다.
- 모델 검증은 시간순 3-fold walk-forward 방식입니다.
- 모델 전략 비교는 가정한 수수료·슬리피지를 반영합니다. 세금과 실제 체결 가능성은 반영하지 않습니다. 기존 전체기간 RSI 백테스트는 비용 미반영입니다.
- Integrated Gradients 기여도는 모델의 인과관계를 증명하지 않고 현재 입력에 대한 민감도를 설명합니다.
- 결과는 미래 수익을 보장하지 않으며 투자 자문이 아닙니다.


직접 조절 가능한 항목:

learning_rate
batch_size
dropout_rate
huber_delta
gradient_clip
patience
max_epochs
## ML·LLM 결과 연결

ML 저장 행의 `id`를 `analysis_id`로 LLM의 모든 페르소나 행에 저장합니다.
대시보드는 최신 ML 행에 연결된 LLM 결과만 표시합니다. 연결된 결과가 없으면
AI 점수는 비어 있고 AI 상세 조회는 분석 대기 또는 실패 안내를 반환합니다.
기존 LLM 기록의 연결값은 추측해 채우지 않습니다. 다음 `llm_stock.py` 실행에서
컬럼이 자동 추가되며 새 분석부터 연결됩니다. 기존 감성 컬럼과 과거 기록은 보존하고
새 감성 값은 NULL로 저장합니다.

## 모델 예측 전략 백테스트

`backtesting.py`는 최종 재학습 모델이 아닌 walk-forward 테스트의 T+1 예측을 사용합니다.
전일 종가 대비 예측 상승률이 0.5% 초과면 다음 거래일 시가에 매수하고, 0% 이하면
같은 방식으로 매도합니다. 사이 구간에서는 기존 포지션을 유지합니다. 공매도·레버리지는
없으며 소수점 주식으로 전액 매수합니다. 임계값은 테스트 성과로 최적화하지 않습니다.

LSTM과 TabPFN의 공통 OOS 날짜에서 신호를 사용합니다. fold 경계 등 신호가 없는 날은
다음 시가에 청산하고 현금으로 대기합니다. 비교 기간은 첫 공통 날짜부터 마지막 공통
날짜까지 연속 거래일입니다. TabPFN을 사용할 수 없으면 LSTM 날짜를 사용합니다.
동일 기간의 RSI(전일 RSI 35 미만 매수, 65 초과 매도)와 단순 보유도 함께 계산합니다.
모든 전략은 마지막 날 종가에 청산합니다.

편도 수수료 10bp(0.10%)와 슬리피지 5bp(0.05%)는 비교용 가정입니다.
세금·환전·배당 현금흐름·호가 유동성은 별도로 모델링하지 않으며 조정 OHLC를 사용합니다.
Sharpe는 무위험 수익률 0, 연 252거래일 기준이고 MDD는 최초 자산 1부터 계산합니다.
이 결과는 과거 모의 성과이며 체결 가능성이나 미래 수익을 보장하지 않습니다.

콘솔과 대시보드에서 수익률·Sharpe·MDD·주문 수를 확인할 수 있습니다.
예측 날짜/가격과 비교 결과(자산 곡선·주문 기록 포함)는 기존 모델 검증 JSON에 저장되며,
LLM에도 비용 조건과 요약 성과가 전달됩니다. 과거 기록에는 이 항목이 없으므로 새 분석부터
표시됩니다. 기존 전체기간 RSI 지표와 새 동일기간 비교 지표는 구분해서 해석해야 합니다.

검증: `python -m unittest test_backtesting`

## T+1 시각화 리포트

`python ml_visualize_t1.py`를 실행하고 `t1_reports/index.html`을 브라우저에서 엽니다.
`--name 삼성전자`, `--db 경로`, `--output 폴더`로 범위를 지정할 수 있습니다.
ML 재학습이나 Ollama 없이 SQLite를 읽기 전용으로 사용합니다. `t1_report.js`는 HTML에
포함되는 차트 코드이며 두 소스 파일을 함께 유지하세요. PNG는 생성하지 않습니다.
Chart.js를 CDN에서 불러오므로 차트 표시에 인터넷 연결이 필요합니다.
기존 기록에 실제 종가 배열이 없으면 성과와 매매 기록만 표시됩니다. 새 ML 실행부터
실제 종가·T+1 예측·매수/매도·현금 구간을 표시합니다.

## 점수·예측 날짜 정확성

LLM 점수는 마지막 비어 있지 않은 줄의 JSON 객체만 검사합니다. 숫자형 score 하나만
허용하며 범위 오류·중복 키·문자열·불리언·추가 텍스트는 파싱 실패(NULL) 처리합니다.
앞쪽 예시 점수로 되돌아가지 않습니다.

`trading_dates.py`는 pandas-market-calendars의 XKRX/NASDAQ 달력으로 완료된 일봉을
선별하고 마지막 일봉 다음 1·4·7번째 거래일을 계산합니다. 달력 계산 오류 시 평일로
추정하지 않고 해당 분석이 실패하도록 합니다. 달력 라이브러리는 향후 임시 휴장 공지에
맞춰 업데이트해야 합니다.
새 분석부터 data_date·target_t1·target_t4·target_t7을 DB에 저장하고 콘솔·LLM·상세 화면에
동일하게 사용합니다. 과거 기록은 날짜를 추측하지 않고 '목표일 미기록'으로 표시합니다.
검증: `python -m unittest test_accuracy test_backtesting`

### 뉴스 선별 및 LLM 정리

- 최근 14일의 회사 관련 기사 중 실적·반도체 수요·계약·투자·규제·생산 위험 키워드와 최신성을 기준으로 매체별 1건을 선택합니다. 점수는 실제 주가 영향의 검증값이 아닌 선별 기준입니다.
- 일반 제목 대체 검색도 영문으로 수행하며 같은 필터를 적용합니다. 조건에 맞는 기사가 없으면 뉴스 부족으로 전달합니다.
- 공통 프롬프트와 결과 저장 처리를 통합했습니다. 전문가 역할 분리와 점수 파싱은 유지합니다.
