# Stock AI 분석 시스템

주가·거시지표·뉴스를 수집하여 LSTM과 TabPFN으로 가격을 예측하고,
Ollama 기반 전문가 페르소나가 투자 의견을 생성하는 로컬 분석 프로젝트입니다.

## 주요 파일

```text
ml_stock.py         데이터 수집, 감성 분석, LSTM·TabPFN 학습 및 결과 저장
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

- 주가와 거시지표 데이터는 `yfinance`의 최근 1년 일봉을 사용합니다.
- 데이터를 시간순으로 학습 70%, 검증 15%, 테스트 15%로 분리합니다.
- LSTM은 최근 20거래일을 입력으로 사용하여 T+1, T+4, T+7 가격을 예측합니다.
- LSTM은 배치 크기 16, 학습률 0.005, dropout 0.2와 Huber 손실을 사용합니다.
- 최대 200 epoch 동안 학습하며 검증 손실이 개선되지 않으면 12회 대기 후 조기 종료합니다.
- TabPFN은 최근 10거래일의 가격·환율·SOX·RSI·EMA20·거래량을 사용합니다.
- TabPFN은 T+1, T+4, T+7별로 별도 회귀 모델을 학습합니다.
- 두 모델 모두 테스트 구간의 MAE, RMSE, MAPE를 계산합니다.
- 검증이 끝난 뒤 최종 예측 모델은 전체 과거 데이터로 다시 학습합니다.
- 거래소 정규장 종료 전에는 당일 미완성 일봉을 학습 데이터에서 제외합니다.
- 거시지표 결측치는 과거 값으로만 채워 미래 데이터가 섞이는 `bfill` 누수를 제거했습니다.
- 한국 종목은 환율·SOX를 1거래일 지연하고, 미국 종목은 환율만 1거래일 지연합니다.
- LSTM의 T+1 예측 기여도는 Integrated Gradients로 실제 계산합니다.

### 뉴스 분석

- 일반 뉴스 감성 점수는 Google News 최신 제목 최대 5개로 계산합니다.
- FinBERT의 `NEUTRAL` 라벨은 긍정값이 아닌 `0.0`으로 계산합니다.
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

공개 감성 모델과 LSTM만 사용할 때는 Hugging Face 토큰이 필수는 아닙니다.
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
- 제목 기반 감성 점수는 기사 전체 문맥을 반영하지 못할 수 있습니다.
- 모델 검증은 시간순 단일 분할이며 아직 walk-forward 검증을 사용하지 않습니다.
- 거래 수수료, 세금, 슬리피지와 실제 주문 체결 가능성은 반영하지 않습니다.
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