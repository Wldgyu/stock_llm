# Stock AI 분석 시스템

주가·환율·반도체 지수·뉴스를 수집하고 **LSTM·TabPFN으로 T+1·T+4·T+7 가격을 예측**하는 Python 프로젝트입니다. 주식전문가와 뉴스기업전문가의 의견을 최종결정자가 종합하고 SQLite에 저장합니다.

**간단 분석은 Ollama, 전문 분석은 NVIDIA API**를 사용합니다. 로컬 GPU의 VRAM 용량·메모리 대역폭에 따른 처리 부담을 줄이고 더 큰 모델로 전문 분석을 수행하도록 NVIDIA API를 추가했습니다. ML 학습은 두 방식 모두 로컬에서 실행합니다.

기본 종목: 삼성전자(`005930.KS`), 엔비디아(`NVDA`), 인텔(`INTC`).

```text
데이터·뉴스 수집 → ML 학습·검증·예측 → 주식전문가 → 뉴스기업전문가 → 최종결정 → DB 저장
```

## 설치

Python 3.12와 Conda 기준입니다. 프로젝트 폴더에서 실행합니다.

```powershell
conda create -n stock_ai_312 python=3.12 -y
conda activate stock_ai_312
python -m pip install -r requirements.txt
```

현재 `requirements.txt`는 Git에서 제외되어 있습니다. 해당 파일이 없다면 다음 명령으로 설치합니다.

```powershell
python -m pip install Flask feedparser googlenewsdecoder huggingface-hub matplotlib numpy pandas pandas-ta python-dotenv requests scikit-learn tabpfn torch trafilatura transformers yfinance pandas-market-calendars
```

CUDA를 지원하는 PyTorch 환경에서는 GPU를 자동 사용하며, 그 외에는 CPU를 사용합니다.

## 설정

`.env.example`을 복사하고 필요한 인증 키를 입력합니다. 이미 `.env`가 있으면 기존 파일을 편집합니다.

```powershell
Copy-Item .env.example .env
```

| 항목 | 설정 |
| --- | --- |
| TabPFN 인증 | `TABPFN_TOKEN`, `HF_TOKEN`에 모델 접근에 필요한 토큰 입력 |
| 간단 분석 | Ollama 서버 실행, 기본 모델 `gemma4:12b` 설치 |
| 전문 분석 | `NVIDIA_TOKEN`에 API 키 입력, 기본 모델 `nvidia/nemotron-3-super-120b-a12b` |
| 응답 대기 | `OLLAMA_TIMEOUT_SECONDS=600`, LLM 요청당 10분 |

간단 분석용 모델 설치:

```powershell
ollama pull gemma4:12b
```

Ollama 앱 또는 `ollama serve`로 서버를 실행합니다. 기본 주소는 `127.0.0.1:11434`이며 다른 서버는 `OLLAMA_HOST`에 호스트/IP만 입력합니다. 전문 분석은 Ollama 없이 실행하며 분석 자료를 NVIDIA API로 전송합니다. `.env`의 실제 키는 Git에 올리지 않습니다.

## 실행

```powershell
python llm_stock.py
```

터미널에서 **1: 간단 / 2: 전문**을 선택합니다(기본 1). 등록된 전체 종목을 분석하며 ML 학습부터 수행하므로 `ml_stock.py`를 먼저 실행할 필요는 없습니다.

방식이나 종목을 직접 지정할 수도 있습니다.

```powershell
# 전체 종목 전문 분석
python llm_stock.py --mode professional

# 삼성전자만 간단 분석
python llm_stock.py --mode simple --name 삼성전자

# ML 분석만 실행
python ml_stock.py

# 저장된 결과 조회
python final_stock.py
```

TabPFN 실행이 실패하면 해당 예측만 생략하고 LSTM 분석을 계속합니다. 전체 실행 시간에는 데이터 수집·ML 학습·세 번의 LLM 호출이 포함됩니다.

## 분석 정보

| 단계 | 주요 입력·처리 |
| --- | --- |
| ML | 최근 3년 일봉·환율·SOX, 시간순 3-fold 검증과 naive baseline 비교 |
| 주식전문가 | 종가·RSI·최근 가격 추이·ML 예측·핵심 검증 요약 |
| 뉴스기업전문가 | 기사별 언론사·제목·게시일·본문 또는 요약 |
| 최종결정자 | 두 전문가의 의견·점수와 실패 상태 |

전문 분석의 주식전문가에는 DB에 저장된 실제 T+1 방향 적중률·평가 건수를 추가합니다. 3년치 원시 일봉과 상세 전략표 전체는 LLM에 전달하지 않습니다.

뉴스는 Google News의 영문 검색으로 Bloomberg·WSJ·Financial Times·Reuters에서 최근 14일의 회사·주가 관련 기사 최대 1건씩 선별합니다. 본문 확보가 어려우면 요약·제목을 사용합니다.

T+1·T+4·T+7은 마지막 확정 일봉 이후의 거래일 기준입니다. OOS 비교는 T+7 실제 정답까지 필요하므로 최신 6거래일이 제외됩니다. 전략 비교에는 수수료·슬리피지를 반영하며 기존 전체기간 RSI 결과와 계산 기준이 다릅니다.

## 결과와 기록 관리

Windows 기본 DB: `%LOCALAPPDATA%\StockAIDashboard\stock_analysis.db`.

분석 정상 종료 시 30일 이전 기록을 정리하며 종목별 최신 결과와 관련 기록은 보존합니다. 수동 정리는 아래 명령을 사용하며 **`--yes` 없이 실행하면 미리보기만 수행**합니다.

```powershell
python cleanup_history.py --yes   # 오래된 기록 정리
python delete_sqlite.py --yes     # 실패한 LLM 기록 정리
```

T+1 차트 리포트는 다음 명령으로 생성하고 `t1_reports/index.html`을 엽니다. ML 재학습 없이 저장 결과를 사용하며 차트 표시에 인터넷 연결이 필요합니다.

```powershell
python ml_visualize_t1.py
```

상세 환경변수, 모델 학습 설정, 백테스트 규칙과 DB 동작은 [IMPLEMENTATION_GUIDE.txt](IMPLEMENTATION_GUIDE.txt)를 참고하세요.

과거 검증과 분석 의견은 미래 수익을 보장하지 않습니다.
