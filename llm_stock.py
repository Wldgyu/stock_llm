# -*- coding: utf-8 -*-
"""
llm_stock.py
───────────────────────────────────────────────────────────────────
[역할] LLM 페르소나 토론 엔진
  - MarketAgentSimulator : Ollama 서버에 3인 페르소나(주식전문가·뉴스기업전문가·최종결정자)를 호출
  - StockAIAgentV4       : ml_stock.StockAIAgentV3 를 상속, 릴레이 토론 및 DB 저장

[의존성] ml_stock.py 가 같은 디렉터리에 있어야 합니다.

[단독 실행 시] python llm_stock.py
"""

import json
import os
import re
import time
import traceback
import uuid
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from database import DB_PATH as DEFAULT_DB_PATH, connect_db

# 프로젝트의 비공개 Ollama·모델 설정을 환경변수로 불러옵니다.
load_dotenv(Path(__file__).with_name(".env"))

# ml_stock.py 에서 공통 유틸 및 V3 에이전트 import
from ml_stock import StockAIAgentV3, get_next_trading_date


# ──────────────────────────────────────────────
# 1. Ollama 페르소나 시뮬레이터
# ──────────────────────────────────────────────
class MarketAgentSimulator:
    """
    Ollama 서버에 3인의 전문 에이전트 페르소나를 순차 호출하여
    릴레이 토론 방식으로 최종 투자 의사결정 점수를 생성합니다.

    페르소나:
      - 주식전문가     : 가격·차트·예측 데이터 분석
      - 뉴스기업전문가 : 뉴스 심리·기업 질적 평가
      - 최종결정자     : 두 전문가 의견을 종합하여 최종 결정
    """

    def __init__(
        self,
        server_ip: Optional[str] = None,
        expert_model: str = "gemma4:12b",
        decision_model: str = "gemma4:12b",
    ):
        self.server_ip      = server_ip or os.environ.get("OLLAMA_HOST", "127.0.0.1")
        self.expert_model   = expert_model
        self.decision_model = decision_model
        self.url            = f"http://{self.server_ip}:11434/api/generate"

        self.personas = {
            "주식전문가":     "계량적 분석 및 주가 차트 흐름 분석 전문가. 가격 예측, 현재 가격 추이.",
            "뉴스기업전문가": "시장 뉴스 심리 및 기업 질적 평가 전문가. 최신 뉴스 3개와 기업 평가(뉴스 감정 점수, 백테스트 수익률, Sharpe, MDD 등)를 종합적으로 분석.",
            "최종결정자":     "최종 투자 의사결정권자. 주식전문가와 뉴스기업전문가의 분석 리포트를 수신하여 투자 방향(점수)과 최종 결정을 내리는 역할.",
        }

    # ── VRAM 캐시 해제 ────────────────────────
    def clear_vram_cache(self, model_name: str):
        """Ollama 서버에 keep_alive=0 신호를 보내 KV 캐시 및 메모리를 강제 회수합니다."""
        payload = {"model": model_name, "keep_alive": 0}
        try:
            requests.post(self.url, json=payload, timeout=15)
            print(f"      🧹 VRAM 캐시 초기화 완료: {model_name}")
        except Exception as e:
            print(f"      ⚠️ VRAM 초기화 통신 오류: {e}")

    # ── 주식전문가 호출 ───────────────────────
    def run_stock_expert(self, ticker_name: str, current_price_desc: str, prediction_info: str):
        """주식전문가 페르소나를 Ollama에 호출합니다."""
        description = self.personas["주식전문가"]
        prompt = (
            f"<start_of_turn>system\n"
            f"당신은 주식 시장 가상 토론에 참여한 계량 분석 및 주가 기술 전문가 '주식전문가'입니다.\n"
            f"당신의 투자 철학: {description}\n"
            f"반드시 주어진 주가 정보와 예측 데이터에 기반하여 논리적인 의견을 전개하되, "
            f"JSON 점수 줄을 제외한 분석 의견은 공백을 포함해 700자 미만으로 작성하세요.\n"
            f"출력 결과의 맨 마지막 줄에는 반드시 다른 텍스트 없이 JSON 형식으로만 점수를 적으세요.\n"
            f'JSON 포맷 예시: {{"score": 0.45}}\n'
            f"점수 범위는 -1.0(극단적 비관/매도)에서 +1.0(극단적 낙관/매수) 사이의 실수여야 합니다.\n<end_of_turn>\n"
            f"<start_of_turn>user\n"
            f"종목: {ticker_name}\n"
            f"현재 주식 정보:\n{current_price_desc}\n"
            f"AI 모델 가격 예측 데이터:\n{prediction_info}\n"
            f"당신의 관점에서 이 종목의 주가 수준과 가격 예측치, 종합적으로 평가하고, "
            f"마지막 줄에 규격화된 JSON 점수를 남겨주세요.\n<end_of_turn>\n"
            f"<start_of_turn>model\n"
            f"전문가 주식전문가의 분석 의견:\n"
        )
        return self._send_request("주식전문가", prompt, model_name=self.expert_model)

    # ── 뉴스기업전문가 호출 ───────────────────
    def run_news_expert(self, ticker_name: str, combined_news: str, company_evaluation: str):
        """뉴스기업전문가 페르소나를 Ollama에 호출합니다."""
        description = self.personas["뉴스기업전문가"]
        prompt = (
            f"<start_of_turn>system\n"
            f"당신은 주식 시장 가상 토론에 참여한 뉴스 심리 및 질적 분석 전문가 '뉴스기업전문가'입니다.\n"
            f"당신의 투자 철학: {description}\n"
            f"반드시 주어진 최신 뉴스 3개와 기업 평가 정보에 기반하여 논리적인 의견을 전개하되, "
            f"JSON 점수 줄을 제외한 분석 의견은 공백을 포함해 700자 미만으로 작성하세요.\n"
            f"출력 결과의 맨 마지막 줄에는 반드시 다른 텍스트 없이 JSON 형식으로만 점수를 적으세요.\n"
            f'JSON 포맷 예시: {{"score": 0.45}}\n'
            f"점수 범위는 -1.0(극단적 비관)에서 +1.0(극단적 낙관) 사이의 실수여야 합니다.\n<end_of_turn>\n"
            f"<start_of_turn>user\n"
            f"종목: {ticker_name}\n"
            f"최신 수집 뉴스:\n{combined_news}\n"
            f"기업 평가 지표:\n{company_evaluation}\n\n"
            f"당신의 관점에서 최근 뉴스의 영향력과 기업 평가 지표를 바탕으로 이 종목의 투자 매력도를 평가하고, "
            f"마지막 줄에 규격화된 JSON 점수를 남겨주세요.\n<end_of_turn>\n"
            f"<start_of_turn>model\n"
            f"전문가 뉴스기업전문가의 분석 의견:\n"
        )
        return self._send_request("뉴스기업전문가", prompt, model_name=self.expert_model)

    # ── 최종결정자 호출 ───────────────────────
    def run_final_decision_maker(
        self,
        ticker_name: str,
        stock_opinion: str,
        stock_score: Optional[float],
        news_opinion: str,
        news_score: Optional[float],
    ):
        """최종결정자 페르소나를 Ollama에 호출합니다."""
        description = self.personas["최종결정자"]
        stock_score_text = f"{stock_score:+.2f}" if stock_score is not None else "분석 실패"
        news_score_text = f"{news_score:+.2f}" if news_score is not None else "분석 실패"
        prompt = (
            f"<start_of_turn>system\n"
            f"당신은 주식 시장 가상 토론의 최종 투자 의사 결정자인 '최종결정자'입니다.\n"
            f"당신의 투자 철학: {description}\n"
            f"주식전문가의 분석 의견과 뉴스기업전문가의 분석 의견을 종합하여 최종 결정을 내리세요. "
            f"출력의 첫 줄은 반드시 '[요약] ' 으로 시작하는 핵심 한 줄 요약을 작성하세요.\n"
            f"JSON 점수 줄을 제외한 전체 의견은 공백을 포함해 500자 이내로 작성하세요.\n"
            f"반드시 다른 텍스트 없이 출력 결과의 맨 마지막 줄에는 JSON 형식으로만 최종 합의 점수를 적으세요.\n"
            f'JSON 포맷 예시: {{"score": 0.45}}\n'
            f"점수 범위는 -1.0(강력 매도/비관)에서 +1.0(강력 매수/낙관) 사이의 실수여야 합니다.\n<end_of_turn>\n"
            f"<start_of_turn>user\n"
            f"종목: {ticker_name}\n"
            f"1. 주식전문가 분석 의견 (점수: {stock_score_text}):\n{stock_opinion}\n\n"
            f"2. 뉴스기업전문가 분석 의견 (점수: {news_score_text}):\n{news_opinion}\n\n"
            f"두 전문가의 분석 결과를 토대로 최종적인 투자 방향성 및 의사결정을 내리고, 마지막 줄에 규격화된 JSON 점수를 남겨주세요.\n<end_of_turn>\n"
            f"<start_of_turn>model\n"
            f"최종결정자의 의사결정 리포트:\n"
        )
        return self._send_request("최종결정자", prompt, model_name=self.decision_model)

    # ── 공통 요청 처리 ────────────────────────
    def _send_request(self, persona_name: str, prompt: str, model_name: str):
        """Ollama 서버에 요청을 보내고 응답 텍스트를 반환합니다."""
        payload = {
            "model":      model_name,
            "prompt":     prompt,
            "stream":     False,
            "options":    {"temperature": 0.3},
            "keep_alive": "5s",
        }
        try:
            response = requests.post(self.url, json=payload, timeout=360)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                print(f"      ❌ {persona_name} 서버 응답 오류 ({response.status_code})")
                return None
        except Exception as e:
            print(f"      ❌ {persona_name} API 호출 중 네트워크 예외 발생: {e}")
            return None
        finally:
            self.clear_vram_cache(model_name)

    # ── JSON 점수 파싱 ────────────────────────
    def parse_score_from_response(self, text: str) -> Optional[float]:
        """응답 텍스트 마지막 JSON 에서 score 값을 추출합니다."""
        if not text:
            return None

        # 마지막 JSON 객체부터 확인해 본문에 있는 다른 숫자와 섞이지 않게 합니다.
        json_blocks = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
        for block in reversed(json_blocks):
            try:
                data = json.loads(block)
                score = float(data["score"])
                if -1.0 <= score <= 1.0:
                    return score
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

        # JSON이 깨졌을 때도 score 키 바로 뒤의 숫자만 제한적으로 복구합니다.
        matches = re.findall(
            r"""["']?score["']?\s*[:=]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))""",
            text,
            flags=re.IGNORECASE,
        )
        if matches:
            score = float(matches[-1])
            if -1.0 <= score <= 1.0:
                return score
        return None

    def parse_response(self, text: str, max_chars: Optional[int] = None):
        """LLM 응답을 의견·요약·점수로 나누며 파싱 실패를 중립과 구분합니다."""
        score = self.parse_score_from_response(text)
        if not text:
            return "분석 실패: 응답 없음", None

        opinion = text.strip()
        for match in reversed(list(re.finditer(r"\{[^{}]*\}", opinion, re.DOTALL))):
            if "score" in match.group(0).lower():
                opinion = (opinion[:match.start()] + opinion[match.end():]).strip()
                break
        opinion = opinion.strip("` \n")
        if score is None:
            opinion = f"분석 실패: 점수 파싱 오류\n{opinion}"
        if max_chars is not None and len(opinion) > max_chars:
            # 프롬프트를 벗어난 긴 응답도 DB 저장 전에 확실히 제한합니다.
            opinion = opinion[:max_chars].rstrip()
        if score is None:
            return opinion, None
        return opinion or "의견 없음", score

    @staticmethod
    def extract_summary(opinion: str) -> str:
        """의견 텍스트에서 [요약] 태그 한 줄을 추출합니다. 없으면 첫 문장을 사용합니다."""
        for line in opinion.split("\n"):
            stripped = line.strip()
            if stripped.startswith("[요약]"):
                return stripped[len("[요약]"):].strip()
        # [요약] 태그가 없으면 첫 문장(마침표·쉼표 기준)을 fallback으로 사용
        first = opinion.split("\n")[0].strip()
        for sep in (".", "。", ","):
            if sep in first:
                return first[:first.index(sep) + 1]
        return first[:80]


# ──────────────────────────────────────────────
# 2. Stock AI Agent V4 (LLM 릴레이 토론)
# ──────────────────────────────────────────────
class StockAIAgentV4(StockAIAgentV3):
    """
    StockAIAgentV3 를 상속하여 LLM 페르소나 릴레이 토론을 추가한 에이전트 (V4).

    흐름:
      1. ml_stock.StockAIAgentV3.run_and_return() 으로 ML 분석 결과 획득
      2. MarketAgentSimulator 로 3인 릴레이 페르소나 토론 실행
      3. 결과를 SQLite (persona_discussion_log) 에 저장
      4. 최종 리포트 출력
    """

    DB_PATH = DEFAULT_DB_PATH

    def __init__(self, server_ip: Optional[str] = None):
        super().__init__()  # StockAIAgentV3 초기화 (NLP 파이프라인 등)

        if not hasattr(self, "db_path"):
            self.db_path = self.DB_PATH

        self.simulator = MarketAgentSimulator(
            server_ip=server_ip,
            expert_model=os.environ.get("OLLAMA_EXPERT_MODEL", "gemma4:12b"),
            decision_model=os.environ.get("OLLAMA_DECISION_MODEL", "gemma4:12b"),
        )
        self.pause_seconds = float(os.environ.get("OLLAMA_PAUSE_SECONDS", "15"))
        self._init_persona_db()

    # ── DB 초기화 ─────────────────────────────
    def _init_persona_db(self):
        """페르소나 토론 로그 테이블을 SQLite에 생성합니다."""
        conn = connect_db(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS persona_discussion_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT,
                ticker_name  TEXT,
                persona      TEXT,
                opinion      TEXT,
                score        REAL,
                run_id       TEXT
            )
        """)
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(persona_discussion_log)")
        }
        if "run_id" not in columns:
            conn.execute("ALTER TABLE persona_discussion_log ADD COLUMN run_id TEXT")
        conn.commit()
        conn.close()

    # ── 페르소나 결과 저장 ────────────────────
    def save_persona_result(
        self,
        ticker_name: str,
        persona: str,
        opinion: str,
        score: Optional[float],
        run_id: str,
    ):
        """단일 페르소나 결과를 즉각 커밋합니다 (Atomic Save)."""
        conn = connect_db(self.db_path)
        conn.execute(
            """
            INSERT INTO persona_discussion_log
            (timestamp, ticker_name, persona, opinion, score, run_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ticker_name,
                persona,
                opinion,
                score,
                run_id,
            ),
        )
        conn.commit()
        conn.close()
        print(f"      💾 [{persona}] 분석 스냅샷 SQLite 저장 완료.")

    # ── 가격 포맷팅 ───────────────────────────
    def format_price(self, val: float, is_krw: bool) -> str:
        """종목 통화에 맞는 가격 문자열을 반환합니다."""
        return f"{val:,.0f}원" if is_krw else f"${val:,.2f}"

    # ── 메인 실행 ─────────────────────────────
    def run(self):
        """ML 분석 → LLM 릴레이 토론 → DB 저장 → 리포트 출력을 일괄 실행합니다."""
        next_date, market_status = get_next_trading_date()

        print(f"📅 분석 기준일: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"🕒 시장 상태: {market_status}")
        print(f"🔮 예측 목표일: {next_date}\n")

        # ML 분석 먼저 실행 (V3 run_and_return 활용)
        print("🤖 ML 분석 엔진 가동 중...")
        ml_results = self.run_and_return()

        for name, data in ml_results.items():
            try:
                df     = data["df"]
                symbol = data["symbol"]
                is_krw = symbol.endswith(".KS")

                bt_ret    = data["bt_ret"]
                sharpe    = data["sharpe"]
                mdd       = data["mdd"]
                sentiment = data["sentiment"]
                titles    = data["titles"]
                news_articles = data.get("news_articles", [])
                preds     = data["preds"]
                tabpfn_preds = data.get("tabpfn_preds")
                run_id = uuid.uuid4().hex

                # ── 페르소나 입력 데이터 구성 ──
                if news_articles:
                    news_blocks = []
                    for article in news_articles:
                        content = article.get("body") or article.get("title", "")
                        news_blocks.append(
                            f"[언론사: {article.get('source', '알 수 없음')}]\n"
                            f"제목: {article.get('title', '')}\n"
                            f"게시일: {article.get('published', '알 수 없음')}\n"
                            f"분석 자료({article.get('content_type', '제목')}): {content}"
                        )
                    combined_news = "\n\n".join(news_blocks)
                else:
                    # 본문을 가져오지 못한 경우 기존 제목 분석으로 대체합니다.
                    combined_news = "\n".join(
                        [f"[언론사 미확인] {title}" for title in titles[:3]]
                    )

                prediction_info = (
                    f"LSTM 예측가: 내일 {self.format_price(preds[0], is_krw)}, "
                    f"4일 뒤 {self.format_price(preds[1], is_krw)}, "
                    f"7일 뒤 {self.format_price(preds[2], is_krw)}"
                )
                if tabpfn_preds:
                    prediction_info += (
                        f"\nTabPFN 예측가: 내일 {self.format_price(tabpfn_preds[0], is_krw)}, "
                        f"4일 뒤 {self.format_price(tabpfn_preds[1], is_krw)}, "
                        f"7일 뒤 {self.format_price(tabpfn_preds[2], is_krw)}"
                    )

                rsi_v         = float(df['RSI'].iloc[-1])
                trend         = "🔥 과매수" if rsi_v > 70 else ("❄️ 과매도" if rsi_v < 30 else "⚖️ 중립 유지")
                recent_prices = df['Close'].tail(5)
                recent_trend  = ", ".join([self.format_price(float(p), is_krw) for p in recent_prices])
                current_price = float(df['Close'].iloc[-1])

                current_price_desc = (
                    f"- 현재가: {self.format_price(current_price, is_krw)}\n"
                    f"- RSI: {rsi_v:.2f} ({trend})\n"
                    f"- 최근 5영업일 종가 추이: {recent_trend}"
                )

                sentiment_desc = (
                    f"긍정 (Score: {sentiment:.2f})"  if sentiment >  0.05 else
                    f"부정 (Score: {sentiment:.2f})"  if sentiment < -0.05 else
                    f"중립 (Score: {sentiment:.2f})"
                )
                company_evaluation = (
                    f"- 뉴스 감정 평가: {sentiment_desc}\n"
                    f"- 과거 전략 백테스트 수익률: {bt_ret:.2f}%\n"
                    f"- Sharpe Ratio (위험 대비 수익성): {sharpe:.2f}\n"
                    f"- MDD (최대 낙폭): {mdd:.2f}%"
                )

                # ── 릴레이 토론 실행 ──────────
                print(f"\n{'─'*53}")
                print(f"🗨️ [{name}] 전문 에이전트 릴레이 의사결정 시작")
                print(f"{'─'*53}")

                # Step 1: 주식전문가
                print("   ▶️ [호출] 주식전문가 에이전트 요청 송신 중...")
                stock_raw   = self.simulator.run_stock_expert(name, current_price_desc, prediction_info)
                stock_opinion, stock_score = self.simulator.parse_response(
                    stock_raw, max_chars=699
                )
                self.save_persona_result(
                    name, "주식전문가", stock_opinion, stock_score, run_id
                )
                time.sleep(self.pause_seconds)

                # Step 2: 뉴스기업전문가
                print("   ▶️ [호출] 뉴스기업전문가 에이전트 요청 송신 중...")
                news_raw    = self.simulator.run_news_expert(name, combined_news, company_evaluation)
                news_opinion, news_score = self.simulator.parse_response(
                    news_raw, max_chars=699
                )
                self.save_persona_result(
                    name, "뉴스기업전문가", news_opinion, news_score, run_id
                )
                time.sleep(self.pause_seconds)

                # Step 3: 최종결정자
                print("   ▶️ [호출] 최종결정자 에이전트 요청 송신 중...")
                final_raw   = self.simulator.run_final_decision_maker(
                    name, stock_opinion, stock_score, news_opinion, news_score
                )
                final_opinion, final_score = self.simulator.parse_response(
                    final_raw, max_chars=500
                )
                self.save_persona_result(
                    name, "최종결정자", final_opinion, final_score, run_id
                )
                time.sleep(self.pause_seconds)

                # ── 리포트 출력 ───────────────
                print(f"\n{'='*65}")
                print(f"🚀 [AI Agent V4.5 리포트: {name}]")
                print(f"{'='*65}")

                print(f"📊 현재가: {self.format_price(current_price, is_krw)} | LSTM 예측: {self.format_price(preds[0], is_krw)}")

                final_score_text = (
                    f"{final_score:+.2f}" if final_score is not None else "분석 실패"
                )
                print(f"\n🎭 [AI 에이전트 토론 및 최종 결정] (최종 의사결정 점수: {final_score_text})")

                def _icon(score):
                    if score is None: return "❌"
                    if score > 0.1:   return "🟢"
                    if score < -0.1:  return "🔴"
                    return "⚖️"

                def _score_text(score):
                    return f"{score:+.2f}점" if score is not None else "분석 실패"

                s_short = stock_opinion[:80].replace("\n", " ") + "..."
                n_short = news_opinion[:80].replace("\n", " ") + "..."
                f_summary = self.simulator.extract_summary(final_opinion)

                print(f"  {_icon(stock_score)} 주식전문가: {_score_text(stock_score)} | {s_short}")
                print(f"  {_icon(news_score)}  뉴스기업전문가: {_score_text(news_score)} | {n_short}")
                print(f"  {_icon(final_score)} 최종결정자: {_score_text(final_score)} | {f_summary}")

                print(f"\n🔮 백테스트 데이터")
                print(f"  - 전략 과거 수익률: {bt_ret:.2f}%")
                print(f"  - Sharpe: {sharpe:.2f} | MDD: {mdd:.2f}%")
                print(f"{'='*65}\n")

            except Exception as e:
                print(f"❌ {name} 분석 실패: {e}")
                traceback.print_exc()


# ──────────────────────────────────────────────
# 단독 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    agent = StockAIAgentV4()
    agent.run()
