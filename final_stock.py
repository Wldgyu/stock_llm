# -*- coding: utf-8 -*-
"""
final_stock.py
───────────────────────────────────────────────────────────────────
[역할] 최종 결과 조회 및 리포트 출력
  - stock_analysis.db 의 analysis_log, persona_discussion_log 조회
  - 종목별 최신 분석 이력 테이블 출력
  - 실패 로그 정리 (opinion = '분석 실패' 삭제)
  - DB 테이블 초기화 유틸리티 (선택 실행)

[실행 전제] ml_stock.py 또는 llm_stock.py 가 먼저 실행되어
           stock_analysis.db 가 생성되어 있어야 합니다.

[단독 실행 시] python final_stock.py
"""

import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "stock_analysis.db"


# ──────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────
def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """SQLite 연결 객체를 반환합니다."""
    return sqlite3.connect(db_path)


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """테이블 존재 여부를 확인합니다."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cur.fetchone() is not None


# ──────────────────────────────────────────────
# 1. ML 분석 이력 조회 (analysis_log)
# ──────────────────────────────────────────────
def show_analysis_log(top_n: int = 10):
    """
    analysis_log 테이블에서 최신 N개의 ML 분석 결과를 출력합니다.

    Args:
        top_n: 출력할 최대 행 수 (기본 10)
    """
    print(f"\n{'='*65}")
    print("📋 [ML 분석 이력 조회: analysis_log]")
    print(f"{'='*65}")

    conn = get_connection()
    try:
        if not table_exists(conn, "analysis_log"):
            print("⚠️  analysis_log 테이블이 없습니다. ml_stock.py 를 먼저 실행하세요.")
            return

        df = pd.read_sql(
            "SELECT * FROM analysis_log ORDER BY timestamp DESC",
            conn
        )
        if df.empty:
            print("📭 저장된 분석 이력이 없습니다.")
            return

        print(f"✅ 총 {len(df)}건 | 최신 {top_n}건 출력\n")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        print(df.head(top_n).to_string(index=False))

    finally:
        conn.close()


# ──────────────────────────────────────────────
# 2. LLM 페르소나 토론 이력 조회 (persona_discussion_log)
# ──────────────────────────────────────────────
def show_persona_log(top_n: int = 20):
    """
    persona_discussion_log 테이블에서 최신 N개의 페르소나 의견을 출력합니다.

    Args:
        top_n: 출력할 최대 행 수 (기본 20)
    """
    print(f"\n{'='*65}")
    print("🎭 [LLM 페르소나 토론 이력 조회: persona_discussion_log]")
    print(f"{'='*65}")

    conn = get_connection()
    try:
        if not table_exists(conn, "persona_discussion_log"):
            print("⚠️  persona_discussion_log 테이블이 없습니다. llm_stock.py 를 먼저 실행하세요.")
            return

        df = pd.read_sql(
            "SELECT * FROM persona_discussion_log ORDER BY id DESC",
            conn
        )
        if df.empty:
            print("📭 저장된 페르소나 이력이 없습니다.")
            return

        print(f"✅ 총 {len(df)}건 | 최신 {top_n}건 출력\n")

        # opinion 컬럼은 길어서 80자로 truncate
        df_display = df.head(top_n).copy()
        df_display['opinion'] = df_display['opinion'].str[:80] + "..."
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        print(df_display.to_string(index=False))

    finally:
        conn.close()


# ──────────────────────────────────────────────
# 3. 종목별 최종 투자 판단 요약
# ──────────────────────────────────────────────
def show_final_summary():
    """
    persona_discussion_log 에서 '최종결정자' 행만 조회하여
    종목별 최신 투자 판단을 요약 출력합니다.
    """
    print(f"\n{'='*65}")
    print("🔮 [종목별 최종 투자 판단 요약]")
    print(f"{'='*65}")

    conn = get_connection()
    try:
        if not table_exists(conn, "persona_discussion_log"):
            print("⚠️  persona_discussion_log 테이블이 없습니다.")
            return

        query = """
            SELECT ticker_name, timestamp, score, opinion
            FROM persona_discussion_log
            WHERE persona = '최종결정자'
            ORDER BY id DESC
        """
        df = pd.read_sql(query, conn)
        if df.empty:
            print("📭 최종결정자 이력이 없습니다.")
            return

        # 종목별 가장 최신 1건만 유지
        df_latest = df.drop_duplicates(subset=['ticker_name'], keep='first')

        for _, row in df_latest.iterrows():
            score = float(row['score'])
            if score > 0.1:
                emoji = "🟢 매수 우세"
            elif score < -0.1:
                emoji = "🔴 매도 우세"
            else:
                emoji = "⚖️ 중립"

            print(f"\n📌 [{row['ticker_name']}]  |  {row['timestamp']}")
            print(f"   최종 점수: {score:+.2f}  →  {emoji}")
            print(f"   의견 요약: {str(row['opinion'])[:120]}...")

    finally:
        conn.close()


# ──────────────────────────────────────────────
# 4. 실패 로그 정리
# ──────────────────────────────────────────────
def clean_failed_logs():
    """persona_discussion_log 에서 opinion = '분석 실패' 데이터를 삭제합니다."""
    print(f"\n{'='*65}")
    print("🧹 [실패 로그 정리]")
    print(f"{'='*65}")

    conn = get_connection()
    cursor = conn.cursor()
    try:
        if not table_exists(conn, "persona_discussion_log"):
            print("⚠️  persona_discussion_log 테이블이 없습니다.")
            return

        cursor.execute(
            "DELETE FROM persona_discussion_log WHERE opinion = '분석 실패'"
        )
        conn.commit()
        print(f"✅ 삭제 완료! 총 {cursor.rowcount}개의 실패 로그가 정리되었습니다.")
    finally:
        conn.close()


# ──────────────────────────────────────────────
# 5. DB 테이블 초기화 (주의: 모든 데이터 삭제)
# ──────────────────────────────────────────────
def drop_all_tables(confirm: bool = False):
    """
    stock_analysis.db 의 모든 테이블을 삭제합니다.
    재실행 시 에이전트가 자동으로 테이블을 다시 생성합니다.

    Args:
        confirm: True 로 명시적으로 설정해야만 실행됩니다.
    """
    if not confirm:
        print("⚠️  drop_all_tables(confirm=True) 로 호출해야 실행됩니다.")
        return

    print(f"\n{'='*65}")
    print("🔥 [DB 테이블 초기화] — 모든 데이터가 삭제됩니다!")
    print(f"{'='*65}")

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DROP TABLE IF EXISTS analysis_log")
        cursor.execute("DROP TABLE IF EXISTS persona_discussion_log")
        conn.commit()
        print("✅ 기존 테이블 삭제 완료! 에이전트 재실행 시 자동으로 새로 생성됩니다.")
    finally:
        conn.close()


# ──────────────────────────────────────────────
# 단독 실행 — 전체 결과 출력
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'#'*65}")
    print(f"  📊 Stock AI 최종 결과 리포트")
    print(f"  실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  DB 경로: {DB_PATH}")
    print(f"{'#'*65}")

    # 1. ML 분석 이력 (최신 10건)
    show_analysis_log(top_n=10)

    # 2. 페르소나 토론 이력 (최신 20건)
    show_persona_log(top_n=20)

    # 3. 종목별 최종 투자 판단 요약
    show_final_summary()

    # 4. 실패 로그 정리 (선택 — 필요 시 주석 해제)
    # clean_failed_logs()

    # 5. DB 전체 초기화 (선택 — 매우 위험! 필요 시만 실행)
    # drop_all_tables(confirm=True)

    print(f"\n{'#'*65}")
    print("  ✅ 리포트 출력 완료.")
    print(f"{'#'*65}\n")
