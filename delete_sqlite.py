# -*- coding: utf-8 -*-
"""
delete_sqlite.py
────────────────────────────────────────────────────────────────
[역할] SQLite DB 에서 분석 실패 레코드를 조회·삭제합니다.

[삭제 기준]
  persona_discussion_log 테이블:
    - opinion 이 '분석 실패' 인 레코드
    - score = 0.0  AND  opinion 이 비어있거나 '분석 실패' 인 레코드

[실행 방법]
  python delete_sqlite.py              -> 분석 실패 레코드 즉시 삭제
  python delete_sqlite.py --all-zero  -> score=0.0 전체 즉시 삭제
  python delete_sqlite.py --dry-run   -> 삭제 없이 미리보기만
"""

import sqlite3
import pathlib
from datetime import datetime

# ── 설정 ─────────────────────────────────────────────
DB_PATH = pathlib.Path(__file__).parent / "stock_analysis.db"


# ── 실패 레코드 조회 ─────────────────────────────────
def fetch_failed_records(conn, all_zero=False):
    if all_zero:
        query = """
            SELECT id, timestamp, ticker_name, persona, opinion, score
            FROM persona_discussion_log
            WHERE score = 0.0
            ORDER BY id
        """
    else:
        query = """
            SELECT id, timestamp, ticker_name, persona, opinion, score
            FROM persona_discussion_log
            WHERE opinion LIKE '%분석 실패%'
               OR (score = 0.0 AND (opinion = '' OR opinion IS NULL OR opinion LIKE '%실패%'))
            ORDER BY id
        """
    return conn.execute(query).fetchall()


# ── 삭제 실행 ─────────────────────────────────────────
def delete_failed_records(conn, ids):
    if not ids:
        return 0
    placeholders = ",".join("?" * len(ids))
    cursor = conn.execute(
        f"DELETE FROM persona_discussion_log WHERE id IN ({placeholders})", ids
    )
    conn.commit()
    return cursor.rowcount


# ── 전체 현황 출력 ────────────────────────────────────
def print_summary(conn):
    total = conn.execute("SELECT COUNT(*) FROM persona_discussion_log").fetchone()[0]
    ok    = conn.execute("SELECT COUNT(*) FROM persona_discussion_log WHERE score != 0.0").fetchone()[0]
    fail  = conn.execute("SELECT COUNT(*) FROM persona_discussion_log WHERE opinion LIKE '%실패%'").fetchone()[0]

    print(f"\n{'='*55}")
    print(f"  [persona_discussion_log 현황]")
    print(f"{'='*55}")
    print(f"  전체 레코드  : {total:>6}행")
    print(f"  정상 레코드  : {ok:>6}행  (score != 0)")
    print(f"  실패 레코드  : {fail:>6}행  (opinion LIKE '%실패%')")
    print(f"{'='*55}\n")


# ── 티커별 집계 ───────────────────────────────────────
def print_ticker_summary(conn):
    rows = conn.execute("""
        SELECT ticker_name,
               COUNT(*) as total,
               SUM(CASE WHEN opinion LIKE '%실패%' THEN 1 ELSE 0 END) as failed,
               SUM(CASE WHEN score != 0.0 THEN 1 ELSE 0 END) as success
        FROM persona_discussion_log
        GROUP BY ticker_name
        ORDER BY ticker_name
    """).fetchall()

    print(f"  {'종목':<12} {'전체':>6} {'실패':>6} {'성공':>6}")
    print(f"  {'-'*36}")
    for ticker, total, failed, success in rows:
        print(f"  {ticker:<12} {total:>6} {failed:>6} {success:>6}")
    print()


# ── 메인 ──────────────────────────────────────────────
def main():
    import sys
    args = sys.argv[1:]
    all_zero = "--all-zero" in args
    dry_run  = "--dry-run"  in args

    # --db 옵션 처리
    db_path = DB_PATH
    if "--db" in args:
        idx = args.index("--db")
        if idx + 1 < len(args):
            db_path = pathlib.Path(args[idx + 1])

    if not db_path.exists():
        print(f"DB 파일을 찾을 수 없습니다: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    print(f"\nDB: {db_path}")
    print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print_summary(conn)
    print_ticker_summary(conn)

    failed = fetch_failed_records(conn, all_zero=all_zero)

    if not failed:
        print("삭제 대상 레코드가 없습니다. DB가 이미 깨끗합니다!")
        conn.close()
        return

    mode = "score=0 전체" if all_zero else "분석 실패"
    print(f"{'─'*55}")
    print(f"  삭제 대상 ({mode}): {len(failed)}행")
    print(f"{'─'*55}")
    for row in failed:
        rid, ts, ticker, persona, opinion, score = row
        short_opinion = (opinion or "")[:30].replace("\n", " ")
        print(f"  id={rid:>4} | {ts} | {ticker:<8} | {persona:<12} | score={score:.2f} | {short_opinion}")
    print()

    if dry_run:
        print("[DRY-RUN] 미리보기 모드 - 실제 삭제 안됨.")
        print("  삭제 실행:         python delete_sqlite.py")
        print("  score=0 전체 제거: python delete_sqlite.py --all-zero")
    else:
        ids_to_delete = [row[0] for row in failed]
        deleted = delete_failed_records(conn, ids_to_delete)
        print(f"✅ {deleted}개 레코드 삭제 완료!")
        print_summary(conn)

    conn.close()


if __name__ == "__main__":
    main()
