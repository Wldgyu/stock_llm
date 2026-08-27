# -*- coding: utf-8 -*-
"""
delete_sqlite.py
────────────────────────────────────────────────────────────────
[역할] SQLite DB 에서 분석 실패 레코드를 조회·삭제합니다.

[삭제 기준]
  persona_discussion_log 테이블:
    - score 가 NULL 이거나 opinion 이 '분석 실패'로 시작하는 레코드
    - opinion 이 비어 있는 레코드

[실행 방법]
  python delete_sqlite.py          -> 삭제 없이 대상 미리보기
  python delete_sqlite.py --yes    -> 확인 후 실패 레코드 삭제
  python delete_sqlite.py --db 경로 -> 다른 DB 대상 미리보기
"""

import pathlib
from datetime import datetime

from database import DB_PATH, connect_db, table_exists


# ── 실패 레코드 조회 ─────────────────────────────────
def fetch_failed_records(conn):
    query = """
        SELECT id, timestamp, ticker_name, persona, opinion, score
        FROM persona_discussion_log
        WHERE score IS NULL
           OR opinion LIKE '분석 실패:%'
           OR opinion = '분석 실패'
           OR opinion IS NULL
           OR TRIM(opinion) = ''
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
    fail = conn.execute(
        """
        SELECT COUNT(*) FROM persona_discussion_log
        WHERE score IS NULL OR opinion LIKE '분석 실패:%'
           OR opinion = '분석 실패' OR opinion IS NULL OR TRIM(opinion) = ''
        """
    ).fetchone()[0]
    ok = total - fail

    print(f"\n{'='*55}")
    print(f"  [persona_discussion_log 현황]")
    print(f"{'='*55}")
    print(f"  전체 레코드  : {total:>6}행")
    print(f"  정상 레코드  : {ok:>6}행")
    print(f"  실패 레코드  : {fail:>6}행  (NULL 또는 명시적 실패)")
    print(f"{'='*55}\n")


# ── 티커별 집계 ───────────────────────────────────────
def print_ticker_summary(conn):
    rows = conn.execute("""
        SELECT ticker_name,
               COUNT(*) as total,
               SUM(CASE WHEN score IS NULL OR opinion LIKE '분석 실패:%'
                         OR opinion = '분석 실패' OR opinion IS NULL
                         OR TRIM(opinion) = '' THEN 1 ELSE 0 END) as failed,
               SUM(CASE WHEN score IS NOT NULL
                         AND opinion NOT LIKE '분석 실패:%'
                         AND opinion != '분석 실패'
                         AND TRIM(COALESCE(opinion, '')) != ''
                        THEN 1 ELSE 0 END) as success
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
    confirmed = "--yes" in args

    if "--all-zero" in args:
        print("안전상 --all-zero 옵션은 제거되었습니다. 중립 점수 0.0은 삭제하지 않습니다.")
        return

    # --db 옵션 처리
    db_path = pathlib.Path(DB_PATH)
    if "--db" in args:
        idx = args.index("--db")
        if idx + 1 < len(args):
            db_path = pathlib.Path(args[idx + 1])

    if not db_path.exists():
        print(f"DB 파일을 찾을 수 없습니다: {db_path}")
        return

    conn = connect_db(str(db_path))
    print(f"\nDB: {db_path}")
    print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not table_exists(conn, "persona_discussion_log"):
        print("persona_discussion_log 테이블이 없습니다.")
        conn.close()
        return

    print_summary(conn)
    print_ticker_summary(conn)

    failed = fetch_failed_records(conn)

    if not failed:
        print("삭제 대상 레코드가 없습니다. DB가 이미 깨끗합니다!")
        conn.close()
        return

    print(f"{'─'*55}")
    print(f"  삭제 대상 (분석 실패): {len(failed)}행")
    print(f"{'─'*55}")
    for row in failed:
        rid, ts, ticker, persona, opinion, score = row
        short_opinion = (opinion or "")[:30].replace("\n", " ")
        score_text = f"{score:.2f}" if score is not None else "NULL"
        print(f"  id={rid:>4} | {ts} | {ticker:<8} | {persona:<12} | score={score_text} | {short_opinion}")
    print()

    if confirmed:
        ids_to_delete = [row[0] for row in failed]
        deleted = delete_failed_records(conn, ids_to_delete)
        print(f"✅ {deleted}개 레코드 삭제 완료!")
        print_summary(conn)
    else:
        print("[미리보기] 실제 데이터는 삭제하지 않았습니다.")
        print("  위 실패 레코드를 삭제하려면: python delete_sqlite.py --yes")

    conn.close()


if __name__ == "__main__":
    main()
