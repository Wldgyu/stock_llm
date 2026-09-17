"""Keep 30 days of analysis history and each stock's latest ML snapshot.
Run: python cleanup_history.py [--db PATH] [--yes]
"""
from contextlib import closing
import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


def cleanup_history(db_path, days=30, apply=False, now=None):
    if days < 1:
        raise ValueError('days must be positive')
    path = Path(db_path).resolve()
    if not path.is_file():
        return {'ml': 0, 'personas': 0, 'missing_db': True}
    cutoff = ((now or datetime.now()) - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
    with closing(sqlite3.connect(path.as_uri() + '?mode=rw', uri=True, timeout=30)) as conn:
        conn.execute('BEGIN IMMEDIATE' if apply else 'BEGIN')
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        ml = list(conn.execute('SELECT id,name,timestamp FROM analysis_log')) if 'analysis_log' in tables else []
        latest = {}
        for id_,name,ts in ml:
            latest[name] = max(id_,latest.get(name,0))
        protected = set(latest.values())
        persons=[]
        linked=False
        if 'persona_discussion_log' in tables:
            linked = 'analysis_id' in {r[1] for r in conn.execute('PRAGMA table_info(persona_discussion_log)')}
            persons=list(conn.execute('SELECT id,timestamp,' + ('analysis_id' if linked else 'NULL') + ' FROM persona_discussion_log'))
        def old(ts):
            try:
                parsed=datetime.strptime(ts,'%Y-%m-%d %H:%M:%S')
                return parsed < datetime.strptime(cutoff,'%Y-%m-%d %H:%M:%S')
            except (ValueError,TypeError):
                return False  # Unknown timestamps are not guessed.
        # Keep ML referenced by a recent or undated persona, too.
        protected.update(aid for _,ts,aid in persons if aid is not None and not old(ts))
        ml_ids=[id_ for id_,_,ts in ml if old(ts) and id_ not in protected]
        persona_ids=[id_ for id_,ts,aid in persons if old(ts) and aid not in protected]
        result={'cutoff':cutoff,'ml':len(ml_ids),'personas':len(persona_ids),'applied':apply,'bytes_before':path.stat().st_size}
        if apply:
            conn.executemany('DELETE FROM persona_discussion_log WHERE id=?',[(id_,) for id_ in persona_ids]) if persona_ids else None
            conn.executemany('DELETE FROM analysis_log WHERE id=?',[(id_,) for id_ in ml_ids]) if ml_ids else None
            conn.commit()
            if ml_ids or persona_ids:
                try:
                    conn.execute('VACUUM')
                    conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                except sqlite3.OperationalError as exc:
                    result['compaction_warning']=str(exc)
        else:
            conn.rollback()
        result['bytes_after']=path.stat().st_size
        return result


def maintain_history(db_path):
    """Run after analysis so retention does not interrupt model work."""
    try:
        result=cleanup_history(db_path,apply=True)
        if result['ml'] or result['personas']:
            print(f"30일 기록 정리: {result}")
    except (OSError,sqlite3.Error,ValueError) as exc:
        print(f"기록 정리 실패(분석 결과는 유지): {exc}")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db')
    parser.add_argument('--yes',action='store_true')
    args=parser.parse_args()
    if not args.db:
        from database import DB_PATH
        args.db=DB_PATH
    print(cleanup_history(args.db,apply=args.yes))
