"""프로젝트 전체에서 공통으로 사용하는 SQLite 설정입니다."""

import os
import shutil
import sqlite3
from pathlib import Path
from typing import Optional


PROJECT_DIR = Path(__file__).parent
LEGACY_DB_PATH = PROJECT_DIR / "stock_analysis.db"
DEFAULT_DATA_DIR = Path(
    os.environ.get("LOCALAPPDATA", PROJECT_DIR)
) / "StockAIDashboard"

if os.environ.get("DB_PATH"):
    DB_PATH = str(Path(os.environ["DB_PATH"]).expanduser().resolve())
else:
    DB_PATH = str(DEFAULT_DATA_DIR / "stock_analysis.db")


def _prepare_database_path(db_path):
    """기존 프로젝트 DB를 안전한 로컬 데이터 폴더로 한 번 복사합니다."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if (
        not os.environ.get("DB_PATH")
        and not path.exists()
        and LEGACY_DB_PATH.exists()
        and path.resolve() != LEGACY_DB_PATH.resolve()
    ):
        shutil.copy2(LEGACY_DB_PATH, path)
    return path


def archive_legacy_workspace_db() -> Optional[Path]:
    """
    로컬 데이터 폴더(DB_PATH)가 활성화된 상태에서 워크스페이스 루트에 구버전
    stock_analysis.db가 남아 있으면 .legacy_backup으로 이름을 바꿔 혼선을 방지합니다.
    """
    active = Path(DB_PATH).resolve()
    legacy = LEGACY_DB_PATH.resolve()
    if active != legacy and active.exists() and legacy.exists():
        backup = LEGACY_DB_PATH.with_name("stock_analysis.db.legacy_backup")
        if not backup.exists():
            LEGACY_DB_PATH.rename(backup)
            return backup
    return None


def connect_db(db_path=None, row_factory=False):
    """잠금 대기와 WAL 모드를 적용한 SQLite 연결을 반환합니다."""
    path = _prepare_database_path(db_path or DB_PATH)
    connection = sqlite3.connect(str(path), timeout=30)
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA journal_mode=WAL")
    if row_factory:
        connection.row_factory = sqlite3.Row
    return connection


def table_exists(connection, table_name):
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None
