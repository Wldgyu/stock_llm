"""Exercise analysis migrations and persistence without downloading models."""
import ast
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


class AnalysisStorageTests(unittest.TestCase):
    def test_existing_table_migrates_and_saves_utc_and_sources(self):
        tree = ast.parse(Path(__file__).with_name('ml_stock.py').read_text(encoding='utf-8'))
        agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'StockAIAgentV3')
        functions = [node for node in agent.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('_init_analysis_db', '_save_analysis_result')]
        namespace = {'closing': closing, 'datetime': datetime, 'timezone': timezone, 'ZoneInfo': ZoneInfo,
                     'np': np, 'json': json, 'connect_db': lambda path: sqlite3.connect(path)}
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'storage', 'exec'), namespace)
        storage = type('Storage', (), {name: namespace[name] for name in ('_init_analysis_db', '_save_analysis_result')})()
        with tempfile.TemporaryDirectory() as directory:
            storage.db_path = str(Path(directory) / 'history.db')
            storage._init_analysis_db()
            with closing(sqlite3.connect(storage.db_path)) as connection, connection:
                connection.execute('ALTER TABLE analysis_log DROP COLUMN timestamp_utc')
                connection.execute('ALTER TABLE analysis_log DROP COLUMN news_sources_json')
            storage._init_analysis_db()
            dates = pd.date_range('2026-01-01', periods=30)
            frame = pd.DataFrame({'RSI': 50., 'Close': np.arange(100, 130), 'Low': 90., 'High': 140.,
                                  'USD_KRW': 1300., 'SOX_Index': 5000.}, index=dates)
            result = {'df': frame, 'preds': [130., 131., 132.], 'symbol': 'NVDA',
                      'lstm_metrics': {}, 'data_date': '2026-01-30', 'target_t1': '2026-02-02',
                      'target_t4': '2026-02-05', 'target_t7': '2026-02-10',
                      'news_sources': [{'id': 'N1', 'url': 'https://example.com'}]}
            storage._save_analysis_result('TEST', result)
            with closing(sqlite3.connect(storage.db_path)) as connection:
                row = connection.execute('SELECT timestamp, timestamp_utc, news_sources_json FROM analysis_log').fetchone()
            self.assertIsNotNone(datetime.fromisoformat(row[1]).utcoffset())
            self.assertEqual(json.loads(row[2])[0]['id'], 'N1')
            self.assertEqual(result['analysis_id'], 1)


if __name__ == '__main__':
    unittest.main()
