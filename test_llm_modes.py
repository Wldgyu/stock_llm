"""Mode routing and saved T+1 metric checks without external API calls."""
import ast
from contextlib import closing
import json
import math
import os
import re
from pathlib import Path
import sqlite3
import tempfile
import time
import types
import unittest
import pandas as pd
from typing import Optional
from unittest.mock import Mock, patch
import forecast_tracking


TREE = ast.parse(Path(__file__).with_name('llm_stock.py').read_text(encoding='utf-8'))


class ModeTests(unittest.TestCase):
    def setUp(self):
        simulator = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'MarketAgentSimulator')
        self.requests = types.SimpleNamespace(
            post=Mock(), exceptions=types.SimpleNamespace(RequestException=Exception))
        namespace = {'os': os, 'math': math, 'Optional': Optional,
                     'requests': self.requests, 'time': time, 'json': json, 're': re}
        exec(compile(ast.Module(body=[simulator], type_ignores=[]), 'simulator', 'exec'), namespace)
        self.simulator = namespace['MarketAgentSimulator']

    def test_simple_uses_local_model(self):
        with patch.dict(os.environ, {'NVIDIA_TOKEN': 'test-key'}):
            sim = self.simulator(analysis_mode='simple')
        sim._send_request = Mock(return_value='local')
        sim._send_nvidia_request = Mock()
        self.assertEqual(sim.run_stock_expert('TEST', 'PRICE', 'PREDICTION'), 'local')
        self.assertEqual(sim._send_request.call_args.kwargs['model_name'], 'gemma4:12b')
        sim._send_nvidia_request.assert_not_called()

    def test_local_request_uses_chat_roles_and_json_format(self):
        sim = self.simulator(analysis_mode='simple')
        self.requests.post.return_value = Mock(status_code=200,
            json=lambda: {'message': {'content': '{"score": 0.2}'}})
        self.assertIn('score', sim.run_news_expert('TEST', 'NEWS'))
        request = self.requests.post.call_args_list[0]
        self.assertTrue(request.args[0].endswith('/api/chat'))
        self.assertEqual(request.kwargs['json']['format'], 'json')
        self.assertEqual([item['role'] for item in request.kwargs['json']['messages']], ['system', 'user'])
        self.assertIn('NEWS', request.kwargs['json']['messages'][1]['content'])

    def test_professional_uses_nvidia_without_key_logging(self):
        with patch.dict(os.environ, {'NVIDIA_TOKEN': 'secret-test-value'}):
            sim = self.simulator(analysis_mode='professional')
        self.requests.post.return_value = Mock(
            json=lambda: {'choices': [{'message': {'content': '의견\n{"score":0.2}'}}]},
            raise_for_status=Mock())
        answer = sim.run_news_expert('TEST', 'NEWS')
        self.assertIn('의견', answer)
        args = self.requests.post.call_args
        self.assertEqual(args.args[0], 'https://integrate.api.nvidia.com/v1/chat/completions')
        self.assertEqual(args.kwargs['json']['model'], 'nvidia/nemotron-3.5-super-vl-120b-a12b')
        self.assertIn('NEWS', args.kwargs['json']['messages'][1]['content'])
        self.assertNotIn('NEWS', args.kwargs['json']['messages'][0]['content'])
        self.assertEqual(args.kwargs['headers']['Authorization'], 'Bearer secret-test-value')

    def test_professional_requires_token(self):
        with patch.dict(os.environ, {'NVIDIA_TOKEN': ''}):
            with self.assertRaises(ValueError):
                self.simulator(analysis_mode='professional')

    def test_saved_t1_metric_deduplicates_and_ignores_late_forecasts(self):
        agent_node = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'StockAIAgentV4')
        namespace = {'StockAIAgentV3': object, 'DEFAULT_DB_PATH': '',
                     'Optional': Optional, 'closing': closing,
                     'math': math, 't1_history': forecast_tracking.t1_history}
        def connect_db(path, row_factory=False):
            conn = sqlite3.connect(path)
            if row_factory:
                conn.row_factory = sqlite3.Row
            return conn
        namespace['connect_db'] = connect_db
        exec(compile(ast.Module(body=[agent_node], type_ignores=[]), 'agent', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'history.db')
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute('CREATE TABLE analysis_log (id INTEGER PRIMARY KEY, name TEXT, '
                             'timestamp TEXT, target_t1 TEXT, t1_actual_close REAL, '
                             'last_close REAL, lstm_t1 REAL, tabpfn_t1 REAL, symbol TEXT, data_date TEXT)')
                conn.executemany('INSERT INTO analysis_log VALUES (?,?,?,?,?,?,?,?,?,?)', [
                    (1, 'TEST', '2026-09-01 10:00', '2026-09-02', 110, 100, 105, 95, 'TEST', '2026-09-01'),
                    (2, 'TEST', '2026-09-02 08:00', '2026-09-02', 110, 100, 106, 94, 'TEST', '2026-09-01'),
                    (3, 'TEST', '2026-09-02 09:00', '2026-09-02', 110, 100, 95, 105, 'TEST', '2026-09-01'),
                ])
            agent = namespace['StockAIAgentV4'].__new__(namespace['StockAIAgentV4'])
            agent.db_path = path
            with patch.object(forecast_tracking, 'saved_before_open',
                              side_effect=lambda timestamp, utc, target, symbol, **kw: timestamp < target + ' 09:00'):
                result = agent._realized_t1_context('TEST')
        self.assertIn('LSTM 방향 적중 100% (1건)', result)
        self.assertIn('TabPFN 방향 적중 0% (1건)', result)

    def test_latest_preopen_pending_is_not_replaced_by_older_result(self):
        agent_node = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'StockAIAgentV4')
        namespace = {'StockAIAgentV3': object, 'DEFAULT_DB_PATH': '', 'Optional': Optional,
                     'closing': closing, 'math': math, 't1_history': forecast_tracking.t1_history}
        def connect_db(path, row_factory=False):
            conn = sqlite3.connect(path)
            if row_factory:
                conn.row_factory = sqlite3.Row
            return conn
        namespace['connect_db'] = connect_db
        exec(compile(ast.Module(body=[agent_node], type_ignores=[]), 'agent', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'history.db')
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute('CREATE TABLE analysis_log (id INTEGER PRIMARY KEY, name TEXT, timestamp TEXT, '
                             'target_t1 TEXT, t1_actual_close REAL, last_close REAL, lstm_t1 REAL, tabpfn_t1 REAL, symbol TEXT, data_date TEXT)')
                conn.executemany('INSERT INTO analysis_log VALUES (?,?,?,?,?,?,?,?,?,?)', [
                    (1, 'TEST', '2026-09-01 10:00', '2026-09-02', 110, 100, 105, 95, 'TEST', '2026-09-01'),
                    (2, 'TEST', '2026-09-02 08:00', '2026-09-02', None, 100, 95, 105, 'TEST', '2026-09-01'),
                ])
            agent = namespace['StockAIAgentV4'].__new__(namespace['StockAIAgentV4'])
            agent.db_path = path
            with patch.object(forecast_tracking, 'saved_before_open', return_value=True), \
                 patch.object(forecast_tracking.yf, 'Ticker', side_effect=RuntimeError('offline')):
                result = agent._realized_t1_context('TEST')
        self.assertIn('LSTM 평가 0건', result)
        self.assertIn('조회 실패 1건', result)

    def test_history_refreshes_and_saves_evaluated_close(self):
        with sqlite3.connect(':memory:') as conn:
            conn.row_factory = sqlite3.Row
            conn.execute('CREATE TABLE analysis_log (id INTEGER PRIMARY KEY, name TEXT, timestamp TEXT, '
                         'timestamp_utc TEXT, target_t1 TEXT, last_close REAL, lstm_t1 REAL, '
                         'tabpfn_t1 REAL, symbol TEXT, data_date TEXT)')
            conn.execute('INSERT INTO analysis_log VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                         ('TEST', '2026-09-01 10:00', None, '2026-09-02', 100., 110., 90., 'NVDA', '2026-09-01'))
            frame = pd.DataFrame({'Close': [50., 55.]}, index=pd.to_datetime(['2026-09-01', '2026-09-02']))
            ticker = Mock()
            ticker.history.return_value = frame
            with patch.object(forecast_tracking.yf, 'Ticker', return_value=ticker), \
                 patch.object(forecast_tracking, 'completed_bars', side_effect=lambda f, s: f):
                history = forecast_tracking.t1_history(conn, 'TEST', unique_targets=True)
            self.assertEqual(history[0]['status'], 'complete')
            self.assertAlmostEqual(history[0]['actual'], 110.)
            self.assertIsNotNone(history[0]['evaluated_at'])
            self.assertTrue(history[0]['models']['lstm']['direction_correct'])
            self.assertAlmostEqual(conn.execute('SELECT t1_actual_close FROM analysis_log').fetchone()[0], 110.)

    def test_evaluate_all_horizons_and_realized_summary_for_llm(self):
        with sqlite3.connect(':memory:') as conn:
            conn.row_factory = sqlite3.Row
            conn.execute('CREATE TABLE analysis_log (id INTEGER PRIMARY KEY, name TEXT, timestamp TEXT, '
                         'timestamp_utc TEXT, target_t1 TEXT, target_t4 TEXT, target_t7 TEXT, '
                         'last_close REAL, lstm_t1 REAL, lstm_t4 REAL, lstm_t7 REAL, '
                         'tabpfn_t1 REAL, tabpfn_t4 REAL, tabpfn_t7 REAL, symbol TEXT, data_date TEXT)')
            conn.execute('INSERT INTO analysis_log VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                         ('TEST', '2026-09-01 10:00', None, '2026-09-02', '2026-09-08', '2026-09-11',
                          100., 105., 108., 112., 98., 101., 104., 'NVDA', '2026-09-01'))
            frame = pd.DataFrame(
                {'Close': [100., 104., 106., 109.]},
                index=pd.to_datetime(['2026-09-01', '2026-09-02', '2026-09-08', '2026-09-11']),
            )
            ticker = Mock()
            ticker.history.return_value = frame
            with patch.object(forecast_tracking.yf, 'Ticker', return_value=ticker), \
                 patch.object(forecast_tracking, 'completed_bars', side_effect=lambda f, s: f):
                histories = forecast_tracking.evaluate_all_horizons(conn, 'TEST', unique_targets=True)
                summary_text = forecast_tracking.realized_summary_for_llm(conn, 'TEST', history=histories[1])
            self.assertEqual(histories[1][0]['status'], 'complete')
            self.assertEqual(histories[4][0]['status'], 'complete')
            self.assertEqual(histories[7][0]['status'], 'complete')
            row = conn.execute('SELECT t1_actual_close, t4_actual_close, t7_actual_close FROM analysis_log').fetchone()
            self.assertAlmostEqual(row[0], 104.)
            self.assertAlmostEqual(row[1], 106.)
            self.assertAlmostEqual(row[2], 109.)
            self.assertIn('판단 근거 참고 비중: ML 예측 40% : 뉴스 60%', summary_text)
            self.assertIn('LSTM 방향 적중 100% (1건)', summary_text)


if __name__ == '__main__':
    unittest.main()
