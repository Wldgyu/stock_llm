"""Mode routing and saved T+1 metric checks without external API calls."""
import ast
from contextlib import closing
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import types
import unittest
from typing import Optional
from unittest.mock import Mock, patch


TREE = ast.parse(Path(__file__).with_name('llm_stock.py').read_text(encoding='utf-8'))


class ModeTests(unittest.TestCase):
    def setUp(self):
        simulator = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'MarketAgentSimulator')
        self.requests = types.SimpleNamespace(
            post=Mock(), exceptions=types.SimpleNamespace(RequestException=Exception))
        namespace = {'os': os, 'math': math, 'Optional': Optional,
                     'requests': self.requests, 'time': time, 'json': json}
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
        self.assertEqual(args.kwargs['json']['model'], 'nvidia/nemotron-3-super-120b-a12b')
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
                     'Optional': Optional, 'closing': closing}
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
                             'last_close REAL, lstm_t1 REAL, tabpfn_t1 REAL)')
                conn.executemany('INSERT INTO analysis_log VALUES (?,?,?,?,?,?,?,?)', [
                    (1, 'TEST', '2026-09-01 10:00', '2026-09-02', 110, 100, 105, 95),
                    (2, 'TEST', '2026-09-01 11:00', '2026-09-02', 110, 100, 106, 94),
                    (3, 'TEST', '2026-09-03 10:00', '2026-09-03', 110, 100, 95, 105),
                ])
            agent = namespace['StockAIAgentV4'].__new__(namespace['StockAIAgentV4'])
            agent.db_path = path
            result = agent._realized_t1_context('TEST')
        self.assertIn('LSTM 방향 적중 100% (1건)', result)
        self.assertIn('TabPFN 방향 적중 0% (1건)', result)


if __name__ == '__main__':
    unittest.main()
