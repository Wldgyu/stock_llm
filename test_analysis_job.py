"""Dashboard launch endpoint: local-only, validated, and single-job."""
import unittest
import json
import sqlite3
from unittest.mock import patch

import app


class FakeThread:
    def __init__(self, target, args=(), **kwargs):
        self.target, self.args = target, args

    def start(self):
        pass


class AnalysisJobTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()
        app._analysis_job = {"status": "idle", "name": None, "mode": None, "message": ""}
        self.poller = patch.object(app, 'start_price_poller', lambda: None)
        self.poller.start()

    def tearDown(self):
        self.poller.stop()

    def post(self, payload, address='127.0.0.1'):
        return self.client.post('/api/analysis-job', json=payload,
                                environ_overrides={'REMOTE_ADDR': address})

    def test_local_start_and_duplicate_prevention(self):
        with patch.object(app.threading, 'Thread', FakeThread):
            response = self.post({'name': '삼성전자', 'mode': 'professional'})
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json['status'], 'running')
            self.assertEqual(self.post({'name': '인텔', 'mode': 'simple'}).status_code, 409)
            self.assertEqual(self.client.get('/api/analysis-job').json['name'], '삼성전자')

    def test_rejects_remote_and_invalid_requests(self):
        self.assertEqual(self.post({'name': '삼성전자', 'mode': 'simple'}, '192.168.1.2').status_code, 403)
        self.assertEqual(self.post({'name': 'unknown', 'mode': 'simple'}).status_code, 400)
        self.assertEqual(self.post(['삼성전자', 'simple']).status_code, 400)

    def test_partial_status_and_news_sources_in_api(self):
        connection = sqlite3.connect(':memory:')
        connection.row_factory = sqlite3.Row
        connection.execute('CREATE TABLE analysis_log (id INTEGER PRIMARY KEY, name TEXT, news_sources_json TEXT)')
        connection.execute('CREATE TABLE persona_discussion_log (id INTEGER PRIMARY KEY, ticker_name TEXT, '
                           'persona TEXT, score REAL, opinion TEXT, timestamp TEXT, run_id TEXT, '
                           'analysis_id INTEGER, analysis_status TEXT)')
        sources = [{'id': 'N1', 'title': '기사', 'source': 'Reuters', 'url': 'https://example.com/news'}]
        connection.execute('INSERT INTO analysis_log VALUES (1, ?, ?)', ('삼성전자', json.dumps(sources)))
        connection.executemany('INSERT INTO persona_discussion_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', [
            (1, '삼성전자', '주식전문가', None, '분석 실패', '', 'run', 1, None),
            (2, '삼성전자', '뉴스기업전문가', .2, '근거 N1', '', 'run', 1, None),
            (3, '삼성전자', '최종결정자', .1, '판단', '', 'run', 1, 'partial'),
        ])
        self.assertEqual(app._latest_persona_map(connection)['삼성전자']['analysis_status'], 'partial')
        with patch.object(app, 'get_connection', return_value=connection):
            response = self.client.get('/api/stock/삼성전자/ai')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['status'], 'partial')
        self.assertEqual(response.json['news_sources'][0]['id'], 'N1')


if __name__ == '__main__':
    unittest.main()
