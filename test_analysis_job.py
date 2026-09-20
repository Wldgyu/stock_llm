"""Dashboard launch endpoint: local-only, validated, and single-job."""
import unittest
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


if __name__ == '__main__':
    unittest.main()
