"""Regression checks without loading ML models or calling Ollama."""
import unittest
import pandas as pd
from trading_dates import prediction_dates, completed_bars, saved_before_open

class AccuracyTests(unittest.TestCase):
    def test_saved_time_uses_exchange_open_and_utc(self):
        self.assertTrue(saved_before_open('2026-09-02 08:59:00', None, '2026-09-02', '005930.KS'))
        self.assertFalse(saved_before_open('2026-09-02 09:00:00', None, '2026-09-02', '005930.KS'))
        self.assertTrue(saved_before_open('2026-09-02 22:29:00', None, '2026-09-02', 'NVDA'))
        self.assertFalse(saved_before_open('2026-09-02 22:30:00', None, '2026-09-02', 'NVDA'))
        self.assertTrue(saved_before_open('ignored', '2026-09-02T13:29:00+00:00', '2026-09-02', 'NVDA'))
        self.assertIsNone(saved_before_open('invalid', None, '2026-09-02', 'NVDA'))
    def test_holiday_and_weekend(self):
        self.assertEqual(prediction_dates('2026-09-04','NVDA')['target_t1'],'2026-09-08')
        self.assertEqual(prediction_dates('2026-09-11','005930.KS')['target_t1'],'2026-09-14')
    def test_completed_bars(self):
        frame=pd.DataFrame({'Close':[1,2]},index=pd.to_datetime(['2026-09-10','2026-09-11']))
        self.assertEqual(len(completed_bars(frame,'005930.KS','2026-09-11 15:29+09:00')),1)
        self.assertEqual(len(completed_bars(frame,'005930.KS','2026-09-11 15:30+09:00')),2)
        early=pd.DataFrame({'Close':[1]},index=pd.to_datetime(['2026-11-27']))
        self.assertEqual(len(completed_bars(early,'NVDA','2026-11-27 12:59-05:00')),0)
        self.assertEqual(len(completed_bars(early,'NVDA','2026-11-27 13:01-05:00')),1)

if __name__=='__main__':unittest.main()
