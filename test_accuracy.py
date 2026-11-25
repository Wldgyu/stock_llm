"""Regression checks without loading ML models or calling Ollama."""
import unittest
import ast
from datetime import datetime, time as clock_time
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from trading_dates import prediction_dates, completed_bars, saved_before_open, forecast_validity, calendar_for, local_date

class AccuracyTests(unittest.TestCase):
    def test_forecast_validity_latest_close_and_target_open(self):
        check = lambda base, target, now: forecast_validity(base, target, 'NVDA', now)[0]
        self.assertTrue(check('2026-09-04', '2026-09-08', '2026-09-08 13:29:59+00:00'))
        self.assertFalse(check('2026-09-04', '2026-09-08', '2026-09-08 13:30:00+00:00'))
        self.assertFalse(check('2026-09-03', '2026-09-04', '2026-09-08 12:00+00:00'))
        self.assertFalse(check('2026-09-04', '2026-09-09', '2026-09-08 12:00+00:00'))
        self.assertTrue(forecast_validity('2026-11-27', '2026-11-30', 'NVDA', '2026-11-27 18:01+00:00')[0])
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

    def test_macro_alignment_uses_early_close_and_market_timezones(self):
        tree = ast.parse(Path(__file__).with_name('ml_stock.py').read_text(encoding='utf-8'))
        agent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StockAIAgentV3')
        methods = [n for n in agent.body if isinstance(n, ast.FunctionDef) and
                   n.name in ('_close_times_utc', '_session_close_times_utc', '_merge_confirmed_macro')]
        cls = ast.ClassDef(name='Macro', bases=[], keywords=[], body=methods, decorator_list=[])
        namespace = {'pd': pd, 'np': np, 'datetime': datetime, 'clock_time': clock_time,
                     'ZoneInfo': ZoneInfo, 'calendar_for': calendar_for, 'local_date': local_date}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), 'macro', 'exec'), namespace)
        macro = namespace['Macro']
        early = pd.to_datetime(['2026-11-27'])
        self.assertEqual(str(macro._session_close_times_utc(early, 'NVDA')[0]), '2026-11-27 18:00:00+00:00')
        stocks = pd.DataFrame(index=early)
        fx = pd.Series([1300., 1400.], index=pd.to_datetime(['2026-11-26', '2026-11-27']))
        sox = pd.Series([5000., 5100.], index=pd.to_datetime(['2026-11-25', '2026-11-27']))
        self.assertEqual(macro._merge_confirmed_macro(stocks, fx, 'NVDA', 'USD_KRW').iloc[0], 1300.)
        self.assertEqual(macro._merge_confirmed_macro(stocks, sox, 'NVDA', 'SOX_Index').iloc[0], 5100.)
        kr = pd.DataFrame(index=pd.to_datetime(['2026-11-27']))
        self.assertEqual(macro._merge_confirmed_macro(kr, sox, '005930.KS', 'SOX_Index').iloc[0], 5000.)

    def test_sequence_and_tabpfn_window_normalization_are_scale_invariant(self):
        tree = ast.parse(Path(__file__).with_name('ml_stock.py').read_text(encoding='utf-8'))
        agent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StockAIAgentV3')
        methods = [n for n in agent.body if isinstance(n, ast.FunctionDef) and
                   n.name in ('_horizon_targets', '_horizon_returns', '_returns_to_prices',
                              '_normalize_sequence_windows', '_normalize_tabpfn_window', '_clean_rss_summary')]
        cls = ast.ClassDef(name='NormHelper', bases=[], keywords=[], body=methods, decorator_list=[])
        from html import unescape
        import re
        namespace = {'np': np, 'unescape': unescape, 're': re}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), 'norm', 'exec'), namespace)
        helper = namespace['NormHelper']

        base_seq = np.array([[100., 1300., 5000.], [110., 1313., 5050.], [121., 1326.13, 5100.5]], dtype=np.float32)
        scaled_seq = base_seq * 10.0
        np.testing.assert_allclose(
            helper._normalize_sequence_windows(base_seq, 3),
            helper._normalize_sequence_windows(scaled_seq, 3),
            atol=1e-5,
        )
        np.testing.assert_allclose(helper._normalize_sequence_windows(base_seq, 3)[0, -1, :], 0.0, atol=1e-6)

        tab_win = np.array([
            [100., 1300., 5000., 30., 98., 1000.],
            [110., 1310., 5100., 70., 105., 2000.],
        ], dtype=np.float64)
        tab_win_scaled = tab_win.copy()
        tab_win_scaled[:, [0, 1, 2, 4, 5]] *= 50.0
        np.testing.assert_allclose(
            helper._normalize_tabpfn_window(tab_win),
            helper._normalize_tabpfn_window(tab_win_scaled),
            atol=1e-5,
        )

        prices = np.array([100., 102., 103., 104., 105., 106., 107., 110.])
        rets = helper._horizon_returns(prices, 0, 1)
        restored = helper._returns_to_prices(100., rets)
        np.testing.assert_allclose(restored, [102., 105., 110.], atol=1e-6)

        entry = {
            'title': 'Samsung chip packaging expansion to spur $30 billion investment',
            'summary': '<p>Samsung chip packaging expansion to spur $30 billion investment - Reuters reported that new advanced packaging lines will be built in South Korea by 2027.</p>',
        }
        summary = helper._clean_rss_summary(entry)
        self.assertIn('Reuters reported that new advanced packaging lines', summary)

if __name__=='__main__':unittest.main()
