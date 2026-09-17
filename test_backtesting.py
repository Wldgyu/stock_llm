import unittest
import numpy as np
import pandas as pd
from backtesting import compare_strategies

class BacktestTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'Open':[100.,120.,120.,120.], 'Close':[100.,120.,120.,120.], 'RSI':[20.,80.,50.,50.]}, index=pd.date_range('2025-01-01',periods=4))
    def test_next_open_not_previous_close(self):
        r=compare_strategies(self.frame(),{'lstm':{'2025-01-02':103.}},fee_bps=0,slippage_bps=0)
        self.assertAlmostEqual(r['strategies']['lstm']['return_pct'],0.)
        self.assertEqual(r['strategies']['lstm']['trades'][0]['price'],120.)
    def test_costs_and_initial_drawdown(self):
        r=compare_strategies(self.frame(),{'lstm':{'2025-01-02':103.}})['strategies']['lstm']
        expected=(1-.0005)*(1-.001)/((1+.0005)*(1+.001))
        self.assertAlmostEqual(r['return_pct'],(expected-1)*100)
        self.assertAlmostEqual(r['mdd_pct'],r['return_pct'])
    def test_no_signal_no_model_trade(self):
        r=compare_strategies(self.frame(),{'lstm':{'2025-01-02':99.}})
        self.assertEqual(r['strategies']['lstm']['orders'],0)
        self.assertEqual(r['strategies']['lstm']['sharpe'],0)
    def test_common_dates_and_gap_exit(self):
        p={'2025-01-02':103.,'2025-01-04':125.}
        r=compare_strategies(self.frame(),{'lstm':p,'tabpfn':dict(p,**{'2025-01-03':125.})})
        self.assertEqual(r['signal_days'],2)
        for name in ('lstm','tabpfn'):
            self.assertEqual(r['strategies'][name]['trades'][1]['date'],'2025-01-03')
            self.assertEqual(r['strategies'][name]['trades'][1]['side'],'sell')
        self.assertEqual(len(r['strategies']['rsi']['equity']),4)
    def test_rsi_uses_previous_day(self):
        r=compare_strategies(self.frame(),{'lstm':{'2025-01-02':99.,'2025-01-03':99.}})
        self.assertEqual(r['strategies']['rsi']['trades'][0]['side'],'buy')
        self.assertEqual(r['strategies']['rsi']['trades'][1]['date'],'2025-01-03')
    def test_bad_prices_fail(self):
        df=self.frame();df.iloc[1,0]=np.nan
        with self.assertRaises(ValueError): compare_strategies(df,{'lstm':{'2025-01-02':103.}})

if __name__=='__main__': unittest.main()
