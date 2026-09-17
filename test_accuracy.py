"""Regression checks without loading ML models or calling Ollama."""
import ast
import json
import unittest
from pathlib import Path
from typing import Optional
import pandas as pd
from trading_dates import prediction_dates, completed_bars

class AccuracyTests(unittest.TestCase):
    def test_score_parser(self):
        tree=ast.parse(Path(__file__).with_name('llm_stock.py').read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='parse_score_from_response')
        namespace={'json':json,'Optional':Optional}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'parser','exec'),namespace)
        parse=lambda text:namespace['parse_score_from_response'](None,text)
        self.assertEqual(parse('opinion\n{"score":0.5}'),0.5)
        for text in ('{"score":0.45}\n{"score":2}', '{"score":true}', '{"score":"0.5"}', '{"score":NaN}', '{"score":0.2,"score":0.3}', '{"score":0.4}\ntrailing'):
            with self.subTest(text=text):self.assertIsNone(parse(text))
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
