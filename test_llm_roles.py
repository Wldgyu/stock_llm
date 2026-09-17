"""Check persona input boundaries without loading ML models or calling Ollama."""
import ast
import math
import os
import json
import re
import time
import uuid
from pathlib import Path
import unittest
from typing import Optional
from unittest.mock import Mock


TREE = ast.parse(Path(__file__).with_name('llm_stock.py').read_text(encoding='utf-8'))


class PersonaInputTests(unittest.TestCase):
    def setUp(self):
        simulator = next(n for n in TREE.body if isinstance(n, ast.ClassDef))
        namespace = {'os': os, 'math': math, 'Optional': Optional}
        exec(compile(ast.Module(body=[simulator], type_ignores=[]), 'simulator', 'exec'), namespace)
        self.sim = namespace['MarketAgentSimulator']()
        self.sim._send_request = Mock(return_value='ok')

    def prompt(self):
        return self.sim._send_request.call_args.args[1]

    def test_news_only(self):
        self.sim.run_news_expert('TEST', 'NEWS_CONTENT')
        self.assertIn('NEWS_CONTENT', self.prompt())
        for value in ('ML', 'MAE', 'naive', '백테스트', 'Sharpe', 'MDD', '비중'):
            self.assertNotIn(value, self.prompt())

    def test_stock_receives_prices_and_predictions(self):
        self.sim.run_stock_expert('TEST', 'PRICE_CONTENT', 'PREDICTION_CONTENT')
        self.assertIn('PRICE_CONTENT', self.prompt())
        self.assertIn('PREDICTION_CONTENT', self.prompt())

    def test_final_receives_expert_reports_and_failure(self):
        self.sim.run_final_decision_maker('TEST', 'STOCK_REPORT', None, 'NEWS_REPORT', 0.2)
        for value in ('STOCK_REPORT', 'NEWS_REPORT', '분석 실패', '+0.20'):
            self.assertIn(value, self.prompt())
        self.assertNotIn('3. 모델 검증 결과', self.prompt())

    def test_main_call_signatures(self):
        agent_class = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'StockAIAgentV4')
        namespace = {'StockAIAgentV3': object, 'DEFAULT_DB_PATH': '', 'Optional': Optional,
                     'uuid': uuid, 'time': time}
        exec(compile(ast.Module(body=[agent_class], type_ignores=[]), 'agent', 'exec'), namespace)
        agent = namespace['StockAIAgentV4'].__new__(namespace['StockAIAgentV4'])
        agent.pause_seconds = 0
        agent._stock_context = Mock(return_value=('PRICE', 'PREDICTION'))
        agent._news_context = Mock(return_value='NEWS')
        agent.simulator = self.sim
        self.sim.run_stock_expert = Mock(return_value='stock raw')
        self.sim.run_news_expert = Mock(return_value='news raw')
        self.sim.run_final_decision_maker = Mock(return_value='final raw')
        self.sim.parse_response = Mock(side_effect=[('stock failed', None), ('news report', .2), ('final report', 0.)])
        agent.save_persona_result = Mock()
        result = agent._discuss_stock('TEST', {'analysis_id': 42})
        self.sim.run_stock_expert.assert_called_once_with('TEST', 'PRICE', 'PREDICTION')
        self.sim.run_news_expert.assert_called_once_with('TEST', 'NEWS')
        self.sim.run_final_decision_maker.assert_called_once_with('TEST', 'stock failed', None, 'news report', .2)
        saved = agent.save_persona_result.call_args_list
        self.assertEqual([c.args[1] for c in saved], ['주식전문가', '뉴스기업전문가', '최종결정자'])
        self.assertEqual(len({c.args[4] for c in saved}), 1)
        self.assertTrue(all(c.args[5] == 42 for c in saved))
        self.assertEqual(result['주식전문가'][1], None)

    def test_invalid_scores_remain_failures(self):
        # Parser methods use these globals at runtime.
        self.sim.parse_score_from_response.__func__.__globals__.update(json=json, re=re)
        for value in ('{"score":true}', '{"score":NaN}', '{"score":2}',
                      '{"score":0.2,"score":0.3}', '{"score":0.3}\ntrailing'):
            self.assertIsNone(self.sim.parse_response(value)[1])
        self.assertEqual(self.sim.parse_response('opinion\n{"score":0.3}'), ('opinion', .3))

    def test_validation_omits_bulk_backtest(self):
        agent = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'StockAIAgentV4')
        fn = next(n for n in agent.body if isinstance(n, ast.FunctionDef) and n.name == 'format_validation_for_llm')
        fn.decorator_list = []
        namespace = {'Optional': Optional}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'formatter', 'exec'), namespace)
        result = namespace[fn.name]('LSTM', {
            'test': {'mae': 2., 'direction_accuracy': 55.},
            'naive_baseline': {'mae': 1.},
            'baseline_warning': '검증 미달',
            'strategy_backtest': {'strategies': 'BULK_BACKTEST'},
        })
        self.assertIn('검증 미달', result)
        self.assertNotIn('BULK_BACKTEST', result)


if __name__ == '__main__':
    unittest.main()
