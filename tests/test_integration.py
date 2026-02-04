# -*- coding: utf-8 -*-

"""
综合集成测试 - 测试整个系统的端到端流程

测试覆盖:
1. 数据层 -> 策略层 -> 回测层 完整流程
2. CLI命令行接口
3. 配置加载
4. 策略注册与创建
"""

import unittest
import sys
import os
import logging
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestConfigLoading(unittest.TestCase):
    """测试配置加载"""
    
    def test_config_import(self):
        """测试配置模块导入"""
        from config import (
            DEFAULT_STRATEGY,
            DEFAULT_DATA_SOURCE,
            STRATEGY_CONFIGS,
            DATA_SOURCE_CONFIGS,
        )
        
        self.assertEqual(DEFAULT_STRATEGY, 'momentum_v2')
        self.assertEqual(DEFAULT_DATA_SOURCE, 'tushare')
        self.assertIn('momentum_v2', STRATEGY_CONFIGS)
        self.assertIn('tushare', DATA_SOURCE_CONFIGS)
    
    def test_strategy_configs(self):
        """测试策略配置完整性"""
        from config.strategy_config import STRATEGY_CONFIGS
        
        # 检查必要的策略配置
        required_strategies = ['momentum_v2', 'random']
        for name in required_strategies:
            self.assertIn(name, STRATEGY_CONFIGS)
            self.assertIn('name', STRATEGY_CONFIGS[name])
    
    def test_data_source_configs(self):
        """测试数据源配置完整性"""
        from config.data_source_config import DATA_SOURCE_CONFIGS
        
        required_sources = ['tushare', 'akshare', 'tencent']
        for name in required_sources:
            self.assertIn(name, DATA_SOURCE_CONFIGS)
            self.assertIn('enabled', DATA_SOURCE_CONFIGS[name])


class TestStrategyRegistry(unittest.TestCase):
    """测试策略注册表"""
    
    def test_list_strategies(self):
        """测试列出所有策略"""
        from strategy import StrategyRegistry
        
        strategies = StrategyRegistry.list_all()
        self.assertIn('momentum_v2', strategies)
        self.assertIn('random', strategies)
    
    def test_create_strategy(self):
        """测试创建策略实例"""
        from strategy import StrategyRegistry
        
        strategy = StrategyRegistry.create('momentum_v2')
        self.assertIsNotNone(strategy)
        # 名称可能是中文或英文
        self.assertIn('momentum', strategy.name.lower().replace('动量', 'momentum'))
    
    def test_strategy_interface(self):
        """测试策略接口完整性"""
        from strategy import StrategyRegistry
        
        strategy = StrategyRegistry.create('momentum_v2')
        
        # 检查必要的方法
        self.assertTrue(hasattr(strategy, 'score'))
        self.assertTrue(hasattr(strategy, 'select'))
        self.assertTrue(hasattr(strategy, 'filter'))
        self.assertTrue(hasattr(strategy, 'name'))


class TestDataManager(unittest.TestCase):
    """测试数据管理器"""
    
    def test_data_manager_creation(self):
        """测试数据管理器创建"""
        from data.manager import DataManager
        
        # 使用缓存，避免网络请求
        manager = DataManager(source='tushare', use_cache=True)
        self.assertIsNotNone(manager)
        self.assertIn('tushare', manager.name.lower())
    
    def test_data_source_interface(self):
        """测试数据管理器实现DataSource接口"""
        from data.manager import DataManager
        from core.interfaces import DataSource
        
        manager = DataManager(source='tushare', use_cache=True)
        
        # 检查是否实现了DataSource接口
        self.assertTrue(isinstance(manager, DataSource))
        self.assertTrue(hasattr(manager, 'get_stock_data'))
        self.assertTrue(hasattr(manager, 'get_index_constituents'))


class TestBacktestEngine(unittest.TestCase):
    """测试回测引擎"""
    
    def test_backtest_config(self):
        """测试回测配置"""
        from core.types import BacktestConfig
        
        config = BacktestConfig(
            start_date='2024-01-01',
            end_date='2024-06-30',
            initial_capital=100000,
        )
        
        self.assertEqual(config.start_date, '2024-01-01')
        self.assertEqual(config.initial_capital, 100000)
    
    def test_backtest_engine_creation(self):
        """测试回测引擎创建"""
        from data.manager import DataManager
        from strategy import StrategyRegistry
        from backtest import BacktestEngine
        from backtest.engine import BacktestConfig as EngineConfig
        
        data_source = DataManager(source='tushare', use_cache=True)
        strategy = StrategyRegistry.create('momentum_v2')
        # 使用backtest.engine中的BacktestConfig，确保有enable_cost属性
        config = EngineConfig(
            start_date='2024-06-01',
            end_date='2024-07-02',
            initial_capital=100000,
        )
        
        engine = BacktestEngine(data_source, strategy, config)
        self.assertIsNotNone(engine)


class TestCLICommands(unittest.TestCase):
    """测试CLI命令（仅测试参数解析，不执行）"""
    
    def test_cli_import(self):
        """测试CLI模块导入"""
        import cli
        self.assertTrue(hasattr(cli, 'main'))
        self.assertTrue(hasattr(cli, 'cmd_backtest'))
        self.assertTrue(hasattr(cli, 'cmd_strategies'))
    
    def test_backtest_entry_module(self):
        """测试backtest入口文件存在"""
        import os
        # backtest.py 与 backtest/ 包同名，直接检查文件存在
        backtest_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'backtest.py'
        )
        self.assertTrue(os.path.exists(backtest_file))
        
        # 验证文件内容包含预期函数
        with open(backtest_file, 'r') as f:
            content = f.read()
        self.assertIn('def run_backtest', content)
        self.assertIn('def run_select', content)


class TestRiskMetrics(unittest.TestCase):
    """测试风险指标计算"""
    
    def test_sharpe_ratio(self):
        """测试夏普比率计算"""
        from backtest.metrics import RiskMetrics
        
        # 模拟月度收益率
        returns = [0.05, -0.02, 0.03, 0.01, -0.01, 0.04]
        
        metrics = RiskMetrics()
        result = metrics.calculate(returns, risk_free_rate=0.03)
        
        self.assertIsNotNone(result.sharpe_ratio)
        self.assertIsNotNone(result.max_drawdown)
    
    def test_max_drawdown(self):
        """测试最大回撤计算"""
        from backtest.metrics import RiskMetrics
        
        # 模拟净值序列：先涨后跌
        returns = [0.1, 0.1, -0.05, -0.1, 0.05]
        
        metrics = RiskMetrics()
        result = metrics.calculate(returns)
        
        # 最大回撤应该大于0
        self.assertGreater(result.max_drawdown, 0)


class TestTradingCost(unittest.TestCase):
    """测试交易成本计算"""
    
    def test_cost_calculation(self):
        """测试交易成本计算"""
        from backtest.cost import TradingCost, CostConfig
        
        config = CostConfig(
            commission_rate=0.00025,
            stamp_tax_rate=0.001,
            slippage=0.001,
        )
        
        cost = TradingCost(config)
        
        # 买入10万元股票，1000股
        buy_cost = cost.calculate_buy_cost(100000, 1000)
        self.assertGreater(buy_cost.total, 0)
        
        # 卖出10万元股票，1000股
        sell_cost = cost.calculate_sell_cost(100000, 1000)
        self.assertGreater(sell_cost.total, 0)
        
        # 卖出成本应该更高（有印花税）
        self.assertGreater(sell_cost.total, buy_cost.total)


class TestEndToEnd(unittest.TestCase):
    """端到端集成测试"""
    
    def test_full_workflow_with_mock(self):
        """测试完整工作流程（使用模拟数据）"""
        from core.types import StockData, ScoreResult
        from core.interfaces import DataSource, Strategy
        from backtest import BacktestEngine
        from backtest.engine import BacktestConfig as EngineConfig
        from typing import List, Optional
        
        # 创建模拟数据源
        class MockDataSource(DataSource):
            @property
            def name(self) -> str:
                return "Mock"
            
            def get_stock_data(self, code: str, date: str) -> Optional[StockData]:
                return StockData(
                    code=code,
                    name=f"股票{code}",
                    price=10.0,
                    change_pct=1.0,
                    pe_ratio=15.0,
                    pb_ratio=2.0,
                    roe=12.0,
                    profit_growth=20.0,
                    momentum_20d=5.0,
                    turnover_rate=2.0,
                    dividend_yield=2.0,
                )
            
            def get_index_constituents(self, index_code: str, date: str = None) -> List[str]:
                return ['000001', '000002', '000003', '000004', '000005']
            
            def get_stock_list(self) -> List[str]:
                return self.get_index_constituents('000300')
            
            def get_index_data(self, index_code: str, start_date: str, end_date: str = None):
                from core.types import IndexData
                return IndexData(
                    code=index_code,
                    name='沪深300',
                    close=4000.0,
                    change_pct=-1.0,
                    return_pct=-1.0,
                )
        
        # 创建模拟策略
        class MockStrategy(Strategy):
            @property
            def name(self) -> str:
                return "Mock Strategy"
            
            def score(self, stock: StockData) -> ScoreResult:
                return ScoreResult(
                    total=50,
                    breakdown={'test': 50},
                    grade='B',
                    risk_flag=False,
                )
            
            def select(self, stocks: List[StockData], top_n: int = 10) -> List[StockData]:
                return stocks[:min(top_n, len(stocks))]
            
            def filter(self, stock: StockData) -> bool:
                return True
        
        # 使用backtest.engine中的BacktestConfig
        config = EngineConfig(
            start_date='2024-06-01',
            end_date='2024-07-02',
            initial_capital=100000,
            top_n=3,
        )
        
        # 执行回测
        engine = BacktestEngine(MockDataSource(), MockStrategy(), config)
        result = engine.run_monthly()
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.total_return)
        self.assertIsNotNone(result.monthly_returns)


if __name__ == '__main__':
    unittest.main(verbosity=2)
