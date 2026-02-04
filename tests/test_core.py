# -*- coding: utf-8 -*-
"""
核心模块测试

验证 core/ 模块的基本功能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_types_import():
    """测试类型导入"""
    print("测试类型导入...")
    
    from core.types import (
        StockData, 
        IndexData, 
        TradeRecord, 
        MonthlyReturn,
        BacktestConfig,
        BacktestResult,
        ScoreResult
    )
    
    print("  ✓ 所有类型导入成功")
    return True

def test_interfaces_import():
    """测试接口导入"""
    print("测试接口导入...")
    
    from core.interfaces import DataSource, Strategy
    
    print("  ✓ 所有接口导入成功")
    return True

def test_stock_data():
    """测试 StockData 类"""
    print("测试 StockData 类...")
    
    from core.types import StockData
    
    # 创建实例
    stock = StockData(
        code='600036',
        name='招商银行',
        price=35.50,
        change_pct=1.25,
        pe_ratio=8.5,
        pb_ratio=1.2,
        momentum_20d=5.5
    )
    
    # 测试 to_dict
    d = stock.to_dict()
    assert d['code'] == '600036'
    assert d['price'] == 35.50
    print("  ✓ to_dict() 正常")
    
    # 测试 from_dict
    stock2 = StockData.from_dict(d)
    assert stock2.code == '600036'
    assert stock2.price == 35.50
    print("  ✓ from_dict() 正常")
    
    # 测试 is_valid
    assert stock.is_valid() == True
    invalid_stock = StockData(code='', price=0)
    assert invalid_stock.is_valid() == False
    print("  ✓ is_valid() 正常")
    
    return True

def test_backtest_config():
    """测试 BacktestConfig 类"""
    print("测试 BacktestConfig 类...")
    
    from core.types import BacktestConfig
    
    config = BacktestConfig(
        start_date='2024-01-01',
        end_date='2024-12-31',
        initial_capital=100000,
        top_n=10
    )
    
    # 测试默认值
    assert config.commission_rate == 0.00025
    assert config.benchmark_code == '000300'
    print("  ✓ 默认值正确")
    
    # 测试 to_dict
    d = config.to_dict()
    assert d['start_date'] == '2024-01-01'
    print("  ✓ to_dict() 正常")
    
    return True

def test_backtest_result():
    """测试 BacktestResult 类"""
    print("测试 BacktestResult 类...")
    
    from core.types import BacktestResult, MonthlyReturn
    
    result = BacktestResult(
        strategy_name='momentum_v2',
        start_date='2024-01-01',
        end_date='2024-12-31',
        total_return=25.5,
        benchmark_return=15.2,
        alpha=10.3,
        max_drawdown=8.5,
        sharpe_ratio=1.8,
        win_rate=65.0
    )
    
    # 测试 summary
    summary = result.summary()
    assert 'momentum_v2' in summary
    assert '25.50%' in summary
    print("  ✓ summary() 正常")
    
    # 测试 to_dict
    d = result.to_dict()
    assert d['total_return'] == 25.5
    print("  ✓ to_dict() 正常")
    
    return True

def test_strategy_interface():
    """测试 Strategy 抽象接口"""
    print("测试 Strategy 接口...")
    
    from core.interfaces import Strategy
    from core.types import StockData, ScoreResult
    
    # 创建一个简单的策略实现
    class SimpleStrategy(Strategy):
        @property
        def name(self) -> str:
            return "simple"
        
        def score(self, stock: StockData) -> ScoreResult:
            # 简单评分：PE越低分越高
            pe = stock.pe_ratio or 50
            score = max(0, 100 - pe * 2)
            return ScoreResult(
                total=score,
                breakdown={'pe_score': score},
                grade=self._get_grade(score),
                risk_flag=False
            )
        
        def filter(self, stock: StockData) -> bool:
            # 简单过滤：价格大于0
            return stock.price > 0
    
    # 测试策略
    strategy = SimpleStrategy()
    assert strategy.name == "simple"
    print("  ✓ name 属性正常")
    
    # 测试评分
    stock = StockData(code='600036', name='招商银行', price=35, pe_ratio=8)
    result = strategy.score(stock)
    assert result.total == 84  # 100 - 8*2 = 84
    assert result.grade == 'A'
    print("  ✓ score() 正常")
    
    # 测试筛选
    assert strategy.filter(stock) == True
    invalid = StockData(code='000001', price=0)
    assert strategy.filter(invalid) == False
    print("  ✓ filter() 正常")
    
    # 测试选股
    stocks = [
        StockData(code='600036', name='招商银行', price=35, pe_ratio=8),
        StockData(code='601398', name='工商银行', price=5, pe_ratio=5),
        StockData(code='600519', name='贵州茅台', price=1800, pe_ratio=30),
    ]
    selected = strategy.select(stocks, top_n=2)
    assert len(selected) == 2
    assert selected[0].code == '601398'  # PE最低，分数最高
    assert selected[0].rank == 1
    print("  ✓ select() 正常")
    
    return True

def test_data_source_interface():
    """测试 DataSource 抽象接口"""
    print("测试 DataSource 接口...")
    
    from core.interfaces import DataSource
    from core.types import StockData, IndexData
    from typing import Optional, List
    
    # 创建一个简单的数据源实现
    class MockDataSource(DataSource):
        @property
        def name(self) -> str:
            return "mock"
        
        def get_stock_data(self, code: str, date: str) -> Optional[StockData]:
            return StockData(code=code, name=f'股票{code}', price=10.0, date=date)
        
        def get_stock_list(self, date: str = None) -> List[str]:
            return ['600036', '601398', '600519']
        
        def get_index_constituents(self, index_code: str, date: str = None) -> List[str]:
            return ['600036', '601398', '600519']
        
        def get_index_data(self, index_code: str, start_date: str, end_date: str) -> Optional[IndexData]:
            return IndexData(
                code=index_code,
                name='沪深300',
                start_date=start_date,
                end_date=end_date,
                start_price=4000,
                end_price=4200,
                return_pct=5.0
            )
    
    # 测试数据源
    source = MockDataSource()
    assert source.name == "mock"
    print("  ✓ name 属性正常")
    
    # 测试获取股票数据
    stock = source.get_stock_data('600036', '2024-01-01')
    assert stock is not None
    assert stock.code == '600036'
    print("  ✓ get_stock_data() 正常")
    
    # 测试批量获取
    stocks = source.batch_get_stock_data(['600036', '601398'], '2024-01-01')
    assert len(stocks) == 2
    print("  ✓ batch_get_stock_data() 正常")
    
    # 测试交易日历
    days = source.get_trading_calendar('2024-01-01', '2024-01-10')
    assert len(days) > 0
    assert '2024-01-06' not in days  # 周六
    print("  ✓ get_trading_calendar() 正常")
    
    # 测试每月首个交易日
    first_days = source.get_first_trading_days('2024-01-01', '2024-03-31')
    assert len(first_days) == 3  # 1月、2月、3月
    print("  ✓ get_first_trading_days() 正常")
    
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("🧪 核心模块测试")
    print("=" * 60)
    
    tests = [
        test_types_import,
        test_interfaces_import,
        test_stock_data,
        test_backtest_config,
        test_backtest_result,
        test_strategy_interface,
        test_data_source_interface,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
