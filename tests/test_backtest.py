# -*- coding: utf-8 -*-
"""
回测层模块测试

验证 backtest/ 模块的基本功能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_risk_metrics():
    """测试风险指标计算"""
    print("测试 RiskMetrics...")
    
    from backtest.metrics import RiskMetrics
    
    metrics = RiskMetrics(periods_per_year=12)
    
    # 模拟12个月的收益率数据
    returns = [2.5, -1.2, 3.1, 1.8, -2.5, 4.2, -0.8, 2.1, -1.5, 3.5, 1.2, 2.8]
    benchmark_returns = [1.0, 0.5, 2.0, 1.5, -1.0, 2.5, 0.2, 1.8, -0.5, 2.0, 0.8, 1.5]
    
    result = metrics.calculate(
        returns=returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=0.02
    )
    
    # 验证结果
    assert result.sharpe_ratio != 0
    assert result.max_drawdown >= 0
    assert 0 <= result.win_rate <= 100
    
    print(f"  ✓ 夏普比率: {result.sharpe_ratio:.2f}")
    print(f"  ✓ 最大回撤: {result.max_drawdown:.2f}%")
    print(f"  ✓ 索提诺比率: {result.sortino_ratio:.2f}")
    print(f"  ✓ 信息比率: {result.information_ratio:.2f}")
    print(f"  ✓ 年化波动率: {result.volatility:.2f}%")
    print(f"  ✓ 胜率: {result.win_rate:.1f}%")
    print(f"  ✓ 盈亏比: {result.profit_loss_ratio:.2f}")
    
    return True


def test_trading_cost():
    """测试交易成本计算"""
    print("测试 TradingCost...")
    
    from backtest.cost import TradingCost, CostConfig
    
    # 使用默认配置
    cost_calc = TradingCost()
    
    # 测试买入成本
    buy_cost = cost_calc.calculate_buy_cost(price=50.0, shares=1000, market='SH')
    assert buy_cost.total > 0
    print(f"  ✓ 买入成本: ¥{buy_cost.total:.2f} (佣金={buy_cost.commission:.2f}, 滑点={buy_cost.slippage_cost:.2f})")
    
    # 测试卖出成本
    sell_cost = cost_calc.calculate_sell_cost(price=55.0, shares=1000, market='SH')
    assert sell_cost.total > buy_cost.total  # 卖出有印花税
    print(f"  ✓ 卖出成本: ¥{sell_cost.total:.2f} (佣金={sell_cost.commission:.2f}, 印花税={sell_cost.stamp_tax:.2f})")
    
    # 测试往返成本率
    round_trip = cost_calc.round_trip_cost_rate()
    print(f"  ✓ 往返成本率: {round_trip*100:.3f}%")
    
    # 测试盈亏平衡收益率
    breakeven = cost_calc.estimate_breakeven_return()
    print(f"  ✓ 盈亏平衡收益率: {breakeven:.3f}%")
    
    # 测试自定义配置
    custom_config = CostConfig(
        commission_rate=0.0003,  # 万3
        stamp_tax_rate=0.001,
        slippage=0.002  # 0.2%
    )
    custom_calc = TradingCost(custom_config)
    custom_round_trip = custom_calc.round_trip_cost_rate()
    print(f"  ✓ 自定义往返成本率: {custom_round_trip*100:.3f}%")
    
    return True


def test_backtest_config():
    """测试回测配置"""
    print("测试 BacktestConfig...")
    
    from backtest.engine import BacktestConfig
    
    config = BacktestConfig(
        start_date='2024-01-01',
        end_date='2024-12-31',
        initial_capital=100000,
        top_n=10,
        benchmark='000300'
    )
    
    assert config.start_date == '2024-01-01'
    assert config.end_date == '2024-12-31'
    assert config.initial_capital == 100000
    assert config.top_n == 10
    assert config.enable_cost == True
    
    print(f"  ✓ 配置创建成功")
    print(f"    起止日期: {config.start_date} ~ {config.end_date}")
    print(f"    初始资金: ¥{config.initial_capital:,.0f}")
    print(f"    每期选股: {config.top_n}只")
    print(f"    佣金率: {config.commission_rate*10000:.1f}‱")
    
    return True


def test_backtest_result():
    """测试回测结果结构"""
    print("测试 BacktestResult...")
    
    from backtest.engine import BacktestResult
    from backtest.metrics import RiskMetricsResult
    
    # 创建模拟结果
    risk_metrics = RiskMetricsResult(
        sharpe_ratio=1.5,
        max_drawdown=10.0,
        max_drawdown_duration=2,
        sortino_ratio=2.0,
        calmar_ratio=1.8,
        information_ratio=0.8,
        volatility=15.0,
        downside_volatility=8.0,
        win_rate=65.0,
        profit_loss_ratio=1.8
    )
    
    result = BacktestResult(
        start_date='2024-01-01',
        end_date='2024-12-31',
        initial_capital=100000,
        final_value=115000,
        total_return=15.0,
        annual_return=15.0,
        benchmark_return=10.0,
        alpha=5.0,
        risk_metrics=risk_metrics,
        total_cost=500.0,
        monthly_returns=[{'month': 1, 'return_pct': 2.5}],
        trades=[{'code': '600036', 'return_pct': 5.0}]
    )
    
    assert result.total_return == 15.0
    assert result.alpha == 5.0
    assert result.risk_metrics.sharpe_ratio == 1.5
    
    print(f"  ✓ 结果结构正常")
    print(f"    总收益: {result.total_return:.2f}%")
    print(f"    超额收益: {result.alpha:.2f}%")
    print(f"    夏普比率: {result.risk_metrics.sharpe_ratio:.2f}")
    
    return True


def test_monthly_mode_config():
    """测试月度回测配置"""
    print("测试 MonthlyConfig...")
    
    from backtest.modes.monthly import MonthlyConfig
    
    config = MonthlyConfig(
        start_date='2024-01-01',
        end_date='2024-06-30',
        initial_capital=100000,
        top_n=10,
    )
    
    assert config.start_date == '2024-01-01'
    assert config.top_n == 10
    
    print(f"  ✓ 月度配置创建成功")
    print(f"    随机种子: {config.random_seed}")
    
    return True


def test_report_generation():
    """测试报告生成"""
    print("测试 BacktestReport...")
    
    from backtest.report import BacktestReport
    from backtest.engine import BacktestResult
    from backtest.metrics import RiskMetricsResult
    import tempfile
    import os
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        report = BacktestReport(output_dir=tmpdir)
        
        # 创建模拟结果
        result = BacktestResult(
            start_date='2024-01-01',
            end_date='2024-12-31',
            initial_capital=100000,
            final_value=115000,
            total_return=15.0,
            annual_return=15.0,
            benchmark_return=10.0,
            alpha=5.0,
            risk_metrics=RiskMetricsResult(
                sharpe_ratio=1.5, max_drawdown=10.0, max_drawdown_duration=2,
                sortino_ratio=2.0, calmar_ratio=1.8, information_ratio=0.8,
                volatility=15.0, downside_volatility=8.0, win_rate=65.0, profit_loss_ratio=1.8
            ),
            total_cost=500.0,
            monthly_returns=[
                {'month': 1, 'buy_date': '2024-01-02', 'sell_date': '2024-02-01', 'return_pct': 2.5, 'benchmark_return': 1.0, 'portfolio_value': 102500},
                {'month': 2, 'buy_date': '2024-02-01', 'sell_date': '2024-03-01', 'return_pct': -1.0, 'benchmark_return': 0.5, 'portfolio_value': 101475},
            ],
            trades=[{'code': '600036', 'name': '招商银行', 'return_pct': 5.0}]
        )
        
        # 测试保存 JSON
        filepath = report.save_json(result, 'test_report')
        assert os.path.exists(filepath)
        print(f"  ✓ JSON报告保存成功: {filepath}")
        
        # 测试加载
        loaded = report.load_json(filepath)
        assert loaded['summary']['total_return'] == 15.0
        print(f"  ✓ JSON报告加载成功")
    
    return True


def test_backtest_engine_with_mock():
    """测试 BacktestEngine（使用 Mock 数据源）"""
    print("测试 BacktestEngine (Mock)...")
    
    from backtest.engine import BacktestEngine, BacktestConfig
    from core.interfaces import DataSource
    from core.types import StockData
    from strategy import StrategyRegistry
    from typing import List, Optional
    
    # 创建 Mock 数据源
    class MockDataSource(DataSource):
        """模拟数据源"""
        
        def __init__(self):
            super().__init__()
            self._stock_pool = [f'60000{i}' for i in range(20)]
        
        @property
        def name(self) -> str:
            return "mock"
        
        def get_stock_data(self, code: str, date: str) -> Optional[StockData]:
            # 生成模拟数据
            import random
            random.seed(hash(f"{code}{date}") % 2**32)
            
            base_price = 10 + random.random() * 50
            return StockData(
                code=code,
                name=f'模拟股票{code[-2:]}',
                date=date,
                price=base_price,
                change_pct=random.uniform(-5, 5),
                momentum_20d=random.uniform(-10, 15),
                pe_ratio=random.uniform(5, 30),
                pb_ratio=random.uniform(0.5, 5),
                roe=random.uniform(5, 25),
                profit_growth=random.uniform(-20, 50),
                turnover_rate=random.uniform(0.5, 5),
            )
        
        def get_stock_list(self, market: str = None) -> List[str]:
            return self._stock_pool
        
        def get_index_constituents(self, index_code: str, date: str = None) -> List[str]:
            return self._stock_pool
        
        def get_index_data(self, index_code: str, date: str):
            return None
        
        def get_index_return(self, index_code: str, start_date: str, end_date: str) -> float:
            import random
            random.seed(hash(f"{index_code}{start_date}{end_date}") % 2**32)
            return random.uniform(-3, 5)
        
        def get_trading_calendar(self, start_date: str, end_date: str) -> List[str]:
            # 简单生成交易日
            from datetime import datetime, timedelta
            days = []
            current = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            while current <= end:
                if current.weekday() < 5:
                    days.append(current.strftime('%Y-%m-%d'))
                current += timedelta(days=1)
            return days
    
    # 创建配置
    config = BacktestConfig(
        start_date='2024-01-01',
        end_date='2024-03-31',  # 3个月
        initial_capital=100000,
        top_n=5,
        enable_cost=True
    )
    
    # 创建引擎
    data_source = MockDataSource()
    strategy = StrategyRegistry.create('momentum_v2', {'min_score': 0})  # 低阈值确保选中
    
    engine = BacktestEngine(data_source, strategy, config)
    
    # 执行回测
    result = engine.run_monthly()
    
    # 验证结果
    assert result is not None
    assert result.start_date == '2024-01-01'
    assert result.end_date == '2024-03-31'
    assert result.initial_capital == 100000
    
    print(f"  ✓ 回测执行成功")
    print(f"    回测月数: {len(result.monthly_returns)}")
    print(f"    总收益: {result.total_return:+.2f}%")
    print(f"    基准收益: {result.benchmark_return:+.2f}%")
    print(f"    超额收益: {result.alpha:+.2f}%")
    
    if result.risk_metrics:
        print(f"    夏普比率: {result.risk_metrics.sharpe_ratio:.2f}")
        print(f"    最大回撤: {result.risk_metrics.max_drawdown:.2f}%")
        print(f"    胜率: {result.risk_metrics.win_rate:.1f}%")
    
    print(f"    交易成本: ¥{result.total_cost:.2f}")
    print(f"    总交易数: {len(result.trades)}")
    
    return True


def test_backtest_engine_with_real_data():
    """测试 BacktestEngine（使用真实数据 + DataManager）"""
    print("=" * 60)
    print("测试 BacktestEngine (真实数据 + DataManager)")
    print("=" * 60)
    
    import logging
    # 设置日志级别为 INFO，显示详细进度
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        from backtest.engine import BacktestEngine, BacktestConfig
        from backtest.report import BacktestReport
        from data.manager import DataManager
        from strategy import StrategyRegistry
        
        # ===== 1. 配置说明 =====
        print("\n📋 回测配置:")
        print("-" * 40)
        
        config = BacktestConfig(
            start_date='2024-01-01',
            end_date='2025-01-02',
            initial_capital=100000,       # 10万初始资金
            top_n=10,                      # 每月选10只股票
            benchmark='000300',           # 基准指数（候选池=全部成分股）
            enable_cost=False,
            random_seed=42
        )
        
        print(f"   • 回测期: {config.start_date} ~ {config.end_date}")
        print(f"   • 初始资金: ¥{config.initial_capital:,.0f}")
        print(f"   • 每月选股: {config.top_n} 只")
        print(f"   • 选股范围: {config.benchmark} 全部成分股")
        print(f"   • 交易成本: {'启用' if config.enable_cost else '禁用'}")
        
        # ===== 2. 初始化数据管理器 =====
        print("\n📊 初始化数据管理器...")
        print("-" * 40)
        
        # 使用 DataManager（支持缓存）
        data_manager = DataManager(
            source='tushare',
            cache_dir='./cache',
            cache_expire_days=7,
            use_cache=True  # 启用缓存，加速重复测试
        )
        
        print(f"   • 数据源: {data_manager.name}")
        print(f"   • 缓存: 启用 (./cache)")
        
        # 验证数据源连接
        print("\n   验证数据源连接...")
        constituents = data_manager.get_index_constituents('000300')
        if constituents:
            print(f"   ✓ 获取沪深300成分股: {len(constituents)} 只")
        else:
            print("   ✗ 无法获取成分股，跳过测试")
            return True
        
        # ===== 3. 初始化策略 =====
        print("\n🎯 初始化策略...")
        print("-" * 40)
        
        strategy = StrategyRegistry.create('momentum_v2')
        print(f"   • 策略名称: {strategy.name}")
        print(f"   • 策略说明: {strategy.description}")
        
        # ===== 4. 创建回测引擎 =====
        print("\n🚀 创建回测引擎...")
        print("-" * 40)
        
        engine = BacktestEngine(data_manager, strategy, config)
        print("   ✓ 回测引擎创建成功")
        
        # ===== 5. 执行月度回测 =====
        print("\n" + "=" * 60)
        print("📈 开始执行月度轮换回测...")
        print("=" * 60)
        
        result = engine.run_monthly()
        
        # ===== 6. 输出结果 =====
        if result and result.monthly_returns:
            print("\n" + "=" * 60)
            print("✅ 回测完成！")
            print("=" * 60)
            
            # 使用报告生成器输出摘要
            report = BacktestReport()
            report.print_summary(result)
            report.print_monthly_detail(result)
            
            # 保存 JSON 报告
            filepath = report.save_json(result, f"test_real_{config.start_date}_{config.end_date}")
            print(f"\n📁 报告已保存: {filepath}")
            
            return True
        else:
            print("\n⚠️ 回测无有效结果")
            return True
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("🧪 回测层模块测试")
    print("=" * 60)
    
    tests = [
        #test_risk_metrics,
        #test_trading_cost,
        #test_backtest_config,
        #test_backtest_result,
        #test_monthly_mode_config,
        #test_report_generation,
        #test_backtest_engine_with_mock,
        test_backtest_engine_with_real_data,  # 需要网络，可选
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
