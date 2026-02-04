#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
回测入口 - 配置驱动版

所有参数从配置文件读取，无需命令行参数。

使用方法:
    1. 修改 config/settings.py 中的 BACKTEST_CONFIG
    2. 运行 python backtest.py

配置示例 (config/settings.py):
    BACKTEST_CONFIG = {
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'initial_capital': 100000,
        'top_n': 10,
        ...
    }

编程使用:
    from backtest import run_backtest, run_select
    
    # 使用配置文件参数执行回测
    result = run_backtest()
    
    # 使用自定义参数
    result = run_backtest(start_date='2024-06-01', end_date='2024-12-31')
"""

import sys
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===== 从配置读取所有参数 =====
from config.settings import BACKTEST_CONFIG
from config.strategy_config import DEFAULT_STRATEGY
from config.data_source_config import DEFAULT_DATA_SOURCE


def run_backtest(
    start_date: str = None,
    end_date: str = None,
    strategy_name: str = None,
    source: str = None,
    initial_capital: float = None,
    top_n: int = None,
    benchmark: str = None,
    save_report: bool = None,
):
    """
    执行月度轮换回测
    
    所有参数都从配置文件读取，传入参数会覆盖配置。
    
    Args:
        start_date: 开始日期，默认从配置读取
        end_date: 结束日期，默认从配置读取
        strategy_name: 策略名称，默认从配置读取
        source: 数据源，默认从配置读取
        initial_capital: 初始资金，默认从配置读取
        top_n: 每月选股数量，默认从配置读取
        benchmark: 基准指数，默认从配置读取
        save_report: 是否保存报告，默认从配置读取
    
    Returns:
        BacktestResult: 回测结果
    """
    from data.manager import DataManager
    from strategy import StrategyRegistry
    from backtest import BacktestEngine
    from backtest.report import BacktestReport
    from backtest.engine import BacktestConfig
    
    # 从配置读取，传入参数可覆盖
    start_date = start_date or BACKTEST_CONFIG['start_date']
    end_date = end_date or BACKTEST_CONFIG['end_date']
    strategy_name = strategy_name or DEFAULT_STRATEGY
    source = source or DEFAULT_DATA_SOURCE
    initial_capital = initial_capital or BACKTEST_CONFIG['initial_capital']
    top_n = top_n or BACKTEST_CONFIG['top_n']
    benchmark = benchmark or BACKTEST_CONFIG['benchmark']
    save_report = save_report if save_report is not None else BACKTEST_CONFIG.get('save_report', True)
    
    logger.info("=" * 60)
    logger.info("📅 月度轮换回测")
    logger.info("=" * 60)
    logger.info(f"📆 日期范围: {start_date} ~ {end_date}")
    logger.info(f"📊 策略: {strategy_name}")
    logger.info(f"📡 数据源: {source}")
    logger.info(f"💰 初始资金: ¥{initial_capital:,.0f}")
    logger.info(f"🎯 每月选股: {top_n} 只")
    logger.info(f"📈 基准指数: {benchmark}")
    logger.info("=" * 60)
    
    # 初始化
    data_source = DataManager(source=source, use_cache=True)
    strategy = StrategyRegistry.create(strategy_name)
    
    # 如果是 ML 策略，需要设置数据源（用于获取日线数据计算技术指标）
    if hasattr(strategy, 'set_data_source'):
        strategy.set_data_source(data_source)
        logger.info(f"🤖 已为策略设置数据源")
    
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        benchmark=benchmark,
        top_n=top_n,
        sample_size=BACKTEST_CONFIG.get('sample_size', 300),
        random_seed=BACKTEST_CONFIG.get('random_seed', 42),
        enable_cost=BACKTEST_CONFIG.get('enable_cost', True),
    )
    
    # 执行回测
    engine = BacktestEngine(data_source, strategy, config)
    result = engine.run_monthly()
    
    # 输出结果
    reporter = BacktestReport()
    reporter.print_summary(result)
    reporter.print_monthly_detail(result)  # 输出月度明细
    
    # 保存报告
    if save_report:
        report_name = f"backtest_{strategy_name}_{start_date}_to_{end_date}"
        reporter.save_json(result, report_name)
    
    return result


def run_select(
    date: str = None,
    strategy_name: str = None,
    source: str = None,
    top_n: int = None,
):
    """
    执行选股
    
    Args:
        date: 选股日期，必须指定
        strategy_name: 策略名称，默认从配置读取
        source: 数据源，默认从配置读取
        top_n: 选股数量，默认从配置读取
    
    Returns:
        List[StockData]: 选中的股票列表
    """
    from data.manager import DataManager
    from strategy import StrategyRegistry
    
    if not date:
        from datetime import datetime
        date = datetime.now().strftime('%Y-%m-%d')
        logger.warning(f"未指定日期，使用今天: {date}")
    
    strategy_name = strategy_name or DEFAULT_STRATEGY
    source = source or DEFAULT_DATA_SOURCE
    top_n = top_n or BACKTEST_CONFIG['top_n']
    
    logger.info(f"🔍 选股日期: {date}")
    logger.info(f"📊 策略: {strategy_name}")
    
    # 初始化
    data_source = DataManager(source=source, use_cache=True)
    strategy = StrategyRegistry.create(strategy_name)
    
    # 如果是 ML 策略，设置数据源
    if hasattr(strategy, 'set_data_source'):
        strategy.set_data_source(data_source)
    
    # 获取候选股票
    stock_codes = data_source.get_csi300_stocks(date)
    logger.info(f"📋 候选股票: {len(stock_codes)} 只")
    
    # 获取股票数据
    stocks = []
    for i, code in enumerate(stock_codes):
        if (i + 1) % 50 == 0:
            logger.info(f"   进度: {i+1}/{len(stock_codes)}")
        stock = data_source.get_stock_data(code, date)
        if stock:
            stocks.append(stock)
    
    logger.info(f"📊 有效股票: {len(stocks)} 只")
    
    # ML 策略：按日期切换模型（若配置了 model_schedule）
    if hasattr(strategy, 'set_current_date'):
        strategy.set_current_date(date)
    
    # 执行选股
    selected = strategy.select(stocks, top_n=top_n)
    
    # 输出结果
    print(f"\n🏆 选出 {len(selected)} 只股票:")
    print("=" * 60)
    
    for i, stock in enumerate(selected, 1):
        score_result = strategy.score(stock)
        print(f"{i:2d}. {stock.name}({stock.code})")
        print(f"    价格: ¥{stock.price:.2f}  涨跌: {stock.change_pct:+.2f}%")
        print(f"    评分: {score_result.total}")
        if score_result.breakdown:
            breakdown_str = ', '.join(f"{k}={v}" for k, v in score_result.breakdown.items())
            print(f"    明细: {breakdown_str}")
        print()
    
    return selected


def run_compare(
    strategy_names: list = None,
    start_date: str = None,
    end_date: str = None,
):
    """
    策略对比回测
    
    Args:
        strategy_names: 策略名称列表，默认 ['momentum_v2', 'random']
        start_date: 开始日期，默认从配置读取
        end_date: 结束日期，默认从配置读取
    
    Returns:
        Dict[str, BacktestResult]: 策略名称 -> 回测结果
    """
    from data.manager import DataManager
    from strategy import StrategyRegistry
    from backtest import BacktestEngine
    from backtest.report import BacktestReport
    from backtest.engine import BacktestConfig
    
    strategy_names = strategy_names or ['momentum_v2', 'random']
    start_date = start_date or BACKTEST_CONFIG['start_date']
    end_date = end_date or BACKTEST_CONFIG['end_date']
    
    logger.info(f"📊 策略对比: {strategy_names}")
    
    # 初始化数据源（共用）
    data_source = DataManager(use_cache=True)
    
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=BACKTEST_CONFIG['initial_capital'],
        benchmark=BACKTEST_CONFIG['benchmark'],
        top_n=BACKTEST_CONFIG['top_n'],
    )
    
    # 执行各策略
    results = {}
    for name in strategy_names:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 执行策略: {name}")
            logger.info(f"{'='*60}")
            
            strategy = StrategyRegistry.create(name)
            # 如果是 ML 策略，设置数据源
            if hasattr(strategy, 'set_data_source'):
                strategy.set_data_source(data_source)
            engine = BacktestEngine(data_source, strategy, config)
            results[name] = engine.run_monthly()
        except Exception as e:
            logger.error(f"❌ 策略 {name} 执行失败: {e}")
    
    # 输出对比报告
    reporter = BacktestReport()
    reporter.compare_strategies(results)
    
    return results


def list_strategies():
    """列出所有可用策略"""
    from strategy import StrategyRegistry
    
    print("\n📋 可用策略:")
    print("=" * 40)
    for name in StrategyRegistry.list_all():
        print(f"  • {name}")
    print()


def show_config():
    """显示当前配置"""
    print("\n⚙️ 当前回测配置 (config/settings.py):")
    print("=" * 50)
    for key, value in BACKTEST_CONFIG.items():
        print(f"  {key}: {value}")
    print()
    print(f"  默认策略: {DEFAULT_STRATEGY}")
    print(f"  默认数据源: {DEFAULT_DATA_SOURCE}")
    print()


def main():
    """
    主函数 - 直接从配置文件读取参数执行回测
    
    使用方法:
        python backtest.py              # 执行回测（使用配置文件参数）
        python backtest.py --config     # 显示当前配置
        python backtest.py --strategies # 列出可用策略
        python backtest.py --compare    # 策略对比
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='JYS股票回测系统 - 配置驱动版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用方法:
  1. 编辑 config/settings.py 中的 BACKTEST_CONFIG
  2. 运行 python backtest.py

示例配置 (config/settings.py):
  BACKTEST_CONFIG = {
      'start_date': '2024-01-01',
      'end_date': '2024-12-31',
      'initial_capital': 100000,
      'top_n': 10,
      ...
  }
        """
    )
    
    # 只保留几个简单的命令选项
    parser.add_argument('--config', action='store_true', help='显示当前配置')
    parser.add_argument('--strategies', action='store_true', help='列出可用策略')
    parser.add_argument('--compare', action='store_true', help='策略对比回测')
    parser.add_argument('--select', metavar='DATE', help='执行选股（指定日期）')
    
    args = parser.parse_args()
    
    try:
        if args.config:
            show_config()
        elif args.strategies:
            list_strategies()
        elif args.compare:
            run_compare()
        elif args.select:
            run_select(date=args.select)
        else:
            # 默认：执行回测
            run_backtest()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
