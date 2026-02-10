#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
命令行接口 - 配置驱动版

所有参数从配置文件读取，提供简洁的命令行操作。

使用方法:
    # 显示当前配置
    python cli.py config
    
    # 执行回测（使用配置文件参数）
    python cli.py backtest
    
    # 策略对比
    python cli.py compare
    
    # 执行选股
    python cli.py select 2024-06-03
    
    # 列出可用策略
    python cli.py strategies
    
    # 列出数据源
    python cli.py sources
    
    # 测试数据源
    python cli.py test-source
"""

import argparse
import sys
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===== 从配置读取所有参数 =====
from config.settings import BACKTEST_CONFIG
from config.strategy_config import DEFAULT_STRATEGY, STRATEGY_CONFIGS
from config.data_source_config import DEFAULT_DATA_SOURCE, DATA_SOURCE_CONFIGS


def cmd_config():
    """显示当前配置"""
    print("\n⚙️ 回测配置 (config/settings.py -> BACKTEST_CONFIG):")
    print("=" * 55)
    for key, value in BACKTEST_CONFIG.items():
        print(f"  {key:20s}: {value}")
    
    print(f"\n📊 策略配置 (config/strategy_config.py):")
    print("=" * 55)
    print(f"  默认策略: {DEFAULT_STRATEGY}")
    
    print(f"\n📡 数据源配置 (config/data_source_config.py):")
    print("=" * 55)
    print(f"  默认数据源: {DEFAULT_DATA_SOURCE}")
    print()


def cmd_backtest():
    """执行回测"""
    from data.manager import DataManager
    from strategy import StrategyRegistry
    from backtest import BacktestEngine
    from backtest.report import BacktestReport
    from backtest.engine import BacktestConfig
    
    # 全部从配置读取
    start_date = BACKTEST_CONFIG['start_date']
    end_date = BACKTEST_CONFIG['end_date']
    initial_capital = BACKTEST_CONFIG['initial_capital']
    benchmark = BACKTEST_CONFIG['benchmark']
    top_n = BACKTEST_CONFIG['top_n']
    
    logger.info("=" * 60)
    logger.info("📅 月度轮换回测")
    logger.info("=" * 60)
    logger.info(f"📆 日期范围: {start_date} ~ {end_date}")
    logger.info(f"📊 策略: {DEFAULT_STRATEGY}")
    logger.info(f"📡 数据源: {DEFAULT_DATA_SOURCE}")
    logger.info(f"💰 初始资金: ¥{initial_capital:,.0f}")
    logger.info(f"🎯 每月选股: {top_n} 只")
    logger.info("=" * 60)
    
    # 初始化
    data_source = DataManager(use_cache=True)
    strategy = StrategyRegistry.create(DEFAULT_STRATEGY)
    
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        benchmark=benchmark,
        top_n=top_n,
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
    if BACKTEST_CONFIG.get('save_report', True):
        report_name = f"backtest_{DEFAULT_STRATEGY}_{start_date}_to_{end_date}"
        reporter.save_json(result, report_name)
    
    return result


def cmd_compare():
    """策略对比回测"""
    from data.manager import DataManager
    from strategy import StrategyRegistry
    from backtest import BacktestEngine
    from backtest.report import BacktestReport
    from backtest.engine import BacktestConfig
    
    strategy_names = ['momentum_v2', 'random']
    
    logger.info(f"📊 策略对比: {strategy_names}")
    
    # 初始化数据源（共用）
    data_source = DataManager(use_cache=True)
    
    config = BacktestConfig(
        start_date=BACKTEST_CONFIG['start_date'],
        end_date=BACKTEST_CONFIG['end_date'],
        initial_capital=BACKTEST_CONFIG['initial_capital'],
        benchmark=BACKTEST_CONFIG['benchmark'],
        top_n=BACKTEST_CONFIG['top_n'],
        random_seed=BACKTEST_CONFIG.get('random_seed', 42),
        enable_cost=BACKTEST_CONFIG.get('enable_cost', True),
    )
    
    # 执行各策略
    results = {}
    for name in strategy_names:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 执行策略: {name}")
            logger.info(f"{'='*60}")
            
            strategy = StrategyRegistry.create(name)
            engine = BacktestEngine(data_source, strategy, config)
            results[name] = engine.run_monthly()
        except Exception as e:
            logger.error(f"❌ 策略 {name} 执行失败: {e}")
    
    # 输出对比报告
    reporter = BacktestReport()
    reporter.compare_strategies(results)
    
    return results


def cmd_select(date: str):
    """执行选股"""
    from data.manager import DataManager
    from strategy import StrategyRegistry
    
    logger.info(f"🔍 选股日期: {date}")
    logger.info(f"📊 策略: {DEFAULT_STRATEGY}")
    
    # 初始化
    data_source = DataManager(use_cache=True)
    strategy = StrategyRegistry.create(DEFAULT_STRATEGY)
    
    # 获取候选股票
    from config.settings import SELECTION_CONFIG
    index_code = SELECTION_CONFIG.get('index_code', '000300')
    index_name = {'000300': '沪深300', '000905': '中证500'}.get(index_code, index_code)
    logger.info(f"📋 获取{index_name}成分股...")
    stock_codes = data_source.get_index_constituents(index_code, date)
    logger.info(f"   候选股票: {len(stock_codes)} 只")
    
    # 获取股票数据
    logger.info("📊 获取股票数据...")
    stocks = []
    for i, code in enumerate(stock_codes):
        if (i + 1) % 50 == 0:
            logger.info(f"   进度: {i+1}/{len(stock_codes)}")
        stock = data_source.get_stock_data(code, date)
        if stock:
            stocks.append(stock)
    
    logger.info(f"   有效股票: {len(stocks)} 只")
    
    # 执行选股
    top_n = BACKTEST_CONFIG['top_n']
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


def cmd_strategies():
    """列出所有可用策略"""
    from strategy import StrategyRegistry
    
    print("\n📋 可用策略列表:")
    print("=" * 55)
    
    for name in StrategyRegistry.list_all():
        try:
            strategy = StrategyRegistry.create(name)
            is_default = "⭐" if name == DEFAULT_STRATEGY else "  "
            print(f"  {is_default} {name}")
            if hasattr(strategy, 'description'):
                print(f"       {strategy.description}")
        except Exception as e:
            print(f"     {name} (加载失败)")
    
    print(f"\n共 {len(StrategyRegistry.list_all())} 个策略，默认: {DEFAULT_STRATEGY}")
    print()


def cmd_sources():
    """列出所有可用数据源"""
    from data.sources import list_sources
    
    print("\n📡 可用数据源列表:")
    print("=" * 55)
    
    for name in list_sources():
        is_default = "⭐" if name == DEFAULT_DATA_SOURCE else "  "
        config = DATA_SOURCE_CONFIGS.get(name, {})
        desc = config.get('description', '')
        print(f"  {is_default} {name}")
        if desc:
            print(f"       {desc}")
    
    print(f"\n共 {len(list_sources())} 个数据源，默认: {DEFAULT_DATA_SOURCE}")
    print()


def cmd_test_source():
    """测试数据源连接"""
    from data.manager import DataManager
    
    print(f"\n🔍 测试数据源: {DEFAULT_DATA_SOURCE}")
    print("=" * 55)
    
    try:
        manager = DataManager(use_cache=False)
        print(f"  ✅ 数据源初始化成功: {manager.name}")
        
        # 测试获取股票数据
        test_code = '000001'
        test_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n  📈 测试获取股票数据: {test_code} @ {test_date}")
        stock = manager.get_stock_data(test_code, test_date)
        
        if stock:
            print(f"    代码: {stock.code}")
            print(f"    名称: {stock.name}")
            print(f"    价格: ¥{stock.price:.2f}")
            print(f"    涨跌: {stock.change_pct:.2f}%")
            print("    ✅ 数据获取成功")
        else:
            print("    ⚠️ 未获取到数据（可能是非交易日）")
            
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='JYS股票分析系统 - 配置驱动版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用方法:
  1. 编辑 config/settings.py 中的 BACKTEST_CONFIG
  2. 运行相应命令

命令示例:
  python cli.py config       # 显示当前配置
  python cli.py backtest     # 执行回测
  python cli.py compare      # 策略对比
  python cli.py select 2024-06-03  # 选股
  python cli.py strategies   # 列出策略
  python cli.py sources      # 列出数据源
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 各命令
    subparsers.add_parser('config', help='显示当前配置')
    subparsers.add_parser('backtest', help='执行回测')
    subparsers.add_parser('compare', help='策略对比回测')
    
    select_parser = subparsers.add_parser('select', help='执行选股')
    select_parser.add_argument('date', help='选股日期 (YYYY-MM-DD)')
    
    subparsers.add_parser('strategies', help='列出可用策略')
    subparsers.add_parser('sources', help='列出数据源')
    subparsers.add_parser('test-source', help='测试数据源')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'config':
            cmd_config()
        elif args.command == 'backtest':
            cmd_backtest()
        elif args.command == 'compare':
            cmd_compare()
        elif args.command == 'select':
            cmd_select(args.date)
        elif args.command == 'strategies':
            cmd_strategies()
        elif args.command == 'sources':
            cmd_sources()
        elif args.command == 'test-source':
            cmd_test_source()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
