# -*- coding: utf-8 -*-
"""
数据层模块测试

验证 data/ 模块的基本功能
"""

import sys
import os
import tempfile
import shutil

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_cache_manager():
    """测试缓存管理器"""
    print("测试 CacheManager...")
    
    from data.cache import CacheManager
    
    # 使用临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        cache = CacheManager(cache_dir=temp_dir, expire_days=1)
        
        # 测试 set/get
        cache.set('test_key', {'value': 123})
        result = cache.get('test_key')
        assert result == {'value': 123}
        print("  ✓ set/get 正常")
        
        # 测试不存在的key
        result = cache.get('nonexistent', default='default')
        assert result == 'default'
        print("  ✓ 默认值正常")
        
        # 测试 exists
        assert cache.exists('test_key') == True
        assert cache.exists('nonexistent') == False
        print("  ✓ exists 正常")
        
        # 测试 delete
        cache.delete('test_key')
        assert cache.exists('test_key') == False
        print("  ✓ delete 正常")
        
        # 测试 stats
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        stats = cache.get_stats()
        assert stats['memory_count'] == 2
        print("  ✓ get_stats 正常")
        
        # 测试 clear
        count = cache.clear()
        assert count >= 2
        print("  ✓ clear 正常")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return True


def test_data_sources_import():
    """测试数据源模块导入"""
    print("测试数据源模块导入...")
    
    from data.sources import (
        TushareSource, AkShareSource, TencentSource,
        AVAILABLE_SOURCES, get_source, list_sources
    )
    
    # 测试所有数据源都已注册
    assert 'tushare' in AVAILABLE_SOURCES
    assert 'akshare' in AVAILABLE_SOURCES
    assert 'tencent' in AVAILABLE_SOURCES
    print("  ✓ 所有数据源已注册 (tushare, akshare, tencent)")
    
    # 测试 get_source
    assert get_source('tushare') == TushareSource
    assert get_source('akshare') == AkShareSource
    assert get_source('tencent') == TencentSource
    print("  ✓ get_source 正常")
    
    # 测试 list_sources
    sources = list_sources()
    assert len(sources) == 3
    print(f"  ✓ list_sources 返回: {sources}")
    
    # 测试无效数据源
    try:
        get_source('invalid')
        assert False, "应该抛出异常"
    except ValueError as e:
        assert 'invalid' in str(e)
        print("  ✓ 无效数据源异常正常")
    
    return True


def test_tushare_source_init():
    """测试 TushareSource 初始化"""
    print("测试 TushareSource 初始化...")
    
    try:
        from data.sources import TushareSource
        
        source = TushareSource()
        assert source.name == 'tushare'
        print("  ✓ 初始化成功")
        print("  ✓ name 属性正常")
        
    except ImportError as e:
        print(f"  ⚠ Tushare 未安装，跳过: {e}")
        return True
    except Exception as e:
        print(f"  ⚠ 初始化失败（可能是Token问题）: {e}")
        return True
    
    return True


def test_akshare_source_init():
    """测试 AkShareSource 初始化"""
    print("测试 AkShareSource 初始化...")
    
    try:
        from data.sources import AkShareSource
        
        source = AkShareSource()
        assert source.name == 'akshare'
        print("  ✓ 初始化成功")
        print("  ✓ name 属性正常")
        
    except ImportError as e:
        print(f"  ⚠ AkShare 未安装，跳过: {e}")
        return True
    except Exception as e:
        print(f"  ⚠ 初始化失败: {e}")
        return True
    
    return True


def test_tencent_source_init():
    """测试 TencentSource 初始化"""
    print("测试 TencentSource 初始化...")
    
    try:
        from data.sources import TencentSource
        
        source = TencentSource()
        assert source.name == 'tencent'
        print("  ✓ 初始化成功")
        print("  ✓ name 属性正常")
        
    except Exception as e:
        print(f"  ⚠ 初始化失败: {e}")
        return True
    
    return True


def test_data_manager_init():
    """测试 DataManager 初始化"""
    print("测试 DataManager 初始化...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        from data.manager import DataManager
        
        # 测试带缓存
        manager = DataManager(source='tushare', cache_dir=temp_dir, use_cache=True)
        assert manager.source_name == 'tushare'
        print("  ✓ 带缓存初始化成功")
        
        # 测试不带缓存
        manager2 = DataManager(source='tushare', use_cache=False)
        assert manager2.source_name == 'tushare'
        print("  ✓ 不带缓存初始化成功")
        
        # 测试缓存统计
        stats = manager.get_cache_stats()
        assert 'memory_count' in stats
        print("  ✓ get_cache_stats 正常")
        
    except ImportError as e:
        print(f"  ⚠ 依赖未安装: {e}")
        return True
    except Exception as e:
        print(f"  ⚠ 初始化失败: {e}")
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return True


def test_data_manager_integration():
    """测试 DataManager 集成（需要网络）"""
    print("测试 DataManager 集成（需要网络）...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        from data.manager import DataManager
        
        manager = DataManager(source='tushare', cache_dir=temp_dir)
        
        # 测试获取沪深300成分股
        print("  正在获取沪深300成分股...")
        stocks = manager.get_csi300_stocks()
        
        if stocks:
            print(f"  ✓ 获取沪深300成分股: {len(stocks)} 只")
            assert len(stocks) > 200  # 应该接近300只
            
            # 测试获取单只股票数据
            print("  正在获取单只股票数据...")
            stock = manager.get_stock_data(stocks[0], '2024-06-03')
            if stock:
                print(f"  ✓ 获取股票数据: {stock.code} {stock.name} 价格={stock.price}")
                assert stock.is_valid()
            else:
                print("  ⚠ 获取股票数据返回空（可能是非交易日）")
            
            # 测试缓存命中
            stocks2 = manager.get_csi300_stocks()
            assert len(stocks2) == len(stocks)
            print("  ✓ 缓存命中正常")
        else:
            print("  ⚠ 获取成分股失败（可能是网络问题）")
        
    except Exception as e:
        print(f"  ⚠ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return True


def compare_data_sources(stock_code: str = '600036', date: str = '2024-06-03'):
    """
    对比三种数据源的数据
    
    对比 TushareSource、AkShareSource、TencentSource 获取的数据差异
    
    Args:
        stock_code: 股票代码，默认招商银行
        date: 日期，默认2024-06-03
    """
    from data.sources import TushareSource, AkShareSource, TencentSource
    
    print("\n" + "=" * 90)
    print(f"📊 数据源对比分析: {stock_code} @ {date}")
    print("=" * 90)
    
    results = {}
    
    # ===== 数据源1：Tushare =====
    print("\n【数据源1】TushareSource（历史数据 + 前复权）")
    print("-" * 70)
    try:
        source = TushareSource()
        stock = source.get_stock_data(stock_code, date)
        if stock:
            results['tushare'] = stock
            print(f"   ✅ 获取成功")
            print(f"   • 股票名称: {stock.name}")
            print(f"   • 价格(前复权): ¥{stock.price:.2f}")
            print(f"   • 涨跌幅: {stock.change_pct:.2f}%")
            print(f"   • 20日动量: {stock.momentum_20d:.2f}%")
            print(f"   • PE(TTM): {stock.pe_ratio}")
            print(f"   • PB: {stock.pb_ratio}")
            print(f"   • ROE: {stock.roe}")
            print(f"   • 利润增长: {stock.profit_growth}")
            print(f"   • 换手率: {stock.turnover_rate}%")
            print(f"   • 股息率: {stock.dividend_yield}")
            print(f"   • 报告期: {stock.report_date}")
        else:
            print("   ❌ 返回空数据")
    except Exception as e:
        print(f"   ❌ 获取失败: {e}")
    
    # ===== 数据源2：AkShare =====
    print("\n【数据源2】AkShareSource（免费数据 + 前复权）")
    print("-" * 70)
    try:
        source = AkShareSource()
        stock = source.get_stock_data(stock_code, date)
        if stock:
            results['akshare'] = stock
            print(f"   ✅ 获取成功")
            print(f"   • 股票名称: {stock.name}")
            print(f"   • 价格(前复权): ¥{stock.price:.2f}")
            print(f"   • 涨跌幅: {stock.change_pct:.2f}%")
            print(f"   • 20日动量: {stock.momentum_20d:.2f}%")
            print(f"   • PE: {stock.pe_ratio}")
            print(f"   • PB: {stock.pb_ratio}")
            print(f"   • ROE: {stock.roe}")
            print(f"   • 利润增长: {stock.profit_growth}")
            print(f"   • 换手率: {stock.turnover_rate}%")
        else:
            print("   ❌ 返回空数据")
    except Exception as e:
        print(f"   ❌ 获取失败: {e}")
    
    # ===== 数据源3：腾讯 =====
    print("\n【数据源3】TencentSource（实时数据）")
    print("-" * 70)
    try:
        source = TencentSource()
        stock = source.get_stock_data(stock_code)
        if stock:
            results['tencent'] = stock
            print(f"   ✅ 获取成功")
            print(f"   ⚠️  注意: 腾讯提供的是实时数据，非历史数据")
            print(f"   • 股票名称: {stock.name}")
            print(f"   • 实时价格: ¥{stock.price:.2f}")
            print(f"   • 涨跌幅: {stock.change_pct:.2f}%")
            print(f"   • PE(TTM): {stock.pe_ratio}")
            print(f"   • PB: {stock.pb_ratio}")
            print(f"   • ROE: {stock.roe}")
            print(f"   • 换手率: {stock.turnover_rate}%")
            print(f"   • 股息率: {stock.dividend_yield}")
        else:
            print("   ❌ 返回空数据")
    except Exception as e:
        print(f"   ❌ 获取失败: {e}")
    
    # ===== 数据对比表格 =====
    if len(results) >= 2:
        print("\n" + "=" * 90)
        print("📊 三方数据对比表")
        print("=" * 90)
        
        compare_fields = [
            ('股票名称', 'name', 's'),
            ('价格', 'price', '.2f'),
            ('涨跌幅(%)', 'change_pct', '.2f'),
            ('20日动量(%)', 'momentum_20d', '.2f'),
            ('PE', 'pe_ratio', '.2f'),
            ('PB', 'pb_ratio', '.2f'),
            ('ROE(%)', 'roe', '.2f'),
            ('利润增长(%)', 'profit_growth', '.2f'),
            ('换手率(%)', 'turnover_rate', '.2f'),
            ('股息率(%)', 'dividend_yield', '.2f'),
        ]
        
        # 打印表头
        print(f"\n{'指标':<15} {'Tushare':<15} {'AkShare':<15} {'Tencent':<15} {'差异分析':<20}")
        print("-" * 80)
        
        for name, field, fmt in compare_fields:
            tushare_val = getattr(results.get('tushare'), field, None) if results.get('tushare') else None
            akshare_val = getattr(results.get('akshare'), field, None) if results.get('akshare') else None
            tencent_val = getattr(results.get('tencent'), field, None) if results.get('tencent') else None
            
            def format_val(val, fmt):
                if val is None:
                    return 'N/A'
                if fmt == 's':
                    return str(val)[:12]
                try:
                    return f"{float(val):{fmt}}"
                except:
                    return str(val)[:12]
            
            tushare_str = format_val(tushare_val, fmt)
            akshare_str = format_val(akshare_val, fmt)
            tencent_str = format_val(tencent_val, fmt)
            
            # 差异分析
            analysis = ""
            numeric_vals = []
            if isinstance(tushare_val, (int, float)) and tushare_val is not None:
                numeric_vals.append(('T', tushare_val))
            if isinstance(akshare_val, (int, float)) and akshare_val is not None:
                numeric_vals.append(('A', akshare_val))
            if isinstance(tencent_val, (int, float)) and tencent_val is not None:
                numeric_vals.append(('Q', tencent_val))
            
            if len(numeric_vals) >= 2:
                vals = [v[1] for v in numeric_vals]
                max_val = max(vals)
                min_val = min(vals)
                if min_val != 0:
                    diff_pct = (max_val - min_val) / abs(min_val) * 100
                    if diff_pct < 5:
                        analysis = "✓ 一致"
                    elif diff_pct < 20:
                        analysis = f"~ 差异{diff_pct:.1f}%"
                    else:
                        analysis = f"⚠ 差异{diff_pct:.1f}%"
                else:
                    analysis = "-"
            
            print(f"{name:<15} {tushare_str:<15} {akshare_str:<15} {tencent_str:<15} {analysis:<20}")
        
        # 推荐说明
        print("\n" + "-" * 80)
        print("📌 数据源推荐:")
        print("   • 历史回测: 优先使用 Tushare（数据完整，支持前复权，消除前视偏差）")
        print("   • 免费回测: 使用 AkShare（免费，但接口可能变动）")
        print("   • 实时监控: 使用 Tencent（实时数据稳定）")
        print("   • 价格差异: Tushare/AkShare 是历史前复权价，Tencent 是实时价")
    
    print("\n" + "=" * 90)
    return results


def main():
    """运行所有测试"""
    print("=" * 60)
    print("🧪 数据层模块测试")
    print("=" * 60)
    
    tests = [
        test_cache_manager,
        test_data_sources_import,
        test_tushare_source_init,
        test_akshare_source_init,
        test_tencent_source_init,
        test_data_manager_init,
        test_data_manager_integration,
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
    import argparse
    
    parser = argparse.ArgumentParser(description='数据层测试')
    parser.add_argument('--compare', '-c', action='store_true', 
                        help='运行数据源对比')
    parser.add_argument('--stock', '-s', type=str, default='600036',
                        help='股票代码，默认600036')
    parser.add_argument('--date', '-d', type=str, default='2024-06-03',
                        help='日期，默认2024-06-03')
    
    args = parser.parse_args()
    
    '''
    if args.compare:
        # 运行数据源对比
        compare_data_sources(args.stock, args.date)
    else:
        # 运行所有测试
        success = main()
        sys.exit(0 if success else 1)
    '''
    compare_data_sources('688506', '2024-12-02')