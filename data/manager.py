# -*- coding: utf-8 -*-
"""
数据管理器

统一的数据获取入口，功能：
- 管理多数据源，支持灵活切换
- 统一缓存管理
- 数据源故障自动回退

所有默认参数从配置文件读取，代码中不包含硬编码默认值。

直接运行本文件时（python data/manager.py）会自动把项目根加入 sys.path，保证可导入 core、config 等。
"""
import os
import sys

# 直接运行本文件时保证项目根在 path 中，避免 ModuleNotFoundError: No module named 'core'
if __name__ == '__main__' or 'core' not in sys.modules:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

import logging
from typing import List, Dict, Optional, Type

from core.interfaces import DataSource
from core.types import StockData, IndexData
from data.cache import CacheManager

# 从配置读取默认值（必须）
from config.data_source_config import DEFAULT_DATA_SOURCE
from config.settings import CACHE_DIR, CACHE_EXPIRE_DAYS

logger = logging.getLogger(__name__)


class DataManager(DataSource):
    """
    数据管理器
    
    作为系统的统一数据入口，实现 DataSource 接口。
    
    功能：
    - 管理底层数据源，支持灵活切换
    - 统一缓存管理，提升性能
    - 可直接传给 BacktestEngine 使用
    
    使用示例:
        # 使用默认数据源（Tushare）
        manager = DataManager()
        
        # 指定数据源
        manager = DataManager(source='tushare')
        
        # 获取数据
        stock = manager.get_stock_data('600036', '2024-01-01')
        constituents = manager.get_index_constituents('000300', '2024-01-01')
        
        # 用于回测
        engine = BacktestEngine(manager, strategy, config)
    """
    
    def __init__(
        self,
        source: str = None,
        cache_dir: str = None,
        cache_expire_days: int = None,
        use_cache: bool = True,
        write_cache: bool = None,
        **source_kwargs
    ):
        """
        初始化数据管理器
        
        Args:
            source: 数据源名称（'tushare', 'akshare'等），默认从配置读取
            cache_dir: 缓存目录，默认从配置读取
            cache_expire_days: 缓存过期天数，默认从配置读取
            use_cache: 是否从缓存读取数据
            write_cache: 是否将获取的数据写入缓存（默认跟随 use_cache）
            **source_kwargs: 传递给数据源的额外参数
            
        缓存模式:
            use_cache=True                     → 读写都启用（默认，向后兼容）
            use_cache=False                    → 完全禁用（向后兼容）
            use_cache=False, write_cache=True  → 不读但写（强制刷新并更新缓存）
            use_cache=True,  write_cache=False → 只读不写
        """
        super().__init__()  # 初始化父类
        
        # 使用配置中的默认值（不在代码中硬编码）
        source = source if source is not None else DEFAULT_DATA_SOURCE
        cache_dir = cache_dir if cache_dir is not None else CACHE_DIR
        cache_expire_days = cache_expire_days if cache_expire_days is not None else CACHE_EXPIRE_DAYS
        
        # write_cache 默认跟随 use_cache（向后兼容）
        if write_cache is None:
            write_cache = use_cache
        
        self._source_name = source
        self._read_cache = use_cache
        self._write_cache = write_cache
        
        # 初始化缓存（只要读或写任一启用，就初始化 CacheManager）
        self._cache = CacheManager(cache_dir, cache_expire_days) if (use_cache or write_cache) else None
        
        # 初始化数据源
        self._source = self._create_source(source, **source_kwargs)
        
        cache_mode = []
        if use_cache:
            cache_mode.append('读')
        if write_cache:
            cache_mode.append('写')
        cache_str = '+'.join(cache_mode) if cache_mode else '禁用'
        logger.info(f"数据管理器初始化: 数据源={source}, 缓存={cache_str}")
        # 交易日历缓存：{(start, end): [trading_days_list]}
        self._trading_cal_cache: Dict[tuple, List[str]] = {}
        # 日期解析缓存：{requested_date: resolved_trading_date}
        self._resolved_dates: Dict[str, str] = {}
    
    def _create_source(self, source_name: str, **kwargs) -> DataSource:
        """创建数据源实例"""
        from data.sources import get_source
        source_cls = get_source(source_name)
        return source_cls(**kwargs)
    
    @property
    def name(self) -> str:
        """DataSource接口：返回数据源名称"""
        return f"manager:{self._source_name}"
    
    @property
    def source(self) -> DataSource:
        """获取当前底层数据源"""
        return self._source
    
    @property
    def source_name(self) -> str:
        """获取数据源名称"""
        return self._source_name
    
    def switch_source(self, source: str, **kwargs):
        """
        切换数据源
        
        Args:
            source: 新数据源名称
            **kwargs: 传递给数据源的参数
        """
        self._source = self._create_source(source, **kwargs)
        self._source_name = source
        logger.info(f"数据源已切换为: {source}")
    
    def _resolve_trading_date(self, date: str) -> str:
        """
        将请求日期解析为最近的实际交易日。
        
        流程：
        1. 校验格式、是否未来日期
        2. 通过 Tushare 交易日历判断是否交易日（含节假日）
        3. 若非交易日，向前搜索最近的交易日
        4. 打一次日志告知用户
        5. 缓存结果，同一日期不重复解析
        
        Args:
            date: 请求日期 (YYYY-MM-DD)
            
        Returns:
            实际交易日 (YYYY-MM-DD)
        """
        # 已解析过，直接返回
        if date in self._resolved_dates:
            return self._resolved_dates[date]
        
        from datetime import datetime as _dt, date as _date_cls, timedelta as _td
        
        # 1. 格式校验
        try:
            d = _dt.strptime(date, '%Y-%m-%d').date()
        except ValueError:
            raise ValueError(f"日期格式异常: '{date}'，应为 YYYY-MM-DD")
        
        # 2. 未来日期校验
        today = _date_cls.today()
        if d > today:
            raise ValueError(
                f"请求日期 {date} 是未来日期（今天 {today}），无法获取行情数据"
            )
        
        # 3. 通过交易日历查找实际交易日
        #    向前搜索最多 30 天（覆盖春节、国庆等长假）
        search_start = d - _td(days=30)
        cal_key = (search_start.strftime('%Y-%m-%d'), date)
        
        if cal_key not in self._trading_cal_cache:
            try:
                cal = self._source.get_trading_calendar(
                    cal_key[0], cal_key[1]
                )
                self._trading_cal_cache[cal_key] = cal
            except Exception as e:
                logger.warning(f"获取交易日历失败: {e}，回退到简单周末判断")
                # 回退：只排除周末
                cal = []
                cur = search_start
                while cur <= d:
                    if cur.weekday() < 5:
                        cal.append(cur.strftime('%Y-%m-%d'))
                    cur += _td(days=1)
                self._trading_cal_cache[cal_key] = cal
        
        trading_days = self._trading_cal_cache[cal_key]
        
        if not trading_days:
            raise ValueError(
                f"在 {search_start} ~ {date} 范围内未找到任何交易日，请检查日期是否合理"
            )
        
        # 找到 <= date 的最近交易日
        resolved = trading_days[-1]  # 已排序，最后一个 <= date
        for td in reversed(trading_days):
            if td <= date:
                resolved = td
                break
        
        # 4. 打日志（只打一次）
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        if resolved != date:
            wd = d.weekday()
            reason = f"{weekday_names[wd]}" if wd >= 5 else "节假日"
            logger.warning(
                f"请求日期 {date} 是{reason}（非交易日），"
                f"自动调整为最近交易日: {resolved}"
            )
        else:
            logger.debug(f"请求日期 {date} 是交易日")
        
        # 5. 缓存解析结果
        self._resolved_dates[date] = resolved
        return resolved
    
    def get_stock_data(self, code: str, date: str) -> Optional[StockData]:
        """
        获取股票数据
        
        流程：
        1. 将请求日期解析为实际交易日（通过交易日历，含节假日判断）
        2. 用实际交易日读缓存
        3. 缓存未命中则从数据源获取
        4. 用实际交易日写缓存
        
        Args:
            code: 股票代码
            date: 日期（YYYY-MM-DD），可以是非交易日，会自动解析
            
        Returns:
            StockData对象
        """
        # 1. 解析为实际交易日（首次会打日志告知用户）
        trading_date = self._resolve_trading_date(date)
        
        logger.debug(f"[get_stock_data] 请求: code={code}, date={date} → trading_date={trading_date}")
        
        # 2. 用实际交易日读缓存
        cache_key = f"stock_{code}_{trading_date}"
        if self._cache and self._read_cache:
            cached = self._cache.get(cache_key)
            if cached:
                logger.debug(f"[get_stock_data] 缓存命中: {code}@{cache_key}")
                return StockData.from_dict(cached) if isinstance(cached, dict) else cached
        
        # 3. 从数据源获取（传入实际交易日，避免数据源内部再次退化）
        stock = self._source.get_stock_data(code, trading_date)
        
        # 4. 用实际交易日写缓存
        if stock and self._cache and self._write_cache:
            self._cache.set(cache_key, stock.to_dict())
            logger.debug(f"[get_stock_data] 获取成功: {code}@{cache_key}, 价格={stock.price:.2f}")
        elif not stock:
            logger.debug(f"[get_stock_data] 获取失败: {code}@{trading_date}")
        
        return stock
    
    def batch_get_stock_data(self, codes: List[str], date: str) -> List[StockData]:
        """
        批量获取股票数据
        
        流程与 get_stock_data 一致：先解析实际交易日，再统一用它操作缓存和数据源。
        
        Args:
            codes: 股票代码列表
            date: 日期（可以是非交易日，会自动解析）
            
        Returns:
            StockData列表
        """
        # 1. 解析为实际交易日
        trading_date = self._resolve_trading_date(date)
        
        # 2. 分离缓存命中和未命中
        results = []
        missing_codes = []
        
        if self._cache and self._read_cache:
            for code in codes:
                cache_key = f"stock_{code}_{trading_date}"
                cached = self._cache.get(cache_key)
                if cached:
                    stock = StockData.from_dict(cached) if isinstance(cached, dict) else cached
                    results.append(stock)
                else:
                    missing_codes.append(code)
        else:
            missing_codes = list(codes)
        
        # 3. 从数据源获取未缓存的（传入实际交易日）
        if missing_codes:
            logger.info(f"从数据源获取 {len(missing_codes)} 只股票数据...")
            new_stocks = self._source.batch_get_stock_data(missing_codes, trading_date)
            
            # 4. 用实际交易日写缓存
            if self._cache and self._write_cache:
                for stock in new_stocks:
                    cache_key = f"stock_{stock.code}_{trading_date}"
                    self._cache.set(cache_key, stock.to_dict())
            
            results.extend(new_stocks)
        
        logger.info(f"数据获取完成: {len(results)} 只 (缓存命中: {len(codes) - len(missing_codes)})")
        return results
    
    def get_index_constituents(self, index_code: str, date: str = None) -> List[str]:
        """
        DataSource接口：获取指数成分股
        
        Args:
            index_code: 指数代码（如 '000300'）
            date: 可选日期
            
        Returns:
            成分股代码列表
        """

        
        stocks = self._source.get_index_constituents(index_code, date)
        

        
        return stocks
    
    def get_csi300_stocks(self, date: str = None) -> List[str]:
        """
        获取沪深300成分股（便捷方法）
        
        Args:
            date: 可选日期
            
        Returns:
            成分股代码列表
        """
        return self.get_index_constituents('000300', date)
    
    def get_csi500_stocks(self, date: str = None) -> List[str]:
        """
        获取中证500成分股（便捷方法）

        与 get_csi300_stocks 一致，不走文件缓存（成分股随调仓日变化，
        且 TushareSource 内部有进程级内存缓存）。
        
        Args:
            date: 可选日期
            
        Returns:
            成分股代码列表
        """
        return self.get_index_constituents('000905', date)
    
    def get_index_data(self, index_code: str, start_date: str, end_date: str = None) -> Optional[IndexData]:
        """
        DataSource接口：获取指数表现数据
        
        Args:
            index_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期（可选，不传则等于start_date）
            
        Returns:
            IndexData对象
        """
        if end_date is None:
            end_date = start_date
            
        cache_key = f"index_{index_code}_{start_date}_{end_date}"
        
        if self._cache and self._read_cache:
            cached = self._cache.get(cache_key)
            if cached:
                return IndexData(**cached) if isinstance(cached, dict) else cached
        
        index_data = self._source.get_index_data(index_code, start_date, end_date)
        
        if index_data and self._cache and self._write_cache:
            self._cache.set(cache_key, index_data.to_dict() if hasattr(index_data, 'to_dict') else index_data)
        
        return index_data
    
    def get_index_return(self, index_code: str, start_date: str, end_date: str) -> float:
        """
        DataSource接口：获取指数区间收益率
        
        注意：不使用文件缓存。该接口调用量小（每月1次），
        但缓存容易出错（API失败时会缓存0.0污染后续计算）。
        
        Args:
            index_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            收益率 (%)
        """
        # 通过 get_index_data 获取收益率（不缓存，每次实时计算）
        index_data = self._source.get_index_data(index_code, start_date, end_date)
        
        if index_data is None:
            logger.warning(f"获取指数收益率失败: {index_code} {start_date}~{end_date}")
            return 0.0
        
        return index_data.return_pct if index_data.return_pct is not None else 0.0

    def get_index_daily(
        self, index_code: str, start_date: str, end_date: str
    ) -> Optional[List[Dict]]:
        """
        获取指数日线数据（用于计算市场环境特征）
        
        Args:
            index_code: 指数代码，如 '000300'
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            
        Returns:
            日线列表，每条包含 trade_date, close 等
        """
        cache_key = f"index_daily_{index_code}_{start_date}_{end_date}"
        if self._cache and self._read_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        if hasattr(self._source, 'get_index_daily'):
            result = self._source.get_index_daily(index_code, start_date, end_date)
        else:
            logger.warning(f"数据源 {self._source.name} 不支持 get_index_daily")
            return None
        if result and self._cache and self._write_cache:
            self._cache.set(cache_key, result)
        return result

    def get_daily_data(
        self, 
        code: str, 
        start_date: str = None, 
        end_date: str = None,
        adj: str = 'qfq',
        days: int = 90
    ) -> Optional[List[Dict]]:
        """
        获取股票日线数据（用于计算技术指标）
        
        Args:
            code: 股票代码
            start_date: 开始日期 YYYY-MM-DD（可选，默认从 end_date 往前推 days 天）
            end_date: 结束日期 YYYY-MM-DD（可选，默认今天）
            adj: 复权类型 'qfq'前复权(默认), 'hfq'后复权, None不复权
            days: 当 start_date 为空时，从 end_date 往前推的天数（默认90）
            
        Returns:
            日线数据列表，每条包含:
            - trade_date: 交易日期
            - open, high, low, close: OHLC价格
            - vol: 成交量
            - amount: 成交额
            - pct_chg: 涨跌幅
        """
        from datetime import datetime, timedelta
        
        # 处理默认日期
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=days)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        cache_key = f"daily_{code}_{start_date}_{end_date}_{adj}"
        
        if self._cache and self._read_cache:
            cached = self._cache.get(cache_key)
            if cached:
                logger.debug(f"[缓存命中] 日线数据 {code} {start_date}-{end_date}")
                return cached
        
        # 检查底层数据源是否支持 get_daily_data
        if hasattr(self._source, 'get_daily_data'):
            data = self._source.get_daily_data(code, start_date, end_date, adj)
        else:
            logger.warning(f"数据源 {self._source_name} 不支持 get_daily_data")
            return None
        
        if data and self._cache and self._write_cache:
            self._cache.set(cache_key, data)
        
        return data

    def get_stock_list(self, market: str = None) -> List[str]:
        """
        DataSource接口：获取股票列表
        
        Args:
            market: 市场筛选（'SH', 'SZ' 或 None 表示全部）
            
        Returns:
            股票代码列表
        """
        cache_key = f"stock_list_{market or 'all'}"
        
        if self._cache and self._read_cache:
            cached = self._cache.get(cache_key)
            if cached:
                return cached
        
        stocks = self._source.get_stock_list(market)
        
        if stocks and self._cache and self._write_cache:
            self._cache.set(cache_key, stocks)
        
        return stocks
    
    def get_trading_calendar(self, start_date: str, end_date: str) -> List[str]:
        """
        获取交易日历
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            交易日列表
        """
        
        
        calendar = self._source.get_trading_calendar(start_date, end_date)
        
        
        return calendar
    
    def get_first_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """
        获取每月第一个交易日
        
        通过交易日历筛选，确保返回的是真正的交易日（与 backtest 逻辑一致）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            每月首个交易日列表
        """
        
        first_days = []
        
        # 通过交易日历获取（与 backtest 逻辑一致）
        try:
            trading_days = self.get_trading_calendar(start_date, end_date)
            
            if trading_days:
                current_month = None
                for day in sorted(trading_days):
                    month = day[:7]  # YYYY-MM
                    if month != current_month:
                        first_days.append(day)
                        current_month = month
                
                logger.debug(f"从交易日历获取到 {len(first_days)} 个月度首日")
        except Exception as e:
            logger.warning(f"无法从交易日历获取首日: {e}")

        
        return first_days
    
    def clear_cache(self):
        """清空缓存"""
        if self._cache:
            count = self._cache.clear()
            logger.info(f"缓存已清空: {count} 个")
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        if self._cache:
            return self._cache.get_stats()
        return {'enabled': False}


# ============================================
# 主要接口测试
# 运行: python -m data.manager
# 或:  from data.manager import run_manager_tests; run_manager_tests('2024-01-01', '2024-06-30')
# ============================================

def _test_get_first_trading_days(dm: DataManager, start: str = '2021-01-01', end: str = '2025-01-30') -> bool:
    """测试 get_first_trading_days：返回每月首个交易日列表"""
    days = dm.get_first_trading_days(start, end)
    assert isinstance(days, list), "应返回列表"
    assert len(days) >= 1, "至少应有一个月"
    for d in days:
        assert len(d) == 10 and d[4] == '-' and d[7] == '-', f"日期格式应为 YYYY-MM-DD: {d}"
    months = set(d[:7] for d in days)
    assert len(months) == len(days), "每月应只有一个首日"
    print(f"  get_first_trading_days({start}, {end}): {len(days)} 个月度首日, 示例 {days[:3]}")
    return True


def _test_get_stock_data(dm: DataManager, code: str = '600036', date: str = '2024-01-02') -> bool:
    """测试 get_stock_data：返回 StockData 或 None"""
    stock = dm.get_stock_data(code, date)
    if stock is None:
        print(f"  get_stock_data({code}, {date}): 无数据")
        return True
    assert hasattr(stock, 'code') and hasattr(stock, 'price'), "应包含 code、price"
    assert stock.code == code or stock.code.replace('.', '')[:6] == code.replace('.', '')[:6]
    print(f"  get_stock_data({code}, {date}): {getattr(stock, 'name', '')} 价格={stock.price:.2f}")
    return True


def _test_get_csi300_stocks(dm: DataManager, date: str = '2024-01-02') -> bool:
    """测试 get_csi300_stocks：返回成分股代码列表"""
    codes = dm.get_csi300_stocks(date)
    assert isinstance(codes, list), "应返回列表"
    assert len(codes) > 0, "成分股不应为空"
    for c in codes[:5]:
        assert isinstance(c, str) and len(c) >= 6, f"代码格式异常: {c}"
    print(f"  get_csi300_stocks({date}): {len(codes)} 只")
    return True


def _test_get_trading_calendar(dm: DataManager, start: str = '2024-01-01', end: str = '2024-01-31') -> bool:
    """测试 get_trading_calendar：返回交易日列表"""
    cal = dm.get_trading_calendar(start, end)
    assert isinstance(cal, list), "应返回列表"
    if cal:
        for d in cal[:3]:
            assert len(d) == 10 or len(d) == 8, f"日期格式异常: {d}"
        print(f"  get_trading_calendar({start}, {end}): {len(cal)} 个交易日")
    else:
        print(f"  get_trading_calendar({start}, {end}): 无数据")
    return True


def _test_get_daily_data(dm: DataManager, code: str = '600036', end_date: str = '2024-01-15', days: int = 30) -> bool:
    """测试 get_daily_data：返回日线列表，元素含 close/trade_date 等"""
    data = dm.get_daily_data(code, end_date=end_date, days=days)
    if data is None:
        print(f"  get_daily_data({code}, end_date={end_date}, days={days}): 无数据")
        return True
    assert isinstance(data, list), "应返回列表"
    if data:
        row = data[0]
        assert isinstance(row, dict), "每行为字典"
        assert 'close' in row or 'trade_date' in row, "应含 close 或 trade_date"
        print(f"  get_daily_data(..., end_date={end_date}, days={days}): {len(data)} 条")
    else:
        print(f"  get_daily_data(..., end_date={end_date}, days={days}): 空列表")
    return True


def run_manager_tests(
    start_date: str = '2021-01-01',
    end_date: str = '2025-01-05',
    stock_code: str = '600036',
    test_date: str = '2021-04-01',
) -> bool:
    """
    运行 DataManager 主要接口的简单测试。
    可传入日期与股票代码以适配不同环境。
    """
    import sys
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    dm = DataManager(use_cache=True)

    ok = True
    try:
        print("DataManager 接口测试")
        print("-" * 50)
        #_test_get_first_trading_days(dm, start_date, end_date)
        #_test_get_trading_calendar(dm, start_date, end_date)
        #_test_get_stock_data(dm, stock_code, test_date)
        _test_get_csi300_stocks(dm, test_date)
        #_test_get_daily_data(dm, code=stock_code, end_date=test_date, days=30)
        print("-" * 50)
        print("全部通过")
    except Exception as e:
        print(f"失败: {e}")
        ok = False
    return ok


if __name__ == '__main__':
    run_manager_tests()
