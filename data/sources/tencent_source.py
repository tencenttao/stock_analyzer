# -*- coding: utf-8 -*-
"""
腾讯财经数据源

基于腾讯财经API实现的数据源，特点：
- 免费，无需Token
- 实时数据稳定
- 适合获取实时行情和基本面快照

注意：腾讯API主要提供实时数据，历史数据支持有限
"""

import time
import random
import logging
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from core.interfaces import DataSource
from core.types import StockData, IndexData

logger = logging.getLogger(__name__)


class TencentSource(DataSource):
    """
    腾讯财经数据源实现
    
    使用腾讯财经API获取实时行情数据。
    
    使用示例:
        source = TencentSource()
        stock = source.get_stock_data('600036', '2024-01-01')
    
    注意：腾讯API主要提供实时数据，历史回测建议使用Tushare或AkShare
    """
    
    def __init__(self):
        """初始化腾讯数据源"""
        # User-Agent池
        self._user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        ]
        
        # 内部缓存
        self._cache: Dict = {}
        
        # 请求间隔控制
        self._last_request_time = 0
        
        logger.info("✅ 腾讯财经数据源初始化成功")
    
    @property
    def name(self) -> str:
        return "tencent"
    
    def _get_headers(self) -> Dict:
        """获取请求头"""
        return {
            'User-Agent': random.choice(self._user_agents),
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive',
            'Referer': 'https://gu.qq.com/'
        }
    
    def _rate_limit(self, min_interval: float = 0.2):
        """API频率限制"""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
    
    def _to_tencent_code(self, code: str) -> str:
        """转换为腾讯格式代码"""
        code = code.strip()
        if code.startswith(('6', '9', '5')):
            return f"sh{code}"
        else:
            return f"sz{code}"
    
    def get_stock_data(self, code: str, date: str = None) -> Optional[StockData]:
        """
        获取股票数据（实时）
        
        注意：腾讯API只提供实时数据，date参数仅作兼容用途。
        如需历史数据请使用Tushare或AkShare。
        
        Args:
            code: 股票代码（6位）
            date: 日期（仅作兼容，实际返回实时数据）
            
        Returns:
            StockData对象
        """
        cache_key = f"stock_{code}"
        
        # 缓存有效期5分钟（实时数据）
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if time.time() - cached_time < 300:
                return cached_data
        
        try:
            self._rate_limit()
            
            symbol = self._to_tencent_code(code)
            url = f"https://qt.gtimg.cn/q={symbol}"
            
            response = requests.get(url, headers=self._get_headers(), timeout=15)
            
            if response.status_code != 200 or 'v_' not in response.text:
                return None
            
            # 解析腾讯返回的数据
            content = response.text
            data_str = content.split('"')[1]
            data_parts = data_str.split('~')
            
            if len(data_parts) < 35:
                return None
            
            # 腾讯API字段说明（2024版）:
            # [1] 名称 [2] 代码 [3] 当前价 [4] 昨收 [5] 今开
            # [6] 成交量 [7] 外盘 [8] 内盘
            # [9-18] 买1-5价和量 [19-28] 卖1-5价和量
            # [30] 时间戳 [31] 涨跌额 [32] 涨跌幅 [33] 最高 [34] 最低
            # [35] 当前价/成交量/成交额 [36] 成交量 [37] 成交额(万)
            # [38] 换手率 [39] PE(TTM) [40] 未知
            # [41] 最高 [42] 最低 [43] 振幅 [44] 流通市值 [45] 总市值
            # [46] PB [47] 涨停价 [48] 跌停价
            
            name = data_parts[1]
            price = self._safe_float(data_parts[3])
            change_pct = self._safe_float(data_parts[32])
            volume = self._safe_float(data_parts[36]) if len(data_parts) > 36 else self._safe_float(data_parts[6])
            turnover = self._safe_float(data_parts[37]) if len(data_parts) > 37 else None  # 成交额(万)
            
            if not price or price <= 0:
                return None
            
            # 解析PE（字段39是PE TTM）
            pe_ratio = None
            if len(data_parts) > 39:
                pe = self._safe_float(data_parts[39])
                if pe and 0 < pe < 500:  # 合理范围
                    pe_ratio = pe
            
            # 解析PB（字段46）
            pb_ratio = None
            if len(data_parts) > 46:
                pb = self._safe_float(data_parts[46])
                if pb and 0 < pb < 100:  # 合理范围检查
                    pb_ratio = pb
            
            # 解析换手率（字段38）
            turnover_rate = None
            if len(data_parts) > 38:
                tr = self._safe_float(data_parts[38])
                if tr and 0 < tr < 100:  # 换手率应该在0-100%
                    turnover_rate = tr
            
            # 解析股息率
            dividend_yield = None
            if len(data_parts) > 53:
                div_data = self._safe_float(data_parts[53])
                if div_data and div_data > 0 and price > 0:
                    # 腾讯返回的是每10股股息
                    per_share_div = div_data / 10
                    dividend_yield = (per_share_div / price) * 100
                    if dividend_yield > 20:  # 异常值检查
                        dividend_yield = None
            
            # 计算ROE（从PB/PE估算）
            roe = None
            if pb_ratio and pe_ratio and pe_ratio > 0:
                roe = (pb_ratio / pe_ratio) * 100
            
            # 组装数据
            stock = StockData(
                code=code,
                name=name,
                price=price,
                change_pct=change_pct or 0,
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio,
                roe=roe,
                turnover_rate=turnover_rate,
                dividend_yield=dividend_yield,
                volume=volume or 0,
                turnover=(turnover or 0) * 10000,  # 转换为元
                date=date or datetime.now().strftime('%Y-%m-%d'),
                data_source='tencent'
            )
            
            # 缓存
            self._cache[cache_key] = (time.time(), stock)
            return stock
            
        except Exception as e:
            logger.debug(f"获取{code}数据失败: {e}")
            return None
    
    def _safe_float(self, value) -> Optional[float]:
        """安全转换为浮点数"""
        if value is None or value == '':
            return None
        try:
            result = float(value)
            return result if result != 0 else None
        except (ValueError, TypeError):
            return None
    
    def get_stock_list(self, date: str = None) -> List[str]:
        """
        获取A股股票列表
        
        注意：腾讯API不提供完整股票列表，返回空列表。
        请使用Tushare或AkShare获取股票列表。
        """
        logger.warning("腾讯数据源不支持获取股票列表，请使用Tushare或AkShare")
        return []
    
    def get_index_constituents(self, index_code: str, date: str = None) -> List[str]:
        """
        获取指数成分股
        
        注意：腾讯API不提供指数成分股，返回空列表。
        请使用Tushare或AkShare获取成分股。
        """
        logger.warning("腾讯数据源不支持获取指数成分股，请使用Tushare或AkShare")
        return []
    
    def get_index_data(self, index_code: str, start_date: str, end_date: str) -> Optional[IndexData]:
        """
        获取指数数据
        
        仅支持获取实时指数点位，不支持历史数据。
        """
        try:
            self._rate_limit()
            
            # 转换指数代码
            if index_code in ['000300', '000001', '000016']:
                symbol = f"sh{index_code}"
            else:
                symbol = f"sz{index_code}"
            
            url = f"https://qt.gtimg.cn/q={symbol}"
            response = requests.get(url, headers=self._get_headers(), timeout=15)
            
            if response.status_code != 200:
                return None
            
            content = response.text
            if 'v_' not in content:
                return None
            
            data_str = content.split('"')[1]
            data_parts = data_str.split('~')
            
            if len(data_parts) < 5:
                return None
            
            name = data_parts[1]
            price = self._safe_float(data_parts[3])
            
            if not price:
                return None
            
            # 腾讯只提供实时数据，无法计算区间收益
            return IndexData(
                code=index_code,
                name=name,
                start_date=start_date,
                end_date=end_date,
                start_price=price,
                end_price=price,
                return_pct=0,
                trading_days=1
            )
            
        except Exception as e:
            logger.error(f"获取指数数据失败: {e}")
            return None
    
    def get_trading_calendar(self, start_date: str, end_date: str) -> List[str]:
        """
        获取交易日历
        
        腾讯API不提供交易日历，使用默认实现。
        """
        return super().get_trading_calendar(start_date, end_date)
    
    def batch_get_stock_data(self, codes: List[str], date: str = None) -> List[StockData]:
        """
        批量获取股票数据
        
        腾讯API支持批量查询，每次最多80只。
        """
        results = []
        batch_size = 80
        
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            batch_results = self._batch_get(batch)
            results.extend(batch_results)
            
            if i + batch_size < len(codes):
                logger.info(f"数据获取进度: {min(i + batch_size, len(codes))}/{len(codes)}")
        
        logger.info(f"批量获取完成: {len(results)}/{len(codes)}")
        return results
    
    def _batch_get(self, codes: List[str]) -> List[StockData]:
        """批量获取一组股票数据"""
        try:
            self._rate_limit()
            
            # 构造批量查询URL
            symbols = [self._to_tencent_code(code) for code in codes]
            url = f"https://qt.gtimg.cn/q={','.join(symbols)}"
            
            response = requests.get(url, headers=self._get_headers(), timeout=30)
            
            if response.status_code != 200:
                return []
            
            results = []
            lines = response.text.strip().split('\n')
            
            for line in lines:
                if 'v_' not in line or '"' not in line:
                    continue
                
                try:
                    # 提取股票代码
                    var_name = line.split('=')[0].strip()
                    code = var_name.replace('v_sh', '').replace('v_sz', '')
                    
                    # 解析数据
                    data_str = line.split('"')[1]
                    data_parts = data_str.split('~')
                    
                    if len(data_parts) < 35:
                        continue
                    
                    name = data_parts[1]
                    price = self._safe_float(data_parts[3])
                    
                    if not price or price <= 0:
                        continue
                    
                    change_pct = self._safe_float(data_parts[32]) or 0
                    
                    # PE（字段39）
                    pe_ratio = None
                    if len(data_parts) > 39:
                        pe = self._safe_float(data_parts[39])
                        if pe and 0 < pe < 500:
                            pe_ratio = pe
                    
                    # PB（字段46）
                    pb_ratio = None
                    if len(data_parts) > 46:
                        pb = self._safe_float(data_parts[46])
                        if pb and 0 < pb < 100:
                            pb_ratio = pb
                    
                    # 换手率（字段38）
                    turnover_rate = None
                    if len(data_parts) > 38:
                        tr = self._safe_float(data_parts[38])
                        if tr and 0 < tr < 100:
                            turnover_rate = tr
                    
                    # 成交额（字段37，单位万）
                    turnover = self._safe_float(data_parts[37]) if len(data_parts) > 37 else None
                    
                    stock = StockData(
                        code=code,
                        name=name,
                        price=price,
                        change_pct=change_pct,
                        pe_ratio=pe_ratio,
                        pb_ratio=pb_ratio,
                        turnover_rate=turnover_rate,
                        turnover=(turnover or 0) * 10000,
                        date=datetime.now().strftime('%Y-%m-%d'),
                        data_source='tencent'
                    )
                    
                    if stock.is_valid():
                        results.append(stock)
                        
                except Exception as e:
                    logger.debug(f"解析行数据失败: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.warning(f"批量获取失败: {e}")
            return []
    
    def clear_cache(self):
        """清空内部缓存"""
        self._cache.clear()
