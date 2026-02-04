# -*- coding: utf-8 -*-
"""
AkShare数据源

基于AkShare库实现的数据源，特点：
- 完全免费，无需Token
- 数据全面，包含财务指标
- 可能不稳定，需要多重回退

注意：AkShare依赖第三方网站，可能存在接口变动
"""

import time
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from core.interfaces import DataSource
from core.types import StockData, IndexData

logger = logging.getLogger(__name__)


class AkShareSource(DataSource):
    """
    AkShare数据源实现
    
    使用AkShare库获取金融数据，免费但可能不稳定。
    
    使用示例:
        source = AkShareSource()
        stock = source.get_stock_data('600036', '2024-01-01')
    """
    
    def __init__(self):
        """初始化AkShare数据源"""
        self._init_api()
        
        # 内部缓存
        self._cache: Dict = {}
        
        # 请求间隔控制
        self._last_request_time = 0
    
    def _init_api(self):
        """初始化AkShare"""
        try:
            import akshare as ak
            self._ak = ak
            logger.info("✅ AkShare数据源初始化成功")
        except ImportError:
            raise ImportError("AkShare未安装，请运行: pip install akshare")
    
    @property
    def name(self) -> str:
        return "akshare"
    
    def _rate_limit(self, min_interval: float = 0.3):
        """API频率限制"""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
    
    def _safe_float(self, value) -> Optional[float]:
        """安全转换为浮点数"""
        import pandas as pd
        import numpy as np
        
        if value is None or pd.isna(value):
            return None
        try:
            result = float(value)
            if np.isnan(result) or np.isinf(result):
                return None
            return result
        except (ValueError, TypeError):
            return None
    
    def get_stock_data(self, code: str, date: str) -> Optional[StockData]:
        """
        获取单只股票在指定日期的数据
        
        Args:
            code: 股票代码（6位）
            date: 日期（YYYY-MM-DD）
            
        Returns:
            StockData对象
        """
        cache_key = f"stock_{code}_{date}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # 1. 获取历史行情数据
            price_data = self._get_historical_price(code, date)
            if not price_data:
                return None
            
            # 2. 获取股票名称
            name = self._get_stock_name(code)
            
            # 3. 获取财务指标
            fin_data = self._get_financial_indicators(code, date)
            
            # 4. 组装数据
            stock = StockData(
                code=code,
                name=name,
                price=price_data.get('close', 0),
                change_pct=price_data.get('pct_chg', 0),
                momentum_20d=price_data.get('momentum_20d', 0),
                volume=price_data.get('volume', 0),
                turnover=price_data.get('amount', 0),
                turnover_rate=price_data.get('turnover_rate'),
                date=date,
                data_source='akshare'
            )
            
            # 合并财务指标
            if fin_data:
                stock.pe_ratio = fin_data.get('pe_ratio')
                stock.pb_ratio = fin_data.get('pb_ratio')
                stock.roe = fin_data.get('roe')
                stock.profit_growth = fin_data.get('profit_growth')
                stock.dividend_yield = fin_data.get('dividend_yield')
            
            # 计算PEG
            if stock.pe_ratio and stock.profit_growth and stock.profit_growth > 0:
                stock.peg = round(stock.pe_ratio / stock.profit_growth, 2)
            
            # 如果没有ROE，尝试从PB/PE估算
            if not stock.roe and stock.pb_ratio and stock.pe_ratio and stock.pe_ratio > 0:
                stock.roe = round(stock.pb_ratio / stock.pe_ratio * 100, 2)
            
            self._cache[cache_key] = stock
            return stock
            
        except Exception as e:
            logger.debug(f"获取{code}数据失败: {e}")
            return None
    
    def _get_historical_price(self, code: str, date: str) -> Optional[Dict]:
        """获取历史价格和动量"""
        try:
            self._rate_limit()
            
            # 计算日期范围（获取60天数据用于计算动量）
            end_dt = datetime.strptime(date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=90)
            
            df = self._ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_dt.strftime('%Y%m%d'),
                end_date=end_dt.strftime('%Y%m%d'),
                adjust="qfq"
            )
            
            if df is None or df.empty:
                return None
            
            # 转换日期格式
            df['日期'] = df['日期'].astype(str)
            target_date = date.replace('-', '')
            
            # 找到目标日期或之前最近的数据
            df = df.sort_values('日期', ascending=False)
            
            latest = None
            for _, row in df.iterrows():
                row_date = row['日期'].replace('-', '')
                if row_date <= target_date:
                    latest = row
                    break
            
            if latest is None:
                return None
            
            # 计算20日动量
            momentum_20d = 0
            prices = df['收盘'].values
            if len(prices) >= 20:
                if prices[19] > 0:
                    momentum_20d = (prices[0] / prices[19] - 1) * 100
            
            return {
                'close': float(latest['收盘']),
                'pct_chg': float(latest.get('涨跌幅', 0)),
                'volume': float(latest.get('成交量', 0)),
                'amount': float(latest.get('成交额', 0)),
                'turnover_rate': self._safe_float(latest.get('换手率')),
                'momentum_20d': momentum_20d
            }
            
        except Exception as e:
            logger.debug(f"获取{code}历史价格失败: {e}")
            return None
    
    def _get_stock_name(self, code: str) -> str:
        """获取股票名称"""
        cache_key = f"name_{code}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            self._rate_limit()
            info = self._ak.stock_individual_info_em(symbol=code)
            if info is not None and not info.empty:
                name_row = info[info['item'] == '股票简称']
                if not name_row.empty:
                    name = str(name_row['value'].iloc[0])
                    self._cache[cache_key] = name
                    return name
        except:
            pass
        
        return ""
    
    def _get_financial_indicators(self, code: str, target_date: str) -> Optional[Dict]:
        """获取财务指标"""
        try:
            self._rate_limit()
            
            # 尝试获取财务分析指标
            df = self._ak.stock_financial_analysis_indicator(symbol=code, start_year="2020")
            
            if df is None or df.empty:
                return None
            
            # 转换报告期格式并筛选
            import pandas as pd
            df['报告期'] = pd.to_datetime(df['报告期'])
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            
            # 只使用公告日期在目标日期之前的数据（消除前视偏差）
            # AkShare的财务指标通常在季报发布后才可用
            report_lag = timedelta(days=60)  # 假设60天发布滞后
            valid_report_date = target_dt - report_lag
            
            df_valid = df[df['报告期'] <= valid_report_date]
            
            if df_valid.empty:
                return None
            
            # 取最新一期
            latest = df_valid.iloc[0]
            
            return {
                'roe': self._safe_float(latest.get('净资产收益率(%)')),
                'profit_growth': self._safe_float(latest.get('主营业务收入增长率(%)')),
                'pe_ratio': self._safe_float(latest.get('市盈率')),
                'pb_ratio': self._safe_float(latest.get('市净率')),
                'dividend_yield': None  # AkShare财务指标中通常没有股息率
            }
            
        except Exception as e:
            logger.debug(f"获取{code}财务指标失败: {e}")
            return None
    
    def get_stock_list(self, date: str = None) -> List[str]:
        """获取A股股票列表"""
        cache_key = 'stock_list'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            self._rate_limit()
            df = self._ak.stock_info_a_code_name()
            
            if df is None or df.empty:
                return []
            
            result = df['code'].tolist()
            self._cache[cache_key] = result
            
            logger.info(f"获取A股列表: {len(result)} 只")
            return result
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
    
    def get_index_constituents(self, index_code: str, date: str = None) -> List[str]:
        """获取指数成分股"""
        cache_key = f"index_{index_code}_{date or 'latest'}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            self._rate_limit()
            
            # AkShare获取指数成分股
            if index_code == '000300':
                # 沪深300
                df = self._ak.index_stock_cons_csindex(symbol="000300")
            elif index_code == '000905':
                # 中证500
                df = self._ak.index_stock_cons_csindex(symbol="000905")
            else:
                df = self._ak.index_stock_cons_csindex(symbol=index_code)
            
            if df is None or df.empty:
                return []
            
            # 提取成分股代码
            if '成分券代码' in df.columns:
                result = df['成分券代码'].astype(str).str.zfill(6).tolist()
            elif '证券代码' in df.columns:
                result = df['证券代码'].astype(str).str.zfill(6).tolist()
            else:
                # 尝试第一列
                result = df.iloc[:, 0].astype(str).str.zfill(6).tolist()
            
            self._cache[cache_key] = result
            logger.info(f"获取{index_code}成分股: {len(result)} 只")
            return result
            
        except Exception as e:
            logger.error(f"获取指数成分股失败: {e}")
            return []
    
    def get_index_data(self, index_code: str, start_date: str, end_date: str) -> Optional[IndexData]:
        """获取指数数据"""
        try:
            self._rate_limit()
            
            # 转换指数代码格式
            if index_code == '000300':
                symbol = 'sh000300'
            elif index_code == '000905':
                symbol = 'sh000905'
            else:
                symbol = f'sh{index_code}'
            
            df = self._ak.stock_zh_index_daily(symbol=symbol)
            
            if df is None or df.empty:
                return None
            
            # 筛选日期范围
            df['date'] = df['date'].astype(str)
            start_str = start_date.replace('-', '')
            end_str = end_date.replace('-', '')
            
            df = df[(df['date'] >= start_str) & (df['date'] <= end_str)]
            
            if df.empty:
                return None
            
            df = df.sort_values('date')
            
            start_price = float(df.iloc[0]['close'])
            end_price = float(df.iloc[-1]['close'])
            return_pct = (end_price / start_price - 1) * 100
            
            return IndexData(
                code=index_code,
                name='沪深300' if index_code == '000300' else index_code,
                start_date=df.iloc[0]['date'],
                end_date=df.iloc[-1]['date'],
                start_price=start_price,
                end_price=end_price,
                return_pct=return_pct,
                trading_days=len(df)
            )
            
        except Exception as e:
            logger.error(f"获取指数数据失败: {e}")
            return None
    
    def get_trading_calendar(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日历"""
        try:
            self._rate_limit()
            
            df = self._ak.tool_trade_date_hist_sina()
            
            if df is None or df.empty:
                return super().get_trading_calendar(start_date, end_date)
            
            # 转换日期格式
            df['trade_date'] = df['trade_date'].astype(str)
            start_str = start_date.replace('-', '')
            end_str = end_date.replace('-', '')
            
            df = df[(df['trade_date'] >= start_str) & (df['trade_date'] <= end_str)]
            
            result = []
            for d in df['trade_date'].tolist():
                d_str = str(d).replace('-', '')
                if len(d_str) == 8:
                    result.append(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}")
            
            return sorted(result)
            
        except Exception as e:
            logger.warning(f"获取交易日历失败，使用默认: {e}")
            return super().get_trading_calendar(start_date, end_date)
    
    def batch_get_stock_data(self, codes: List[str], date: str) -> List[StockData]:
        """批量获取股票数据"""
        results = []
        
        for i, code in enumerate(codes):
            if i > 0 and i % 50 == 0:
                logger.info(f"数据获取进度: {i}/{len(codes)}")
            
            stock = self.get_stock_data(code, date)
            if stock and stock.is_valid():
                results.append(stock)
        
        logger.info(f"批量获取完成: {len(results)}/{len(codes)}")
        return results
    
    def clear_cache(self):
        """清空内部缓存"""
        self._cache.clear()
