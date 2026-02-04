# -*- coding: utf-8 -*-
"""
Tushare数据源

基于Tushare Pro API实现的数据源，特点：
- 数据稳定可靠
- 支持历史数据，无前视偏差
- 需要付费API Token

注意：此模块从原 tushare_fetcher.py 重构而来
"""

import os
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from core.interfaces import DataSource
from core.types import StockData, IndexData

# 从配置读取 Token（必须）
from config.data_source_config import TUSHARE_TOKEN

logger = logging.getLogger(__name__)


class TushareSource(DataSource):
    """
    Tushare数据源实现
    
    使用Tushare Pro API获取金融数据。
    
    使用示例:
        source = TushareSource(token='your_token')
        stock = source.get_stock_data('600036', '2024-01-01')
    """
    
    def __init__(self, token: str = None):
        """
        初始化Tushare数据源
        
        Args:
            token: Tushare API Token，不传则使用配置文件中的值
        """
        # 优先使用传入的token，其次环境变量，最后配置文件
        self._token = token or os.environ.get('TUSHARE_TOKEN') or TUSHARE_TOKEN
        self._init_api()
        
        # 请求频率控制
        self._request_count = 0
        self._last_request_time = 0
        
        # 内部缓存（避免重复请求）
        self._cache: Dict = {}
    
    def _init_api(self):
        """初始化Tushare API"""
        try:
            import tushare as ts
            self._ts = ts
            ts.set_token(self._token)
            self._pro = ts.pro_api()
            logger.info("✅ Tushare数据源初始化成功")
        except ImportError:
            raise ImportError("Tushare未安装，请运行: pip install tushare")
        except Exception as e:
            raise RuntimeError(f"Tushare初始化失败: {e}")
    
    @property
    def name(self) -> str:
        return "tushare"
    
    def _rate_limit(self):
        """API频率限制"""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < 0.2:  # 至少间隔200ms
            time.sleep(0.2 - elapsed)
        self._last_request_time = time.time()
        self._request_count += 1
    
    def _to_ts_code(self, code: str) -> str:
        """转换为Tushare格式代码"""
        code = code.strip()
        if '.' in code:
            return code
        if code.startswith(('6', '9')):
            return f"{code}.SH"
        elif code.startswith(('0', '2', '3')):
            return f"{code}.SZ"
        elif code.startswith('8') or code.startswith('4'):
            return f"{code}.BJ"
        return f"{code}.SH"
    
    def _from_ts_code(self, ts_code: str) -> str:
        """从Tushare格式转换为普通代码"""
        return ts_code.split('.')[0] if '.' in ts_code else ts_code
    
    def _get_stock_name(self, code: str) -> str:
        """获取股票名称"""
        info = self._get_stock_basic_info(code)
        return info.get('name', '') if info else ''
    
    def _get_stock_basic_info(self, code: str) -> Optional[Dict]:
        """获取股票基本信息（名称、行业、上市日期等）"""
        cache_key = f"basic_info_{code}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            self._rate_limit()
            ts_code = self._to_ts_code(code)
            
            df = self._pro.stock_basic(
                ts_code=ts_code,
                fields='ts_code,name,industry,market,list_date'
            )
            
            if df is not None and not df.empty:
                row = df.iloc[0]
                info = {
                    'name': row.get('name', ''),
                    'industry': row.get('industry', ''),
                    'market': row.get('market', ''),  # 主板/创业板/科创板
                    'list_date': row.get('list_date', ''),
                }
                self._cache[cache_key] = info
                return info
        except Exception as e:
            logger.debug(f"获取{code}基本信息失败: {e}")
        
        return None
    
    def get_all_stock_basic_info(self) -> Dict[str, Dict]:
        """
        获取所有股票的基本信息（用于批量获取行业等）
        
        Returns:
            {code: {name, industry, market, list_date}, ...}
        """
        cache_key = "all_stock_basic"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            self._rate_limit()
            df = self._pro.stock_basic(
                fields='ts_code,name,industry,market,list_date'
            )
            
            if df is not None and not df.empty:
                result = {}
                for _, row in df.iterrows():
                    code = row['ts_code'][:6]
                    result[code] = {
                        'name': row.get('name', ''),
                        'industry': row.get('industry') or 'unknown',
                        'market': row.get('market', ''),
                        'list_date': row.get('list_date', ''),
                    }
                self._cache[cache_key] = result
                logger.info(f"获取股票基本信息: {len(result)} 只")
                return result
        except Exception as e:
            logger.warning(f"获取所有股票基本信息失败: {e}")
        
        return {}
    
    def get_stock_data(self, code: str, date: str) -> Optional[StockData]:
        """
        获取单只股票在指定日期的完整数据
        
        Args:
            code: 股票代码（6位）
            date: 日期（YYYY-MM-DD）
            
        Returns:
            StockData对象
        """
        logger.debug(f"[get_stock_data] 请求: code={code}, date={date}")
        
        cache_key = f"stock_{code}_{date}"
        if cache_key in self._cache:
            logger.debug(f"[get_stock_data] 缓存命中: {code}@{date}")
            return self._cache[cache_key]
        
        try:
            # 1. 获取日线数据（价格、涨跌幅、动量）- 使用前复权
            daily_data = self._get_daily_data(code, date, adj='qfq')
            if not daily_data:
                return None
            
            # 2. 获取股票基本信息（名称、行业、板块）
            basic_info = self._get_stock_basic_info(code)
            name = basic_info.get('name', '') if basic_info else ''
            
            # 3. 获取每日指标（PE、PB、换手率、市值等）
            basic_data = self._get_daily_basic(code, date)
            
            # 4. 获取财务指标（ROE、利润增长）
            fin_data = self._get_financial_indicator(code, date)
            
            # 5. 计算上市天数
            list_days = None
            if basic_info and basic_info.get('list_date'):
                try:
                    list_date = datetime.strptime(basic_info['list_date'], '%Y%m%d')
                    target_date = datetime.strptime(date, '%Y-%m-%d')
                    list_days = (target_date - list_date).days
                except:
                    pass
            
            # 6. 组装数据
            stock = StockData(
                code=code,
                name=name,
                price=daily_data.get('close', 0),
                change_pct=daily_data.get('pct_chg', 0),
                momentum_20d=daily_data.get('momentum_20d', 0),
                momentum_60d=daily_data.get('momentum_60d', 0),
                volume=daily_data.get('vol', 0),
                turnover=daily_data.get('amount', 0) * 1000,  # 转换为元
                industry=basic_info.get('industry') if basic_info else None,
                market=basic_info.get('market') if basic_info else None,
                list_days=list_days,
                date=date,
                data_source='tushare'
            )
            
            # 合并基本面数据
            if basic_data:
                stock.pe_ratio = basic_data.get('pe_ttm')
                stock.pb_ratio = basic_data.get('pb')
                stock.turnover_rate = basic_data.get('turnover_rate')
                stock.dividend_yield = basic_data.get('dv_ttm')
                stock.total_mv = basic_data.get('total_mv')      # 总市值
                stock.circ_mv = basic_data.get('circ_mv')        # 流通市值
                stock.volume_ratio = basic_data.get('volume_ratio')  # 量比
            
            # 合并财务指标
            if fin_data:
                stock.roe = fin_data.get('roe')
                stock.profit_growth = fin_data.get('profit_growth')
                stock.report_date = fin_data.get('report_date')
            
            # 计算PEG
            if stock.pe_ratio and stock.profit_growth and stock.profit_growth > 0:
                stock.peg = round(stock.pe_ratio / stock.profit_growth, 2)
            
            # 如果没有ROE，使用PB/PE估算
            if not stock.roe and stock.pb_ratio and stock.pe_ratio and stock.pe_ratio > 0:
                stock.roe = round(stock.pb_ratio / stock.pe_ratio * 100, 2)
            
            self._cache[cache_key] = stock
            return stock
            
        except Exception as e:
            logger.debug(f"获取{code}数据失败: {e}")
            return None
    
    def _get_daily_data(self, code: str, date: str, adj: str = 'qfq') -> Optional[Dict]:
        """获取日线数据和动量（支持前复权）
        
        Args:
            code: 股票代码
            date: 目标日期 YYYY-MM-DD
            adj: 复权类型 'qfq'前复权(默认), 'hfq'后复权, None不复权
            
        Returns:
            包含价格和动量的字典（前复权价格）
        """
        try:
            self._rate_limit()
            ts_code = self._to_ts_code(code)
            
            # 获取约120天数据（需满足60日动量：60交易日≈90+日历日，加节假日预留取120天）
            start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=120)).strftime('%Y%m%d')
            end_date = date.replace('-', '')
            
            logger.debug(f"[_get_daily_data] {code}: 目标日期={date}, 查询范围={start_date}-{end_date}")
            
            # 获取不复权日线数据
            df = self._pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or df.empty:
                logger.debug(f"[_get_daily_data] {code}: 无数据返回")
                return None
            
            # 如果需要复权，获取复权因子并计算
            if adj in ['qfq', 'hfq']:
                try:
                    self._rate_limit()
                    # 获取复权因子（扩展到最新日期以获取最新因子）
                    adj_df = self._pro.adj_factor(
                        ts_code=ts_code,
                        start_date=start_date,
                        end_date=datetime.now().strftime('%Y%m%d')  # 获取到最新
                    )
                    
                    if adj_df is not None and not adj_df.empty:
                        # 合并复权因子
                        df = df.merge(adj_df[['trade_date', 'adj_factor']], on='trade_date', how='left')
                        
                        if adj == 'qfq':
                            # 前复权：以最新复权因子为基准
                            latest_factor = adj_df['adj_factor'].max()
                            if latest_factor and latest_factor > 0:
                                df['close'] = df['close'] * df['adj_factor'] / latest_factor
                                df['open'] = df['open'] * df['adj_factor'] / latest_factor
                                df['high'] = df['high'] * df['adj_factor'] / latest_factor
                                df['low'] = df['low'] * df['adj_factor'] / latest_factor
                        elif adj == 'hfq':
                            # 后复权：以最早复权因子为基准
                            earliest_factor = adj_df['adj_factor'].min()
                            if earliest_factor and earliest_factor > 0:
                                df['close'] = df['close'] * df['adj_factor'] / earliest_factor
                                df['open'] = df['open'] * df['adj_factor'] / earliest_factor
                                df['high'] = df['high'] * df['adj_factor'] / earliest_factor
                                df['low'] = df['low'] * df['adj_factor'] / earliest_factor
                        
                        # 删除复权因子列
                        if 'adj_factor' in df.columns:
                            df = df.drop(columns=['adj_factor'])
                            
                        logger.debug(f"{code} 使用{adj}复权价格")
                except Exception as e:
                    logger.debug(f"获取{code}复权因子失败，使用不复权价格: {e}")
            
            # 找到目标日期或之前最近的数据
            target_date = date.replace('-', '')
            df_sorted = df.sort_values('trade_date', ascending=False)
            
            latest = None
            for _, row in df_sorted.iterrows():
                if row['trade_date'] <= target_date:
                    latest = row
                    break
            
            if latest is None:
                logger.debug(f"[_get_daily_data] {code}: 目标日期{date}之前无数据")
                return None
            
            actual_date = latest['trade_date']
            if actual_date != target_date:
                logger.debug(f"[_get_daily_data] {code}: 目标日期{date}非交易日, 实际使用{actual_date}")
            
            # 计算动量（使用复权价格，从目标日期开始往前算）
            # 找到目标日期在数据中的位置
            target_idx = df_sorted[df_sorted['trade_date'] == actual_date].index
            if len(target_idx) > 0:
                # 从目标日期位置开始取数据
                idx_pos = df_sorted.index.get_loc(target_idx[0])
                prices_from_target = df_sorted.iloc[idx_pos:]['close'].values
            else:
                prices_from_target = df_sorted['close'].values
            
            momentum_20d = 0
            momentum_60d = 0
            if len(prices_from_target) >= 20 and prices_from_target[19] > 0:
                momentum_20d = (prices_from_target[0] / prices_from_target[19] - 1) * 100
            if len(prices_from_target) >= 60 and prices_from_target[59] > 0:
                momentum_60d = (prices_from_target[0] / prices_from_target[59] - 1) * 100
            
            logger.debug(f"[_get_daily_data] {code}: 返回日期={actual_date}, 收盘价={latest['close']:.2f}, 共{len(df_sorted)}条数据")
            
            return {
                'close': float(latest['close']),
                'pct_chg': float(latest.get('pct_chg', 0)),
                'vol': float(latest.get('vol', 0)),
                'amount': float(latest.get('amount', 0)),
                'momentum_20d': momentum_20d,
                'momentum_60d': momentum_60d,
            }
            
        except Exception as e:
            logger.debug(f"获取{code}日线数据失败: {e}")
            return None
    
    def _get_daily_basic(self, code: str, date: str) -> Optional[Dict]:
        """获取每日指标（含市值、换手率、估值等）"""
        try:
            self._rate_limit()
            ts_code = self._to_ts_code(code)
            trade_date = date.replace('-', '')
            
            # 增加 total_mv(总市值), circ_mv(流通市值), volume_ratio(量比)
            fields = 'ts_code,trade_date,close,turnover_rate,pe,pe_ttm,pb,dv_ratio,dv_ttm,total_mv,circ_mv,volume_ratio'
            
            df = self._pro.daily_basic(
                ts_code=ts_code,
                trade_date=trade_date,
                fields=fields
            )
            
            if df is None or df.empty:
                # 尝试获取前几天的数据
                for offset in range(1, 8):
                    try_date = datetime.strptime(date, '%Y-%m-%d') - timedelta(days=offset)
                    df = self._pro.daily_basic(
                        ts_code=ts_code,
                        trade_date=try_date.strftime('%Y%m%d'),
                        fields=fields
                    )
                    if df is not None and not df.empty:
                        break
            
            if df is None or df.empty:
                return None
            
            row = df.iloc[0]
            return {
                'pe_ttm': row.get('pe_ttm'),
                'pb': row.get('pb'),
                'turnover_rate': row.get('turnover_rate'),
                'dv_ttm': row.get('dv_ttm'),
                'total_mv': row.get('total_mv'),      # 总市值（万元）
                'circ_mv': row.get('circ_mv'),        # 流通市值（万元）
                'volume_ratio': row.get('volume_ratio'),  # 量比
            }
            
        except Exception as e:
            logger.debug(f"获取{code}每日指标失败: {e}")
            return None
    
    def _get_financial_indicator(self, code: str, target_date: str) -> Optional[Dict]:
        """获取财务指标（消除前视偏差）"""
        try:
            self._rate_limit()
            ts_code = self._to_ts_code(code)
            
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            start_date = (target_dt - timedelta(days=550)).strftime('%Y%m%d')
            end_date = target_dt.strftime('%Y%m%d')
            
            df = self._pro.fina_indicator(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,ann_date,end_date,roe,netprofit_yoy'
            )
            
            if df is None or df.empty:
                return None
            
            # 只保留公告日期在目标日期之前的（消除前视偏差）
            target_dt_str = target_date.replace('-', '')
            df = df[df['ann_date'] <= target_dt_str]
            
            if df.empty:
                return None
            
            # 取最新的报告
            df = df.sort_values('end_date', ascending=False)
            row = df.iloc[0]
            
            return {
                'roe': row.get('roe'),
                'profit_growth': row.get('netprofit_yoy'),
                'report_date': row.get('end_date')
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
            df = self._pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name'
            )
            
            if df is None or df.empty:
                return []
            
            result = [self._from_ts_code(code) for code in df['ts_code'].tolist()]
            self._cache[cache_key] = result
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
            
            # 转换指数代码
            if index_code == '000300':
                ts_index_code = '399300.SZ'
            elif index_code == '000905':
                ts_index_code = '000905.SH'
            else:
                ts_index_code = f"{index_code}.SH"
            
            trade_date = date.replace('-', '') if date else datetime.now().strftime('%Y%m%d')
            
            df = self._pro.index_weight(
                index_code=ts_index_code,
                start_date=trade_date,
                end_date=trade_date
            )
            
            if df is None or df.empty:
                # 获取最近的成分股
                df = self._pro.index_weight(
                    index_code=ts_index_code,
                    start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                    end_date=datetime.now().strftime('%Y%m%d')
                )
            
            if df is None or df.empty:
                return []
            
            # 取最新日期的数据
            latest_date = df['trade_date'].max()
            df = df[df['trade_date'] == latest_date]
            
            result = [self._from_ts_code(code) for code in df['con_code'].tolist()]
            self._cache[cache_key] = result
            
            logger.info(f"获取{index_code}成分股 {len(result)} 只 (日期: {latest_date})")
            return result
            
        except Exception as e:
            logger.error(f"获取指数成分股失败: {e}")
            return []
    
    def get_index_data(self, index_code: str, start_date: str, end_date: str) -> Optional[IndexData]:
        """获取指数表现数据"""
        try:
            self._rate_limit()
            
            # 转换指数代码
            if index_code == '000300':
                ts_code = '000300.SH'
            else:
                ts_code = f"{index_code}.SH"
            
            df = self._pro.index_daily(
                ts_code=ts_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', '')
            )
            
            if df is None or df.empty:
                return None
            
            df = df.sort_values('trade_date')
            
            start_price = float(df.iloc[0]['close'])
            end_price = float(df.iloc[-1]['close'])
            return_pct = (end_price / start_price - 1) * 100
            
            return IndexData(
                code=index_code,
                name='沪深300' if index_code == '000300' else index_code,
                start_date=df.iloc[0]['trade_date'],
                end_date=df.iloc[-1]['trade_date'],
                start_price=start_price,
                end_price=end_price,
                return_pct=return_pct,
                trading_days=len(df)
            )
            
        except Exception as e:
            logger.error(f"获取指数数据失败: {e}")
            return None
    
    def get_index_daily(self, index_code: str, start_date: str, end_date: str) -> Optional[List[Dict]]:
        """
        获取指数日线数据（用于计算市场环境特征）
        
        Args:
            index_code: 指数代码，如 '000300'
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            
        Returns:
            日线列表，每条包含 trade_date, open, high, low, close, vol 等
        """
        try:
            self._rate_limit()
            ts_code = '000300.SH' if index_code == '000300' else f"{index_code}.SH"
            start_ = start_date.replace('-', '')
            end_ = end_date.replace('-', '')
            df = self._pro.index_daily(ts_code=ts_code, start_date=start_, end_date=end_)
            if df is None or df.empty:
                return None
            df = df.sort_values('trade_date', ascending=False)
            records = []
            for _, row in df.iterrows():
                records.append({
                    'trade_date': str(row['trade_date']),
                    'open': float(row.get('open', 0)),
                    'high': float(row.get('high', 0)),
                    'low': float(row.get('low', 0)),
                    'close': float(row.get('close', 0)),
                    'vol': float(row.get('vol', 0)),
                })
            return records
        except Exception as e:
            logger.debug(f"获取指数日线失败: {e}")
            return None
    
    def get_trading_calendar(self, start_date: str, end_date: str) -> List[str]:
        """获取真实交易日历"""
        try:
            self._rate_limit()
            
            df = self._pro.trade_cal(
                exchange='SSE',
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                is_open='1'
            )
            
            if df is None or df.empty:
                # 回退到默认实现
                return super().get_trading_calendar(start_date, end_date)
            
            df = df.sort_values('cal_date')
            result = [
                f"{d[:4]}-{d[4:6]}-{d[6:8]}"
                for d in df['cal_date'].tolist()
            ]
            
            return result
            
        except Exception as e:
            logger.warning(f"获取交易日历失败，使用默认: {e}")
            return super().get_trading_calendar(start_date, end_date)
    
    def batch_get_stock_data(self, codes: List[str], date: str) -> List[StockData]:
        """批量获取股票数据（优化版）"""
        results = []
        
        # 先批量获取每日指标
        basic_data_map = self._batch_get_daily_basic(codes, date)
        
        for i, code in enumerate(codes):
            if i > 0 and i % 50 == 0:
                logger.info(f"数据获取进度: {i}/{len(codes)}")
            
            stock = self.get_stock_data(code, date)
            if stock and stock.is_valid():
                # 补充批量获取的数据
                if code in basic_data_map:
                    basic = basic_data_map[code]
                    if not stock.pe_ratio:
                        stock.pe_ratio = basic.get('pe_ttm')
                    if not stock.pb_ratio:
                        stock.pb_ratio = basic.get('pb')
                    if not stock.turnover_rate:
                        stock.turnover_rate = basic.get('turnover_rate')
                    if not stock.dividend_yield:
                        stock.dividend_yield = basic.get('dv_ttm')
                    if not stock.total_mv:
                        stock.total_mv = basic.get('total_mv')
                    if not stock.circ_mv:
                        stock.circ_mv = basic.get('circ_mv')
                    if not stock.volume_ratio:
                        stock.volume_ratio = basic.get('volume_ratio')
                
                results.append(stock)
        
        logger.info(f"批量获取完成: {len(results)}/{len(codes)}")
        return results
    
    def _batch_get_daily_basic(self, codes: List[str], date: str) -> Dict[str, Dict]:
        """批量获取每日指标"""
        try:
            self._rate_limit()
            trade_date = date.replace('-', '')
            
            # 与 _get_daily_basic 保持一致的字段
            fields = 'ts_code,trade_date,close,turnover_rate,pe,pe_ttm,pb,dv_ratio,dv_ttm,total_mv,circ_mv,volume_ratio'
            
            df = self._pro.daily_basic(
                trade_date=trade_date,
                fields=fields
            )
            
            if df is None or df.empty:
                return {}
            
            result = {}
            for _, row in df.iterrows():
                code = self._from_ts_code(row['ts_code'])
                if code in codes:
                    result[code] = {
                        'pe_ttm': row.get('pe_ttm'),
                        'pb': row.get('pb'),
                        'turnover_rate': row.get('turnover_rate'),
                        'dv_ttm': row.get('dv_ttm'),
                        'total_mv': row.get('total_mv'),
                        'circ_mv': row.get('circ_mv'),
                        'volume_ratio': row.get('volume_ratio'),
                    }
            
            logger.debug(f"批量获取每日指标: {len(result)}/{len(codes)}")
            return result
            
        except Exception as e:
            logger.warning(f"批量获取每日指标失败: {e}")
            return {}
    
    def get_daily_data(self, code: str, start_date: str, end_date: str, 
                        adj: str = 'qfq') -> Optional[List[Dict]]:
        """获取日线数据（公开接口，供回测引擎使用）
        
        Args:
            code: 股票代码
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            adj: 复权类型 'qfq'前复权(默认), 'hfq'后复权, None不复权
            
        Returns:
            日线数据列表（前复权价格）
        """
        logger.debug(f"[get_daily_data] 请求: code={code}, 范围={start_date} ~ {end_date}, adj={adj}")
        
        try:
            self._rate_limit()
            ts_code = self._to_ts_code(code)
            
            # 获取不复权日线数据
            df = self._pro.daily(
                ts_code=ts_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', '')
            )
            
            if df is None or df.empty:
                logger.debug(f"[get_daily_data] {code}: 范围内无数据")
                return None
            
            # 如果需要复权，获取复权因子并计算
            if adj in ['qfq', 'hfq']:
                try:
                    self._rate_limit()
                    # 获取复权因子（扩展到最新日期）
                    adj_df = self._pro.adj_factor(
                        ts_code=ts_code,
                        start_date=start_date.replace('-', ''),
                        end_date=datetime.now().strftime('%Y%m%d')
                    )
                    
                    if adj_df is not None and not adj_df.empty:
                        df = df.merge(adj_df[['trade_date', 'adj_factor']], on='trade_date', how='left')
                        
                        if adj == 'qfq':
                            latest_factor = adj_df['adj_factor'].max()
                            if latest_factor and latest_factor > 0:
                                for col in ['close', 'open', 'high', 'low']:
                                    df[col] = df[col] * df['adj_factor'] / latest_factor
                        elif adj == 'hfq':
                            earliest_factor = adj_df['adj_factor'].min()
                            if earliest_factor and earliest_factor > 0:
                                for col in ['close', 'open', 'high', 'low']:
                                    df[col] = df[col] * df['adj_factor'] / earliest_factor
                        
                        if 'adj_factor' in df.columns:
                            df = df.drop(columns=['adj_factor'])
                except Exception as e:
                    logger.debug(f"获取{code}复权因子失败: {e}")
            
            # 按日期降序排列（最新在前）
            df = df.sort_values('trade_date', ascending=False)
            records = df.to_dict('records')
            
            if records:
                first_date = records[-1].get('trade_date', '')  # 最早
                last_date = records[0].get('trade_date', '')    # 最新
                logger.debug(f"[get_daily_data] {code}: 返回{len(records)}条, 实际范围={first_date} ~ {last_date}")
            
            return records
            
        except Exception as e:
            logger.debug(f"[get_daily_data] {code}: 获取失败 - {e}")
            return None
    
    def get_daily_basic(self, code: str, date: str) -> Optional[Dict]:
        """获取每日指标（公开接口）
        
        Args:
            code: 股票代码
            date: 日期 YYYY-MM-DD
            
        Returns:
            每日指标字典
        """
        result = self._get_daily_basic(code, date)
        if result:
            return {
                'code': code,
                'date': date,
                'pe_ratio': result.get('pe_ttm'),
                'pb_ratio': result.get('pb'),
                'turnover_rate': result.get('turnover_rate'),
                'dividend_yield': result.get('dv_ttm'),
            }
        return None
    
    def get_financial_indicator(self, code: str, target_date: str = None) -> Optional[Dict]:
        """获取财务指标（公开接口）
        
        Args:
            code: 股票代码
            target_date: 目标日期(YYYY-MM-DD)，获取该日期之前最新的财务报告
            
        Returns:
            财务指标字典
        """
        result = self._get_financial_indicator(code, target_date or datetime.now().strftime('%Y-%m-%d'))
        if result:
            return {
                'code': code,
                'report_date': result.get('report_date'),
                'roe': result.get('roe'),
                'profit_growth': result.get('profit_growth'),
            }
        return None
    
    def clear_cache(self):
        """清空内部缓存"""
        self._cache.clear()
