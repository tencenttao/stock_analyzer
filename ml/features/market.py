# -*- coding: utf-8 -*-
"""
市场环境特征计算

计算市场整体动量、波动率等指标，以及个股与大盘的相关性和 Beta。
用于 build_training_data.py、回测/选股时的特征构造。

========== 逻辑梳理 ==========

1) 大盘环境特征（compute_market_features）
   - 数据：沪深300(000300) 日线，date 前约 180 天 ~ date（含 date）
   - 排序：按 trade_date 升序，closes[-1] = 当日收盘
   - market_momentum_20d: (closes[-1] - closes[-20]) / closes[-20] * 100（%）
     注意：使用 closes[-period] 而非 closes[-period-1]，与 tushare_source 个股动量口径一致
     即 momentum_20d 实际是 "当前 vs 19个交易日前"（间隔 19 天）
   - market_momentum_60d: 同上，使用 closes[-60]
   - market_volatility_20d: 最近 21 根 K 的日收益率标准差，年化 sqrt(252) 后 * 100（%）
   - market_trend: 1=上涨(current>MA20>MA60), -1=下跌(current<MA20<MA60), 0=震荡

2) 个股与大盘关系（compute_stock_market_relation）
   - 输入：个股日线、指数日线（均按 trade_date 升序），period=20
   - 取各自最近 21 根收盘，算日收益率，对齐长度后：
   - stock_market_correlation_20d: 日收益率相关系数，[-1,1]
   - stock_beta_20d: cov(股,指)/var(指)，[-5,5]

3) 衍生/兼容
   - calc_annualized_volatility_from_daily: 从日线算年化波动率，供 engineer 的 volatility_ratio_20d（个股波动率/大盘波动率）
   - calc_volume_ratio / calc_recent_max_drawdown: 兼容旧版，当前模型未用

========== 模型用到的市场特征 ==========

- 全量特征（train / quarterly_selector 默认 full）：以下 9 个市场相关特征都会进模型
  - 大盘环境: market_momentum_20d, market_momentum_60d, market_volatility_20d, market_trend
  - 相对/关系: relative_momentum_20d, relative_momentum_60d, volatility_ratio_20d,
               stock_market_correlation_20d, stock_beta_20d
- 特征子集 market_aware（strategy_optimizer）：主要用 market_momentum_20d, market_trend, stock_beta_20d
- 特征子集 low_volatility：主要用 stock_beta_20d, volatility_ratio_20d
"""

import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


def compute_market_features(data_source, date: str) -> Optional[Dict[str, Any]]:
    """
    计算市场环境特征
    
    Args:
        data_source: DataManager 实例
        date: 日期 (YYYY-MM-DD)
        
    Returns:
        市场特征字典，包含:
        - market_momentum_20d: 20日大盘动量 (%)
        - market_momentum_60d: 60日大盘动量 (%)
        - market_volatility_20d: 20日大盘波动率 (%)
        - market_trend: 趋势标识 (1=上涨, 0=震荡, -1=下跌)
    """
    from datetime import datetime, timedelta
    
    try:
        # 获取沪深300指数日线数据（取120天确保够用）
        end_date = date
        start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=180)).strftime('%Y-%m-%d')
        
        index_daily = data_source.get_index_daily('000300', start_date, end_date)
        
        if not index_daily or len(index_daily) < 60:
            logger.warning(f"指数数据不足: {len(index_daily) if index_daily else 0} 条")
            return None
        
        # 按日期排序（升序）
        index_daily = sorted(index_daily, key=lambda x: x.get('trade_date', x.get('date', '')))
        
        # 提取收盘价序列
        closes = [d.get('close', 0) for d in index_daily]
        
        if len(closes) < 60:
            return None
        
        # 计算动量
        market_momentum_20d = _calc_momentum(closes, 20)
        market_momentum_60d = _calc_momentum(closes, 60)
        
        # 计算波动率
        market_volatility_20d = _calc_volatility(closes, 20)
        
        # 计算趋势
        market_trend = _calc_trend(closes, 20)
        
        return {
            'market_momentum_20d': round(market_momentum_20d, 2),
            'market_momentum_60d': round(market_momentum_60d, 2),
            'market_volatility_20d': round(market_volatility_20d, 2),
            'market_trend': market_trend,
        }
        
    except Exception as e:
        logger.error(f"计算市场特征失败: {e}")
        return None


def compute_stock_market_relation(
    stock_daily: List[Dict],
    index_daily: List[Dict],
    period: int = 20
) -> Tuple[float, float]:
    """
    计算个股与大盘的相关系数和Beta
    
    Args:
        stock_daily: 个股日线数据列表
        index_daily: 指数日线数据列表
        period: 计算周期（天）
        
    Returns:
        (correlation, beta): 相关系数和Beta值
    """
    import numpy as np
    
    try:
        if not stock_daily or not index_daily:
            return 0.0, 1.0
        
        # 按日期排序并取最近 period 天
        stock_daily = sorted(stock_daily, key=lambda x: x.get('trade_date', x.get('date', '')))
        index_daily = sorted(index_daily, key=lambda x: x.get('trade_date', x.get('date', '')))
        
        # 提取收盘价
        stock_closes = [d.get('close', 0) for d in stock_daily[-period-1:]]
        index_closes = [d.get('close', 0) for d in index_daily[-period-1:]]
        
        if len(stock_closes) < period + 1 or len(index_closes) < period + 1:
            return 0.0, 1.0
        
        # 计算日收益率
        stock_returns = np.diff(stock_closes) / np.array(stock_closes[:-1])
        index_returns = np.diff(index_closes) / np.array(index_closes[:-1])
        
        # 对齐长度
        min_len = min(len(stock_returns), len(index_returns))
        stock_returns = stock_returns[-min_len:]
        index_returns = index_returns[-min_len:]
        
        if min_len < 5:
            return 0.0, 1.0
        
        # 清理异常值
        stock_returns = np.nan_to_num(stock_returns, nan=0.0, posinf=0.0, neginf=0.0)
        index_returns = np.nan_to_num(index_returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 计算相关系数
        corr_matrix = np.corrcoef(stock_returns, index_returns)
        correlation = corr_matrix[0, 1] if corr_matrix.shape == (2, 2) else 0.0
        
        # 计算 Beta (协方差 / 市场方差)
        covariance = np.cov(stock_returns, index_returns)[0, 1]
        market_variance = np.var(index_returns)
        beta = covariance / market_variance if market_variance > 1e-10 else 1.0
        
        # 限制范围
        correlation = max(-1.0, min(1.0, correlation))
        beta = max(-5.0, min(5.0, beta))
        
        return round(correlation, 4), round(beta, 4)
        
    except Exception as e:
        logger.error(f"计算个股-大盘关系失败: {e}")
        return 0.0, 1.0


def _calc_momentum(closes: List[float], period: int) -> float:
    """
    计算动量 (%)，与 tushare_source / quarterly_data 口径一致。
    
    口径说明：
    - 个股 momentum_20d: prices[0] / prices[19] (降序数组，间隔 19 个交易日)
    - 本函数: closes[-1] / closes[-period] (升序数组，间隔 period-1 个交易日)
    - 两者等价：都是"当前 vs period-1 天前"
    
    closes 需已按时间升序排列，closes[-1] 为最近日。
    """
    if len(closes) < period:
        return 0.0
    
    current = closes[-1]
    past = closes[-period]  # 与个股动量口径一致
    
    if past <= 0:
        return 0.0
    
    return (current - past) / past * 100


def _calc_volatility(closes: List[float], period: int) -> float:
    """
    计算波动率 (年化 %)，与 quarterly_data 口径一致。
    使用最近 period 个交易日的日收益率标准差，年化系数 sqrt(252)。
    
    注意：波动率需要 period+1 个数据点来计算 period 个日收益率。
    """
    import numpy as np
    
    if len(closes) < period + 1:
        return 0.0
    
    # 计算日收益率：需要 period+1 个点来算 period 个收益率
    returns = np.diff(closes[-period-1:]) / np.array(closes[-period-1:-1])
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 年化波动率
    daily_vol = np.std(returns)
    annual_vol = daily_vol * np.sqrt(252)
    
    return annual_vol * 100


def calc_annualized_volatility_from_daily(daily_data: List[Dict], period: int = 20) -> float:
    """
    从日线数据计算年化波动率 (%)，与 _calc_volatility 口径一致。
    用于 engineer.f_volatility_ratio_20d：个股波动率 / 大盘波动率。
    
    Args:
        daily_data: 日线列表，每条含 close 或 trade_date
        period: 计算周期（天）
        
    Returns:
        年化波动率 (%)
    """
    import numpy as np
    
    if not daily_data or len(daily_data) < period + 1:
        return 0.0
    
    daily_data = sorted(daily_data, key=lambda x: x.get('trade_date', x.get('date', '')))
    closes = [d.get('close', 0) for d in daily_data[-period - 1:]]
    
    if len(closes) < period + 1:
        return 0.0
    
    returns = np.diff(closes) / np.array(closes[:-1])
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    daily_vol = np.std(returns)
    annual_vol = daily_vol * np.sqrt(252)
    return round(annual_vol * 100, 2)


def _calc_trend(closes: List[float], period: int) -> int:
    """
    计算趋势
    
    Returns:
        1: 上涨趋势 (20日均线上穿60日均线 或 价格在20日均线上方)
        0: 震荡
        -1: 下跌趋势
    """
    if len(closes) < 60:
        return 0
    
    ma20 = sum(closes[-20:]) / 20
    ma60 = sum(closes[-60:]) / 60
    current = closes[-1]
    
    # 判断趋势
    if current > ma20 and ma20 > ma60:
        return 1  # 上涨趋势
    elif current < ma20 and ma20 < ma60:
        return -1  # 下跌趋势
    else:
        return 0  # 震荡


# 兼容旧版调用（可能有些地方直接调用这些函数）
def calc_volume_ratio(data_source, date: str, short_period: int = 5, long_period: int = 20) -> float:
    """计算成交量比率（短期/长期）"""
    from datetime import datetime, timedelta
    
    try:
        end_date = date
        start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=60)).strftime('%Y-%m-%d')
        
        index_daily = data_source.get_index_daily('000300', start_date, end_date)
        
        if not index_daily or len(index_daily) < long_period:
            return 1.0
        
        index_daily = sorted(index_daily, key=lambda x: x.get('trade_date', x.get('date', '')))
        volumes = [d.get('vol', d.get('volume', 0)) for d in index_daily]
        
        if len(volumes) < long_period:
            return 1.0
        
        short_avg = sum(volumes[-short_period:]) / short_period
        long_avg = sum(volumes[-long_period:]) / long_period
        
        return short_avg / long_avg if long_avg > 0 else 1.0
        
    except Exception:
        return 1.0


def calc_recent_max_drawdown(data_source, date: str, period: int = 20) -> float:
    """计算最近N天的最大回撤 (%)"""
    from datetime import datetime, timedelta
    
    try:
        end_date = date
        start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=60)).strftime('%Y-%m-%d')
        
        index_daily = data_source.get_index_daily('000300', start_date, end_date)
        
        if not index_daily or len(index_daily) < period:
            return 0.0
        
        index_daily = sorted(index_daily, key=lambda x: x.get('trade_date', x.get('date', '')))
        closes = [d.get('close', 0) for d in index_daily[-period:]]
        
        if not closes:
            return 0.0
        
        peak = closes[0]
        max_dd = 0.0
        
        for close in closes:
            peak = max(peak, close)
            dd = (peak - close) / peak * 100
            max_dd = max(max_dd, dd)
        
        return round(max_dd, 2)
        
    except Exception:
        return 0.0
