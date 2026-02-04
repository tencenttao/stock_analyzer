# -*- coding: utf-8 -*-
"""
特征定义模块

定义所有可用的特征及其属性。
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class FeatureDefinition:
    """
    特征定义
    
    Attributes:
        name: 特征名称
        source: 来源字段（StockData 属性名，'_computed' 表示需计算，'_derived' 表示衍生）
        dtype: 类型 ('numeric' 或 'categorical')
        default: 默认值
        transform: 可选的转换函数
        description: 描述
    """
    name: str
    source: str
    dtype: str = 'numeric'
    default: Any = 0
    transform: Optional[Callable] = None
    description: str = ""


# ============================================
# 基础特征（直接从 StockData 获取）
# ============================================

BASIC_FEATURES = {
    'price': FeatureDefinition(
        name='price', source='price', dtype='numeric', default=0,
        description='股价'
    ),
    'change_pct': FeatureDefinition(
        name='change_pct', source='change_pct', dtype='numeric', default=0,
        description='买入日涨跌幅'
    ),
    'pe_ratio': FeatureDefinition(
        name='pe_ratio', source='pe_ratio', dtype='numeric', default=0,
        transform=lambda x: max(0, x) if x else 0,
        description='市盈率'
    ),
    'pb_ratio': FeatureDefinition(
        name='pb_ratio', source='pb_ratio', dtype='numeric', default=0,
        transform=lambda x: max(0, x) if x else 0,
        description='市净率'
    ),
    'turnover_rate': FeatureDefinition(
        name='turnover_rate', source='turnover_rate', dtype='numeric', default=0,
        description='换手率'
    ),
    'roe': FeatureDefinition(
        name='roe', source='roe', dtype='numeric', default=0,
        description='ROE'
    ),
    'profit_growth': FeatureDefinition(
        name='profit_growth', source='profit_growth', dtype='numeric', default=0,
        description='净利润增长率'
    ),
    'revenue_growth': FeatureDefinition(
        name='revenue_growth', source='revenue_growth', dtype='numeric', default=0,
        description='营收增长率'
    ),
    'dividend_yield': FeatureDefinition(
        name='dividend_yield', source='dividend_yield', dtype='numeric', default=0,
        description='股息率'
    ),
    'peg': FeatureDefinition(
        name='peg', source='peg', dtype='numeric', default=0,
        description='PEG指标'
    ),
    'turnover': FeatureDefinition(
        name='turnover', source='turnover', dtype='numeric', default=0,
        description='成交额（元）'
    ),
    'volume': FeatureDefinition(
        name='volume', source='volume', dtype='numeric', default=0,
        description='成交量（股）'
    ),
    'volume_ratio': FeatureDefinition(
        name='volume_ratio', source='volume_ratio', dtype='numeric', default=0,
        description='量比'
    ),
    'total_mv': FeatureDefinition(
        name='total_mv', source='total_mv', dtype='numeric', default=0,
        description='总市值（万元）'
    ),
    'circ_mv': FeatureDefinition(
        name='circ_mv', source='circ_mv', dtype='numeric', default=0,
        description='流通市值（万元）'
    ),
    'list_days': FeatureDefinition(
        name='list_days', source='list_days', dtype='numeric', default=0,
        description='上市天数'
    ),
}


# ============================================
# 技术指标特征
# ============================================

TECHNICAL_FEATURES = {
    # 可从 StockData 直接获取的
    'momentum_20d': FeatureDefinition(
        name='momentum_20d', source='momentum_20d', dtype='numeric', default=0,
        description='20日动量'
    ),
    'momentum_60d': FeatureDefinition(
        name='momentum_60d', source='momentum_60d', dtype='numeric', default=0,
        description='60日动量'
    ),
    # 需要从日线数据计算的
    'volatility_20d': FeatureDefinition(
        name='volatility_20d', source='_computed', dtype='numeric', default=0,
        description='20日波动率（需要历史数据计算）'
    ),
    'rsi_14': FeatureDefinition(
        name='rsi_14', source='_computed', dtype='numeric', default=50,
        description='14日RSI（需要历史数据计算）'
    ),
    'ma_deviation_20': FeatureDefinition(
        name='ma_deviation_20', source='_computed', dtype='numeric', default=0,
        description='价格偏离20日均线百分比'
    ),
}


# ============================================
# 分类特征
# ============================================

CATEGORICAL_FEATURES = {
    'industry': FeatureDefinition(
        name='industry', source='industry', dtype='categorical', default='unknown',
        description='行业'
    ),
    'market': FeatureDefinition(
        name='market', source='market', dtype='categorical', default='unknown',
        description='市场板块（主板/创业板/科创板）'
    ),
}


# ============================================
# 市场环境特征（需传入 market_data，用于预测相对收益）
# ============================================

MARKET_FEATURES = {
    'market_momentum_20d': FeatureDefinition(
        name='market_momentum_20d', source='_market', dtype='numeric', default=0,
        description='沪深300近20日涨跌幅(%)'
    ),
    'market_momentum_60d': FeatureDefinition(
        name='market_momentum_60d', source='_market', dtype='numeric', default=0,
        description='沪深300近60日涨跌幅(%)'
    ),
    'market_volatility_20d': FeatureDefinition(
        name='market_volatility_20d', source='_market', dtype='numeric', default=0,
        description='沪深300近20日波动率(%)'
    ),
    'market_trend': FeatureDefinition(
        name='market_trend', source='_market', dtype='numeric', default=0,
        description='市场趋势: 1=上涨, 0=震荡, -1=下跌'
    ),
}

# ============================================
# 相对收益相关特征（个股 vs 大盘，需 market_data）
# ============================================

RELATIVE_FEATURES = {
    'relative_momentum_20d': FeatureDefinition(
        name='relative_momentum_20d', source='_market', dtype='numeric', default=0,
        description='相对动量20日 = 个股20日动量 - 大盘20日动量(%)'
    ),
    'relative_momentum_60d': FeatureDefinition(
        name='relative_momentum_60d', source='_market', dtype='numeric', default=0,
        description='相对动量60日 = 个股60日动量 - 大盘60日动量(%)'
    ),
    'volatility_ratio_20d': FeatureDefinition(
        name='volatility_ratio_20d', source='_market', dtype='numeric', default=1.0,
        description='波动率比 = 个股20日波动率 / 大盘20日波动率'
    ),
    'stock_market_correlation_20d': FeatureDefinition(
        name='stock_market_correlation_20d', source='_market', dtype='numeric', default=0,
        description='近20日个股与大盘收益相关系数'
    ),
    'stock_beta_20d': FeatureDefinition(
        name='stock_beta_20d', source='_market', dtype='numeric', default=1.0,
        description='近20日个股相对大盘的Beta（弹性）'
    ),
}

# ============================================
# 衍生特征（由其他特征计算得出）
# ============================================

DERIVED_FEATURES = {
    'pe_percentile': FeatureDefinition(
        name='pe_percentile', source='_derived', dtype='numeric', default=0.5,
        description='PE在所有股票中的百分位'
    ),
    'momentum_rank': FeatureDefinition(
        name='momentum_rank', source='_derived', dtype='numeric', default=0.5,
        description='动量在所有股票中的排名百分比'
    ),
    'value_score': FeatureDefinition(
        name='value_score', source='_derived', dtype='numeric', default=0,
        description='价值得分 = 1/PE + 股息率'
    ),
    'quality_score': FeatureDefinition(
        name='quality_score', source='_derived', dtype='numeric', default=0,
        description='质量得分 = ROE + 利润增长'
    ),
}


# ============================================
# 所有特征汇总
# ============================================

ALL_FEATURES = {
    **BASIC_FEATURES,
    **TECHNICAL_FEATURES,
    **CATEGORICAL_FEATURES,
    **MARKET_FEATURES,
    **RELATIVE_FEATURES,
    **DERIVED_FEATURES,
}
