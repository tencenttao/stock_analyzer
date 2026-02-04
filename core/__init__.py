# -*- coding: utf-8 -*-
"""
核心抽象层

提供系统的核心接口和类型定义：
- types: 数据类型定义（StockData, BacktestResult等）
- interfaces: 抽象接口（DataSource, Strategy）
"""

from core.types import (
    StockData,
    IndexData,
    TradeRecord,
    MonthlyReturn,
    BacktestResult,
    BacktestConfig,
)

from core.interfaces import (
    DataSource,
    Strategy,
)

__all__ = [
    # 类型
    'StockData',
    'IndexData',
    'TradeRecord',
    'MonthlyReturn',
    'BacktestResult',
    'BacktestConfig',
    # 接口
    'DataSource',
    'Strategy',
]
