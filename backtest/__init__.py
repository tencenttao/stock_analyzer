# -*- coding: utf-8 -*-
"""
回测层

提供回测引擎和相关工具：
- BacktestEngine: 回测引擎主类
- BacktestConfig: 回测配置
- BacktestResult: 回测结果
- 各种回测模式（月度轮换、单日、多日）
"""

from backtest.engine import BacktestEngine, BacktestConfig, BacktestResult
from backtest.metrics import RiskMetrics
from backtest.cost import TradingCost

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'RiskMetrics',
    'TradingCost',
]
