# -*- coding: utf-8 -*-
"""
回测模式

提供不同的回测执行模式：
- MonthlyMode: 月度轮换回测
- SingleDayMode: 单日回测
"""

from backtest.modes.monthly import MonthlyMode

__all__ = [
    'MonthlyMode',
]
