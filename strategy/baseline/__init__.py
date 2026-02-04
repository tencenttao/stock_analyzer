# -*- coding: utf-8 -*-
"""
基线策略（对照组）

用于策略有效性验证的基线策略：
- RandomStrategy: 随机选股策略
- EqualWeightStrategy: 等权选股策略（预留）
"""

from strategy.baseline.random_select import RandomStrategy

__all__ = [
    'RandomStrategy',
]
