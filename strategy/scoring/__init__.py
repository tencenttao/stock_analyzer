# -*- coding: utf-8 -*-
"""
评分类选股策略

基于多因子评分的选股策略：
- MomentumV2Strategy: 动量优先策略（40%动量 + 25%成长 + 20%估值 + 10%质量 + 5%安全）
- AdaptiveStrategy: 自适应策略（根据市场状态动态调整权重）
- ValueFirstStrategy: 价值优先策略（预留）
- BalancedStrategy: 平衡策略（预留）
"""

from strategy.scoring.momentum_v2 import MomentumV2Strategy
from strategy.scoring.adaptive import AdaptiveStrategy, MarketState

__all__ = [
    'MomentumV2Strategy',
    'AdaptiveStrategy',
    'MarketState',
]
