# -*- coding: utf-8 -*-
"""
策略层

提供可插拔的选股策略：
- StrategyRegistry: 策略注册表，管理所有可用策略
- MomentumV2Strategy: 动量优先策略（默认）
- RandomStrategy: 随机选股策略（基线对照）

所有默认参数从配置文件读取，代码中不包含硬编码默认值。
"""

from strategy.registry import StrategyRegistry, register_strategy
from strategy.scoring.momentum_v2 import MomentumV2Strategy
from strategy.baseline.random_select import RandomStrategy
from strategy.ml_strategy import MLStrategy

# 从配置读取默认策略（必须）
from config.strategy_config import DEFAULT_STRATEGY

__all__ = [
    'StrategyRegistry',
    'register_strategy',
    'MomentumV2Strategy',
    'RandomStrategy',
    'MLStrategy',
    'DEFAULT_STRATEGY',
]
