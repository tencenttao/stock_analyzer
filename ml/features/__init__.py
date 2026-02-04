# -*- coding: utf-8 -*-
"""
特征工程模块

提供灵活的特征定义、计算和处理功能。

模块结构:
    - definitions.py: 特征定义（FeatureDefinition, BASIC_FEATURES, ...）
    - indicators.py: 技术指标计算（TechnicalIndicators）
    - engineer.py: 特征配置和工程器（FeatureConfig, FeatureEngineer）

使用示例:
    from ml.features import FeatureEngineer, FeatureConfig, register_feature

    # 1. 按配置提取全部特征（推荐入口）
    engineer = FeatureEngineer(config)
    features = engineer.extract(stock, daily_data=None)

    # 2. 一步到位（含日线技术指标）
    features = FeatureEngineer.extract_with_history(stock, daily_data)

    # 3. 扩展新特征：定义函数并注册
    def my_feature(stock, daily_data=None):
        return getattr(stock, 'pe_ratio', 0) or 0
    register_feature('my_feature', my_feature)
    # 然后在 FeatureConfig 的 basic_features 等中加入 'my_feature'
"""

# 特征定义
from .definitions import (
    FeatureDefinition,
    BASIC_FEATURES,
    TECHNICAL_FEATURES,
    CATEGORICAL_FEATURES,
    DERIVED_FEATURES,
    ALL_FEATURES,
)

# 技术指标计算
from .indicators import TechnicalIndicators

# 特征配置和工程器
from .engineer import (
    FeatureConfig,
    FeatureEngineer,
    DEFAULT_FEATURE_CONFIG,
    FULL_FEATURE_CONFIG,
    SIMPLE_FEATURE_CONFIG,
    get_full_numeric_feature_names,
    FEATURE_EXTRACTORS,
    register_feature,
    compute_batch_derived,
    extract_features,
    batch_extract_features,
)

__all__ = [
    # 特征定义
    'FeatureDefinition',
    'BASIC_FEATURES',
    'TECHNICAL_FEATURES',
    'CATEGORICAL_FEATURES',
    'DERIVED_FEATURES',
    'ALL_FEATURES',
    # 技术指标
    'TechnicalIndicators',
    # 配置和工程器
    'FeatureConfig',
    'FeatureEngineer',
    'DEFAULT_FEATURE_CONFIG',
    'FULL_FEATURE_CONFIG',
    'SIMPLE_FEATURE_CONFIG',
    'get_full_numeric_feature_names',
    'FEATURE_EXTRACTORS',
    'register_feature',
    'compute_batch_derived',
    # 便捷函数
    'extract_features',
    'batch_extract_features',
]
