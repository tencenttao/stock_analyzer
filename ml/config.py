# -*- coding: utf-8 -*-
"""
ML 模块配置

统一管理特征配置、模型配置、标签定义。
"""

from dataclasses import dataclass, field
from typing import Optional
from ml.features import FeatureConfig, DEFAULT_FEATURE_CONFIG


@dataclass
class MLConfig:
    """
    机器学习配置
    
    Attributes:
        feature_config: 特征配置
        model_type: 模型类型 ('random_forest', 'xgboost', 'logistic')
        label_type: 标签类型
            - 'absolute': 绝对收益（涨跌幅超过阈值）
            - 'relative': 相对收益（相对指数涨跌幅超过阈值）
        threshold_up: 上涨阈值（%）
        threshold_down: 下跌阈值（%）
        test_size: 测试集比例
    """
    feature_config: FeatureConfig = field(default_factory=lambda: DEFAULT_FEATURE_CONFIG)
    model_type: str = 'random_forest'
    label_type: str = 'absolute'  # 'absolute' | 'relative'
    threshold_up: float = 5.0
    threshold_down: float = -5.0
    test_size: float = 0.2
    
    def get_numeric_feature_names(self):
        """获取数值型特征名（排除分类特征）"""
        all_names = self.feature_config.get_all_feature_names()
        categorical = self.feature_config.categorical_features
        return [n for n in all_names if n not in categorical]


# 预定义配置
DEFAULT_ML_CONFIG = MLConfig()

RELATIVE_ML_CONFIG = MLConfig(
    label_type='relative',
    threshold_up=3.0,   # 跑赢指数 3%
    threshold_down=-3.0 # 跑输指数 3%
)
