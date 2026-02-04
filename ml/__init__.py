# -*- coding: utf-8 -*-
"""
机器学习模块

核心组件：
- MLConfig: ML 配置（特征、模型、标签定义）
- StockPredictor: 预测器
- FeatureEngineer: 特征工程
- models: 模型实现

使用流程：
1. python ml/build_training_data.py --start 2021-01-01 --end 2024-01-01 --output data/training.json
2. python ml/train.py --data data/training.json --output models/predictor.pkl
"""

from .config import MLConfig, DEFAULT_ML_CONFIG, RELATIVE_ML_CONFIG
from .predictor import StockPredictor, PredictionResult
from .models import create_model, BaseModel, MODEL_REGISTRY

from .features import (
    FeatureEngineer,
    FeatureConfig,
    DEFAULT_FEATURE_CONFIG,
    FULL_FEATURE_CONFIG,
    SIMPLE_FEATURE_CONFIG,
)

__all__ = [
    # 配置
    'MLConfig',
    'DEFAULT_ML_CONFIG',
    'RELATIVE_ML_CONFIG',
    # 预测器
    'StockPredictor',
    'PredictionResult',
    # 模型
    'create_model',
    'BaseModel',
    'MODEL_REGISTRY',
    # 特征
    'FeatureEngineer',
    'FeatureConfig',
    'DEFAULT_FEATURE_CONFIG',
    'FULL_FEATURE_CONFIG',
    'SIMPLE_FEATURE_CONFIG',
]
