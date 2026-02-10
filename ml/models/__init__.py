# -*- coding: utf-8 -*-
"""
模型模块

统一入口 create_model：
- create_model(model_type: str, **kwargs) 或 create_model(config: dict) 均返回 BaseModel，
  内部含 StandardScaler，调用方传入原始 X 即可。predictor、ml_strategy、strategy_optimizer、
  quarterly_selector 均使用同一套。
"""

from .base import BaseModel
from .random_forest import RandomForestModel
from .random_forest_regressor import RandomForestRegressorModel
from .hgb_regressor import HGBRegressorModel
from .random_baseline import RandomBaselineModel

# ========== 预设配置（create_model(config) 时映射到 BaseModel） ==========
REGRESSOR_PRESETS = {
    'hgb_shallow': {
        'model': 'hgb',
        'params': {'max_depth': 3, 'learning_rate': 0.05, 'max_iter': 300, 'random_state': 42},
    },
    'hgb_medium': {
        'model': 'hgb',
        'params': {'max_depth': 5, 'learning_rate': 0.05, 'max_iter': 200, 'random_state': 42},
    },
    'hgb_deep': {
        'model': 'hgb',
        'params': {'max_depth': 8, 'learning_rate': 0.03, 'max_iter': 500, 'random_state': 42},
    },
    'rf_100': {
        'model': 'rf',
        'params': {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
    },
    'rf_200': {
        'model': 'rf',
        'params': {'n_estimators': 200, 'max_depth': 8, 'random_state': 42},
          },
}


def _create_from_preset(config):
    """根据预设配置创建 BaseModel（与 predictor 同一套，内部含 scaler）。"""
    if isinstance(config, dict) and 'model' in config and 'params' in config:
        kind, params = config['model'], config['params']
    else:
        kind, params = 'hgb', (config or {})
    if kind == 'hgb':
        return HGBRegressorModel(**params)
    if kind == 'rf':
        return RandomForestRegressorModel(**params)
    raise ValueError(f"不支持的预设类型: {kind}, 可选: hgb, rf")


# 模型注册表（基础模型始终可用）
MODEL_REGISTRY = {
    'random_forest': RandomForestModel,
    'rf_regressor': RandomForestRegressorModel,
    'rf': RandomForestRegressorModel,       # 别名
    'rf_100': RandomForestRegressorModel,   # 别名
    'rf_200': RandomForestRegressorModel,   # 别名
    'hgb_regressor': HGBRegressorModel,     # 推荐：最优模型
    'hgb': HGBRegressorModel,               # 别名
    'hgb_shallow': HGBRegressorModel,       # 别名
    'hgb_deep': HGBRegressorModel,          # 别名
    'random': RandomBaselineModel,
}

# 可选模型（依赖可能未安装或运行时库缺失）
def _try_register(name, module_name, class_name):
    """尝试注册模型，验证依赖是否真正可用"""
    try:
        import importlib
        module = importlib.import_module(f'.{module_name}', package='ml.models')
        cls = getattr(module, class_name)
        # 尝试实例化以验证运行时依赖
        cls()
        MODEL_REGISTRY[name] = cls
    except Exception:
        pass

_try_register('xgboost', 'xgboost_model', 'XGBoostModel')
_try_register('lightgbm', 'lightgbm_model', 'LightGBMModel')


def create_model(model_type_or_config, **kwargs):
    """
    统一入口：根据类型或预设配置创建模型，均返回 BaseModel（内部含 StandardScaler）。

    - create_model(model_type: str, **kwargs) → BaseModel
      如 create_model('hgb_regressor', max_depth=8)。
    - create_model(config: dict) → BaseModel
      config 为 REGRESSOR_PRESETS 中一项（含 'model' 与 'params'），如 create_model(REGRESSOR_PRESETS['hgb_deep'])。
      调用方传入原始 X，模型内部做标准化。

    Returns:
        BaseModel 实例（fit/predict 接受原始 X）
    """
    if isinstance(model_type_or_config, dict):
        return _create_from_preset(model_type_or_config)
    model_type = model_type_or_config
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"不支持的模型类型: {model_type}, 可选: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type](**kwargs)


def list_available_models():
    """列出当前可用的模型"""
    return list(MODEL_REGISTRY.keys())


def create_sklearn_regressor(config):
    """create_model(config) 的别名，兼容旧调用。返回 BaseModel，非裸 sklearn。"""
    return create_model(config)


__all__ = [
    'BaseModel',
    'RandomForestModel',
    'RandomBaselineModel',
    'MODEL_REGISTRY',
    'create_model',
    'list_available_models',
    'REGRESSOR_PRESETS',
    'create_sklearn_regressor',
]
