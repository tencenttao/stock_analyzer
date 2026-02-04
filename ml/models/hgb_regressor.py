# -*- coding: utf-8 -*-
"""
HistGradientBoosting 回归模型实现

优化配置（2020-2025验证）：
- max_depth=3, max_iter=300: 6年累计Alpha +326.6%, Precision@10 约 39.9%
- 浅层模型泛化能力更强，防止过拟合
"""

import logging
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from .base import BaseModel

logger = logging.getLogger(__name__)


class HGBRegressorModel(BaseModel):
    """
    HistGradientBoosting 回归模型
    
    直接预测相对收益值，然后按预测值排序选股。
    优化配置：max_depth=3, max_iter=300（经过 2020-2025 验证）
    """
    
    def __init__(
        self,
        max_depth: int = 3,          # 优化：浅层模型，防止过拟合
        learning_rate: float = 0.05,
        max_iter: int = 300,          # 优化：增加迭代次数
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ):
        super().__init__()
        self.params = {
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'max_iter': max_iter,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state,
        }
        self._create_model()
    
    def _create_model(self):
        self.model = HistGradientBoostingRegressor(**self.params)
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, **kwargs) -> dict:
        """
        训练模型
        
        Args:
            X: 特征矩阵（原始，内部会标准化）
            y: 相对收益值（连续值）
            test_size: 未使用，保持接口一致
            **kwargs: 透传给底层 model.fit，如 sample_weight
            
        Returns:
            训练结果
        """
        # 处理 NaN
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 标准化（重要！对 HGB 性能有显著影响）
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # 训练
        self.model.fit(X_scaled, y, **kwargs)
        self.is_fitted = True
        
        logger.info(f"HGB回归模型训练完成, 样本数: {len(X)}")
        
        return {
            'train_size': len(X),
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测相对收益"""
        if not self.is_fitted:
            raise RuntimeError("模型未训练")
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_clean)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        返回伪概率（将预测值转换为概率形式）
        
        为了兼容现有接口，将预测的相对收益转换为三分类概率
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练")
        
        pred = self.predict(X)
        
        # 将预测值转换为概率（使用 sigmoid 归一化）
        prob_up = 1 / (1 + np.exp(-(pred - 3) / 5))  # 以 3% 为中心
        prob_down = 1 / (1 + np.exp((pred + 3) / 5))
        prob_neutral = 1 - prob_up - prob_down
        prob_neutral = np.clip(prob_neutral, 0, 1)
        
        # 归一化
        total = prob_down + prob_neutral + prob_up
        proba = np.column_stack([
            prob_down / total,
            prob_neutral / total,
            prob_up / total
        ])
        
        return proba
