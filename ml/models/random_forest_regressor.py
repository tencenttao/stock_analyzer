# -*- coding: utf-8 -*-
"""
RandomForest 回归模型实现

直接预测相对收益，用于选股排序。
"""

import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from .base import BaseModel

logger = logging.getLogger(__name__)


class RandomForestRegressorModel(BaseModel):
    """
    随机森林回归模型
    
    直接预测相对收益值，然后按预测值排序选股。
    优点：在不同市场环境下更稳定。
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 15,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ):
        super().__init__()
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state,
            'n_jobs': -1,
        }
        self._create_model()
    
    def _create_model(self):
        self.model = RandomForestRegressor(**self.params)
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
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练
        self.model.fit(X_scaled, y, **kwargs)
        self.is_fitted = True
        
        logger.info(f"回归模型训练完成, 样本数: {len(X)}")
        
        return {
            'train_size': len(X),
            'feature_importance': self.model.feature_importances_.tolist(),
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测相对收益"""
        if not self.is_fitted:
            raise RuntimeError("模型未训练")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        返回伪概率（将预测值转换为概率形式）
        
        为了兼容现有接口，将预测的相对收益转换为三分类概率：
        - 概率越高表示预测的相对收益越高
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练")
        
        pred = self.predict(X)
        
        # 将预测值转换为概率（使用 sigmoid 归一化）
        # 预测值 > 3% 倾向于上涨，< -3% 倾向于下跌
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
