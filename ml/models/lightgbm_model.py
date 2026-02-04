# -*- coding: utf-8 -*-
"""LightGBM 模型"""

import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .base import BaseModel

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    """LightGBM 分类器"""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        super().__init__()
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'n_jobs': -1,
            'verbose': -1,
        }
        self._create_model()
    
    def _create_model(self):
        from lightgbm import LGBMClassifier
        self.model = LGBMClassifier(**self.params)
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> dict:
        X_scaled = self.scaler.fit_transform(X)
        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X_scaled, y
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info(f"LightGBM 训练完成, 样本数: {len(X_train)}")
        
        return {'accuracy': 0, 'train_size': len(X_train)}
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("模型未训练")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
