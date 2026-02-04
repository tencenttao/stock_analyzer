# -*- coding: utf-8 -*-
"""
RandomForest 模型实现
"""

import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from .base import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    随机森林分类器
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        class_weight: dict = None,  # 支持自定义权重
    ):
        super().__init__()
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state,
            'n_jobs': -1,
            'class_weight': class_weight or 'balanced',
        }
        self._create_model()
    
    def _create_model(self):
        self.model = RandomForestClassifier(**self.params)
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> dict:
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签数组
            test_size: 测试集比例，0 表示使用全部数据训练
            
        Returns:
            训练结果
        """
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        if test_size > 0:
            # 划分数据集
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            # 使用全部数据训练
            X_train, y_train = X_scaled, y
            X_test, y_test = X_scaled, y
        
        # 训练
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # 评估
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"训练完成, 样本数: {len(X_train)}")
        
        return {
            'accuracy': accuracy,
            'train_size': len(X_train),
            'feature_importance': self.model.feature_importances_.tolist(),
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise RuntimeError("模型未训练")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
