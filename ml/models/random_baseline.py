# -*- coding: utf-8 -*-
"""完全随机预测 Baseline 模型"""

import logging
import numpy as np

from .base import BaseModel

logger = logging.getLogger(__name__)


class RandomBaselineModel(BaseModel):
    """
    完全随机预测模型（Baseline）
    
    用于对比其他模型是否优于随机猜测。
    每个类别的预测概率完全随机（期望均匀），不考虑任何特征或先验分布。
    """
    
    def __init__(self, random_state=None):
        super().__init__()
        self.random_state = random_state
        self.n_classes = 3
        self._create_model()
    
    def _create_model(self):
        self.model = None  # 无需底层模型
        self.scaler = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> dict:
        """训练（仅记录类别数，不学习任何模式）"""
        self.n_classes = len(np.unique(y))
        self.is_fitted = True
        logger.info(f"Random Baseline: 完全随机预测, {self.n_classes} 个类别, 每个类别期望概率 = {1/self.n_classes:.1%}")
        
        return {'accuracy': 1/self.n_classes, 'train_size': len(X), 'n_classes': self.n_classes}
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        完全随机预测
        
        使用均匀 Dirichlet 分布生成随机概率，每个类别的期望概率相等。
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练")
        
        n_samples = X.shape[0]
        
        # 均匀 Dirichlet 分布：alpha = [1, 1, 1, ...]
        # 生成的概率期望为 [1/n, 1/n, 1/n, ...]
        alpha = np.ones(self.n_classes)
        proba = np.random.dirichlet(alpha, size=n_samples)
        
        return proba
    
    def save(self, path: str, extra_data: dict = None):
        """保存（只需保存类别数）"""
        import pickle
        data = {
            'model': None,
            'scaler': None,
            'is_fitted': self.is_fitted,
            'n_classes': self.n_classes,
        }
        if extra_data:
            data.update(extra_data)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str) -> dict:
        """加载"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.n_classes = data.pop('n_classes')
        self.is_fitted = data.pop('is_fitted')
        data.pop('model', None)
        data.pop('scaler', None)
        return data
