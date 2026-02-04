# -*- coding: utf-8 -*-
"""
模型基类

定义所有模型的统一接口。
"""

import pickle
import logging
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    模型基类
    
    所有具体模型需实现以下方法：
    - fit(X, y): 训练
    - predict_proba(X): 预测概率
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_fitted = False
    
    @abstractmethod
    def _create_model(self):
        """创建底层模型（子类实现）"""
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签数组
            
        Returns:
            训练结果（准确率、特征重要性等）
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            概率矩阵 (n_samples, n_classes)
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def save(self, path: str, extra_data: dict = None):
        """保存模型（保存整个包装器对象以保留类型信息）"""
        data = {
            'wrapper': self,  # 保存整个包装器对象
        }
        if extra_data:
            data.update(extra_data)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"模型已保存: {path}")
    
    @classmethod
    def load_from_file(cls, path: str) -> Tuple['BaseModel', dict]:
        """从文件加载模型，返回 (模型包装器, 额外数据)"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if 'wrapper' in data:
            # 新格式：直接返回保存的包装器
            wrapper = data.pop('wrapper')
            logger.info(f"模型已加载: {path}")
            return wrapper, data
        else:
            # 旧格式：无法恢复正确的包装器类型，返回原始数据
            logger.warning(f"旧格式模型，可能需要重新训练")
            return None, data
    
    def load(self, path: str) -> dict:
        """加载模型（兼容旧接口）"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if 'wrapper' in data:
            # 新格式：从包装器复制属性
            wrapper = data.pop('wrapper')
            self.__dict__.update(wrapper.__dict__)
        else:
            # 旧格式
            self.model = data.pop('model')
            self.scaler = data.pop('scaler')
            self.is_fitted = data.pop('is_fitted')
        
        logger.info(f"模型已加载: {path}")
        return data
    
    def get_feature_importance(self) -> dict:
        """获取特征重要性（若模型支持）"""
        if hasattr(self.model, 'feature_importances_'):
            return dict(enumerate(self.model.feature_importances_))
        return {}
