# -*- coding: utf-8 -*-
"""
股票预测器（精简版）

根据 MLConfig 统一管理特征、模型、标签定义。
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

from .config import MLConfig, DEFAULT_ML_CONFIG
from .models import create_model

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """预测结果"""
    code: str
    name: str
    prob_up: float
    prob_down: float
    prob_neutral: float
    confidence: float
    prediction: str  # 'up', 'down', 'neutral'
    predicted_return: Optional[float] = None  # 回归模型预测的相对收益(%)，用于阈值过滤


class StockPredictor:
    """
    股票预测器
    
    使用示例:
        config = MLConfig(label_type='relative', threshold_up=3.0)
        predictor = StockPredictor(config)
        X, y = predictor.prepare_data(training_data)
        predictor.train(X, y)
        results = predictor.predict(features_list)
    """
    
    CLASS_LABELS = {0: 'down', 1: 'neutral', 2: 'up'}
    
    def __init__(self, config: MLConfig = None, feature_names: List[str] = None):
        self.config = config or DEFAULT_ML_CONFIG
        # 支持自定义特征列表（优先级高于 config）
        self.feature_names = feature_names or self.config.get_numeric_feature_names()
        self.model = create_model(self.config.model_type)
        logger.info(f"预测器初始化: 模型={self.config.model_type}, 特征数={len(self.feature_names)}")
    
    def prepare_data(
        self,
        training_data: List[Dict],
        regression: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从训练数据准备 X, y
        
        Args:
            training_data: build_training_data 的输出
                每条记录: {features: dict, return_pct: float, index_return_pct?: float}
            regression: 是否为回归模式（返回相对收益值而非分类标签）
                
        Returns:
            X: 特征矩阵
            y: 标签数组（分类）或相对收益值（回归）
        """
        X_list = []
        y_list = []
        
        for record in training_data:
            features = record.get('features', {})
            
            # 按 feature_names 顺序提取特征
            feature_array = np.array([
                float(features.get(name, 0) or 0)
                for name in self.feature_names
            ], dtype=np.float32)
            
            # 处理异常值
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            X_list.append(feature_array)
            
            if regression:
                # 回归模式：返回相对收益值
                return_pct = record.get('return_pct', 0) or 0
                index_return = record.get('index_return_pct', 0) or 0
                relative_return = return_pct - index_return
                y_list.append(relative_return)
            else:
                # 分类模式：返回标签
                label = self._compute_label(record)
                y_list.append(label)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        if regression:
            logger.info(f"准备数据: {len(X)} 条, 相对收益: 均值={y.mean():.2f}%, 标准差={y.std():.2f}%")
        else:
            logger.info(f"准备数据: {len(X)} 条, 标签分布: 下跌={sum(y==0)}, 持平={sum(y==1)}, 上涨={sum(y==2)}")
        
        return X, y
    
    def _compute_label(self, record: Dict) -> int:
        """
        根据配置计算标签
        
        Returns:
            0=下跌, 1=持平, 2=上涨
        """
        return_pct = record.get('return_pct', 0) or 0
        
        if self.config.label_type == 'relative':
            # 相对收益：减去指数收益
            index_return = record.get('index_return_pct', 0) or 0
            relative_return = return_pct - index_return
            value = relative_return
        else:
            # 绝对收益
            value = return_pct
        
        if value > self.config.threshold_up:
            return 2  # 上涨
        elif value < self.config.threshold_down:
            return 0  # 下跌
        else:
            return 1  # 持平
    
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """训练模型"""
        return self.model.fit(X, y, test_size=self.config.test_size)
    
    def predict(self, features_list: List[Dict]) -> List[PredictionResult]:
        """
        预测
        
        Args:
            features_list: 特征字典列表，每个字典包含 code, name 和特征值
            
        Returns:
            预测结果列表。回归模型时含 predicted_return（相对收益%），供阈值过滤使用。
        """
        if not features_list:
            return []
        
        # 批量构建特征矩阵
        X_list = []
        for features in features_list:
            arr = np.array([
                float(features.get(name, 0) or 0)
                for name in self.feature_names
            ], dtype=np.float32)
            X_list.append(arr)
        X = np.nan_to_num(np.array(X_list), nan=0.0, posinf=0.0, neginf=0.0)
        
        # 回归模型：取原始预测值（相对收益%）
        raw_pred = None
        if hasattr(self.model, 'predict') and callable(getattr(self.model, 'predict', None)):
            try:
                out = self.model.predict(X)
                if getattr(out, 'ndim', 0) == 1 and len(out) == len(features_list):
                    raw_pred = np.asarray(out).ravel()
            except Exception:
                pass
        
        # 概率/伪概率
        proba = self.model.predict_proba(X)
        
        results = []
        for i, features in enumerate(features_list):
            p = proba[i] if proba.ndim > 1 else proba
            prob_down = float(p[0] if len(p) > 0 else 0)
            prob_neutral = float(p[1] if len(p) > 1 else 0)
            prob_up = float(p[2] if len(p) > 2 else 0)
            max_idx = np.argmax(p)
            pred_ret = float(raw_pred[i]) if raw_pred is not None and i < len(raw_pred) else None
            
            result = PredictionResult(
                code=features.get('code', ''),
                name=features.get('name', ''),
                prob_up=prob_up,
                prob_down=prob_down,
                prob_neutral=prob_neutral,
                confidence=float(p[max_idx]),
                prediction=self.CLASS_LABELS.get(max_idx, 'neutral'),
                predicted_return=pred_ret,
            )
            results.append(result)
        
        # 排序：有 predicted_return 时按预测收益排，否则按 prob_up
        def sort_key(r):
            if r.predicted_return is not None:
                return r.predicted_return
            return r.prob_up
        results.sort(key=sort_key, reverse=True)
        return results
    
    def filter_high_confidence(
        self,
        predictions: List[PredictionResult],
        min_confidence: float = 0.5,
        prediction_type: str = 'up',
    ) -> List[PredictionResult]:
        """筛选高置信度预测"""
        if prediction_type == 'up':
            return [p for p in predictions if p.prob_up >= min_confidence]
        elif prediction_type == 'down':
            return [p for p in predictions if p.prob_down >= min_confidence]
        else:
            return [p for p in predictions if p.confidence >= min_confidence]
    
    def save(self, path: str):
        """保存模型（含配置和特征名）"""
        extra_data = {
            'feature_names': self.feature_names,
            'label_type': self.config.label_type,
            'threshold_up': self.config.threshold_up,
            'threshold_down': self.config.threshold_down,
            'model_type': self.config.model_type,
        }
        self.model.save(path, extra_data)
    
    def load(self, path: str):
        """加载模型（恢复特征名和配置）"""
        from .models.base import BaseModel
        
        # 尝试使用新的加载方法
        wrapper, extra = BaseModel.load_from_file(path)
        if wrapper is not None:
            # 新格式：使用加载的包装器替换当前模型
            self.model = wrapper
        else:
            # 旧格式：先读取 model_type，创建正确的包装器再加载
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            model_type = data.get('model_type', 'random_forest')
            if model_type in ('hgb_regressor', 'hgb', 'hgb_shallow', 'hgb_deep'):
                from .models.hgb_regressor import HGBRegressorModel
                self.model = HGBRegressorModel()
            elif model_type in ('rf_regressor', 'rf', 'rf_100', 'rf_200'):
                from .models.random_forest_regressor import RandomForestRegressorModel
                self.model = RandomForestRegressorModel()
            # 其他类型保持默认包装器
            
            # 加载模型数据到包装器
            self.model.model = data.pop('model')
            self.model.scaler = data.pop('scaler')
            self.model.is_fitted = data.pop('is_fitted', True)
            extra = data
        
        # 恢复配置参数
        if 'model_type' in extra:
            self.config.model_type = extra['model_type']
        if 'label_type' in extra:
            self.config.label_type = extra['label_type']
        if 'threshold_up' in extra:
            self.config.threshold_up = extra['threshold_up']
        if 'threshold_down' in extra:
            self.config.threshold_down = extra['threshold_down']
        
        if 'feature_names' in extra:
            self.feature_names = extra['feature_names']
        
        logger.info(f"模型已加载: {path}")
        logger.info(f"  模型类型: {self.config.model_type}, 特征数: {len(self.feature_names)}")
