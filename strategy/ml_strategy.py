# -*- coding: utf-8 -*-
"""
机器学习选股策略

使用训练好的 ML 模型进行选股：
- 加载预训练模型
- 提取股票特征
- 预测上涨概率
- 按概率排序选股

使用示例:
    # 方式1: 使用默认模型
    strategy = MLStrategy()
    
    # 方式2: 指定模型路径
    strategy = MLStrategy(model_path='models/my_model.pkl')
    
    # 选股
    selected = strategy.select(stocks, top_n=10)
"""

import os
import logging
from typing import Dict, List, Any, Optional

from core.interfaces import Strategy
from core.types import StockData, ScoreResult
from strategy.registry import register_strategy

logger = logging.getLogger(__name__)


# 默认配置
DEFAULT_CONFIG = {
    'model_path': 'models/predictor.pkl',  # 默认模型路径
    'min_prob_up': 0.5,                    # 最低上涨概率阈值（0.5 可达到约70%精确率）
    'min_pred_threshold': None,            # 回归模型：最小预测相对收益(%)，低于此值不选入（如 2 表示只选预测跑赢基准2%+）
    'min_price': 2.0,                      # 最低股价
    'max_stocks': 10,                      # 最大选股数量
}


@register_strategy('ml', '机器学习选股策略 - 基于上涨概率预测')
class MLStrategy(Strategy):
    """
    机器学习选股策略
    
    特点：
    - 使用预训练的 ML 模型预测股票涨跌
    - 以上涨概率作为评分依据
    - 选择上涨概率最高的股票
    
    评分逻辑：
    - 总分 = 上涨概率 * 100 (0-100分)
    """
    
    def __init__(self, config: Dict[str, Any] = None, model_path: str = None, data_source=None):
        """
        初始化 ML 策略
        
        Args:
            config: 策略配置
            model_path: 模型文件路径（优先级高于 config）
            data_source: 数据源（用于获取日线数据，可选）
        """
        merged_config = DEFAULT_CONFIG.copy()
        if config:
            merged_config.update(config)
        
        super().__init__(merged_config)
        
        # 默认模型路径（无 schedule 或未匹配时使用）
        self._default_model_path = model_path or merged_config.get('model_path')
        self._model_path = self._default_model_path
        
        # 按月份切换模型：{"YYYY-MM": "path", "default": "path"}，回测/选股时按当前日期匹配
        self._model_schedule = merged_config.get('model_schedule') or None
        self._current_date = None       # 回测/选股时由调用方 set_current_date 设置
        self._loaded_model_path = None  # 当前已加载的模型路径，用于判断是否需要重载
        
        # 数据源（用于获取日线数据）
        self._data_source = data_source
        
        # 延迟加载模型
        self._predictor = None
        self._feature_engineer = None
        self._model_loaded = False
        self._need_daily_data = False
        self._need_market_data = False
    
    def set_current_date(self, date: str):
        """
        设置当前选股/回测日期，用于按 model_schedule 切换模型。
        回测引擎在每月选股前会调用，选股单次调用时也可在 select 前调用。
        """
        self._current_date = date
    
    def get_model_path_for_date(self, date: str) -> str:
        """
        根据日期从 model_schedule 解析出应使用的模型路径。
        未配置 schedule 或未匹配时返回默认 model_path。
        """
        if not self._model_schedule or not date:
            return self._default_model_path
        month_key = date[:7]  # YYYY-MM
        return (
            self._model_schedule.get(month_key)
            or self._model_schedule.get('default')
            or self._default_model_path
        )
    
    def _load_model(self):
        """延迟加载模型（或按当前日期切换后加载）"""
        if self._model_loaded and self._loaded_model_path == self._model_path:
            return
        
        from ml.predictor import StockPredictor
        from ml.features import FeatureEngineer, FULL_FEATURE_CONFIG
        
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"模型文件不存在: {self._model_path}")
        
        # 加载预测器
        self._predictor = StockPredictor()
        self._predictor.load(self._model_path)
        
        # 初始化特征工程器（与训练时 FULL_FEATURE_CONFIG 一致）
        self._feature_engineer = FeatureEngineer(FULL_FEATURE_CONFIG)
        
        # 检查是否需要日线数据（技术指标）
        tech_features = {'rsi_14', 'volatility_20d', 'ma_deviation_20'}
        self._need_daily_data = bool(tech_features & set(self._predictor.feature_names))
        
        # 检查是否需要市场/相对特征
        # 如果模型特征名中包含市场特征，则需要计算
        market_feature_names = {
            'market_momentum_20d', 'market_momentum_60d', 'market_volatility_20d', 'market_trend',
            'relative_momentum_20d', 'relative_momentum_60d', 'volatility_ratio_20d',
            'stock_market_correlation_20d', 'stock_beta_20d',
        }
        self._need_market_data = bool(market_feature_names & set(self._predictor.feature_names))
        
        logger.info(f"[ML策略] 模型加载成功: {self._model_path}")
        logger.info(f"[ML策略] 特征数量: {len(self._predictor.feature_names)}")
        logger.info(f"[ML策略] 需要日线数据: {'是' if self._need_daily_data else '否'}")
        logger.info(f"[ML策略] 需要市场/相对特征: {'是' if self._need_market_data else '否'}")
        
        self._loaded_model_path = self._model_path
        self._model_loaded = True
    
    @property
    def name(self) -> str:
        return "ml"
    
    @property
    def description(self) -> str:
        return f"机器学习选股策略 (模型: {self._model_path})"
    
    def _extract_features(
        self,
        stock: StockData,
        daily_data: List = None,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """从 StockData 提取特征（与 build_training_data 逻辑一致）
        
        Args:
            stock: 股票数据
            daily_data: 日线数据（技术指标 + 相关系数/Beta 用，可选）
            market_data: 市场环境与相对特征（与训练时一致，可选）
        """
        features = self._feature_engineer.extract(
            stock, daily_data=daily_data, market_data=market_data
        )
        features['code'] = stock.code
        features['name'] = stock.name
        return features
    
    def _compute_market_data(self, date: str) -> Optional[Dict[str, Any]]:
        """计算市场环境特征（与 build_training_data 中 compute_market_features 一致）"""
        if not self._data_source:
            return None
        try:
            from ml.features.market import compute_market_features
            return compute_market_features(self._data_source, date)
        except Exception as e:
            logger.warning(f"[ML策略] 计算市场特征失败: {e}")
            return None
    
    def _compute_stock_market_relation(
        self, stock_daily: List, index_daily: List, period: int = 20
    ) -> tuple:
        """计算个股与大盘相关系数、Beta（与 build_training_data 一致）"""
        if not stock_daily or not index_daily:
            return 0.0, 1.0
        try:
            from ml.features.market import compute_stock_market_relation
            return compute_stock_market_relation(stock_daily, index_daily, period)
        except Exception as e:
            logger.warning(f"[ML策略] 计算个股-大盘关系失败: {e}")
            return 0.0, 1.0
    
    def set_data_source(self, data_source):
        """设置数据源（用于获取日线数据）"""
        self._data_source = data_source
    
    def score(self, stock: StockData) -> ScoreResult:
        """
        使用 ML 模型对股票评分
        
        评分 = 上涨概率 * 100
        """
        self._load_model()
        
        # 提取特征
        features = self._extract_features(stock)
        
        # 预测
        predictions = self._predictor.predict([features])
        
        if not predictions:
            return ScoreResult(
                total=0,
                breakdown={'prob_up': 0, 'prob_down': 0, 'prob_neutral': 0},
                grade='D',
                risk_flag=True
            )
        
        pred = predictions[0]
        
        # 总分 = 上涨概率 * 100
        total = pred.prob_up * 100
        
        # 风险标记：下跌概率 > 50%
        risk_flag = pred.prob_down > 0.5
        
        # 评级
        grade = self._calculate_grade(total)
        
        return ScoreResult(
            total=total,
            breakdown={
                'prob_up': round(pred.prob_up * 100, 1),
                'prob_down': round(pred.prob_down * 100, 1),
                'prob_neutral': round(pred.prob_neutral * 100, 1),
                'confidence': round(pred.confidence * 100, 1),
            },
            grade=grade,
            risk_flag=risk_flag
        )
    
    def _calculate_grade(self, total: float) -> str:
        """计算评级"""
        if total >= 60:
            return 'A+'
        elif total >= 50:
            return 'A'
        elif total >= 45:
            return 'B+'
        elif total >= 40:
            return 'B'
        elif total >= 35:
            return 'C'
        else:
            return 'D'
    
    def filter(self, stock: StockData) -> bool:
        """基本筛选"""
        # 1. 排除价格过低
        min_price = self._config.get('min_price', 2.0)
        if stock.price < min_price:
            return False
        
        # 2. 排除停牌股票
        if stock.change_pct == 0 and (stock.turnover_rate is None or stock.turnover_rate < 0.1):
            return False
        
        # 3. 排除跌停股票
        if stock.change_pct is not None and stock.change_pct <= -9.8:
            return False
        
        return True
    
    def select(self, stocks: List[StockData], top_n: int = 10, data_source=None) -> List[StockData]:
        """
        使用 ML 模型选股
        
        Args:
            stocks: 候选股票列表
            top_n: 选择数量
            data_source: 数据源（用于获取日线数据，可选）
            
        Returns:
            选中的股票列表（按上涨概率排序）
        """
        if not stocks:
            return []
        
        # 如果传入了 data_source，更新
        if data_source:
            self._data_source = data_source
        
        # 按日期切换模型（配置了 model_schedule 且已 set_current_date 时）
        if self._model_schedule and self._current_date:
            path = self.get_model_path_for_date(self._current_date)
            if path != self._loaded_model_path:
                self._model_path = path
                self._model_loaded = False
                logger.info(f"[ML策略] 按日期切换模型: {self._current_date[:7]} -> {path}")
        
        self._load_model()
        
        # 1. 去重
        unique_stocks = {}
        for stock in stocks:
            if stock.code not in unique_stocks:
                unique_stocks[stock.code] = stock
        stocks = list(unique_stocks.values())
        
        logger.info(f"[ML策略] 候选股票: {len(stocks)} 只")
        
        # 2. 基本筛选
        filtered = [s for s in stocks if self.filter(s)]
        logger.info(f"[ML策略] 基本筛选后: {len(filtered)} 只")
        
        if not filtered:
            return []
        
        # 3. 市场特征（与 build_training_data 一致：月度一次大盘 + 每只股票 correlation/beta）
        market_data_base = None
        index_daily_for_relation = None
        if self._need_market_data and self._data_source:
            ref_date = getattr(filtered[0], 'date', None)
            if ref_date:
                market_data_base = self._compute_market_data(ref_date)
                if market_data_base:
                    from datetime import datetime, timedelta
                    start_120 = (datetime.strptime(ref_date, '%Y-%m-%d') - timedelta(days=120)).strftime('%Y-%m-%d')
                    index_daily_for_relation = self._data_source.get_index_daily('000300', start_120, ref_date)
                    logger.info(f"[ML策略] 市场特征: 20d动量={market_data_base.get('market_momentum_20d', 0):.1f}%, 趋势={market_data_base.get('market_trend', 0)}")
            if not market_data_base:
                logger.warning("[ML策略] 未获取到市场特征，市场/相对特征将使用默认值")
        
        # 4. 批量提取特征（需要日线或市场特征时拉取 120 天日线，与训练一致）
        need_any_daily = self._need_daily_data or self._need_market_data
        logger.info(f"[ML策略] 提取特征中...")
        
        features_list = []
        if need_any_daily and self._data_source:
            for i, s in enumerate(filtered):
                if (i + 1) % 100 == 0 or i + 1 == len(filtered):
                    logger.info(f"[ML策略]   进度: {i+1}/{len(filtered)}")
                daily_data = self._data_source.get_daily_data(s.code, end_date=getattr(s, 'date', None), days=120)
                market_data = None
                if self._need_market_data and market_data_base is not None:
                    market_data = dict(market_data_base)
                    if index_daily_for_relation and daily_data:
                        corr, beta = self._compute_stock_market_relation(daily_data, index_daily_for_relation, 20)
                        market_data['stock_market_correlation_20d'] = corr
                        market_data['stock_beta_20d'] = beta
                features = self._extract_features(s, daily_data=daily_data, market_data=market_data)
                features_list.append(features)
        else:
            if need_any_daily and not self._data_source:
                logger.warning("[ML策略] 需要日线/市场数据但未设置数据源，相关特征将使用默认值")
            market_data = market_data_base if (self._need_market_data and market_data_base) else None
            features_list = [self._extract_features(s, market_data=market_data) for s in filtered]
        
        # 5. 批量预测
        logger.info(f"[ML策略] 模型预测中...")
        predictions = self._predictor.predict(features_list)
        
        # 6. 构建 code -> prediction 映射
        pred_map = {p.code: p for p in predictions}
        
        # 7. 筛选候选（支持两种阈值，与 quarterly_selector 一致）
        min_prob = self._config.get('min_prob_up', 0.5)
        min_pred_threshold = self._config.get('min_pred_threshold')  # 回归模型：预测相对收益(%) 阈值
        candidates = []
        
        for stock in filtered:
            pred = pred_map.get(stock.code)
            if not pred:
                continue
            # 回归阈值：只选预测相对收益 >= 阈值的
            if min_pred_threshold is not None:
                if pred.predicted_return is None:
                    continue
                if pred.predicted_return < min_pred_threshold:
                    continue
                score = pred.predicted_return  # 用预测收益%作为排序分
            else:
                if pred.prob_up < min_prob:
                    continue
                score = pred.prob_up * 100
            stock.strength_score = score if min_pred_threshold is not None else pred.prob_up * 100
            stock.strength_grade = self._calculate_grade(stock.strength_score)
            stock.score_breakdown = {
                'prob_up': round(pred.prob_up * 100, 1),
                'prob_down': round(pred.prob_down * 100, 1),
                'prob_neutral': round(pred.prob_neutral * 100, 1),
            }
            if pred.predicted_return is not None:
                stock.score_breakdown['pred_return_pct'] = round(pred.predicted_return, 2)
            stock.selection_reason = (
                f"🤖 ML预测相对收益: {pred.predicted_return:.1f}%" if pred.predicted_return is not None and min_pred_threshold is not None
                else f"🤖 ML预测上涨概率: {pred.prob_up:.1%}"
            )
            candidates.append(stock)
        
        if min_pred_threshold is not None:
            logger.info(f"[ML策略] 满足预测阈值候选: {len(candidates)} 只 (pred_return >= {min_pred_threshold}%)")
        else:
            logger.info(f"[ML策略] 高概率候选: {len(candidates)} 只 (prob_up >= {min_prob:.0%})")
        
        # 无满足阈值时：退化为按预测收益/概率取 Top N（与 quarterly_selector 一致）
        if not candidates and filtered:
            for stock in filtered:
                pred = pred_map.get(stock.code)
                if pred:
                    score = pred.predicted_return if pred.predicted_return is not None else pred.prob_up * 100
                    stock.strength_score = score
                    stock.strength_grade = self._calculate_grade(stock.strength_score)
                    stock.score_breakdown = {
                        'prob_up': round(pred.prob_up * 100, 1),
                        'prob_down': round(pred.prob_down * 100, 1),
                        'prob_neutral': round(pred.prob_neutral * 100, 1),
                    }
                    if pred.predicted_return is not None:
                        stock.score_breakdown['pred_return_pct'] = round(pred.predicted_return, 2)
                    stock.selection_reason = f"🤖 ML预测相对收益: {pred.predicted_return:.1f}%" if pred.predicted_return is not None else f"🤖 ML预测上涨概率: {pred.prob_up:.1%}"
                    candidates.append(stock)
            candidates.sort(key=lambda x: x.strength_score, reverse=True)
            candidates = candidates[:min(top_n, self._config.get('max_stocks', top_n))]
            logger.info(f"[ML策略] 无满足阈值，按预测排序取 Top {len(candidates)}")
        
        # 8. 按得分排序（预测收益% 或 概率）
        candidates.sort(key=lambda x: x.strength_score, reverse=True)
        
        # 9. 取 Top N
        max_stocks = self._config.get('max_stocks', top_n)
        selected = candidates[:min(top_n, max_stocks)]
        
        # 10. 添加排名
        for i, stock in enumerate(selected):
            stock.rank = i + 1
        
        logger.info(f"[ML策略] ✅ 选出 {len(selected)} 只股票")
        
        return selected
