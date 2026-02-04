# -*- coding: utf-8 -*-
"""
随机选股策略

基线对照策略，用于验证其他策略是否真正有效。

如果动量策略的表现与随机策略相当，说明动量因子可能无效。
如果动量策略显著优于随机策略，说明动量因子有选股价值。
"""

import random
import logging
from typing import Dict, List, Any

from core.interfaces import Strategy
from core.types import StockData, ScoreResult
from strategy.registry import register_strategy

logger = logging.getLogger(__name__)


# 默认配置
DEFAULT_CONFIG = {
    'seed': 42,           # 随机种子（确保可重复）
    'min_price': 2.0,     # 最低股价
    'max_stocks': 10,     # 最大选股数量
}


@register_strategy('random', '随机选股策略 - 基线对照')
class RandomStrategy(Strategy):
    """
    随机选股策略
    
    从候选股票中随机选择，用于作为策略效果的基线对照。
    
    使用示例:
        strategy = RandomStrategy(config={'seed': 42})
        selected = strategy.select(stocks, top_n=10)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化策略
        
        Args:
            config: 策略配置
        """
        merged_config = DEFAULT_CONFIG.copy()
        if config:
            merged_config.update(config)
        
        super().__init__(merged_config)
        
        # 初始化随机数生成器
        self._seed = merged_config.get('seed', 42)
        self._rng = random.Random(self._seed)
    
    @property
    def name(self) -> str:
        return "random"
    
    @property
    def description(self) -> str:
        return f"随机选股策略 (seed={self._seed}) - 基线对照"
    
    def score(self, stock: StockData) -> ScoreResult:
        """
        随机评分
        
        随机策略的评分是随机的，仅用于排序。
        """
        # 生成 0-100 的随机分数
        random_score = self._rng.uniform(0, 100)
        
        return ScoreResult(
            total=random_score,
            breakdown={'random': random_score},
            grade=self._calculate_grade(random_score),
            risk_flag=False
        )
    
    def _calculate_grade(self, total: float) -> str:
        """计算评级（与其他策略保持一致）"""
        if total >= 80:
            return 'A+'
        elif total >= 70:
            return 'A'
        elif total >= 60:
            return 'B+'
        elif total >= 50:
            return 'B'
        elif total >= 40:
            return 'C'
        else:
            return 'D'
    
    def filter(self, stock: StockData) -> bool:
        """
        基本筛选
        
        即使是随机策略，也应排除明显不可买的股票。
        """
        # 1. 排除亏损股
        if stock.pe_ratio is not None and stock.pe_ratio < 0:
            return False
        
        # 2. 排除停牌股票
        if stock.change_pct == 0 and (stock.turnover_rate is None or stock.turnover_rate < 0.1):
            return False
        
        # 3. 排除跌停股票
        if stock.change_pct is not None and stock.change_pct <= -9.8:
            return False
        
        # 4. 排除仙股
        min_price = self._config.get('min_price', 2.0)
        if stock.price < min_price:
            return False
        
        return True
    
    def select(self, stocks: List[StockData], top_n: int = 10) -> List[StockData]:
        """
        随机选择股票
        
        Args:
            stocks: 候选股票列表
            top_n: 选择数量
            
        Returns:
            随机选中的股票列表
        """
        if not stocks:
            return []
        
        # 重置随机数生成器（确保每次调用结果一致）
        self._rng = random.Random(self._seed)
        
        # 1. 去重
        unique_stocks = {}
        for stock in stocks:
            if stock.code not in unique_stocks:
                unique_stocks[stock.code] = stock
        stocks = list(unique_stocks.values())
        
        logger.info(f"[随机策略] 候选股票: {len(stocks)} 只")
        
        # 2. 基本筛选
        filtered = [s for s in stocks if self.filter(s)]
        logger.info(f"[随机策略] 基本筛选后: {len(filtered)} 只")
        
        if not filtered:
            return []
        
        # 3. 随机选择
        max_stocks = self._config.get('max_stocks', top_n)
        select_n = min(top_n, max_stocks, len(filtered))
        
        selected = self._rng.sample(filtered, select_n)
        
        # 4. 添加随机评分和排名
        for i, stock in enumerate(selected):
            score_result = self.score(stock)
            stock.strength_score = score_result.total
            stock.strength_grade = score_result.grade
            stock.score_breakdown = score_result.breakdown
            stock.rank = i + 1
            stock.selection_reason = "🎲 随机选择"
        
        logger.info(f"[随机策略] ✅ 随机选择 {len(selected)} 只股票")
        
        return selected
    
    def reset_seed(self, seed: int = None):
        """
        重置随机种子
        
        Args:
            seed: 新的随机种子，不传则使用配置中的种子
        """
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)
        logger.info(f"[随机策略] 重置随机种子: {self._seed}")
