# -*- coding: utf-8 -*-
"""
自适应策略 - 根据市场状态动态调整权重

市场状态判断：
- 牛市: 指数在20日均线上方，且20日均线向上
- 熊市: 指数在20日均线下方，且20日均线向下
- 震荡: 其他情况

权重自适应：
- 牛市: 动量权重最高，追涨强势股
- 熊市: 价值/安全权重最高，防守为主
- 震荡: 平衡配置，兼顾各方面
"""

import logging
from typing import Dict, List, Any, Optional
from enum import Enum

from core.interfaces import Strategy
from core.types import StockData, ScoreResult, IndexData
from strategy.registry import register_strategy
from strategy.scoring.momentum_v2 import MomentumV2Strategy

logger = logging.getLogger(__name__)


class MarketState(Enum):
    """市场状态枚举"""
    BULL = "bull"       # 牛市
    BEAR = "bear"       # 熊市
    SIDEWAYS = "sideways"  # 震荡


# 不同市场状态的权重配置
MARKET_WEIGHTS = {
    MarketState.BULL: {
        'momentum': 45,   # 牛市：动量权重最高
        'growth': 25,
        'valuation': 15,
        'quality': 10,
        'safety': 5
    },
    MarketState.BEAR: {
        'momentum': 15,   # 熊市：降低动量，提升安全
        'growth': 20,
        'valuation': 25,
        'quality': 20,
        'safety': 20
    },
    MarketState.SIDEWAYS: {
        'momentum': 30,   # 震荡：平衡配置
        'growth': 25,
        'valuation': 20,
        'quality': 15,
        'safety': 10
    }
}

# 市场状态中文名
MARKET_STATE_NAMES = {
    MarketState.BULL: "牛市",
    MarketState.BEAR: "熊市",
    MarketState.SIDEWAYS: "震荡市"
}


@register_strategy('adaptive', '自适应策略 - 根据市场状态动态调整权重')
class AdaptiveStrategy(MomentumV2Strategy):
    """
    自适应选股策略
    
    特点：
    - 自动判断市场状态（牛/熊/震荡）
    - 根据市场状态动态调整评分权重
    - 牛市追涨、熊市防守、震荡平衡
    
    使用示例:
        # 方式1：手动指定市场状态
        strategy = AdaptiveStrategy(market_state=MarketState.BULL)
        
        # 方式2：传入指数数据，自动判断市场状态
        strategy = AdaptiveStrategy()
        strategy.update_market_state(index_data)
        
        # 选股
        selected = strategy.select(stocks, top_n=10)
    """
    
    def __init__(self, config: Dict[str, Any] = None, market_state: MarketState = None):
        """
        初始化自适应策略
        
        Args:
            config: 策略配置
            market_state: 指定市场状态，None则默认为震荡市
        """
        # 设置市场状态（默认震荡市）
        self._market_state = market_state or MarketState.SIDEWAYS
        
        # 状态历史记录（用于平滑，避免频繁切换）
        self._state_history: List[MarketState] = []
        self._state_history_max = 3  # 保留最近3次状态
        
        # 根据市场状态设置权重
        weights = MARKET_WEIGHTS[self._market_state].copy()
        
        # 合并配置
        merged_config = {
            'weights': weights,
            'min_price': 2.0,
            'min_score': 35,
            'max_stocks': 10,
        }
        if config:
            merged_config.update(config)
            # 如果配置中没有指定权重，使用市场状态对应的权重
            if 'weights' not in config:
                merged_config['weights'] = weights
        
        # 调用父类初始化
        super().__init__(merged_config)
        
        logger.info(f"🎯 自适应策略初始化: 市场状态={MARKET_STATE_NAMES[self._market_state]}, "
                   f"权重={weights}")
    
    @property
    def name(self) -> str:
        return "adaptive"
    
    @property
    def description(self) -> str:
        state_name = MARKET_STATE_NAMES[self._market_state]
        weights = self._config.get('weights', {})
        return f"自适应策略（{state_name}模式）- 动量{weights.get('momentum', 0)}% + 成长{weights.get('growth', 0)}% + 估值{weights.get('valuation', 0)}%"
    
    @property
    def market_state(self) -> MarketState:
        """获取当前市场状态"""
        return self._market_state
    
    def update_market_state(self, index_data: IndexData = None, 
                           index_prices: List[float] = None,
                           force_state: MarketState = None) -> MarketState:
        """
        更新市场状态（带平滑机制）
        
        Args:
            index_data: 指数数据（包含收盘价和历史数据）
            index_prices: 历史收盘价列表（至少20个，从旧到新）
            force_state: 强制指定市场状态
            
        Returns:
            更新后的市场状态
        """
        detected_state = None
        
        if force_state:
            detected_state = force_state
            logger.info(f"🎯 强制设置市场状态: {MARKET_STATE_NAMES[force_state]}")
            # 强制设置时清空历史
            self._state_history = []
        elif index_prices and len(index_prices) >= 20:
            detected_state = self._detect_market_state(index_prices)
        elif index_data and hasattr(index_data, 'close_prices') and len(index_data.close_prices) >= 20:
            detected_state = self._detect_market_state(index_data.close_prices)
        else:
            logger.warning("⚠️ 无法判断市场状态，保持当前状态")
            return self._market_state
        
        # 状态平滑：需要连续2次相同状态才切换（避免频繁切换）
        self._state_history.append(detected_state)
        if len(self._state_history) > self._state_history_max:
            self._state_history.pop(0)
        
        # 判断是否需要切换状态
        if len(self._state_history) >= 2:
            # 如果最近2次检测结果相同，且与当前状态不同，则切换
            if (self._state_history[-1] == self._state_history[-2] and 
                self._state_history[-1] != self._market_state):
                old_state = self._market_state
                self._market_state = detected_state
                logger.info(f"🔄 市场状态切换: {MARKET_STATE_NAMES[old_state]} → {MARKET_STATE_NAMES[self._market_state]}")
            elif self._state_history[-1] != self._market_state:
                logger.info(f"⏳ 检测到{MARKET_STATE_NAMES[detected_state]}，等待确认（当前仍为{MARKET_STATE_NAMES[self._market_state]}）")
        else:
            # 历史记录不足，直接使用检测结果
            self._market_state = detected_state
        
        # 更新权重
        self._update_weights()
        
        return self._market_state
    
    def _detect_market_state(self, prices: List[float]) -> MarketState:
        """
        根据价格序列判断市场状态（改进版）
        
        判断维度：
        1. 价格位置：当前价格相对MA20/MA60的位置
        2. 均线趋势：MA20相对MA60的位置和变化方向
        3. 波动率：用于调整判断阈值
        4. 动量：近期涨跌幅
        
        Args:
            prices: 价格序列（至少20个，从旧到新）
            
        Returns:
            市场状态
        """
        if len(prices) < 20:
            return MarketState.SIDEWAYS
        
        current_price = prices[-1]
        
        # ===== 1. 计算均线 =====
        ma20 = sum(prices[-20:]) / 20
        
        # 如果有足够数据，计算MA60
        if len(prices) >= 60:
            ma60 = sum(prices[-60:]) / 60
        else:
            ma60 = ma20  # 数据不足时用MA20代替
        
        # ===== 2. 计算MA20趋势（当前MA20 vs 5天前MA20）=====
        if len(prices) >= 25:
            # 5天前的MA20：取prices[-25:-5]的最后20个数据的均值
            ma20_5days_ago = sum(prices[-25:-5]) / 20
            ma20_change = (ma20 - ma20_5days_ago) / ma20_5days_ago * 100
        else:
            ma20_change = 0
        
        # ===== 3. 计算波动率（用于自适应阈值）=====
        if len(prices) >= 20:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] * 100 
                      for i in range(-19, 0)]
            volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
        else:
            volatility = 2.0  # 默认波动率
        
        # 自适应阈值（波动率越高，阈值越宽松）
        base_threshold = 3.0
        adaptive_threshold = max(base_threshold, volatility * 1.5)
        
        # ===== 4. 计算近期动量 =====
        if len(prices) >= 20:
            momentum_20d = (current_price - prices[-20]) / prices[-20] * 100
        else:
            momentum_20d = 0
        
        # ===== 5. 综合评分 =====
        bull_score = 0
        bear_score = 0
        
        # 维度1：价格相对MA20位置（权重30%）
        price_vs_ma20 = (current_price - ma20) / ma20 * 100
        if price_vs_ma20 > adaptive_threshold:
            bull_score += 3
        elif price_vs_ma20 > 0:
            bull_score += 1
        elif price_vs_ma20 < -adaptive_threshold:
            bear_score += 3
        elif price_vs_ma20 < 0:
            bear_score += 1
        
        # 维度2：MA20相对MA60位置（权重25%）
        ma20_vs_ma60 = (ma20 - ma60) / ma60 * 100
        if ma20_vs_ma60 > 2:
            bull_score += 2.5
        elif ma20_vs_ma60 > 0:
            bull_score += 1
        elif ma20_vs_ma60 < -2:
            bear_score += 2.5
        elif ma20_vs_ma60 < 0:
            bear_score += 1
        
        # 维度3：MA20趋势（权重25%）
        if ma20_change > 1:
            bull_score += 2.5
        elif ma20_change > 0:
            bull_score += 1
        elif ma20_change < -1:
            bear_score += 2.5
        elif ma20_change < 0:
            bear_score += 1
        
        # 维度4：20日动量（权重20%）
        if momentum_20d > 5:
            bull_score += 2
        elif momentum_20d > 0:
            bull_score += 1
        elif momentum_20d < -5:
            bear_score += 2
        elif momentum_20d < 0:
            bear_score += 1
        
        # ===== 6. 状态判定 =====
        # 使用分数差而非绝对阈值，更稳健
        score_diff = bull_score - bear_score
        
        if score_diff >= 4:
            state = MarketState.BULL
        elif score_diff <= -4:
            state = MarketState.BEAR
        else:
            state = MarketState.SIDEWAYS
        
        logger.info(f"📊 市场状态检测: "
                   f"价格/MA20={price_vs_ma20:+.1f}%, "
                   f"MA20/MA60={ma20_vs_ma60:+.1f}%, "
                   f"MA20趋势={ma20_change:+.1f}%, "
                   f"20日动量={momentum_20d:+.1f}%, "
                   f"波动率={volatility:.1f}% "
                   f"→ 牛分={bull_score:.1f}, 熊分={bear_score:.1f} "
                   f"→ {MARKET_STATE_NAMES[state]}")
        
        return state
    
    def _update_weights(self):
        """根据市场状态更新评分权重"""
        new_weights = MARKET_WEIGHTS[self._market_state].copy()
        self._config['weights'] = new_weights
        
        logger.info(f"📈 权重已更新: 市场={MARKET_STATE_NAMES[self._market_state]}, "
                   f"动量={new_weights['momentum']}%, "
                   f"成长={new_weights['growth']}%, "
                   f"估值={new_weights['valuation']}%, "
                   f"质量={new_weights['quality']}%, "
                   f"安全={new_weights['safety']}%")
    
    def score(self, stock: StockData) -> ScoreResult:
        """
        计算股票评分（使用动态权重）
        
        与 MomentumV2Strategy.score 的区别：
        - 各维度分数计算逻辑相同
        - 但最终权重根据市场状态动态调整
        
        Args:
            stock: 股票数据
            
        Returns:
            ScoreResult 评分结果
        """
        # 获取当前权重配置
        weights = self._config.get('weights', MARKET_WEIGHTS[MarketState.SIDEWAYS])
        
        # 计算各维度原始分（满分100分制，然后按权重缩放）
        raw_scores = {
            'momentum': self._score_momentum(stock),      # 满分40
            'growth': self._score_growth(stock),          # 满分25
            'valuation': self._score_valuation(stock),    # 满分20
            'quality': self._score_quality(stock),        # 满分10
            'safety': self._score_safety(stock)           # 满分5
        }
        
        # 将原始分转换为百分制
        max_scores = {'momentum': 40, 'growth': 25, 'valuation': 20, 'quality': 10, 'safety': 5}
        normalized = {k: (raw_scores[k] / max_scores[k]) * 100 if max_scores[k] > 0 else 0 
                     for k in raw_scores}
        
        # 按权重计算最终得分
        breakdown = {}
        total = 0
        for key in weights:
            # 按权重分配分数
            weighted_score = normalized.get(key, 0) * weights[key] / 100
            breakdown[key] = round(weighted_score, 1)
            total += weighted_score
        
        total = round(total, 1)
        
        # 风险检查
        risk_flag = False
        profit_growth = stock.profit_growth or 0
        if profit_growth < -30:
            risk_flag = True
            if total > 50:
                total = 50
        
        # 评级
        grade = self._calculate_grade(total)
        
        return ScoreResult(
            total=total,
            breakdown=breakdown,
            grade=grade,
            risk_flag=risk_flag
        )
    
    def select(self, stocks: List[StockData], top_n: int = 10, 
               index_prices: List[float] = None) -> List[StockData]:
        """
        选择股票（支持传入指数价格自动更新市场状态）
        
        Args:
            stocks: 候选股票列表
            top_n: 选择数量
            index_prices: 指数历史价格（可选，用于更新市场状态）
            
        Returns:
            选中的股票列表
        """
        # 如果传入了指数价格，更新市场状态
        if index_prices and len(index_prices) >= 20:
            self.update_market_state(index_prices=index_prices)
        
        logger.info(f"🎯 自适应策略选股: 市场状态={MARKET_STATE_NAMES[self._market_state]}")
        
        # 调用父类的选股逻辑
        return super().select(stocks, top_n)


def detect_market_state_from_returns(monthly_returns: List[float], 
                                     threshold: float = 5.0) -> MarketState:
    """
    根据近期月度收益判断市场状态（辅助函数）
    
    Args:
        monthly_returns: 最近几个月的收益率列表
        threshold: 判断阈值（%）
        
    Returns:
        市场状态
    """
    if not monthly_returns or len(monthly_returns) < 2:
        return MarketState.SIDEWAYS
    
    avg_return = sum(monthly_returns) / len(monthly_returns)
    
    if avg_return > threshold:
        return MarketState.BULL
    elif avg_return < -threshold:
        return MarketState.BEAR
    else:
        return MarketState.SIDEWAYS
