# -*- coding: utf-8 -*-
"""
动量优先策略 V2

评分体系（100分）- 针对月度轮换策略优化：
- 动量/趋势 (40分): 20日动量(25) + 涨跌幅(10) + 成交活跃度(5)  ← 核心指标
- 成长性 (25分): 利润增长(15) + ROE(10)  ← 重视增长
- 估值 (20分): PE(8) + PB(7) + PEG(5)  ← 降低权重，避免价值陷阱
- 质量 (10分): ROE质量(6) + 换手率适中(4)
- 安全性 (5分): 股息率(3) + 风险控制(2)  ← 大幅降低

设计理念：
1. 动量优先：追涨强势股，顺势而为
2. 成长为王：高增长比低估值更重要
3. 减少价值陷阱：不过度偏好低PE/低PB
4. 风险可控：通过预筛选过滤高风险股
"""

import logging
from typing import Dict, List, Any, Optional

from core.interfaces import Strategy
from core.types import StockData, ScoreResult
from strategy.registry import register_strategy

logger = logging.getLogger(__name__)


# 默认配置
DEFAULT_CONFIG = {
    # 权重配置
    'weights': {
        'momentum': 40,   # 动量/趋势
        'growth': 25,     # 成长性
        'valuation': 20,  # 估值
        'quality': 10,    # 质量
        'safety': 5       # 安全性
    },
    # 筛选配置
    'min_price': 2.0,           # 最低股价
    'min_score': 35,            # 最低分数
    'max_stocks': 10,           # 最大选股数量
    'use_dynamic_threshold': False,  # 是否使用动态阈值
}


@register_strategy('momentum_v2', '动量优先策略V2 - 适合月度轮换')
class MomentumV2Strategy(Strategy):
    """
    动量优先选股策略 V2
    
    特点：
    - 动量为核心（40%权重）
    - 重视成长性（25%权重）
    - 适度考虑估值（20%权重）
    - 适合月度轮换回测
    
    使用示例:
        strategy = MomentumV2Strategy()
        selected = strategy.select(stocks, top_n=10)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化策略
        
        Args:
            config: 策略配置，可覆盖默认配置
        """
        # 合并配置
        merged_config = DEFAULT_CONFIG.copy()
        if config:
            merged_config.update(config)
        
        super().__init__(merged_config)
    
    @property
    def name(self) -> str:
        return "momentum_v2"
    
    @property
    def description(self) -> str:
        return "动量优先策略V2 - 40%动量 + 25%成长 + 20%估值 + 10%质量 + 5%安全"
    
    def score(self, stock: StockData) -> ScoreResult:
        """
        计算股票评分
        
        Args:
            stock: 股票数据
            
        Returns:
            ScoreResult 评分结果
        """
        breakdown = {
            'momentum': 0,       # 动量/趋势 (40分)
            'growth': 0,         # 成长性 (25分)
            'valuation': 0,      # 估值 (20分)
            'quality': 0,        # 质量 (10分)
            'safety': 0          # 安全性 (5分)
        }
        
        risk_flag = False
        
        # ===== 风险预检查 =====
        profit_growth = stock.profit_growth or 0
        
        # 业绩暴雷预警：利润增长 < -30%
        if profit_growth < -30:
            risk_flag = True
        
        # ===== 1. 动量/趋势得分 (40分) =====
        breakdown['momentum'] = self._score_momentum(stock)
        
        # ===== 2. 成长性得分 (25分) =====
        breakdown['growth'] = self._score_growth(stock)
        
        # ===== 3. 估值得分 (20分) =====
        breakdown['valuation'] = self._score_valuation(stock)
        
        # ===== 4. 质量得分 (10分) =====
        breakdown['quality'] = self._score_quality(stock)
        
        # ===== 5. 安全性得分 (5分) =====
        breakdown['safety'] = self._score_safety(stock)
        
        # 计算总分
        total = sum(breakdown.values())
        
        # 风险惩罚：业绩暴雷时总分上限50分
        if risk_flag and total > 50:
            total = 50
        
        # 评级
        grade = self._calculate_grade(total)
        
        return ScoreResult(
            total=total,
            breakdown=breakdown,
            grade=grade,
            risk_flag=risk_flag
        )
    
    def _score_momentum(self, stock: StockData) -> int:
        """计算动量得分 (40分满分)"""
        score = 0
        
        # 1.1 20日动量 (25分)
        momentum = stock.momentum_20d or 0
        
        if momentum > 25:
            score += 25
        elif momentum > 20:
            score += 22
        elif momentum > 15:
            score += 18
        elif momentum > 10:
            score += 14
        elif momentum > 5:
            score += 10
        elif momentum > 0:
            score += 5
        elif momentum > -5:
            score += 2
        
        # 1.2 当日涨跌幅 (10分)
        change_pct = stock.change_pct or 0
        
        if change_pct > 7:
            score += 10
        elif change_pct > 5:
            score += 8
        elif change_pct > 3:
            score += 6
        elif change_pct > 1:
            score += 4
        elif change_pct > 0:
            score += 2
        elif change_pct > -2:
            score += 1
        
        # 1.3 成交活跃度 (5分)
        turnover_rate = stock.turnover_rate or 0
        
        if 2 <= turnover_rate < 5:
            score += 5
        elif 1 <= turnover_rate < 2:
            score += 4
        elif 5 <= turnover_rate < 8:
            score += 3
        elif 0.5 <= turnover_rate < 1:
            score += 2
        elif turnover_rate >= 8:
            score += 1
        
        return score
    
    def _score_growth(self, stock: StockData) -> int:
        """计算成长性得分 (25分满分)"""
        score = 0
        
        # 2.1 净利润增长率 (15分)
        profit_growth = stock.profit_growth or 0
        
        if profit_growth > 50:
            score += 15
        elif profit_growth > 30:
            score += 12
        elif profit_growth > 20:
            score += 10
        elif profit_growth > 10:
            score += 7
        elif profit_growth > 0:
            score += 4
        elif profit_growth > -10:
            score += 1
        
        # 2.2 ROE (10分)
        roe = stock.roe or 0
        
        if roe > 25:
            score += 10
        elif roe > 20:
            score += 8
        elif roe > 15:
            score += 6
        elif roe > 10:
            score += 4
        elif roe > 5:
            score += 2
        
        return score
    
    def _score_valuation(self, stock: StockData) -> int:
        """计算估值得分 (20分满分)"""
        score = 0
        
        # 3.1 PE估值 (8分)
        pe = stock.pe_ratio or 0
        
        if pe and 10 <= pe < 25:
            score += 8
        elif pe and 5 <= pe < 10:
            score += 6
        elif pe and 25 <= pe < 40:
            score += 4
        elif pe and 0 < pe < 5:
            score += 3
        elif pe and 40 <= pe < 60:
            score += 2
        
        # 3.2 PB估值 (7分)
        pb = stock.pb_ratio or 0
        
        if pb and 1 <= pb < 3:
            score += 7
        elif pb and 3 <= pb < 5:
            score += 5
        elif pb and 0.5 <= pb < 1:
            score += 4
        elif pb and 5 <= pb < 8:
            score += 3
        elif pb and 0 < pb < 0.5:
            score += 2
        elif pb and 8 <= pb < 12:
            score += 1
        
        # 3.3 PEG (5分)
        peg = stock.peg or 0
        
        if peg and 0 < peg < 0.5:
            score += 5
        elif peg and 0.5 <= peg < 1:
            score += 4
        elif peg and 1 <= peg < 1.5:
            score += 2
        elif peg and 1.5 <= peg < 2:
            score += 1
        
        return score
    
    def _score_quality(self, stock: StockData) -> int:
        """计算质量得分 (10分满分)"""
        score = 0
        
        # 4.1 ROE质量 (6分)
        roe = stock.roe or 0
        
        if roe > 20:
            score += 6
        elif roe > 15:
            score += 4
        elif roe > 10:
            score += 2
        
        # 4.2 换手率适中 (4分) - 筹码结构
        # 最佳区间 1-3%，其次 3-5%，再次 0.5-1%
        turnover_rate = stock.turnover_rate or 0
        
        if 1 <= turnover_rate < 3:
            score += 4
        elif 3 <= turnover_rate < 5:
            score += 3
        elif 0.5 <= turnover_rate < 1:
            score += 2
        elif 5 <= turnover_rate < 8:
            score += 1
        # turnover_rate >= 8 或 < 0.5 不得分
        
        return score
    
    def _score_safety(self, stock: StockData) -> int:
        """计算安全性得分 (5分满分)"""
        score = 0
        
        # 5.1 股息率 (3分)
        dividend_yield = stock.dividend_yield or 0
        
        if dividend_yield > 4:
            score += 3
        elif dividend_yield > 2:
            score += 2
        elif dividend_yield > 1:
            score += 1
        
        # 5.2 风险控制 (2分)
        # 低换手+正涨幅 = 稳健
        turnover_rate = stock.turnover_rate or 0
        change_pct = stock.change_pct or 0
        
        if turnover_rate < 3 and change_pct > 0:
            score += 2
        elif turnover_rate < 5 and change_pct >= 0:
            score += 1
        
        return score
    
    def _calculate_grade(self, total: float) -> str:
        """计算评级"""
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
        预筛选：判断股票是否满足基本条件
        
        所有过滤条件从配置读取，支持的配置项：
        - min_price: 最低股价
        - max_pe: PE上限
        - max_pb: PB上限
        - max_momentum_20d: 20日动量上限（避免追高）
        - min_momentum_20d: 20日动量下限
        - max_change_pct: 买入日涨幅上限（避免追涨）
        - max_turnover_rate: 换手率上限（筹码稳定）
        - min_turnover_rate: 换手率下限
        - min_dividend_yield: 最低股息率
        
        Args:
            stock: 股票数据
            
        Returns:
            True表示通过筛选
        """
        filters = self._config.get('filters', {})
        
        # 1. 排除亏损股（PE < 0）
        if stock.pe_ratio is not None and stock.pe_ratio < 0:
            return False
        
        # 2. 排除停牌股票
        if stock.change_pct == 0 and (stock.turnover_rate is None or stock.turnover_rate < 0.1):
            return False
        
        # 3. 排除跌停股票
        if stock.change_pct is not None and stock.change_pct <= -9.8:
            return False
        
        # 4. 排除仙股
        min_price = filters.get('min_price', self._config.get('min_price', 2.0))
        if stock.price < min_price:
            return False
        
        # 5. PE上限过滤
        max_pe = filters.get('max_pe')
        if max_pe and stock.pe_ratio and stock.pe_ratio > max_pe:
            return False
        
        # 6. PB上限过滤
        max_pb = filters.get('max_pb')
        if max_pb and stock.pb_ratio and stock.pb_ratio > max_pb:
            return False
        
        # 7. 动量过滤（避免追高/追跌）
        max_momentum = filters.get('max_momentum_20d')
        if max_momentum is not None and stock.momentum_20d and stock.momentum_20d > max_momentum:
            return False
        
        min_momentum = filters.get('min_momentum_20d')
        if min_momentum is not None and stock.momentum_20d and stock.momentum_20d < min_momentum:
            return False
        
        # 8. 买入日涨幅过滤（避免追涨）
        max_change = filters.get('max_change_pct')
        if max_change is not None and stock.change_pct and stock.change_pct > max_change:
            return False
        
        # 9. 换手率过滤
        max_turnover = filters.get('max_turnover_rate')
        if max_turnover and stock.turnover_rate and stock.turnover_rate > max_turnover:
            return False
        
        min_turnover = filters.get('min_turnover_rate')
        if min_turnover and stock.turnover_rate and stock.turnover_rate < min_turnover:
            return False
        
        # 10. 股息率过滤
        min_dividend = filters.get('min_dividend_yield')
        if min_dividend and (stock.dividend_yield is None or stock.dividend_yield < min_dividend):
            return False
        
        return True
    
    def select(self, stocks: List[StockData], top_n: int = 10) -> List[StockData]:
        """
        选择最终的推荐股票
        
        Args:
            stocks: 候选股票列表
            top_n: 选择数量
            
        Returns:
            选中的股票列表（已排序、已评分）
        """
        if not stocks:
            return []
        
        # 1. 去重
        unique_stocks = {}
        for stock in stocks:
            if stock.code not in unique_stocks:
                unique_stocks[stock.code] = stock
        stocks = list(unique_stocks.values())
        
        logger.info(f"去重后股票数量: {len(stocks)}")
        
        # 2. 计算评分
        logger.info("📊 第1步：计算所有股票的评分...")
        for stock in stocks:
            score_result = self.score(stock)
            stock.strength_score = score_result.total
            stock.strength_grade = score_result.grade
            stock.score_breakdown = score_result.breakdown
        
        # 统计评分分布
        scores = [s.strength_score for s in stocks]
        if scores:
            avg_score = sum(scores) / len(scores)
            logger.info(f"   评分分布: 平均={avg_score:.1f}, 最高={max(scores):.1f}, 最低={min(scores):.1f}")
        
        # 3. 硬性过滤
        logger.info("📊 第2步：应用硬性过滤条件...")
        filtered = [s for s in stocks if self.filter(s)]
        logger.info(f"   硬性过滤后: {len(filtered)} 只股票")
        
        if not filtered:
            return []
        
        # 4. 分数阈值筛选
        logger.info("📊 第3步：应用评分阈值筛选...")
        min_score = self._config.get('min_score', 35)
        filtered = [s for s in filtered if s.strength_score >= min_score]
        logger.info(f"   评分筛选后: {len(filtered)} 只股票 (阈值: {min_score}分)")
        
        if not filtered:
            return []
        
        # 5. 按分数排序
        filtered.sort(key=lambda x: x.strength_score, reverse=True)
        
        # 6. 选择前N只
        max_stocks = self._config.get('max_stocks', top_n)
        final_n = min(top_n, max_stocks)
        selected = filtered[:final_n]
        
        # 7. 添加排名和选择理由
        for i, stock in enumerate(selected):
            stock.rank = i + 1
            stock.selection_reason = self._generate_reason(stock)
        
        logger.info(f"✅ 最终选择 {len(selected)} 只股票")
        
        # 打印选中股票评分明细
        if selected:
            logger.info("📋 选中股票评分明细:")
            for stock in selected[:5]:
                breakdown = stock.score_breakdown
                risk_mark = " ⚠️风险" if self.score(stock).risk_flag else ""
                logger.info(f"   {stock.name}({stock.code}): "
                           f"总分={stock.strength_score:.0f}{risk_mark} "
                           f"[动量={breakdown.get('momentum', 0)}, "
                           f"成长={breakdown.get('growth', 0)}, "
                           f"估值={breakdown.get('valuation', 0)}, "
                           f"质量={breakdown.get('quality', 0)}, "
                           f"安全={breakdown.get('safety', 0)}]")
        
        return selected
    
    def _generate_reason(self, stock: StockData) -> str:
        """生成选择理由"""
        reasons = []
        
        breakdown = stock.score_breakdown or {}
        
        # 动量亮点
        momentum_score = breakdown.get('momentum', 0)
        if momentum_score >= 30:
            reasons.append(f"🚀强势动量({stock.momentum_20d:.1f}%)")
        elif momentum_score >= 20:
            reasons.append(f"📈趋势向上({stock.momentum_20d:.1f}%)")
        
        # 成长亮点
        growth_score = breakdown.get('growth', 0)
        if growth_score >= 20:
            reasons.append(f"🌱高成长(增长{stock.profit_growth:.1f}%)")
        
        # 估值亮点
        valuation_score = breakdown.get('valuation', 0)
        if valuation_score >= 15:
            reasons.append(f"💰估值合理(PE={stock.pe_ratio:.1f})")
        
        # 质量亮点
        if stock.roe and stock.roe > 15:
            reasons.append(f"⭐优质(ROE={stock.roe:.1f}%)")
        
        # 组合理由
        if reasons:
            return ", ".join(reasons)
        else:
            return f"综合评分{stock.strength_score:.0f}分"
