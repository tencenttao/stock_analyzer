# -*- coding: utf-8 -*-
"""
风险指标计算模块

计算回测结果的各种风险指标：
- 夏普比率 (Sharpe Ratio)
- 最大回撤 (Maximum Drawdown)
- 索提诺比率 (Sortino Ratio)
- 信息比率 (Information Ratio)
- 卡尔马比率 (Calmar Ratio)
"""

import logging
import math
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskMetricsResult:
    """风险指标计算结果"""
    sharpe_ratio: float            # 夏普比率
    max_drawdown: float            # 最大回撤 (%)
    max_drawdown_duration: int     # 最大回撤持续期 (月)
    sortino_ratio: float           # 索提诺比率
    calmar_ratio: float            # 卡尔马比率
    information_ratio: float       # 信息比率
    volatility: float              # 年化波动率 (%)
    downside_volatility: float     # 下行波动率 (%)
    win_rate: float                # 胜率 (%)
    profit_loss_ratio: float       # 盈亏比


class RiskMetrics:
    """
    风险指标计算器
    
    使用示例:
        metrics = RiskMetrics()
        result = metrics.calculate(
            returns=[2.5, -1.2, 3.1, ...],  # 月度收益率
            benchmark_returns=[1.0, 0.5, ...],  # 基准收益率
            risk_free_rate=0.02  # 年化无风险利率
        )
        print(f"夏普比率: {result.sharpe_ratio:.2f}")
        print(f"最大回撤: {result.max_drawdown:.2f}%")
    """
    
    def __init__(self, periods_per_year: int = 12):
        """
        初始化
        
        Args:
            periods_per_year: 每年的周期数（月度=12，日度=252）
        """
        self.periods_per_year = periods_per_year
    
    def calculate(self, 
                  returns: List[float],
                  benchmark_returns: List[float] = None,
                  risk_free_rate: float = 0.02) -> RiskMetricsResult:
        """
        计算所有风险指标
        
        Args:
            returns: 策略收益率序列 (%)
            benchmark_returns: 基准收益率序列 (%)
            risk_free_rate: 年化无风险利率 (如 0.02 = 2%)
            
        Returns:
            RiskMetricsResult 包含所有风险指标
        """
        if not returns:
            return self._empty_result()
        
        # 转换为小数形式
        returns_decimal = [r / 100 for r in returns]
        
        # 基准收益
        if benchmark_returns:
            benchmark_decimal = [r / 100 for r in benchmark_returns]
        else:
            benchmark_decimal = [0] * len(returns_decimal)
        
        # 每期无风险利率
        rf_per_period = risk_free_rate / self.periods_per_year
        
        # 计算各项指标
        sharpe = self._sharpe_ratio(returns_decimal, rf_per_period)
        max_dd, max_dd_duration = self._max_drawdown(returns_decimal)
        sortino = self._sortino_ratio(returns_decimal, rf_per_period)
        calmar = self._calmar_ratio(returns_decimal, max_dd)
        info_ratio = self._information_ratio(returns_decimal, benchmark_decimal)
        volatility = self._annualized_volatility(returns_decimal)
        downside_vol = self._downside_volatility(returns_decimal, rf_per_period)
        win_rate = self._win_rate(returns)
        pl_ratio = self._profit_loss_ratio(returns)
        
        return RiskMetricsResult(
            sharpe_ratio=sharpe,
            max_drawdown=max_dd * 100,  # 转回百分比
            max_drawdown_duration=max_dd_duration,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            volatility=volatility * 100,
            downside_volatility=downside_vol * 100,
            win_rate=win_rate,
            profit_loss_ratio=pl_ratio
        )
    
    def _sharpe_ratio(self, returns: List[float], rf_per_period: float) -> float:
        """
        计算夏普比率
        
        Sharpe = (E[R] - Rf) / σ * sqrt(periods_per_year)
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = [r - rf_per_period for r in returns]
        mean_excess = sum(excess_returns) / len(excess_returns)
        
        variance = sum((r - mean_excess) ** 2 for r in excess_returns) / (len(excess_returns) - 1)
        std = math.sqrt(variance) if variance > 0 else 0
        
        if std == 0:
            return 0.0
        
        return (mean_excess / std) * math.sqrt(self.periods_per_year)
    
    def _max_drawdown(self, returns: List[float]) -> tuple:
        """
        计算最大回撤及持续期
        
        Returns:
            (max_drawdown, duration) - 最大回撤和持续月数
        """
        if not returns:
            return 0.0, 0
        
        # 计算累计净值
        cumulative = [1.0]
        for r in returns:
            cumulative.append(cumulative[-1] * (1 + r))
        
        # 计算回撤
        peak = cumulative[0]
        max_dd = 0.0
        max_dd_duration = 0
        current_dd_start = 0
        
        for i, value in enumerate(cumulative):
            if value > peak:
                peak = value
                current_dd_start = i
            
            dd = (peak - value) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_duration = i - current_dd_start
        
        return max_dd, max_dd_duration
    
    def _sortino_ratio(self, returns: List[float], rf_per_period: float) -> float:
        """
        计算索提诺比率
        
        Sortino = (E[R] - Rf) / σ_downside * sqrt(periods_per_year)
        只考虑下行波动
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = [r - rf_per_period for r in returns]
        mean_excess = sum(excess_returns) / len(excess_returns)
        
        # 只计算负收益的波动
        downside = [r for r in excess_returns if r < 0]
        if not downside:
            return float('inf') if mean_excess > 0 else 0.0
        
        downside_variance = sum(r ** 2 for r in downside) / len(downside)
        downside_std = math.sqrt(downside_variance)
        
        if downside_std == 0:
            return 0.0
        
        return (mean_excess / downside_std) * math.sqrt(self.periods_per_year)
    
    def _calmar_ratio(self, returns: List[float], max_dd: float) -> float:
        """
        计算卡尔马比率
        
        Calmar = 年化收益率 / 最大回撤
        """
        if max_dd == 0:
            return 0.0
        
        # 计算年化收益
        cumulative_return = 1.0
        for r in returns:
            cumulative_return *= (1 + r)
        
        years = len(returns) / self.periods_per_year
        if years == 0:
            return 0.0
        
        annual_return = cumulative_return ** (1 / years) - 1
        
        return annual_return / max_dd
    
    def _information_ratio(self, returns: List[float], benchmark: List[float]) -> float:
        """
        计算信息比率
        
        IR = E[R - B] / σ(R - B) * sqrt(periods_per_year)
        """
        if len(returns) < 2 or len(returns) != len(benchmark):
            return 0.0
        
        # 计算超额收益
        excess = [r - b for r, b in zip(returns, benchmark)]
        mean_excess = sum(excess) / len(excess)
        
        variance = sum((e - mean_excess) ** 2 for e in excess) / (len(excess) - 1)
        std = math.sqrt(variance) if variance > 0 else 0
        
        if std == 0:
            return 0.0
        
        return (mean_excess / std) * math.sqrt(self.periods_per_year)
    
    def _annualized_volatility(self, returns: List[float]) -> float:
        """计算年化波动率"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(variance)
        
        return std * math.sqrt(self.periods_per_year)
    
    def _downside_volatility(self, returns: List[float], rf_per_period: float) -> float:
        """计算下行波动率"""
        if len(returns) < 2:
            return 0.0
        
        downside = [r - rf_per_period for r in returns if r < rf_per_period]
        if not downside:
            return 0.0
        
        variance = sum(r ** 2 for r in downside) / len(downside)
        return math.sqrt(variance) * math.sqrt(self.periods_per_year)
    
    def _win_rate(self, returns: List[float]) -> float:
        """计算胜率（盈利周期占比）"""
        if not returns:
            return 0.0
        
        wins = len([r for r in returns if r > 0])
        return (wins / len(returns)) * 100
    
    def _profit_loss_ratio(self, returns: List[float]) -> float:
        """计算盈亏比（平均盈利/平均亏损）"""
        profits = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]
        
        if not profits or not losses:
            return 0.0
        
        avg_profit = sum(profits) / len(profits)
        avg_loss = sum(losses) / len(losses)
        
        if avg_loss == 0:
            return float('inf')
        
        return avg_profit / avg_loss
    
    def _empty_result(self) -> RiskMetricsResult:
        """返回空结果"""
        return RiskMetricsResult(
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            information_ratio=0.0,
            volatility=0.0,
            downside_volatility=0.0,
            win_rate=0.0,
            profit_loss_ratio=0.0
        )
