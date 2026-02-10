# -*- coding: utf-8 -*-
"""
回测引擎核心

提供统一的回测执行入口，整合：
- 数据源
- 选股策略
- 回测模式
- 风险指标
- 交易成本
- 报告生成

所有默认参数从配置文件读取，代码中不包含硬编码默认值。
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from core.interfaces import DataSource, Strategy
from backtest.metrics import RiskMetrics, RiskMetricsResult
from backtest.cost import TradingCost, CostConfig
from backtest.modes.monthly import MonthlyMode, MonthlyConfig, MonthlyResult

# 从配置读取默认值（必须）
from config.settings import BACKTEST_DEFAULTS

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """
    回测配置
    
    统一的配置类，支持各种回测模式。
    默认值全部从 config.settings.BACKTEST_DEFAULTS 读取。
    """
    # 基本配置
    start_date: str                         # 开始日期 (YYYY-MM-DD)
    end_date: str                           # 结束日期 (YYYY-MM-DD)
    initial_capital: float = field(default_factory=lambda: BACKTEST_DEFAULTS['initial_capital'])
    benchmark: str = field(default_factory=lambda: BACKTEST_DEFAULTS['benchmark'])
    
    # 选股配置（候选池=基准指数全部成分股，不采样）
    top_n: int = field(default_factory=lambda: BACKTEST_DEFAULTS['top_n'])
    random_seed: int = field(default_factory=lambda: BACKTEST_DEFAULTS.get('random_seed', 42))
    
    # 交易成本配置
    commission_rate: float = field(default_factory=lambda: BACKTEST_DEFAULTS['commission_rate'])
    stamp_tax_rate: float = field(default_factory=lambda: BACKTEST_DEFAULTS['stamp_tax_rate'])
    slippage: float = field(default_factory=lambda: BACKTEST_DEFAULTS['slippage'])
    enable_cost: bool = field(default_factory=lambda: BACKTEST_DEFAULTS['enable_cost'])
    
    # 风险指标配置
    risk_free_rate: float = 0.02            # 无风险利率


@dataclass
class BacktestResult:
    """
    回测结果
    
    包含完整的回测统计信息
    """
    # 基本信息
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    
    # 收益指标
    total_return: float                     # 总收益率 (%)
    annual_return: float                    # 年化收益率 (%)
    benchmark_return: float                 # 基准收益率 (%)
    alpha: float                            # 超额收益 (%)
    
    # 风险指标
    risk_metrics: Optional[RiskMetricsResult] = None
    
    # 成本
    total_cost: float = 0.0                 # 总交易成本
    
    # 详细数据
    monthly_returns: List[Dict] = field(default_factory=list)   # 月度收益
    trades: List[Dict] = field(default_factory=list)            # 交易记录
    
    # 配置
    config: Dict = field(default_factory=dict)


class BacktestEngine:
    """
    回测引擎
    
    统一的回测执行入口
    
    使用示例:
        from data.manager import DataManager
        from strategy import StrategyRegistry
        from backtest import BacktestEngine, BacktestConfig
        
        # 初始化
        data_source = DataManager(source='tushare')
        strategy = StrategyRegistry.create('momentum_v2')
        config = BacktestConfig(
            start_date='2024-01-01',
            end_date='2024-12-31',
            initial_capital=100000
        )
        
        # 创建引擎
        engine = BacktestEngine(data_source, strategy, config)
        
        # 执行月度回测
        result = engine.run_monthly()
        
        # 查看结果
        print(f"总收益: {result.total_return:.2f}%")
        print(f"夏普比率: {result.risk_metrics.sharpe_ratio:.2f}")
    """
    
    def __init__(self,
                 data_source: DataSource,
                 strategy: Strategy,
                 config: BacktestConfig):
        """
        初始化回测引擎
        
        Args:
            data_source: 数据源
            strategy: 选股策略
            config: 回测配置
        """
        self.data_source = data_source
        self.strategy = strategy
        self.config = config
        
        # 初始化组件
        self.risk_calculator = RiskMetrics(periods_per_year=12)
        
        if config.enable_cost:
            cost_config = CostConfig(
                commission_rate=config.commission_rate,
                stamp_tax_rate=config.stamp_tax_rate,
                slippage=config.slippage
            )
            self.cost_calculator = TradingCost(cost_config)
        else:
            self.cost_calculator = None
    
    def run_monthly(self) -> BacktestResult:
        """
        执行月度轮换回测
        
        Returns:
            BacktestResult 回测结果
        """
        logger.info("🚀 开始月度轮换回测...")
        
        # 构建月度配置
        monthly_config = MonthlyConfig(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=self.config.initial_capital,
            top_n=self.config.top_n,
            random_seed=self.config.random_seed,
            benchmark=self.config.benchmark
        )
        
        # 执行月度回测
        mode = MonthlyMode(self.data_source, self.strategy, monthly_config)
        monthly_results = mode.run()
        
        if not monthly_results:
            logger.error("❌ 回测失败，没有有效结果")
            return self._empty_result()
        
        # 整合结果
        return self._aggregate_monthly_results(monthly_results)
    
    def _aggregate_monthly_results(self, monthly_results: List[MonthlyResult]) -> BacktestResult:
        """整合月度结果"""
        # 提取收益序列
        returns = [r.return_pct for r in monthly_results]
        benchmark_returns = [r.benchmark_return for r in monthly_results]
        
        # 计算总收益
        final_value = monthly_results[-1].portfolio_value
        final_benchmark = monthly_results[-1].benchmark_value
        
        total_return = (final_value / self.config.initial_capital - 1) * 100
        benchmark_return = (final_benchmark / self.config.initial_capital - 1) * 100
        
        # 计算年化收益
        months = len(monthly_results)
        years = months / 12
        if years > 0:
            annual_return = ((final_value / self.config.initial_capital) ** (1 / years) - 1) * 100
        else:
            annual_return = total_return
        
        # 计算风险指标
        risk_metrics = self.risk_calculator.calculate(
            returns=returns,
            benchmark_returns=benchmark_returns,
            risk_free_rate=self.config.risk_free_rate
        )
        
        # 应用交易成本
        total_cost = 0.0
        if self.cost_calculator:
            # 每月调仓算一次往返交易
            cost_rate = self.cost_calculator.round_trip_cost_rate()
            total_cost = self.config.initial_capital * cost_rate * months
            
            # 调整收益
            adjusted_final = final_value - total_cost
            total_return = (adjusted_final / self.config.initial_capital - 1) * 100
        
        # 整理交易记录
        all_trades = []
        for r in monthly_results:
            all_trades.extend(r.trades)
        
        # 整理月度收益
        monthly_data = []
        for r in monthly_results:
            monthly_data.append({
                'month': r.month,
                'buy_date': r.buy_date,
                'sell_date': r.sell_date,
                'return_pct': r.return_pct,
                'benchmark_return': r.benchmark_return,
                'alpha': r.alpha,
                'portfolio_value': r.portfolio_value,
                'trades_count': r.successful_trades
            })
        
        return BacktestResult(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=self.config.initial_capital,
            final_value=final_value,
            total_return=total_return,
            annual_return=annual_return,
            benchmark_return=benchmark_return,
            alpha=total_return - benchmark_return,
            risk_metrics=risk_metrics,
            total_cost=total_cost,
            monthly_returns=monthly_data,
            trades=all_trades,
            config={
                'strategy': self.strategy.name,
                'top_n': self.config.top_n,
                'enable_cost': self.config.enable_cost,
                'commission_rate': self.config.commission_rate,
                'benchmark': self.config.benchmark
            }
        )
    
    def _empty_result(self) -> BacktestResult:
        """返回空结果"""
        return BacktestResult(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=self.config.initial_capital,
            final_value=self.config.initial_capital,
            total_return=0.0,
            annual_return=0.0,
            benchmark_return=0.0,
            alpha=0.0,
            risk_metrics=None,
            total_cost=0.0,
            monthly_returns=[],
            trades=[],
            config={}
        )
    
    def compare_strategies(self, 
                           strategies: List[Strategy],
                           names: List[str] = None) -> List[BacktestResult]:
        """
        对比多个策略
        
        Args:
            strategies: 策略列表
            names: 策略名称列表
            
        Returns:
            每个策略的回测结果列表
        """
        if names is None:
            names = [s.name for s in strategies]
        
        results = []
        
        for i, strategy in enumerate(strategies):
            logger.info(f"\n{'='*70}")
            logger.info(f"📊 策略 {i+1}/{len(strategies)}: {names[i]}")
            logger.info(f"{'='*70}")
            
            # 临时替换策略
            original_strategy = self.strategy
            self.strategy = strategy
            
            try:
                result = self.run_monthly()
                results.append(result)
            finally:
                self.strategy = original_strategy
        
        return results
