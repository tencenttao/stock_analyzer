# -*- coding: utf-8 -*-
"""
交易成本模拟模块

模拟真实交易中的各种成本：
- 佣金（买卖双向）
- 印花税（卖出单向）
- 滑点（买卖双向）
- 过户费（沪市股票）

所有默认参数从配置文件读取，代码中不包含硬编码默认值。
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

# 从配置读取默认值（必须）
from config.settings import BACKTEST_DEFAULTS

logger = logging.getLogger(__name__)


@dataclass
class CostConfig:
    """交易成本配置，默认值从 config.settings.BACKTEST_DEFAULTS 读取"""
    commission_rate: float = field(default_factory=lambda: BACKTEST_DEFAULTS['commission_rate'])
    commission_min: float = field(default_factory=lambda: BACKTEST_DEFAULTS['min_commission'])
    stamp_tax_rate: float = field(default_factory=lambda: BACKTEST_DEFAULTS['stamp_tax_rate'])
    slippage: float = field(default_factory=lambda: BACKTEST_DEFAULTS['slippage'])
    transfer_fee_rate: float = field(default_factory=lambda: BACKTEST_DEFAULTS['transfer_fee_rate'])


@dataclass
class TradeCost:
    """单笔交易成本明细"""
    commission: float     # 佣金
    stamp_tax: float      # 印花税
    slippage_cost: float  # 滑点成本
    transfer_fee: float   # 过户费
    total: float          # 总成本
    cost_rate: float      # 成本率（相对交易金额）


class TradingCost:
    """
    交易成本计算器
    
    使用示例:
        cost_calc = TradingCost()
        
        # 计算买入成本
        buy_cost = cost_calc.calculate_buy_cost(
            price=50.0, 
            shares=1000, 
            market='SH'
        )
        print(f"买入成本: {buy_cost.total:.2f}元")
        
        # 计算卖出成本
        sell_cost = cost_calc.calculate_sell_cost(
            price=55.0, 
            shares=1000, 
            market='SH'
        )
        print(f"卖出成本: {sell_cost.total:.2f}元")
        
        # 计算往返成本率
        round_trip = cost_calc.round_trip_cost_rate()
        print(f"往返成本率: {round_trip*100:.3f}%")
    """
    
    def __init__(self, config: CostConfig = None):
        """
        初始化
        
        Args:
            config: 成本配置，不传则使用默认配置
        """
        self.config = config or CostConfig()
    
    def calculate_buy_cost(self, 
                           price: float, 
                           shares: int,
                           market: str = 'SH') -> TradeCost:
        """
        计算买入成本
        
        买入时产生的成本：
        - 佣金（双向）
        - 滑点（向上滑）
        - 过户费（仅沪市）
        
        Args:
            price: 买入价格
            shares: 股数
            market: 市场 ('SH'=沪市, 'SZ'=深市)
            
        Returns:
            TradeCost 成本明细
        """
        trade_value = price * shares
        
        # 佣金
        commission = max(
            trade_value * self.config.commission_rate,
            self.config.commission_min
        )
        
        # 印花税（买入不收）
        stamp_tax = 0.0
        
        # 滑点（买入向上滑）
        slippage_cost = trade_value * self.config.slippage
        
        # 过户费（仅沪市）
        transfer_fee = 0.0
        if market.upper() == 'SH':
            transfer_fee = trade_value * self.config.transfer_fee_rate
        
        total = commission + stamp_tax + slippage_cost + transfer_fee
        cost_rate = total / trade_value if trade_value > 0 else 0
        
        return TradeCost(
            commission=commission,
            stamp_tax=stamp_tax,
            slippage_cost=slippage_cost,
            transfer_fee=transfer_fee,
            total=total,
            cost_rate=cost_rate
        )
    
    def calculate_sell_cost(self, 
                            price: float, 
                            shares: int,
                            market: str = 'SH') -> TradeCost:
        """
        计算卖出成本
        
        卖出时产生的成本：
        - 佣金（双向）
        - 印花税（单向，千分之一）
        - 滑点（向下滑）
        - 过户费（仅沪市）
        
        Args:
            price: 卖出价格
            shares: 股数
            market: 市场 ('SH'=沪市, 'SZ'=深市)
            
        Returns:
            TradeCost 成本明细
        """
        trade_value = price * shares
        
        # 佣金
        commission = max(
            trade_value * self.config.commission_rate,
            self.config.commission_min
        )
        
        # 印花税（卖出收取）
        stamp_tax = trade_value * self.config.stamp_tax_rate
        
        # 滑点（卖出向下滑）
        slippage_cost = trade_value * self.config.slippage
        
        # 过户费（仅沪市）
        transfer_fee = 0.0
        if market.upper() == 'SH':
            transfer_fee = trade_value * self.config.transfer_fee_rate
        
        total = commission + stamp_tax + slippage_cost + transfer_fee
        cost_rate = total / trade_value if trade_value > 0 else 0
        
        return TradeCost(
            commission=commission,
            stamp_tax=stamp_tax,
            slippage_cost=slippage_cost,
            transfer_fee=transfer_fee,
            total=total,
            cost_rate=cost_rate
        )
    
    def round_trip_cost_rate(self, market: str = 'SH') -> float:
        """
        计算往返交易成本率
        
        Args:
            market: 市场
            
        Returns:
            往返成本率（如 0.003 表示 0.3%）
        """
        # 买入成本
        buy_rate = (
            self.config.commission_rate +
            self.config.slippage +
            (self.config.transfer_fee_rate if market.upper() == 'SH' else 0)
        )
        
        # 卖出成本
        sell_rate = (
            self.config.commission_rate +
            self.config.stamp_tax_rate +
            self.config.slippage +
            (self.config.transfer_fee_rate if market.upper() == 'SH' else 0)
        )
        
        return buy_rate + sell_rate
    
    def apply_cost_to_return(self, 
                             gross_return: float,
                             num_trades: int = 1) -> float:
        """
        将成本应用到收益率
        
        Args:
            gross_return: 毛收益率 (%)
            num_trades: 交易次数（每次包含买入+卖出）
            
        Returns:
            扣除成本后的净收益率 (%)
        """
        cost_rate = self.round_trip_cost_rate() * num_trades
        net_return = gross_return - (cost_rate * 100)
        return net_return
    
    def estimate_breakeven_return(self, num_trades: int = 1) -> float:
        """
        估算盈亏平衡所需的收益率
        
        Args:
            num_trades: 交易次数
            
        Returns:
            盈亏平衡收益率 (%)
        """
        cost_rate = self.round_trip_cost_rate() * num_trades
        return cost_rate * 100
    
    def print_cost_summary(self):
        """打印成本配置摘要"""
        logger.info("📊 交易成本配置:")
        logger.info(f"   • 佣金率: {self.config.commission_rate*10000:.1f}‱ (万分之)")
        logger.info(f"   • 最低佣金: ¥{self.config.commission_min:.0f}")
        logger.info(f"   • 印花税: {self.config.stamp_tax_rate*1000:.1f}‰ (千分之，卖出)")
        logger.info(f"   • 滑点: {self.config.slippage*100:.2f}%")
        logger.info(f"   • 过户费: {self.config.transfer_fee_rate*10000:.1f}‱ (沪市)")
        logger.info(f"   • 往返成本: {self.round_trip_cost_rate()*100:.3f}%")
