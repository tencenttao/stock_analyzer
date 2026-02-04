# -*- coding: utf-8 -*-
"""
月度轮换回测模式

策略：
- 每月第一个交易日选股
- 持有至下月第一个交易日
- 月度调仓
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from core.interfaces import DataSource, Strategy
from core.types import StockData

logger = logging.getLogger(__name__)


@dataclass
class MonthlyConfig:
    """月度回测配置"""
    start_date: str                     # 开始日期 (YYYY-MM-DD)
    end_date: str                       # 结束日期 (YYYY-MM-DD)
    initial_capital: float = 100000     # 初始资金
    top_n: int = 10                     # 每月选股数量
    sample_size: int = 100              # 每月采样股票数
    random_seed: int = 42               # 随机种子
    benchmark: str = '000300'           # 基准指数


@dataclass
class MonthlyResult:
    """单月回测结果"""
    month: int                      # 第几个月
    buy_date: str                   # 买入日期
    sell_date: str                  # 卖出日期
    hold_days: int                  # 持有天数
    selected_stocks: int            # 选中股票数
    successful_trades: int          # 成功交易数
    return_pct: float               # 组合收益率 (%)
    benchmark_return: float         # 基准收益率 (%)
    alpha: float                    # 超额收益 (%)
    portfolio_value: float          # 组合价值
    benchmark_value: float          # 基准价值
    trades: List[Dict]              # 交易明细
    best_stock: float               # 最佳个股收益
    worst_stock: float              # 最差个股收益


class MonthlyMode:
    """
    月度轮换回测模式
    
    使用示例:
        mode = MonthlyMode(data_source, strategy, config)
        results = mode.run()
    """
    
    def __init__(self, 
                 data_source: DataSource,
                 strategy: Strategy,
                 config: MonthlyConfig):
        """
        初始化
        
        Args:
            data_source: 数据源
            strategy: 选股策略
            config: 回测配置
        """
        self.data_source = data_source
        self.strategy = strategy
        self.config = config
    
    def run(self) -> List[MonthlyResult]:
        """
        执行月度轮换回测
        
        Returns:
            月度结果列表
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"📅 月度轮换策略回测: {self.config.start_date} ~ {self.config.end_date}")
        logger.info(f"{'='*70}")
        
        # 生成所有月份的第一个交易日
        monthly_dates = self._get_first_trading_days()
        
        if len(monthly_dates) < 2:
            logger.error("❌ 至少需要2个月的数据进行回测")
            return []
        
        logger.info(f"📊 回测周期: {len(monthly_dates)-1}个月")
        logger.info(f"   调仓日期: {', '.join(monthly_dates[:5])}" + 
                   (f" ... {monthly_dates[-1]}" if len(monthly_dates) > 5 else ""))
        
        monthly_results = []
        portfolio_value = self.config.initial_capital
        benchmark_value = self.config.initial_capital
        
        for i in range(len(monthly_dates) - 1):
            buy_date = monthly_dates[i]
            sell_date = monthly_dates[i + 1]
            
            result = self._run_single_month(
                month_index=i + 1,
                buy_date=buy_date,
                sell_date=sell_date,
                portfolio_value=portfolio_value,
                benchmark_value=benchmark_value
            )
            
            if result:
                monthly_results.append(result)
                portfolio_value = result.portfolio_value
                benchmark_value = result.benchmark_value
            
            time.sleep(0.5)  # 避免请求过快
        
        return monthly_results
    
    def _run_single_month(self,
                          month_index: int,
                          buy_date: str,
                          sell_date: str,
                          portfolio_value: float,
                          benchmark_value: float) -> Optional[MonthlyResult]:
        """执行单月回测"""
        hold_days = self._count_trading_days(buy_date, sell_date)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📅 第{month_index}个月: {buy_date} → {sell_date}")
        logger.info(f"   持有期: {hold_days}个交易日")
        logger.info(f"{'='*70}")
        
        # 步骤1: 获取股票池
        stock_list = self.data_source.get_index_constituents(
            self.config.benchmark, 
            buy_date
        )
        
        if not stock_list:
            logger.error("无法获取指数成分股列表")
            return None
        
        # 采样
        import random
        random.seed(self.config.random_seed + month_index)
        if self.config.sample_size < len(stock_list):
            sampled_stocks = random.sample(stock_list, self.config.sample_size)
        else:
            sampled_stocks = stock_list
        
        # 步骤2: 获取买入日数据
        logger.info(f"📊 获取 {buy_date} 的股票数据...")
        
        stock_data_list = []
        for j, code in enumerate(sampled_stocks):
            if j % 20 == 0:
                logger.info(f"   进度: {j+1}/{len(sampled_stocks)}")
            
            stock_data = self.data_source.get_stock_data(code, buy_date)
            if stock_data:
                stock_data_list.append(stock_data)
            
            if j % 20 == 19:
                time.sleep(0.3)
        
        if len(stock_data_list) < 10:
            logger.warning(f"⚠️  {buy_date} 数据不足，跳过本月")
            return None
        
        # 步骤3: 使用策略选股
        logger.info(f"\n🔍 使用策略选股...")
        
        # 如果是自适应策略，先更新市场状态
        if hasattr(self.strategy, 'update_market_state'):
            index_prices = self._get_index_prices_for_state(buy_date)
            if index_prices:
                self.strategy.update_market_state(index_prices=index_prices)
        
        # ML 策略：按月份切换模型（若配置了 model_schedule）
        if hasattr(self.strategy, 'set_current_date'):
            self.strategy.set_current_date(buy_date)
        
        selected_stocks = self.strategy.select(stock_data_list, top_n=self.config.top_n)
        
        if not selected_stocks:
            logger.warning(f"⚠️  {buy_date} 未能选出股票，跳过本月")
            return None
        
        logger.info(f"\n🏆 选出 {len(selected_stocks)} 只股票:")
        for stock in selected_stocks[:5]:
            logger.info(f"   • {stock.name} ({stock.code}): "
                       f"¥{stock.price:.2f}, 分数={stock.strength_score:.0f}")
        
        # 步骤4: 计算月度收益
        logger.info(f"\n💰 计算月度收益...")
        month_returns = []
        successful_trades = []
        
        for stock in selected_stocks:
            sell_data = self.data_source.get_stock_data(stock.code, sell_date)
            if sell_data:
                buy_price = stock.price
                sell_price = sell_data.price
                return_pct = (sell_price / buy_price - 1) * 100
                month_returns.append(return_pct)
                
                successful_trades.append({
                    'code': stock.code,
                    'name': stock.name,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'return_pct': return_pct
                })
                
                emoji = "📈" if return_pct > 0 else "📉" if return_pct < 0 else "➖"
                logger.info(f"   {emoji} {stock.name}: "
                           f"¥{buy_price:.2f} → ¥{sell_price:.2f} ({return_pct:+.2f}%)")
        
        if not month_returns:
            logger.warning(f"⚠️  无法计算收益，跳过本月")
            return None
        
        # 计算平均收益
        avg_return = sum(month_returns) / len(month_returns)
        
        # 步骤5: 获取基准收益
        benchmark_return = self.data_source.get_index_return(
            self.config.benchmark, buy_date, sell_date
        )
        
        # 步骤6: 更新组合价值
        new_portfolio_value = portfolio_value * (1 + avg_return / 100)
        new_benchmark_value = benchmark_value * (1 + benchmark_return / 100)
        
        alpha = avg_return - benchmark_return
        
        # 输出月度统计
        logger.info(f"\n📊 本月统计:")
        logger.info(f"   • 策略收益: {avg_return:+.2f}%")
        logger.info(f"   • 沪深300: {benchmark_return:+.2f}%")
        logger.info(f"   • 超额收益: {alpha:+.2f}% {'✅' if alpha > 0 else '❌'}")
        logger.info(f"   • 组合价值: ¥{new_portfolio_value:,.2f}")
        logger.info(f"   • 基准价值: ¥{new_benchmark_value:,.2f}")
        
        return MonthlyResult(
            month=month_index,
            buy_date=buy_date,
            sell_date=sell_date,
            hold_days=hold_days,
            selected_stocks=len(selected_stocks),
            successful_trades=len(successful_trades),
            return_pct=avg_return,
            benchmark_return=benchmark_return,
            alpha=alpha,
            portfolio_value=new_portfolio_value,
            benchmark_value=new_benchmark_value,
            trades=successful_trades,
            best_stock=max(month_returns),
            worst_stock=min(month_returns)
        )
    
    def _get_first_trading_days(self) -> List[str]:
        """获取每个月的第一个交易日"""
        first_days = []
        
        # 尝试使用数据源的交易日历
        try:
            trading_days = self.data_source.get_trading_calendar(
                self.config.start_date,
                self.config.end_date
            )
            
            if trading_days:
                # 按月分组
                current_month = None
                for day in sorted(trading_days):
                    month = day[:7]  # YYYY-MM
                    if month != current_month:
                        first_days.append(day)
                        current_month = month
                
                return first_days
        except Exception as e:
            logger.warning(f"无法获取交易日历，使用简单方法: {e}")
        
        # 简单方法：每月1号或之后的第一个工作日
        start = datetime.strptime(self.config.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.config.end_date, '%Y-%m-%d')
        
        current = start.replace(day=1)
        while current <= end:
            # 找到该月第一个工作日
            first_day = current
            while first_day.weekday() >= 5:  # 跳过周末
                first_day += timedelta(days=1)
            
            if first_day >= start and first_day <= end:
                first_days.append(first_day.strftime('%Y-%m-%d'))
            
            # 下个月
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return first_days
    
    def _count_trading_days(self, start_date: str, end_date: str) -> int:
        """计算两个日期之间的交易日数量"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        trading_days = 0
        current = start
        
        while current < end:
            if current.weekday() < 5:  # 周一到周五
                trading_days += 1
            current += timedelta(days=1)
        
        return trading_days
    
    def _get_index_prices_for_state(self, date: str, lookback_days: int = 60) -> List[float]:
        """
        获取指数历史价格，用于自适应策略判断市场状态
        
        Args:
            date: 当前日期
            lookback_days: 回看天数（默认60个交易日）
            
        Returns:
            价格列表（从旧到新）
        """
        try:
            # 计算开始日期
            end_dt = datetime.strptime(date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=lookback_days * 2)  # 多取一些，覆盖非交易日
            start_date = start_dt.strftime('%Y-%m-%d')
            
            # 获取指数数据
            index_data = self.data_source.get_index_data(
                self.config.benchmark,
                start_date,
                date
            )
            
            if index_data and hasattr(index_data, 'close_prices') and index_data.close_prices:
                prices = index_data.close_prices
                # 取最近 lookback_days 个
                if len(prices) > lookback_days:
                    prices = prices[-lookback_days:]
                logger.info(f"📊 获取指数价格用于市场状态判断: {len(prices)}个数据点")
                return prices
            
        except Exception as e:
            logger.warning(f"⚠️ 无法获取指数价格用于市场状态判断: {e}")
        
        return []
