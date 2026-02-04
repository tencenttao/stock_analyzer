# -*- coding: utf-8 -*-
"""
回测报告生成模块

生成各种格式的回测报告：
- 控制台输出
- JSON 文件
- 详细分析报告
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class BacktestReport:
    """
    回测报告生成器
    
    使用示例:
        report = BacktestReport(output_dir='./logs/backtest')
        
        # 打印控制台摘要
        report.print_summary(result)
        
        # 保存 JSON 报告
        report.save_json(result, 'monthly_2024')
    """
    
    def __init__(self, output_dir: str = './logs/backtest'):
        """
        初始化
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def print_summary(self, result: 'BacktestResult'):
        """
        打印回测结果摘要到控制台
        
        Args:
            result: BacktestResult 回测结果
        """
        logger.info("\n" + "=" * 70)
        logger.info("📊 回测结果汇总")
        logger.info("=" * 70)
        
        # 基本信息
        logger.info(f"\n📅 回测周期:")
        logger.info(f"   • 开始日期: {result.start_date}")
        logger.info(f"   • 结束日期: {result.end_date}")
        logger.info(f"   • 回测月数: {len(result.monthly_returns)}")
        
        # 收益表现
        logger.info(f"\n💰 收益表现:")
        logger.info(f"   • 初始资金: ¥{result.initial_capital:,.2f}")
        logger.info(f"   • 最终价值: ¥{result.final_value:,.2f}")
        logger.info(f"   • 总收益率: {result.total_return:+.2f}%")
        logger.info(f"   • 年化收益: {result.annual_return:+.2f}%")
        
        if result.monthly_returns:
            returns = [m['return_pct'] for m in result.monthly_returns]
            logger.info(f"   • 平均月收益: {sum(returns)/len(returns):+.2f}%")
            logger.info(f"   • 最佳月份: {max(returns):+.2f}%")
            logger.info(f"   • 最差月份: {min(returns):+.2f}%")
        
        # 风险指标
        if result.risk_metrics:
            metrics = result.risk_metrics
            logger.info(f"\n📈 风险指标:")
            logger.info(f"   • 夏普比率: {metrics.sharpe_ratio:.2f}")
            logger.info(f"   • 最大回撤: {metrics.max_drawdown:.2f}%")
            logger.info(f"   • 索提诺比率: {metrics.sortino_ratio:.2f}")
            logger.info(f"   • 年化波动率: {metrics.volatility:.2f}%")
            logger.info(f"   • 胜率: {metrics.win_rate:.1f}%")
            logger.info(f"   • 盈亏比: {metrics.profit_loss_ratio:.2f}")
        
        # 与基准对比
        logger.info(f"\n📊 与沪深300对比:")
        logger.info(f"   • 基准收益: {result.benchmark_return:+.2f}%")
        logger.info(f"   • 超额收益 (Alpha): {result.alpha:+.2f}%")
        
        if result.alpha > 0:
            logger.info(f"   • 结论: ✅ 跑赢大盘 {abs(result.alpha):.2f}%")
        else:
            logger.info(f"   • 结论: ❌ 跑输大盘 {abs(result.alpha):.2f}%")
        
        if result.risk_metrics:
            logger.info(f"   • 信息比率: {result.risk_metrics.information_ratio:.2f}")
        
        # 交易统计
        if result.trades:
            logger.info(f"\n🔄 交易统计:")
            logger.info(f"   • 总交易次数: {len(result.trades)}")
            wins = len([t for t in result.trades if t.get('return_pct', 0) > 0])
            logger.info(f"   • 盈利交易: {wins}")
            logger.info(f"   • 亏损交易: {len(result.trades) - wins}")
            
            if result.total_cost:
                logger.info(f"   • 总交易成本: ¥{result.total_cost:,.2f}")
        
        logger.info("=" * 70)
    
    def print_monthly_detail(self, result: 'BacktestResult'):
        """打印逐月明细"""
        if not result.monthly_returns:
            return
        
        logger.info(f"\n📋 逐月明细:")
        logger.info(f"{'月份':<6} {'日期范围':<25} {'策略':<10} {'基准':<10} {'Alpha':<10} {'组合价值':<15}")
        logger.info("-" * 90)
        
        for i, m in enumerate(result.monthly_returns, 1):
            alpha = m['return_pct'] - m.get('benchmark_return', 0)
            alpha_emoji = "✅" if alpha > 0 else "❌"
            date_range = f"{m['buy_date']} → {m['sell_date']}"
            logger.info(
                f"{i:<6} {date_range:<25} {m['return_pct']:>+7.2f}%  "
                f"{m.get('benchmark_return', 0):>+7.2f}%  {alpha:>+7.2f}% {alpha_emoji} "
                f"¥{m.get('portfolio_value', 0):>12,.0f}"
            )
        
        logger.info("-" * 90)
    
    def save_json(self, result: 'BacktestResult', name: str = None) -> str:
        """
        保存回测结果为 JSON 文件
        
        Args:
            result: BacktestResult 回测结果
            name: 文件名（不含扩展名），不传则自动生成
            
        Returns:
            保存的文件路径
        """
        if name is None:
            name = f"backtest_{result.start_date}_to_{result.end_date}"
        
        filepath = os.path.join(self.output_dir, f"{name}.json")
        
        # 转换为可序列化的字典
        data = self._to_dict(result)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📁 回测报告已保存: {filepath}")
        return filepath
    
    def _to_dict(self, result: 'BacktestResult') -> Dict:
        """将 BacktestResult 转换为字典"""
        data = {
            'summary': {
                'start_date': result.start_date,
                'end_date': result.end_date,
                'initial_capital': result.initial_capital,
                'final_value': result.final_value,
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'benchmark_return': result.benchmark_return,
                'alpha': result.alpha,
                'total_cost': result.total_cost,
            },
            'risk_metrics': None,
            'monthly_returns': result.monthly_returns,
            'trades': result.trades,
            'config': result.config,
            'generated_at': datetime.now().isoformat(),
        }
        
        if result.risk_metrics:
            data['risk_metrics'] = asdict(result.risk_metrics)
        
        return data
    
    def load_json(self, filepath: str) -> Dict:
        """
        加载 JSON 报告
        
        Args:
            filepath: 文件路径
            
        Returns:
            报告数据字典
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def compare_results(self, results: List['BacktestResult'], names: List[str] = None):
        """
        对比多个回测结果
        
        Args:
            results: 回测结果列表
            names: 结果名称列表
        """
        if not results:
            return
        
        if names is None:
            names = [f"策略{i+1}" for i in range(len(results))]
        
        logger.info("\n" + "=" * 70)
        logger.info("📊 策略对比")
        logger.info("=" * 70)
        
        # 表头
        header = f"{'指标':<20}"
        for name in names:
            header += f"{name:<15}"
        logger.info(header)
        logger.info("-" * (20 + 15 * len(names)))
        
        # 对比指标
        metrics = [
            ('总收益率', 'total_return', '+.2f%'),
            ('年化收益', 'annual_return', '+.2f%'),
            ('基准收益', 'benchmark_return', '+.2f%'),
            ('超额收益', 'alpha', '+.2f%'),
            ('夏普比率', 'sharpe_ratio', '.2f'),
            ('最大回撤', 'max_drawdown', '.2f%'),
            ('胜率', 'win_rate', '.1f%'),
        ]
        
        for label, attr, fmt in metrics:
            row = f"{label:<20}"
            for result in results:
                if attr in ['sharpe_ratio', 'max_drawdown', 'win_rate']:
                    value = getattr(result.risk_metrics, attr, 0) if result.risk_metrics else 0
                else:
                    value = getattr(result, attr, 0)
                row += f"{value:{fmt}:<15}"
            logger.info(row)
        
        logger.info("=" * (20 + 15 * len(names)))
