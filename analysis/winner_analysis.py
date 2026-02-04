# -*- coding: utf-8 -*-
"""
赢家特征分析模块

分析每月涨幅最大的股票（赢家）在买入时的特征，找出规律。

使用方法:
    python -m analysis.winner_analysis 2024-01-01 2024-12-31
    
    或:
    from analysis.winner_analysis import run_winner_analysis
    result = run_winner_analysis('2024-01-01', '2024-12-31')
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import statistics

from data.manager import DataManager
from core.types import StockData

logger = logging.getLogger(__name__)


@dataclass
class WinnerFeatures:
    """赢家特征数据"""
    code: str
    name: str
    month_date: str
    monthly_return: float
    rank_in_month: int
    
    # 买入时的特征
    price: float = 0.0
    change_pct: float = 0.0
    turnover_rate: float = 0.0
    momentum_20d: float = 0.0
    momentum_60d: float = 0.0
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    peg: float = 0.0
    roe: float = 0.0
    profit_growth: float = 0.0
    revenue_growth: float = 0.0
    dividend_yield: float = 0.0
    industry: str = ""
    
    # 策略对比
    strategy_score: float = 0.0
    was_selected: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MonthlyWinnersStats:
    """单月赢家统计"""
    month_date: str
    winners_count: int
    avg_return: float
    best_return: float
    worst_return: float
    
    avg_momentum_20d: float = 0.0
    avg_pe_ratio: float = 0.0
    avg_roe: float = 0.0
    avg_profit_growth: float = 0.0
    avg_turnover_rate: float = 0.0
    
    strategy_selected_count: int = 0
    strategy_hit_rate: float = 0.0
    
    winners: List[WinnerFeatures] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['winners'] = [w.to_dict() for w in self.winners]
        return result


@dataclass
class FeatureDistribution:
    """特征分布统计"""
    feature_name: str
    count: int
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    percentile_25: float
    percentile_75: float
    buckets: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WinnerAnalysisResult:
    """赢家分析结果"""
    start_date: str
    end_date: str
    top_n: int
    total_months: int
    total_winners: int
    avg_winner_return: float
    median_winner_return: float
    
    feature_distributions: Dict[str, FeatureDistribution] = field(default_factory=dict)
    monthly_stats: List[MonthlyWinnersStats] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['feature_distributions'] = {k: v.to_dict() for k, v in self.feature_distributions.items()}
        result['monthly_stats'] = [m.to_dict() for m in self.monthly_stats]
        return result
    
    def save_to_json(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"分析结果已保存到: {filepath}")


class WinnerAnalyzer:
    """赢家特征分析器"""
    
    def __init__(self, data_source: DataManager = None, strategy=None):
        self.data_source = data_source or DataManager(use_cache=True)
        self.strategy = strategy
        
        if self.strategy is None:
            from strategy import StrategyRegistry
            self.strategy = StrategyRegistry.create('momentum_v2')
    
    def analyze(
        self,
        start_date: str,
        end_date: str,
        top_n: int = 10,
        index_code: str = '000300',
        save_path: str = None,
    ) -> WinnerAnalysisResult:
        """执行赢家特征分析"""
        logger.info("=" * 60)
        logger.info("🏆 赢家特征分析")
        logger.info("=" * 60)
        logger.info(f"📅 分析期间: {start_date} ~ {end_date}")
        logger.info(f"🎯 每月选取涨幅前{top_n}的股票")
        logger.info(f"📊 股票池: 指数{index_code}成分股")
        logger.info("=" * 60)
        
        # 获取每月第一个交易日
        trading_days = self.data_source.get_first_trading_days(start_date, end_date)
        if not trading_days or len(trading_days) < 2:
            logger.error("交易日历获取失败或月份不足")
            return None
        
        logger.info(f"📆 回测周期: {len(trading_days)-1}个月")
        
        # 逐月分析
        monthly_stats_list = []
        all_winners = []
        
        for i in range(len(trading_days) - 1):
            buy_date = trading_days[i]
            sell_date = trading_days[i + 1]
            
            logger.info(f"\n📅 分析第{i+1}个月: {buy_date} → {sell_date}")
            
            monthly_stats = self._analyze_single_month(buy_date, sell_date, top_n, index_code)
            
            if monthly_stats:
                monthly_stats_list.append(monthly_stats)
                all_winners.extend(monthly_stats.winners)
        
        if not all_winners:
            logger.error("未找到任何赢家数据")
            return None
        
        # 汇总分析
        logger.info("\n" + "=" * 60)
        logger.info("📊 汇总分析特征分布...")
        
        feature_distributions = self._calculate_feature_distributions(all_winners)
        key_findings = self._generate_key_findings(all_winners, feature_distributions)
        suggestions = self._generate_suggestions(all_winners, feature_distributions)
        
        returns = [w.monthly_return for w in all_winners]
        result = WinnerAnalysisResult(
            start_date=start_date,
            end_date=end_date,
            top_n=top_n,
            total_months=len(monthly_stats_list),
            total_winners=len(all_winners),
            avg_winner_return=statistics.mean(returns),
            median_winner_return=statistics.median(returns),
            feature_distributions=feature_distributions,
            monthly_stats=monthly_stats_list,
            key_findings=key_findings,
            suggestions=suggestions,
        )
        
        self._print_summary(result)
        
        if save_path:
            result.save_to_json(save_path)
        
        return result
    
    def _analyze_single_month(
        self,
        buy_date: str,
        sell_date: str,
        top_n: int,
        index_code: str,
    ) -> Optional[MonthlyWinnersStats]:
        """分析单个月份的赢家"""
        
        # 1. 获取成分股
        stock_codes = self.data_source.get_index_constituents(index_code, buy_date)
        if not stock_codes:
            logger.warning(f"   无法获取{buy_date}的成分股")
            return None
        
        logger.info(f"   成分股: {len(stock_codes)} 只")
        
        # 2. 获取买入日数据（逐个获取，从缓存读取时很快）
        logger.info(f"   📥 获取买入日({buy_date})数据...")
        buy_data_list = []
        for i, code in enumerate(stock_codes):
            if (i + 1) % 100 == 0:
                logger.info(f"      进度: {i+1}/{len(stock_codes)}")
            stock = self.data_source.get_stock_data(code, buy_date)
            if stock and stock.price > 0:
                buy_data_list.append(stock)
        logger.info(f"   ✅ 买入日有效数据: {len(buy_data_list)} 只")
        
        if not buy_data_list:
            return None
        
        # 3. 获取卖出日数据
        logger.info(f"   📥 获取卖出日({sell_date})数据...")
        sell_data_map = {}
        for i, buy_stock in enumerate(buy_data_list):
            if (i + 1) % 100 == 0:
                logger.info(f"      进度: {i+1}/{len(buy_data_list)}")
            sell_stock = self.data_source.get_stock_data(buy_stock.code, sell_date)
            if sell_stock and sell_stock.price > 0:
                sell_data_map[buy_stock.code] = sell_stock
        logger.info(f"   ✅ 卖出日有效数据: {len(sell_data_map)} 只")
        
        # 4. 计算收益
        stock_returns = []
        for buy_stock in buy_data_list:
            if buy_stock.code in sell_data_map:
                sell_stock = sell_data_map[buy_stock.code]
                ret = (sell_stock.price - buy_stock.price) / buy_stock.price * 100
                stock_returns.append((buy_stock, ret))
        
        if not stock_returns:
            return None
        
        # 5. 按收益排序，选出赢家
        stock_returns.sort(key=lambda x: x[1], reverse=True)
        top_winners = stock_returns[:top_n]
        
        # 6. 策略对比
        strategy_selected_codes = set()
        if self.strategy:
            try:
                selected = self.strategy.select(buy_data_list, top_n=top_n)
                strategy_selected_codes = {s.code for s in selected}
            except Exception as e:
                logger.warning(f"   策略评分失败: {e}")
        
        # 7. 构建赢家特征
        winners = []
        for rank, (stock, ret) in enumerate(top_winners, 1):
            strategy_score = 0
            if self.strategy:
                try:
                    score_result = self.strategy.score(stock)
                    strategy_score = score_result.total
                except:
                    pass
            
            winner = WinnerFeatures(
                code=stock.code,
                name=stock.name,
                month_date=buy_date,
                monthly_return=ret,
                rank_in_month=rank,
                price=stock.price,
                change_pct=stock.change_pct or 0,
                turnover_rate=stock.turnover_rate or 0,
                momentum_20d=stock.momentum_20d or 0,
                momentum_60d=stock.momentum_60d or 0,
                pe_ratio=stock.pe_ratio or 0,
                pb_ratio=stock.pb_ratio or 0,
                peg=stock.peg or 0,
                roe=stock.roe or 0,
                profit_growth=stock.profit_growth or 0,
                revenue_growth=stock.revenue_growth or 0,
                dividend_yield=stock.dividend_yield or 0,
                industry=stock.industry or "",
                strategy_score=strategy_score,
                was_selected=stock.code in strategy_selected_codes,
            )
            winners.append(winner)
        
        # 统计
        returns = [w.monthly_return for w in winners]
        hit_count = sum(1 for w in winners if w.was_selected)
        
        stats = MonthlyWinnersStats(
            month_date=buy_date,
            winners_count=len(winners),
            avg_return=statistics.mean(returns),
            best_return=max(returns),
            worst_return=min(returns),
            avg_momentum_20d=statistics.mean([w.momentum_20d for w in winners]),
            avg_pe_ratio=statistics.mean([w.pe_ratio for w in winners if w.pe_ratio > 0] or [0]),
            avg_roe=statistics.mean([w.roe for w in winners if w.roe > 0] or [0]),
            avg_profit_growth=statistics.mean([w.profit_growth for w in winners]),
            avg_turnover_rate=statistics.mean([w.turnover_rate for w in winners]),
            strategy_selected_count=hit_count,
            strategy_hit_rate=hit_count / len(winners) * 100 if winners else 0,
            winners=winners,
        )
        
        # 打印当月赢家
        logger.info(f"   🏆 当月涨幅前{top_n}:")
        for w in winners[:5]:
            selected_mark = "✓策略选中" if w.was_selected else ""
            logger.info(f"      {w.rank_in_month}. {w.name}({w.code}): "
                       f"+{w.monthly_return:.2f}% "
                       f"[动量20d={w.momentum_20d:.1f}%, PE={w.pe_ratio:.1f}] "
                       f"策略分={w.strategy_score:.0f} {selected_mark}")
        
        logger.info(f"   📊 策略命中率: {hit_count}/{len(winners)} = {stats.strategy_hit_rate:.1f}%")
        
        return stats
    
    def _calculate_feature_distributions(self, winners: List[WinnerFeatures]) -> Dict[str, FeatureDistribution]:
        """计算特征分布"""
        features_config = {
            'momentum_20d': {'name': '20日动量', 'buckets': [
                ('<-10%', lambda x: x < -10),
                ('-10~0%', lambda x: -10 <= x < 0),
                ('0~10%', lambda x: 0 <= x < 10),
                ('10~20%', lambda x: 10 <= x < 20),
                ('20~30%', lambda x: 20 <= x < 30),
                ('>30%', lambda x: x >= 30),
            ]},
            'change_pct': {'name': '买入日涨幅', 'buckets': [
                ('<-5%', lambda x: x < -5),
                ('-5~0%', lambda x: -5 <= x < 0),
                ('0~3%', lambda x: 0 <= x < 3),
                ('3~5%', lambda x: 3 <= x < 5),
                ('5~7%', lambda x: 5 <= x < 7),
                ('>7%', lambda x: x >= 7),
            ]},
            'pe_ratio': {'name': '市盈率(PE)', 'buckets': [
                ('<0(亏损)', lambda x: x < 0),
                ('0~15', lambda x: 0 <= x < 15),
                ('15~25', lambda x: 15 <= x < 25),
                ('25~40', lambda x: 25 <= x < 40),
                ('40~60', lambda x: 40 <= x < 60),
                ('>60', lambda x: x >= 60),
            ]},
            'pb_ratio': {'name': '市净率(PB)', 'buckets': [
                ('<1', lambda x: x < 1),
                ('1~2', lambda x: 1 <= x < 2),
                ('2~3', lambda x: 2 <= x < 3),
                ('3~5', lambda x: 3 <= x < 5),
                ('>5', lambda x: x >= 5),
            ]},
            'roe': {'name': 'ROE', 'buckets': [
                ('<5%', lambda x: x < 5),
                ('5~10%', lambda x: 5 <= x < 10),
                ('10~15%', lambda x: 10 <= x < 15),
                ('15~20%', lambda x: 15 <= x < 20),
                ('>20%', lambda x: x >= 20),
            ]},
            'profit_growth': {'name': '利润增长率', 'buckets': [
                ('<-30%', lambda x: x < -30),
                ('-30~0%', lambda x: -30 <= x < 0),
                ('0~30%', lambda x: 0 <= x < 30),
                ('30~50%', lambda x: 30 <= x < 50),
                ('>50%', lambda x: x >= 50),
            ]},
            'turnover_rate': {'name': '换手率', 'buckets': [
                ('<1%', lambda x: x < 1),
                ('1~3%', lambda x: 1 <= x < 3),
                ('3~5%', lambda x: 3 <= x < 5),
                ('5~8%', lambda x: 5 <= x < 8),
                ('>8%', lambda x: x >= 8),
            ]},
            'strategy_score': {'name': '策略评分', 'buckets': [
                ('<30', lambda x: x < 30),
                ('30~40', lambda x: 30 <= x < 40),
                ('40~50', lambda x: 40 <= x < 50),
                ('50~60', lambda x: 50 <= x < 60),
                ('>60', lambda x: x >= 60),
            ]},
        }
        
        distributions = {}
        
        for feature_key, config in features_config.items():
            values = [getattr(w, feature_key) for w in winners if getattr(w, feature_key) is not None]
            valid_values = [v for v in values if v != 0 or feature_key in ['change_pct', 'momentum_20d', 'profit_growth']]
            
            if not valid_values:
                continue
            
            sorted_values = sorted(valid_values)
            n = len(sorted_values)
            
            dist = FeatureDistribution(
                feature_name=config['name'],
                count=n,
                mean=statistics.mean(valid_values),
                median=statistics.median(valid_values),
                std=statistics.stdev(valid_values) if n > 1 else 0,
                min_val=min(valid_values),
                max_val=max(valid_values),
                percentile_25=sorted_values[n // 4] if n >= 4 else sorted_values[0],
                percentile_75=sorted_values[3 * n // 4] if n >= 4 else sorted_values[-1],
                buckets={},
            )
            
            for bucket_name, bucket_func in config['buckets']:
                count = sum(1 for v in valid_values if bucket_func(v))
                dist.buckets[bucket_name] = count
            
            distributions[feature_key] = dist
        
        return distributions
    
    def _generate_key_findings(self, winners: List[WinnerFeatures], distributions: Dict[str, FeatureDistribution]) -> List[str]:
        """生成关键发现"""
        findings = []
        
        if 'momentum_20d' in distributions:
            dist = distributions['momentum_20d']
            positive_count = sum(v for k, v in dist.buckets.items() if not k.startswith('<') and not k.startswith('-'))
            findings.append(f"📈 动量特征: 赢家买入时20日动量均值={dist.mean:.1f}%, 中位数={dist.median:.1f}% (正动量占比={positive_count/dist.count*100:.0f}%)")
        
        if 'pe_ratio' in distributions:
            dist = distributions['pe_ratio']
            pe_15_40 = dist.buckets.get('15~25', 0) + dist.buckets.get('25~40', 0)
            findings.append(f"💰 估值特征: PE均值={dist.mean:.1f}, PE在15-40区间占比={pe_15_40/dist.count*100:.1f}%")
        
        if 'profit_growth' in distributions:
            dist = distributions['profit_growth']
            positive = sum(v for k, v in dist.buckets.items() if not k.startswith('<') and not k.startswith('-'))
            findings.append(f"🌱 成长特征: 利润增长率均值={dist.mean:.1f}%, 正增长占比={positive/dist.count*100:.1f}%")
        
        hit_count = sum(1 for w in winners if w.was_selected)
        hit_rate = hit_count / len(winners) * 100
        findings.append(f"🎯 策略命中率: 当前策略能选中{hit_rate:.1f}%的赢家 ({hit_count}/{len(winners)})")
        
        if 'turnover_rate' in distributions:
            dist = distributions['turnover_rate']
            findings.append(f"📊 交易特征: 换手率均值={dist.mean:.1f}%, 中位数={dist.median:.1f}%")
        
        return findings
    
    def _generate_suggestions(self, winners: List[WinnerFeatures], distributions: Dict[str, FeatureDistribution]) -> List[str]:
        """生成策略优化建议"""
        suggestions = []
        
        if 'momentum_20d' in distributions:
            dist = distributions['momentum_20d']
            if dist.median > 5:
                suggestions.append(f"✅ 继续重视动量指标，赢家的20日动量中位数为{dist.median:.1f}%")
            elif dist.median < 0:
                suggestions.append(f"⚠️ 注意：赢家动量并不总是正的(中位数={dist.median:.1f}%)，可能需要关注反转机会")
        
        if 'pe_ratio' in distributions:
            dist = distributions['pe_ratio']
            high_pe = dist.buckets.get('>60', 0) + dist.buckets.get('40~60', 0)
            if high_pe / dist.count > 0.3:
                suggestions.append(f"💡 考虑放宽PE筛选条件，{high_pe/dist.count*100:.1f}%的赢家PE>40")
        
        if 'profit_growth' in distributions:
            dist = distributions['profit_growth']
            if dist.mean < 10:
                suggestions.append(f"💡 高增长不是必要条件，赢家利润增长均值仅{dist.mean:.1f}%")
        
        if 'strategy_score' in distributions:
            dist = distributions['strategy_score']
            low_score = dist.buckets.get('<30', 0) + dist.buckets.get('30~40', 0)
            if low_score / dist.count > 0.4:
                suggestions.append(f"⚠️ 当前策略评分体系可能有偏差，{low_score/dist.count*100:.1f}%的赢家评分<40")
        
        hit_rate = sum(1 for w in winners if w.was_selected) / len(winners) * 100
        if hit_rate < 20:
            suggestions.append(f"🔧 策略命中率仅{hit_rate:.1f}%，需要重新审视选股逻辑")
        elif hit_rate > 40:
            suggestions.append(f"👍 策略命中率{hit_rate:.1f}%，方向正确，可微调参数优化")
        
        return suggestions
    
    def _print_summary(self, result: WinnerAnalysisResult):
        """打印汇总报告"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 赢家特征分析汇总报告")
        logger.info("=" * 60)
        
        logger.info(f"\n📅 分析期间: {result.start_date} ~ {result.end_date}")
        logger.info(f"📈 分析月数: {result.total_months} 个月")
        logger.info(f"🏆 赢家总数: {result.total_winners} 只次")
        logger.info(f"💰 赢家平均收益: {result.avg_winner_return:.2f}%")
        logger.info(f"💰 赢家收益中位数: {result.median_winner_return:.2f}%")
        
        logger.info("\n📊 关键特征分布:")
        logger.info("-" * 50)
        for key, dist in result.feature_distributions.items():
            logger.info(f"  {dist.feature_name}:")
            logger.info(f"    均值={dist.mean:.2f}, 中位数={dist.median:.2f}, 范围=[{dist.min_val:.2f}, {dist.max_val:.2f}]")
            bucket_str = ", ".join(f"{k}:{v}" for k, v in dist.buckets.items() if v > 0)
            logger.info(f"    分布: {bucket_str}")
        
        logger.info("\n🔍 关键发现:")
        logger.info("-" * 50)
        for finding in result.key_findings:
            logger.info(f"  {finding}")
        
        logger.info("\n💡 策略优化建议:")
        logger.info("-" * 50)
        for suggestion in result.suggestions:
            logger.info(f"  {suggestion}")
        
        logger.info("\n" + "=" * 60)


def run_winner_analysis(
    start_date: str = '2024-01-01',
    end_date: str = '2024-12-31',
    top_n: int = 10,
    save_path: str = None,
) -> WinnerAnalysisResult:
    """运行赢家分析"""
    analyzer = WinnerAnalyzer()
    
    if save_path is None:
        os.makedirs('logs/analysis', exist_ok=True)
        save_path = f"logs/analysis/winner_analysis_{start_date}_{end_date}.json"
    
    return analyzer.analyze(
        start_date=start_date,
        end_date=end_date,
        top_n=top_n,
        save_path=save_path,
    )


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    start_date = sys.argv[1] if len(sys.argv) > 1 else '2024-01-01'
    end_date = sys.argv[2] if len(sys.argv) > 2 else '2024-12-31'
    
    result = run_winner_analysis(start_date=start_date, end_date=end_date, top_n=10)
    
    if result:
        print(f"\n✅ 分析完成，共分析 {result.total_months} 个月，{result.total_winners} 只赢家")
