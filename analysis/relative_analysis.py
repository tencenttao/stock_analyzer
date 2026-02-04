# -*- coding: utf-8 -*-
"""
相对收益分析模块

分析跑赢/跑输大盘的股票特征差异，找出有效的预测指标。

核心思路：
1. 将股票按相对收益（vs大盘）分为三组：跑赢、持平、跑输
2. 对比各组在买入时的特征差异
3. 找出能有效区分跑赢/跑输的指标
4. 为策略优化提供数据支撑

使用方法:
    python -m analysis.relative_analysis 2024-01-01 2024-12-31
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import statistics

from data.manager import DataManager
from core.types import StockData

logger = logging.getLogger(__name__)


@dataclass
class StockPerformance:
    """股票表现数据"""
    code: str
    name: str
    month_date: str
    
    # 收益数据
    stock_return: float      # 股票收益率
    benchmark_return: float  # 基准收益率
    relative_return: float   # 相对收益（超额收益）
    
    # 买入时特征
    price: float = 0.0
    change_pct: float = 0.0
    turnover_rate: float = 0.0
    momentum_20d: float = 0.0
    momentum_60d: float = 0.0
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    roe: float = 0.0
    profit_growth: float = 0.0
    dividend_yield: float = 0.0
    
    # 分组
    group: str = ""  # 'outperform', 'neutral', 'underperform'
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroupStats:
    """组别统计"""
    group_name: str
    count: int
    pct: float  # 占比
    
    # 收益统计
    avg_return: float
    avg_relative: float
    
    # 各指标均值
    avg_momentum_20d: float = 0.0
    avg_change_pct: float = 0.0
    avg_pe_ratio: float = 0.0
    avg_pb_ratio: float = 0.0
    avg_roe: float = 0.0
    avg_profit_growth: float = 0.0
    avg_turnover_rate: float = 0.0
    avg_dividend_yield: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class FeatureEffectiveness:
    """特征有效性分析"""
    feature_name: str
    
    # 跑赢组 vs 跑输组的均值差异
    outperform_avg: float
    underperform_avg: float
    diff: float
    diff_pct: float  # 差异百分比
    
    # 与相对收益的相关性
    correlation: float = 0.0
    
    # 区分能力评分 (0-100)
    effectiveness_score: float = 0.0
    
    # 建议
    suggestion: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelativeAnalysisResult:
    """相对收益分析结果"""
    start_date: str
    end_date: str
    total_months: int
    total_samples: int
    
    # 分组统计
    outperform_stats: GroupStats = None
    neutral_stats: GroupStats = None
    underperform_stats: GroupStats = None
    
    # 特征有效性排名
    feature_effectiveness: List[FeatureEffectiveness] = field(default_factory=list)
    
    # 月度统计
    monthly_outperform_rate: List[float] = field(default_factory=list)
    
    # 关键发现
    key_findings: List[str] = field(default_factory=list)
    
    # 策略建议
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.outperform_stats:
            result['outperform_stats'] = self.outperform_stats.to_dict()
        if self.neutral_stats:
            result['neutral_stats'] = self.neutral_stats.to_dict()
        if self.underperform_stats:
            result['underperform_stats'] = self.underperform_stats.to_dict()
        result['feature_effectiveness'] = [f.to_dict() for f in self.feature_effectiveness]
        return result
    
    def save_to_json(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"分析结果已保存到: {filepath}")


class RelativeAnalyzer:
    """相对收益分析器"""
    
    # 跑输阈值（相对收益低于此值视为跑输）
    UNDERPERFORM_THRESHOLD = -5.0
    
    def __init__(self, data_source: DataManager = None):
        self.data_source = data_source or DataManager(use_cache=True)
    
    def analyze(
        self,
        start_date: str,
        end_date: str,
        index_code: str = '000300',
        save_path: str = None,
    ) -> RelativeAnalysisResult:
        """执行相对收益分析"""
        logger.info("=" * 60)
        logger.info("📊 相对收益分析（跑赢/跑输大盘）")
        logger.info("=" * 60)
        logger.info(f"📅 分析期间: {start_date} ~ {end_date}")
        logger.info(f"📈 基准指数: {index_code}")
        logger.info("=" * 60)
        
        # 获取每月第一个交易日
        trading_days = self.data_source.get_first_trading_days(start_date, end_date)
        if not trading_days or len(trading_days) < 2:
            logger.error("交易日历获取失败")
            return None
        
        logger.info(f"📆 分析周期: {len(trading_days)-1}个月")
        
        # 收集所有股票表现数据
        all_performances = []
        monthly_outperform_rates = []
        
        for i in range(len(trading_days) - 1):
            buy_date = trading_days[i]
            sell_date = trading_days[i + 1]
            
            logger.info(f"\n📅 第{i+1}个月: {buy_date} → {sell_date}")
            
            performances, outperform_rate = self._analyze_single_month(
                buy_date, sell_date, index_code
            )
            
            if performances:
                all_performances.extend(performances)
                monthly_outperform_rates.append(outperform_rate)
        
        if not all_performances:
            logger.error("无有效数据")
            return None
        
        # 分组统计
        logger.info("\n" + "=" * 60)
        logger.info("📊 汇总统计...")
        
        outperform = [p for p in all_performances if p.group == 'outperform']
        neutral = [p for p in all_performances if p.group == 'neutral']
        underperform = [p for p in all_performances if p.group == 'underperform']
        
        total = len(all_performances)
        
        outperform_stats = self._calc_group_stats('跑赢组', outperform, total)
        neutral_stats = self._calc_group_stats('持平组', neutral, total)
        underperform_stats = self._calc_group_stats('跑输组', underperform, total)
        
        # 特征有效性分析
        feature_effectiveness = self._analyze_feature_effectiveness(
            outperform, underperform, all_performances
        )
        
        # 生成发现和建议
        key_findings = self._generate_findings(
            outperform_stats, underperform_stats, feature_effectiveness
        )
        suggestions = self._generate_suggestions(feature_effectiveness)
        
        # 构建结果
        result = RelativeAnalysisResult(
            start_date=start_date,
            end_date=end_date,
            total_months=len(trading_days) - 1,
            total_samples=total,
            outperform_stats=outperform_stats,
            neutral_stats=neutral_stats,
            underperform_stats=underperform_stats,
            feature_effectiveness=feature_effectiveness,
            monthly_outperform_rate=monthly_outperform_rates,
            key_findings=key_findings,
            suggestions=suggestions,
        )
        
        # 打印报告
        self._print_report(result)
        
        if save_path:
            result.save_to_json(save_path)
        
        return result
    
    def _analyze_single_month(
        self,
        buy_date: str,
        sell_date: str,
        index_code: str,
    ) -> Tuple[List[StockPerformance], float]:
        """分析单月表现"""
        
        # 获取成分股
        stock_codes = self.data_source.get_index_constituents(index_code, buy_date)
        if not stock_codes:
            return [], 0.0
        
        # 获取基准收益
        benchmark_return = self.data_source.get_index_return(index_code, buy_date, sell_date)
        logger.info(f"   基准收益: {benchmark_return:+.2f}%")
        
        # 获取买入日数据
        buy_data_list = []
        for code in stock_codes:
            stock = self.data_source.get_stock_data(code, buy_date)
            if stock and stock.price > 0:
                buy_data_list.append(stock)
        
        # 获取卖出日数据，计算收益
        performances = []
        for buy_stock in buy_data_list:
            sell_stock = self.data_source.get_stock_data(buy_stock.code, sell_date)
            if sell_stock and sell_stock.price > 0:
                stock_return = (sell_stock.price - buy_stock.price) / buy_stock.price * 100
                relative_return = stock_return - benchmark_return
                
                # 分组
                if relative_return > 0:
                    group = 'outperform'
                elif relative_return > self.UNDERPERFORM_THRESHOLD:
                    group = 'neutral'
                else:
                    group = 'underperform'
                
                perf = StockPerformance(
                    code=buy_stock.code,
                    name=buy_stock.name,
                    month_date=buy_date,
                    stock_return=stock_return,
                    benchmark_return=benchmark_return,
                    relative_return=relative_return,
                    price=buy_stock.price,
                    change_pct=buy_stock.change_pct or 0,
                    turnover_rate=buy_stock.turnover_rate or 0,
                    momentum_20d=buy_stock.momentum_20d or 0,
                    momentum_60d=buy_stock.momentum_60d or 0,
                    pe_ratio=buy_stock.pe_ratio or 0,
                    pb_ratio=buy_stock.pb_ratio or 0,
                    roe=buy_stock.roe or 0,
                    profit_growth=buy_stock.profit_growth or 0,
                    dividend_yield=buy_stock.dividend_yield or 0,
                    group=group,
                )
                performances.append(perf)
        
        # 统计
        outperform_count = sum(1 for p in performances if p.group == 'outperform')
        underperform_count = sum(1 for p in performances if p.group == 'underperform')
        outperform_rate = outperform_count / len(performances) * 100 if performances else 0
        
        logger.info(f"   有效样本: {len(performances)}")
        logger.info(f"   🟢 跑赢: {outperform_count} ({outperform_rate:.1f}%)")
        logger.info(f"   🔴 跑输: {underperform_count} ({underperform_count/len(performances)*100:.1f}%)")
        
        return performances, outperform_rate
    
    def _calc_group_stats(
        self,
        name: str,
        performances: List[StockPerformance],
        total: int,
    ) -> GroupStats:
        """计算组别统计"""
        if not performances:
            return GroupStats(
                group_name=name, count=0, pct=0,
                avg_return=0, avg_relative=0
            )
        
        n = len(performances)
        
        def safe_mean(values):
            valid = [v for v in values if v != 0 and v is not None]
            return statistics.mean(valid) if valid else 0
        
        return GroupStats(
            group_name=name,
            count=n,
            pct=n / total * 100,
            avg_return=statistics.mean([p.stock_return for p in performances]),
            avg_relative=statistics.mean([p.relative_return for p in performances]),
            avg_momentum_20d=safe_mean([p.momentum_20d for p in performances]),
            avg_change_pct=safe_mean([p.change_pct for p in performances]),
            avg_pe_ratio=safe_mean([p.pe_ratio for p in performances if 0 < p.pe_ratio < 200]),
            avg_pb_ratio=safe_mean([p.pb_ratio for p in performances if 0 < p.pb_ratio < 20]),
            avg_roe=safe_mean([p.roe for p in performances]),
            avg_profit_growth=safe_mean([p.profit_growth for p in performances if -100 < p.profit_growth < 200]),
            avg_turnover_rate=safe_mean([p.turnover_rate for p in performances]),
            avg_dividend_yield=safe_mean([p.dividend_yield for p in performances]),
        )
    
    def _analyze_feature_effectiveness(
        self,
        outperform: List[StockPerformance],
        underperform: List[StockPerformance],
        all_data: List[StockPerformance],
    ) -> List[FeatureEffectiveness]:
        """分析各特征的有效性"""
        
        features = [
            ('momentum_20d', '20日动量'),
            ('change_pct', '买入日涨幅'),
            ('pe_ratio', '市盈率PE'),
            ('pb_ratio', '市净率PB'),
            ('roe', 'ROE'),
            ('profit_growth', '利润增长率'),
            ('turnover_rate', '换手率'),
            ('dividend_yield', '股息率'),
        ]
        
        results = []
        
        for attr, name in features:
            # 获取两组的值
            out_values = [getattr(p, attr) for p in outperform if getattr(p, attr) != 0]
            under_values = [getattr(p, attr) for p in underperform if getattr(p, attr) != 0]
            
            # 过滤异常值
            if attr == 'pe_ratio':
                out_values = [v for v in out_values if 0 < v < 200]
                under_values = [v for v in under_values if 0 < v < 200]
            if attr == 'profit_growth':
                out_values = [v for v in out_values if -100 < v < 300]
                under_values = [v for v in under_values if -100 < v < 300]
            
            if not out_values or not under_values:
                continue
            
            out_avg = statistics.mean(out_values)
            under_avg = statistics.mean(under_values)
            diff = out_avg - under_avg
            
            # 避免除零
            base = abs(under_avg) if under_avg != 0 else abs(out_avg) if out_avg != 0 else 1
            diff_pct = diff / base * 100 if base != 0 else 0
            
            # 计算相关性（简化版：用差异比例作为区分能力评分）
            effectiveness = min(100, abs(diff_pct))
            
            # 生成建议
            if effectiveness > 30:
                if diff > 0:
                    suggestion = f"✅ 选择高{name}股票"
                else:
                    suggestion = f"✅ 选择低{name}股票"
            elif effectiveness > 15:
                suggestion = f"⚠️ {name}有一定区分能力"
            else:
                suggestion = f"❌ {name}区分能力弱"
            
            results.append(FeatureEffectiveness(
                feature_name=name,
                outperform_avg=out_avg,
                underperform_avg=under_avg,
                diff=diff,
                diff_pct=diff_pct,
                effectiveness_score=effectiveness,
                suggestion=suggestion,
            ))
        
        # 按有效性排序
        results.sort(key=lambda x: x.effectiveness_score, reverse=True)
        
        return results
    
    def _generate_findings(
        self,
        outperform: GroupStats,
        underperform: GroupStats,
        features: List[FeatureEffectiveness],
    ) -> List[str]:
        """生成关键发现"""
        findings = []
        
        findings.append(f"📊 跑赢大盘比例: {outperform.pct:.1f}% ({outperform.count}只)")
        findings.append(f"📊 跑输大盘比例: {underperform.pct:.1f}% ({underperform.count}只)")
        findings.append(f"💰 跑赢组平均相对收益: +{outperform.avg_relative:.2f}%")
        findings.append(f"💰 跑输组平均相对收益: {underperform.avg_relative:.2f}%")
        
        # 最有效的特征
        if features:
            top_feature = features[0]
            findings.append(f"🎯 最有效区分指标: {top_feature.feature_name} "
                          f"(跑赢组={top_feature.outperform_avg:.2f}, "
                          f"跑输组={top_feature.underperform_avg:.2f})")
        
        return findings
    
    def _generate_suggestions(
        self,
        features: List[FeatureEffectiveness],
    ) -> List[str]:
        """生成策略建议"""
        suggestions = []
        
        # 根据有效特征生成建议
        effective_features = [f for f in features if f.effectiveness_score > 20]
        
        if effective_features:
            suggestions.append("📋 有效的筛选条件：")
            for f in effective_features[:5]:
                if f.diff > 0:
                    suggestions.append(f"   • 优选高{f.feature_name}：跑赢组均值={f.outperform_avg:.2f}")
                else:
                    suggestions.append(f"   • 优选低{f.feature_name}：跑赢组均值={f.outperform_avg:.2f}")
        
        # 风险控制建议
        weak_features = [f for f in features if f.effectiveness_score < 10]
        if weak_features:
            names = [f.feature_name for f in weak_features[:3]]
            suggestions.append(f"⚠️ 以下指标区分能力弱，不宜作为主要筛选条件：{', '.join(names)}")
        
        return suggestions
    
    def _print_report(self, result: RelativeAnalysisResult):
        """打印分析报告"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 相对收益分析报告")
        logger.info("=" * 70)
        
        logger.info(f"\n📅 分析期间: {result.start_date} ~ {result.end_date}")
        logger.info(f"📈 分析月数: {result.total_months}")
        logger.info(f"📊 总样本数: {result.total_samples}")
        
        # 分组统计
        logger.info("\n📊 分组统计：")
        logger.info("-" * 60)
        logger.info(f"  {'组别':<8} {'数量':<8} {'占比':<10} {'平均收益':<12} {'相对收益':<12}")
        logger.info("-" * 60)
        
        for stats in [result.outperform_stats, result.neutral_stats, result.underperform_stats]:
            if stats:
                logger.info(f"  {stats.group_name:<8} {stats.count:<8} "
                          f"{stats.pct:.1f}%{'':<6} "
                          f"{stats.avg_return:+.2f}%{'':<6} "
                          f"{stats.avg_relative:+.2f}%")
        
        # 特征对比
        logger.info("\n📊 跑赢组 vs 跑输组 特征对比：")
        logger.info("-" * 70)
        logger.info(f"  {'特征':<12} {'跑赢组':<12} {'跑输组':<12} {'差异':<12} {'有效性':<10}")
        logger.info("-" * 70)
        
        for f in result.feature_effectiveness:
            logger.info(f"  {f.feature_name:<12} "
                       f"{f.outperform_avg:>10.2f} "
                       f"{f.underperform_avg:>10.2f} "
                       f"{f.diff:>+10.2f} "
                       f"{f.effectiveness_score:>8.1f}")
        
        # 关键发现
        logger.info("\n🔍 关键发现：")
        logger.info("-" * 60)
        for finding in result.key_findings:
            logger.info(f"  {finding}")
        
        # 策略建议
        logger.info("\n💡 策略建议：")
        logger.info("-" * 60)
        for suggestion in result.suggestions:
            logger.info(f"  {suggestion}")
        
        logger.info("\n" + "=" * 70)


def run_relative_analysis(
    start_date: str = '2024-01-01',
    end_date: str = '2024-12-31',
    save_path: str = None,
) -> RelativeAnalysisResult:
    """运行相对收益分析"""
    analyzer = RelativeAnalyzer()
    
    if save_path is None:
        os.makedirs('logs/analysis', exist_ok=True)
        save_path = f"logs/analysis/relative_analysis_{start_date}_{end_date}.json"
    
    return analyzer.analyze(
        start_date=start_date,
        end_date=end_date,
        save_path=save_path,
    )


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    start_date = sys.argv[1] if len(sys.argv) > 1 else '2024-01-01'
    end_date = sys.argv[2] if len(sys.argv) > 2 else '2024-12-31'
    
    result = run_relative_analysis(start_date=start_date, end_date=end_date)
    
    if result:
        print(f"\n✅ 分析完成")
