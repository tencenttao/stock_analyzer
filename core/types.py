# -*- coding: utf-8 -*-
"""
核心数据类型定义

使用 dataclass 定义系统中的核心数据结构，确保：
- 类型安全
- 数据一致性
- 易于序列化
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class StockData:
    """
    股票数据标准格式
    
    所有数据源都应将数据转换为此格式，确保系统内部数据一致性。
    """
    # 基本信息
    code: str                                    # 股票代码（6位）
    name: str = ""                               # 股票名称
    
    # 价格数据
    price: float = 0.0                           # 当前/收盘价
    change_pct: float = 0.0                      # 涨跌幅（%）
    
    # 估值指标
    pe_ratio: Optional[float] = None             # 市盈率（PE）
    pb_ratio: Optional[float] = None             # 市净率（PB）
    peg: Optional[float] = None                  # PEG
    
    # 基本面指标
    roe: Optional[float] = None                  # 净资产收益率（%）
    profit_growth: Optional[float] = None        # 净利润增长率（%）
    revenue_growth: Optional[float] = None       # 营收增长率（%）
    
    # 交易指标
    turnover: Optional[float] = None             # 成交额（元）
    turnover_rate: Optional[float] = None        # 换手率（%）
    volume: Optional[float] = None               # 成交量（股）
    volume_ratio: Optional[float] = None         # 量比
    
    # 市值指标
    total_mv: Optional[float] = None             # 总市值（万元）
    circ_mv: Optional[float] = None              # 流通市值（万元）
    
    # 技术指标
    momentum_20d: Optional[float] = None         # 20日动量（%）
    momentum_60d: Optional[float] = None         # 60日动量（%）
    
    # 分红指标
    dividend_yield: Optional[float] = None       # 股息率（%）
    
    # 行业/板块信息
    industry: Optional[str] = None               # 所属行业
    market: Optional[str] = None                 # 市场板块（主板/创业板/科创板）
    list_days: Optional[int] = None              # 上市天数
    
    # 策略评分（由策略填充）
    strength_score: float = 0.0                  # 综合评分
    strength_grade: str = ""                     # 评级（A+/A/B+/B/C/D）
    score_breakdown: Dict[str, float] = field(default_factory=dict)  # 分项得分
    selection_reason: str = ""                   # 选择理由
    rank: int = 0                                # 排名
    
    # 元数据
    date: str = ""                               # 数据日期（YYYY-MM-DD）
    data_source: str = ""                        # 数据来源
    report_date: Optional[str] = None            # 财报期
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StockData':
        """从字典创建"""
        # 只保留 StockData 定义的字段
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    def is_valid(self) -> bool:
        """检查数据是否有效"""
        return bool(self.code) and self.price > 0


@dataclass
class IndexData:
    """指数数据"""
    code: str                          # 指数代码
    name: str = ""                     # 指数名称
    start_date: str = ""               # 开始日期
    end_date: str = ""                 # 结束日期
    start_price: float = 0.0           # 起始价格
    end_price: float = 0.0             # 结束价格
    return_pct: float = 0.0            # 收益率（%）
    trading_days: int = 0              # 交易日数
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradeRecord:
    """交易记录"""
    code: str                          # 股票代码
    name: str                          # 股票名称
    buy_date: str                      # 买入日期
    buy_price: float                   # 买入价格
    sell_date: str                     # 卖出日期
    sell_price: float                  # 卖出价格
    return_pct: float                  # 收益率（%）
    hold_days: int = 0                 # 持有天数
    
    # 可选信息
    pe_ratio: Optional[float] = None
    strength_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MonthlyReturn:
    """月度收益记录"""
    month: int                         # 第几个月（从1开始）
    buy_date: str                      # 买入日期
    sell_date: str                     # 卖出日期
    hold_days: int                     # 持有天数
    
    # 股票信息
    selected_count: int                # 选中股票数
    successful_trades: int             # 成功交易数
    
    # 收益信息
    strategy_return: float             # 策略收益（%）
    benchmark_return: float            # 基准收益（%）
    alpha: float                       # 超额收益（%）
    
    # 组合价值
    portfolio_value: float             # 组合价值
    benchmark_value: float             # 基准价值
    
    # 个股表现
    best_stock_return: float = 0.0     # 最佳个股收益
    worst_stock_return: float = 0.0    # 最差个股收益
    trades: List[TradeRecord] = field(default_factory=list)  # 交易明细
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['trades'] = [t.to_dict() if hasattr(t, 'to_dict') else t for t in self.trades]
        return result


@dataclass
class BacktestConfig:
    """回测配置"""
    # 时间范围
    start_date: str                              # 开始日期（YYYY-MM-DD）
    end_date: str                                # 结束日期（YYYY-MM-DD）
    
    # 资金配置
    initial_capital: float = 100000.0            # 初始资金
    
    # 交易成本
    commission_rate: float = 0.00025             # 佣金费率（万2.5）
    stamp_tax_rate: float = 0.001                # 印花税率（千1，卖出）
    slippage: float = 0.001                      # 滑点（0.1%）
    
    # 基准配置
    benchmark_code: str = '000300'               # 基准指数代码（沪深300）
    
    # 股票池配置
    stock_pool: str = 'csi300'                   # 股票池（csi300/csi500/all）
    
    # 选股配置
    top_n: int = 10                              # 选择股票数量
    
    # 其他配置
    random_seed: int = 42                        # 随机种子
    use_cache: bool = True                       # 是否使用缓存
    cache_expire_days: int = 7                   # 缓存过期天数
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    """回测结果"""
    # 基本信息
    strategy_name: str                           # 策略名称
    start_date: str                              # 开始日期
    end_date: str                                # 结束日期
    
    # 收益指标
    total_return: float                          # 总收益率（%）
    benchmark_return: float                      # 基准收益率（%）
    alpha: float                                 # 超额收益（%）
    
    # 风险指标
    max_drawdown: float = 0.0                    # 最大回撤（%）
    sharpe_ratio: float = 0.0                    # 夏普比率
    information_ratio: float = 0.0              # 信息比率
    volatility: float = 0.0                      # 年化波动率（%）
    
    # 胜率指标
    win_rate: float = 0.0                        # 胜率（%）
    beat_benchmark_rate: float = 0.0             # 跑赢基准比率（%）
    
    # 资金曲线
    initial_capital: float = 100000.0
    final_value: float = 0.0                     # 最终价值
    benchmark_final_value: float = 0.0           # 基准最终价值
    
    # 交易统计
    total_trades: int = 0                        # 总交易次数
    winning_trades: int = 0                      # 盈利交易次数
    
    # 月度明细
    monthly_returns: List[MonthlyReturn] = field(default_factory=list)
    
    # 配置信息
    config: Optional[BacktestConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['monthly_returns'] = [
            m.to_dict() if hasattr(m, 'to_dict') else m 
            for m in self.monthly_returns
        ]
        if self.config:
            result['config'] = self.config.to_dict()
        return result
    
    def summary(self) -> str:
        """生成结果摘要"""
        lines = [
            f"策略: {self.strategy_name}",
            f"回测期间: {self.start_date} ~ {self.end_date}",
            f"总收益: {self.total_return:+.2f}%",
            f"基准收益: {self.benchmark_return:+.2f}%",
            f"超额收益(Alpha): {self.alpha:+.2f}%",
            f"最大回撤: {self.max_drawdown:.2f}%",
            f"夏普比率: {self.sharpe_ratio:.2f}",
            f"胜率: {self.win_rate:.1f}%",
            f"最终价值: ¥{self.final_value:,.2f}",
        ]
        return "\n".join(lines)


@dataclass
class ScoreResult:
    """评分结果"""
    total: float                                 # 总分
    breakdown: Dict[str, float]                  # 分项得分
    grade: str                                   # 评级
    risk_flag: bool = False                      # 风险标记
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
