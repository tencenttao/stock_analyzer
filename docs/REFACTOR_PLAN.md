# 📋 JYSstock_analyzer 重构方案

> **目标**：实现灵活可配置的选股策略系统，方便执行月度回测，代码模块化可维护。

---

## 一、当前问题分析

### 1.1 代码耦合度高，文件过长

| 文件 | 行数 | 问题 |
|------|------|------|
| `run_backtest_optimized.py` | 1734行 | 混合了数据获取、回测执行、策略选股、报告生成、缓存管理 |
| `src/data/data_fetcher.py` | 1698行 | 混合了腾讯/东方财富/新浪/AkShare多个API |
| `src/analysis/stock_filter.py` | 640行 | 评分策略硬编码，无法切换 |

### 1.2 策略不可配置
- 选股策略（V2动量优先）硬编码在 `stock_filter.py`
- 无法灵活切换不同策略进行对比回测
- 无法快速实验新策略

### 1.3 数据源切换不灵活
- 虽然有多个数据源（Tushare、AkShare、腾讯API），但：
  - 切换逻辑分散在各处
  - 没有统一的数据接口抽象
  - 回退逻辑复杂且重复

### 1.4 回测系统问题
- 单日/多日/月度回测逻辑分散且重复
- 缺少统一的回测框架
- 风险指标（最大回撤、夏普比率）计算不完整
- 没有交易成本模拟

---

## 二、重构目标

### 2.1 核心目标
1. **策略可插拔**：支持多种选股策略，可通过配置切换
2. **数据源可切换**：统一数据接口，灵活切换数据源
3. **回测标准化**：统一回测框架，支持多种回测模式
4. **代码模块化**：单一职责，文件不超过300行

### 2.2 设计原则
- **开闭原则**：对扩展开放，对修改关闭（新增策略无需改核心代码）
- **依赖倒置**：依赖抽象而非具体实现
- **单一职责**：每个模块只做一件事

---

## 三、新架构设计

### 3.1 目录结构

```
stock_analyzer/
├── core/                           # 🎯 核心抽象层
│   ├── __init__.py
│   ├── interfaces.py              # 核心接口定义（Strategy, DataSource）
│   └── types.py                   # 类型定义（StockData, BacktestResult）
│
├── data/                           # 📊 数据层
│   ├── __init__.py
│   ├── sources/                   # 数据源实现
│   │   ├── __init__.py
│   │   ├── base.py               # DataSource抽象基类
│   │   ├── tushare_source.py     # Tushare数据源 (~200行)
│   │   ├── akshare_source.py     # AkShare数据源 (~200行)
│   │   └── tencent_source.py     # 腾讯API数据源 (~200行)
│   ├── manager.py                 # 数据管理器（统一入口）
│   └── cache.py                   # 缓存管理
│
├── strategy/                       # 🎯 策略层
│   ├── __init__.py
│   ├── base.py                    # Strategy抽象基类
│   ├── registry.py                # 策略注册表
│   ├── scoring/                   # 评分类策略
│   │   ├── __init__.py
│   │   ├── momentum_v2.py        # 动量优先策略（当前）
│   │   ├── value_first.py        # 价值优先策略
│   │   └── balanced.py           # 平衡策略
│   ├── baseline/                  # 基线策略（对照组）
│   │   ├── __init__.py
│   │   ├── random_select.py      # 随机选股
│   │   └── equal_weight.py       # 等权选股
│   └── ml/                        # 机器学习策略（预留）
│       ├── __init__.py
│       └── xgboost_strategy.py
│
├── backtest/                       # 📈 回测层
│   ├── __init__.py
│   ├── engine.py                  # 回测引擎核心
│   ├── metrics.py                 # 风险指标计算
│   ├── cost.py                    # 交易成本模拟
│   ├── report.py                  # 报告生成
│   └── modes/                     # 回测模式
│       ├── __init__.py
│       ├── base.py               # 回测模式基类
│       ├── single_day.py         # 单日回测
│       ├── multi_day.py          # 多日回测
│       └── monthly.py            # 月度轮换回测
│
├── config/                         # ⚙️ 配置层
│   ├── __init__.py
│   ├── settings.py                # 全局设置
│   ├── data_source.py             # 数据源配置
│   ├── strategy.py                # 策略配置
│   └── backtest.py                # 回测配置
│
├── analysis/                       # 📊 分析层（精简）
│   ├── __init__.py
│   └── market_analyzer.py         # 市场分析器
│
├── notification/                   # 📧 通知层（保留）
│   └── email_sender.py
│
├── scheduler/                      # ⏰ 调度层（保留）
│   └── task_scheduler.py
│
├── cli.py                          # 🖥️ 命令行接口
├── main.py                         # 入口文件（实盘）
└── backtest.py                     # 回测入口（精简版）
```

### 3.2 核心接口设计

#### 3.2.1 数据源接口 (`core/interfaces.py`)

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class StockData:
    """股票数据标准格式"""
    code: str
    name: str
    price: float
    change_pct: float
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    roe: Optional[float] = None
    profit_growth: Optional[float] = None
    momentum_20d: Optional[float] = None
    turnover_rate: Optional[float] = None
    dividend_yield: Optional[float] = None
    turnover: Optional[float] = None  # 成交额
    # ... 其他字段

class DataSource(ABC):
    """数据源抽象基类"""
    
    @abstractmethod
    def get_stock_data(self, code: str, date: str) -> Optional[StockData]:
        """获取单只股票数据"""
        pass
    
    @abstractmethod
    def get_index_constituents(self, index_code: str, date: str = None) -> List[str]:
        """获取指数成分股"""
        pass
    
    @abstractmethod
    def get_index_return(self, index_code: str, start_date: str, end_date: str) -> float:
        """获取指数收益率"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """数据源名称"""
        pass
```

#### 3.2.2 策略接口 (`strategy/base.py`)

```python
from abc import ABC, abstractmethod
from typing import List, Dict
from core.types import StockData

class Strategy(ABC):
    """选股策略抽象基类"""
    
    @abstractmethod
    def score(self, stock: StockData) -> Dict:
        """
        计算股票评分
        
        Returns:
            {
                'total': float,        # 总分
                'breakdown': Dict,     # 分项得分
                'grade': str,          # 评级
                'risk_flag': bool      # 风险标记
            }
        """
        pass
    
    @abstractmethod
    def select(self, stocks: List[StockData], top_n: int = 10) -> List[StockData]:
        """
        从候选池中选出股票
        
        Args:
            stocks: 候选股票列表
            top_n: 选择数量
            
        Returns:
            选中的股票列表（已排序）
        """
        pass
    
    @abstractmethod
    def filter(self, stock: StockData) -> bool:
        """
        预筛选：判断股票是否满足基本条件
        
        Returns:
            True表示通过筛选，False表示排除
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass
    
    @property
    def config(self) -> Dict:
        """策略配置"""
        return {}
```

#### 3.2.3 回测引擎接口 (`backtest/engine.py`)

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from core.interfaces import DataSource
from strategy.base import Strategy

@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: str
    end_date: str
    initial_capital: float = 100000
    commission_rate: float = 0.00025  # 佣金万2.5
    stamp_tax_rate: float = 0.001     # 印花税千1
    slippage: float = 0.001           # 滑点0.1%
    benchmark: str = '000300'         # 基准指数

@dataclass  
class BacktestResult:
    """回测结果"""
    total_return: float               # 总收益率
    benchmark_return: float           # 基准收益率
    alpha: float                      # 超额收益
    sharpe_ratio: float              # 夏普比率
    max_drawdown: float              # 最大回撤
    win_rate: float                  # 胜率
    trades: List[Dict]               # 交易记录
    monthly_returns: List[Dict]      # 月度收益
    # ... 更多指标

class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, 
                 data_source: DataSource,
                 strategy: Strategy,
                 config: BacktestConfig):
        self.data_source = data_source
        self.strategy = strategy
        self.config = config
    
    def run_monthly(self) -> BacktestResult:
        """执行月度轮换回测"""
        pass
    
    def run_single_day(self, date: str, hold_days: int = 1) -> BacktestResult:
        """执行单日回测"""
        pass
```

### 3.3 策略注册与配置

#### 策略注册表 (`strategy/registry.py`)

```python
from typing import Dict, Type
from strategy.base import Strategy

class StrategyRegistry:
    """策略注册表 - 管理所有可用策略"""
    
    _strategies: Dict[str, Type[Strategy]] = {}
    
    @classmethod
    def register(cls, name: str):
        """装饰器：注册策略"""
        def decorator(strategy_cls: Type[Strategy]):
            cls._strategies[name] = strategy_cls
            return strategy_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[Strategy]:
        """获取策略类"""
        if name not in cls._strategies:
            raise ValueError(f"未知策略: {name}, 可用策略: {list(cls._strategies.keys())}")
        return cls._strategies[name]
    
    @classmethod
    def list_all(cls) -> List[str]:
        """列出所有已注册策略"""
        return list(cls._strategies.keys())
```

#### 策略配置 (`config/strategy.py`)

```python
# 策略配置
STRATEGY_CONFIGS = {
    # 动量优先策略（当前V2）
    'momentum_v2': {
        'name': '动量优先V2',
        'weights': {
            'momentum': 40,   # 动量/趋势
            'growth': 25,     # 成长性
            'valuation': 20,  # 估值
            'quality': 10,    # 质量
            'safety': 5       # 安全性
        },
        'filters': {
            'max_pe': 50,
            'min_price': 1.0,
            'min_turnover': 30000000,  # 3000万
        },
        'top_n': 10
    },
    
    # 价值优先策略
    'value_first': {
        'name': '价值优先',
        'weights': {
            'valuation': 40,
            'safety': 25,
            'quality': 20,
            'growth': 10,
            'momentum': 5
        },
        'filters': {
            'max_pe': 20,
            'max_pb': 3,
            'min_dividend_yield': 2.0
        },
        'top_n': 10
    },
    
    # 随机策略（基线对照）
    'random': {
        'name': '随机选股',
        'description': '随机选择N只股票，用于对比策略有效性',
        'top_n': 10,
        'seed': 42  # 固定随机种子确保可重复
    },
    
    # 等权策略（全部持有）
    'equal_weight': {
        'name': '等权持有',
        'description': '等权持有所有股票，代表市场平均水平',
    }
}

# 默认策略
DEFAULT_STRATEGY = 'momentum_v2'
```

### 3.4 使用示例

#### 执行月度回测

```python
from data.manager import DataManager
from strategy.registry import StrategyRegistry
from backtest.engine import BacktestEngine, BacktestConfig

# 1. 初始化数据源
data_manager = DataManager(source='tushare')  # 或 'akshare'

# 2. 选择策略
strategy = StrategyRegistry.get('momentum_v2')()

# 3. 配置回测
config = BacktestConfig(
    start_date='2024-01-01',
    end_date='2024-12-31',
    initial_capital=100000,
    benchmark='000300'
)

# 4. 运行回测
engine = BacktestEngine(data_manager, strategy, config)
result = engine.run_monthly()

# 5. 查看结果
print(f"总收益: {result.total_return:.2%}")
print(f"超额收益: {result.alpha:.2%}")
print(f"夏普比率: {result.sharpe_ratio:.2f}")
print(f"最大回撤: {result.max_drawdown:.2%}")
```

#### 策略对比回测

```python
# 对比多个策略
strategies = ['momentum_v2', 'value_first', 'random']
results = {}

for strategy_name in strategies:
    strategy = StrategyRegistry.get(strategy_name)()
    engine = BacktestEngine(data_manager, strategy, config)
    results[strategy_name] = engine.run_monthly()

# 输出对比报告
for name, result in results.items():
    print(f"{name}: 收益={result.total_return:.2%}, Alpha={result.alpha:.2%}")
```

#### 命令行使用

```bash
# 执行月度回测（默认策略）
python backtest.py monthly --start 2024-01-01 --end 2024-12-31

# 使用指定策略
python backtest.py monthly --strategy value_first --start 2024-01-01 --end 2024-12-31

# 策略对比
python backtest.py compare --strategies momentum_v2,value_first,random --start 2024-01-01 --end 2024-12-31

# 列出可用策略
python backtest.py list-strategies

# 单日回测
python backtest.py single --date 2024-06-03 --hold-days 5
```

---

## 四、重构实施计划

### 4.1 分阶段实施

#### 阶段一：核心抽象层 (预计工作量: 中)
1. 创建 `core/` 目录，定义核心接口
2. 定义 `StockData`, `BacktestResult` 等数据类型
3. 定义 `DataSource`, `Strategy` 抽象基类

#### 阶段二：数据层重构 (预计工作量: 大)
1. 拆分 `data_fetcher.py` 为多个数据源实现
2. 实现 `DataManager` 统一数据入口
3. 优化缓存管理

#### 阶段三：策略层重构 (预计工作量: 中)
1. 实现策略注册表
2. 将当前评分逻辑迁移到 `MomentumV2Strategy`
3. 实现 `RandomStrategy` 等基线策略

#### 阶段四：回测层重构 (预计工作量: 大)
1. 实现新的 `BacktestEngine`
2. 添加风险指标计算
3. 实现月度回测模式
4. 添加交易成本模拟

#### 阶段五：整合与测试 (预计工作量: 中)
1. 重构入口文件
2. 添加CLI命令
3. 编写测试用例
4. 更新文档

### 4.2 迁移策略

采用**渐进式重构**，保持系统可用：

1. **新旧并存**：新模块与旧代码并存，通过配置切换
2. **逐步迁移**：一个模块一个模块地迁移
3. **保留兼容**：旧的调用方式仍可使用
4. **完全切换**：所有测试通过后，移除旧代码

---

## 五、文件拆分计划

### 5.1 `run_backtest_optimized.py` 拆分

| 原代码位置 | 新文件位置 | 说明 |
|-----------|-----------|------|
| `OptimizedBacktest.__init__` | `data/manager.py` | 数据管理器初始化 |
| `get_stock_data_for_date` | `data/sources/tushare_source.py` | Tushare数据获取 |
| `get_csi300_stocks` | `data/manager.py` | 成分股获取 |
| `backtest_monthly_rotation` | `backtest/modes/monthly.py` | 月度回测逻辑 |
| `backtest_single_day` | `backtest/modes/single_day.py` | 单日回测逻辑 |
| `_print_monthly_summary` | `backtest/report.py` | 报告生成 |
| `analyze_single_stock` | `analysis/stock_analyzer.py` | 单股分析 |
| 缓存相关代码 | `data/cache.py` | 缓存管理 |

### 5.2 `data_fetcher.py` 拆分

| 原代码位置 | 新文件位置 | 说明 |
|-----------|-----------|------|
| `get_stock_realtime_data` | `data/sources/tencent_source.py` | 腾讯API |
| `get_stock_fundamental_data` | `data/sources/tencent_source.py` | 腾讯基本面 |
| `_fetch_eastmoney_realtime_data` | `data/sources/eastmoney_source.py` | 东方财富API |
| `get_sina_stock_data` | `data/sources/sina_source.py` | 新浪API |
| `get_historical_fundamental_data` | `data/sources/akshare_source.py` | AkShare历史 |
| `get_market_overview` | `analysis/market_overview.py` | 市场概况 |
| `batch_get_stock_data` | `data/manager.py` | 批量获取 |

### 5.3 `stock_filter.py` 拆分

| 原代码位置 | 新文件位置 | 说明 |
|-----------|-----------|------|
| `calculate_strength_score` | `strategy/scoring/momentum_v2.py` | 动量V2策略 |
| `filter_by_pe_ratio` | `strategy/base.py` | 策略基类filter |
| `select_top_stocks` | `strategy/base.py` | 策略基类select |
| `_apply_hard_filters` | `strategy/base.py` | 硬性过滤 |

---

## 六、风险与注意事项

### 6.1 风险
1. **数据兼容性**：重构后数据格式需要保持兼容
2. **回测结果一致性**：重构前后回测结果应保持一致
3. **性能影响**：抽象层可能带来轻微性能开销

### 6.2 缓解措施
1. 编写充分的单元测试
2. 对比重构前后回测结果
3. 保留性能关键路径的优化

---

## 七、实施进度

### ✅ 阶段一：核心抽象层（已完成 2026-01-24）

**创建的文件：**
- `core/__init__.py` - 包初始化和导出
- `core/types.py` - 数据类型定义
  - `StockData` - 股票数据标准格式
  - `IndexData` - 指数数据
  - `TradeRecord` - 交易记录
  - `MonthlyReturn` - 月度收益
  - `BacktestConfig` - 回测配置
  - `BacktestResult` - 回测结果
  - `ScoreResult` - 评分结果
- `core/interfaces.py` - 抽象接口
  - `DataSource` - 数据源基类
  - `Strategy` - 策略基类
- `tests/test_core.py` - 核心模块测试（7个测试全部通过）

### ✅ 阶段二：数据层重构（已完成 2026-01-24）

**创建的文件：**
- `data/__init__.py` - 数据层包初始化
- `data/cache.py` - 缓存管理器（~200行）
  - `CacheManager` - 支持内存+文件双层缓存
- `data/sources/__init__.py` - 数据源注册表
  - `get_source()` - 获取数据源类
  - `list_sources()` - 列出可用数据源
- `data/sources/tushare_source.py` - Tushare数据源（~350行）
  - `TushareSource` - 付费稳定，支持历史数据
  - 内置请求频率控制、批量获取优化
- `data/sources/akshare_source.py` - AkShare数据源（~320行）
  - `AkShareSource` - 免费数据源
  - 支持股票列表、指数成分股、财务指标
- `data/sources/tencent_source.py` - 腾讯数据源（~350行）
  - `TencentSource` - 实时行情数据源
  - 支持批量查询（每次80只）
- `data/manager.py` - 数据管理器（~220行）
  - `DataManager` - 统一数据入口
  - 支持数据源切换、自动缓存
- `tests/test_data.py` - 数据层测试（7个测试全部通过）

**数据源对比：**
| 数据源 | 费用 | 历史数据 | 实时数据 | 推荐场景 |
|--------|------|----------|----------|----------|
| Tushare | 付费 | ✅ 完整 | ✅ | 历史回测（推荐）|
| AkShare | 免费 | ✅ 部分 | ❌ | 免费回测 |
| Tencent | 免费 | ❌ | ✅ 稳定 | 实时监控 |

**重构亮点：**
- 从原 `data_fetcher.py`（1698行）+ `tushare_fetcher.py`（584行）提取核心逻辑
- 实现标准化 `DataSource` 接口，支持灵活切换
- 统一缓存管理，减少重复代码
- 三种数据源覆盖不同使用场景

### ✅ 阶段三：策略层重构（已完成 2026-01-24）

**创建的文件：**
- `strategy/__init__.py` - 策略层包初始化
- `strategy/registry.py` - 策略注册表（~150行）
  - `StrategyRegistry` - 管理所有可用策略
  - `register_strategy` - 装饰器快捷注册
- `strategy/scoring/__init__.py` - 评分策略包
- `strategy/scoring/momentum_v2.py` - 动量优先策略（~400行）
  - `MomentumV2Strategy` - 从 stock_filter.py 迁移评分逻辑
  - 40%动量 + 25%成长 + 20%估值 + 10%质量 + 5%安全
- `strategy/baseline/__init__.py` - 基线策略包
- `strategy/baseline/random_select.py` - 随机选股策略（~180行）
  - `RandomStrategy` - 用于策略效果对照
  - 支持固定随机种子确保可重复
- `tests/test_strategy.py` - 策略层测试（4个测试全部通过）

**使用示例：**
```python
from strategy import StrategyRegistry

# 获取并创建策略
strategy = StrategyRegistry.create('momentum_v2')

# 选股
selected = strategy.select(stocks, top_n=10)

# 列出所有策略
print(StrategyRegistry.list_all())  # ['momentum_v2', 'random']
```

**策略对比验证：**
- 动量策略选股基于评分（动量高的股票优先）
- 随机策略选股完全随机（用于效果对照）
- 测试中两策略选股0重叠，说明动量策略确实有选股倾向性

### ✅ 阶段四：回测层重构（已完成 2026-01-24）

**创建的文件：**
- `backtest/__init__.py` - 回测层包初始化
- `backtest/metrics.py` - 风险指标计算（~280行）
  - `RiskMetrics` - 计算夏普比率、最大回撤、索提诺比率等
  - `RiskMetricsResult` - 风险指标结果数据类
- `backtest/cost.py` - 交易成本模拟（~200行）
  - `TradingCost` - 计算佣金、印花税、滑点、过户费
  - `CostConfig` - 成本配置数据类
- `backtest/report.py` - 报告生成（~250行）
  - `BacktestReport` - 控制台输出、JSON保存、策略对比
- `backtest/modes/__init__.py` - 回测模式包
- `backtest/modes/monthly.py` - 月度轮换回测（~300行）
  - `MonthlyMode` - 月度轮换执行逻辑
  - `MonthlyConfig` - 月度回测配置
  - `MonthlyResult` - 单月结果
- `backtest/engine.py` - 回测引擎核心（~280行）
  - `BacktestEngine` - 统一回测入口
  - `BacktestConfig` - 通用回测配置
  - `BacktestResult` - 回测结果
- `tests/test_backtest.py` - 回测层测试（6个测试全部通过）

**风险指标支持：**
| 指标 | 说明 |
|------|------|
| 夏普比率 | 风险调整后收益 |
| 最大回撤 | 最大亏损幅度 |
| 索提诺比率 | 下行风险调整收益 |
| 卡尔马比率 | 收益/最大回撤 |
| 信息比率 | 超额收益稳定性 |
| 年化波动率 | 收益波动程度 |
| 胜率 | 盈利周期占比 |
| 盈亏比 | 平均盈利/平均亏损 |

**交易成本模拟：**
- 佣金：万2.5（买卖双向）
- 印花税：千1（卖出单向）
- 滑点：0.1%（买卖双向）
- 过户费：万0.1（沪市）
- 往返成本率：约0.35%

**使用示例：**
```python
from data.manager import DataManager
from strategy import StrategyRegistry
from backtest import BacktestEngine, BacktestConfig

# 初始化
data_source = DataManager(source='tushare')
strategy = StrategyRegistry.create('momentum_v2')
config = BacktestConfig(
    start_date='2024-01-01',
    end_date='2024-12-31',
    initial_capital=100000,
    top_n=10
)

# 执行回测
engine = BacktestEngine(data_source, strategy, config)
result = engine.run_monthly()

# 查看结果
print(f"总收益: {result.total_return:.2f}%")
print(f"夏普比率: {result.risk_metrics.sharpe_ratio:.2f}")
print(f"最大回撤: {result.risk_metrics.max_drawdown:.2f}%")
```

### ✅ 阶段五：整合与测试（已完成 2026-01-24）

**创建的文件：**
- `cli.py` - 命令行接口（~350行）
  - `backtest monthly` - 月度轮换回测
  - `backtest compare` - 策略对比回测
  - `strategy list` - 列出可用策略
  - `data list-sources` - 列出数据源
  - `data test` - 测试数据源
  - `select` - 执行选股
- `backtest.py` - 精简回测入口（~250行）
  - `run_monthly()` - 执行月度回测
  - `compare_strategies()` - 策略对比
  - `select_stocks()` - 执行选股
  - 支持命令行和编程调用
- `config/__init__.py` - 配置层初始化
- `config/settings.py` - 全局设置（~50行）
- `config/strategy_config.py` - 策略配置（~120行）
- `config/data_source_config.py` - 数据源配置（~100行）
- `tests/test_integration.py` - 综合集成测试（~320行）
  - 16个测试用例全部通过

**命令行使用示例：**
```bash
# 月度回测（默认策略）
python backtest.py --start 2024-01-01 --end 2024-12-31

# 使用指定策略
python backtest.py --strategy random --start 2024-06-01 --end 2024-12-31

# 策略对比
python backtest.py --compare momentum_v2,random --start 2024-01-01 --end 2024-06-30

# 执行选股
python backtest.py --select --date 2024-06-03

# 列出策略
python cli.py strategy list

# 测试数据源
python cli.py data test --source tushare
```

**编程使用示例：**
```python
from data.manager import DataManager
from strategy import StrategyRegistry
from backtest import BacktestEngine
from backtest.engine import BacktestConfig

# 初始化
data_source = DataManager(source='tushare', use_cache=True)
strategy = StrategyRegistry.create('momentum_v2')
config = BacktestConfig(
    start_date='2024-01-01',
    end_date='2024-12-31',
    initial_capital=100000,
    top_n=10
)

# 执行回测
engine = BacktestEngine(data_source, strategy, config)
result = engine.run_monthly()

# 查看结果
print(f"总收益: {result.total_return:.2f}%")
print(f"夏普比率: {result.risk_metrics.sharpe_ratio:.2f}")
```

---

## 八、重构完成总结

### 8.1 重构成果

| 阶段 | 内容 | 状态 | 完成日期 |
|------|------|------|----------|
| 阶段一 | 核心抽象层 | ✅ 完成 | 2026-01-24 |
| 阶段二 | 数据层重构 | ✅ 完成 | 2026-01-24 |
| 阶段三 | 策略层重构 | ✅ 完成 | 2026-01-24 |
| 阶段四 | 回测层重构 | ✅ 完成 | 2026-01-24 |
| 阶段五 | 整合与测试 | ✅ 完成 | 2026-01-24 |

### 8.2 新架构目录结构

```
stock_analyzer/
├── core/                    # 核心抽象层
│   ├── __init__.py
│   ├── interfaces.py       # DataSource, Strategy 接口
│   └── types.py            # StockData, BacktestResult 等类型
│
├── data/                    # 数据层
│   ├── __init__.py
│   ├── cache.py            # CacheManager
│   ├── manager.py          # DataManager (统一入口)
│   └── sources/
│       ├── __init__.py
│       ├── tushare_source.py
│       ├── akshare_source.py
│       └── tencent_source.py
│
├── strategy/                # 策略层
│   ├── __init__.py
│   ├── registry.py         # StrategyRegistry
│   ├── scoring/
│   │   ├── __init__.py
│   │   └── momentum_v2.py  # MomentumV2Strategy
│   └── baseline/
│       ├── __init__.py
│       └── random_select.py # RandomStrategy
│
├── backtest/                # 回测层
│   ├── __init__.py
│   ├── engine.py           # BacktestEngine
│   ├── metrics.py          # RiskMetrics
│   ├── cost.py             # TradingCost
│   ├── report.py           # BacktestReport
│   └── modes/
│       ├── __init__.py
│       └── monthly.py      # MonthlyMode
│
├── config/                  # 配置层
│   ├── __init__.py
│   ├── settings.py
│   ├── strategy_config.py
│   ├── data_source_config.py
│   └── backtest_config.py
│
├── tests/                   # 测试
│   ├── test_core.py
│   ├── test_data.py
│   ├── test_strategy.py
│   ├── test_backtest.py
│   └── test_integration.py
│
├── cli.py                   # 命令行接口
└── backtest.py              # 回测入口
```

### 8.3 主要改进

1. **代码模块化**
   - 原 `run_backtest_optimized.py`（1734行）→ 拆分为多个模块，每个<300行
   - 原 `data_fetcher.py`（1698行）→ 三个数据源独立实现

2. **策略可插拔**
   - 支持通过注册表动态添加策略
   - 策略对比回测一行代码实现

3. **数据源可切换**
   - Tushare（付费稳定）/ AkShare（免费）/ Tencent（实时）
   - 统一 DataSource 接口

4. **风险指标完善**
   - 夏普比率、最大回撤、索提诺比率、卡尔马比率
   - 胜率、盈亏比、年化波动率

5. **交易成本模拟**
   - 佣金、印花税、滑点、过户费

6. **测试覆盖**
   - 35+ 测试用例覆盖各模块

### 8.4 后续可扩展

- [ ] 实现 `ValueFirstStrategy`（价值优先策略）
- [ ] 实现 `BalancedStrategy`（平衡策略）
- [ ] 添加机器学习策略 `ml/`
- [ ] 添加 Web UI 界面
- [ ] 实时监控与自动交易
