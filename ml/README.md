# ML 模块架构说明

## 模块结构

```
ml/
├── __init__.py                 # 模块导出
├── build_training_data.py      # 构建训练数据集
├── stock_predictor.py          # 股票预测器（模型）
├── train_and_evaluate.py      # 训练和评估脚本
├── test_features.py           # 特征提取测试
└── features/                  # 特征工程子模块
    ├── __init__.py
    ├── definitions.py         # 特征定义
    ├── engineer.py            # 特征工程器
    └── indicators.py          # 技术指标计算
```

## 核心组件

### 1. 特征工程模块 (`features/`)

#### 1.1 特征定义 (`definitions.py`)
- **FeatureDefinition**: 特征定义数据类
- **BASIC_FEATURES**: 基础特征（从 StockData 直接获取）
  - `price`, `change_pct`, `pe_ratio`, `pb_ratio`, `turnover_rate`, `roe`, `profit_growth`, `revenue_growth`, `dividend_yield`
- **TECHNICAL_FEATURES**: 技术指标特征
  - `momentum_20d`, `momentum_60d` (可从 StockData 获取)
  - `volatility_20d`, `rsi_14`, `ma_deviation_20` (需要日线数据计算)
- **CATEGORICAL_FEATURES**: 分类特征
  - `industry`
- **DERIVED_FEATURES**: 衍生特征
  - `value_score`, `quality_score` (可单独计算)
  - `pe_percentile`, `momentum_rank` (需要批量计算)

#### 1.2 技术指标计算 (`indicators.py`)
- **TechnicalIndicators**: 技术指标计算器（静态方法）
  - `calc_rsi()`: RSI 相对强弱指数
  - `calc_macd()`: MACD 指标
  - `calc_volatility()`: 波动率
  - `calc_bollinger_position()`: 布林带位置
  - `calc_atr()`: 平均真实波幅
  - `calc_volume_ratio()`: 成交量比率
  - `compute_all()`: 计算所有技术指标

#### 1.3 特征工程器 (`engineer.py`)
- **FeatureConfig**: 特征配置类
  - `basic_features`: 基础特征列表
  - `technical_features`: 技术指标列表
  - `categorical_features`: 分类特征列表
  - `derived_features`: 衍生特征列表
  - 预定义配置：`DEFAULT_FEATURE_CONFIG`, `FULL_FEATURE_CONFIG`, `SIMPLE_FEATURE_CONFIG`

- **FeatureEngineer**: 特征工程器
  - `extract()`: 从 StockData 提取基础特征
  - `batch_extract()`: 批量提取特征
  - `compute_derived()`: 计算衍生特征（需要批量数据）
  - `enrich_with_daily_data()`: 添加技术指标（需要日线数据）
  - `extract_with_history()`: 静态方法，一步到位提取完整特征
  - `to_array()`: 转换为模型可用的数组

### 2. 训练数据构建 (`build_training_data.py`)

**功能**: 从历史数据构建机器学习训练数据集

**流程**:
1. 获取每月第一个交易日（与回测逻辑一致）
2. 对每个月：
   - 获取沪深300成分股
   - 对每只股票：
     - 获取买入时点数据（月初第一个交易日）
     - 获取卖出时点数据（下月初第一个交易日）
     - 计算持仓收益
     - **直接构建特征字典**（不使用 FeatureEngineer，简化流程）
3. 保存为 JSON 文件

**特征字段**（与 `StockFeatures` 保持一致）:
```python
{
    'code', 'name', 'buy_date', 'sell_date', 'buy_price', 'sell_price',
    'change_pct',           # 价格相关
    'pe_ratio', 'pb_ratio', # 估值特征
    'momentum_20d', 'momentum_60d', # 动量特征
    'turnover_rate',        # 交易特征
    'roe', 'profit_growth', 'dividend_yield' # 财务特征
}
```

**输出格式**:
```json
[
    {
        "features": {...},
        "return_pct": 5.2,
        "holding_days": 22
    },
    ...
]
```

### 3. 股票预测器 (`stock_predictor.py`)

#### 3.1 数据类
- **StockFeatures**: 股票特征数据类
  - 与 `StockData` 字段保持一致
  - `FEATURE_NAMES`: 特征名称列表（9个特征）
  - `to_array()`: 转换为特征数组
  - `from_stock_data()`: 从 StockData 创建

- **PredictionResult**: 预测结果数据类
  - `prob_up`, `prob_down`, `prob_neutral`: 三类概率
  - `confidence`: 置信度（最高概率）
  - `prediction`: 预测类别

#### 3.2 预测器类
- **StockPredictor**: 主预测器
  - 支持模型类型：`random_forest`, `xgboost`, `logistic`
  - `train()`: 训练模型
  - `predict()`: 预测单只股票
  - `predict_batch()`: 批量预测
  - `prepare_training_data()`: 准备训练数据

- **RandomPredictor**: 随机预测器（用于对比基准）

### 4. 训练和评估 (`train_and_evaluate.py`)

**功能**: 训练模型并在历史数据上回测评估

**流程**:
1. 加载训练数据
2. 训练模型（`train_model()`）
3. 在测试集上回测（`backtest_with_ml()`）
   - 按月份分组
   - 对每个月：预测所有股票，筛选高置信度股票
   - 计算策略收益
4. 输出评估报告

## 数据流

### 训练数据构建流程
```
历史数据 (DataManager)
    ↓
每月第一个交易日
    ↓
沪深300成分股
    ↓
获取买入/卖出时点数据
    ↓
构建特征字典 (直接构建，不使用 FeatureEngineer)
    ↓
计算收益标签
    ↓
保存为 JSON (ml_training_data.json)
```

### 模型训练流程
```
训练数据 (JSON)
    ↓
加载数据
    ↓
准备特征和标签 (StockPredictor.prepare_training_data)
    ↓
训练模型 (RandomForest/XGBoost/Logistic)
    ↓
保存模型 (pickle)
```

### 预测流程
```
StockData
    ↓
转换为 StockFeatures
    ↓
提取特征数组
    ↓
模型预测
    ↓
PredictionResult (概率 + 置信度)
```

## 特征一致性

### 核心特征列表（9个）
与 `StockFeatures.FEATURE_NAMES` 保持一致：
1. `change_pct` - 买入日涨跌幅
2. `pe_ratio` - 市盈率
3. `pb_ratio` - 市净率
4. `momentum_20d` - 20日动量
5. `momentum_60d` - 60日动量
6. `turnover_rate` - 换手率
7. `roe` - ROE
8. `profit_growth` - 净利润增长率
9. `dividend_yield` - 股息率

### 数据源
- **StockData**: 核心数据类，包含所有基础数据
- **DataManager**: 统一数据获取接口
- **特征提取**: 
  - `build_training_data.py` 中直接构建特征字典（简化）
  - `test_features.py` 中使用 `FeatureEngineer`（完整功能）

## 使用示例

### 1. 构建训练数据
```python
python ml/build_training_data.py
# 生成 data/ml_training_data.json
```

### 2. 训练模型
```python
from ml.train_and_evaluate import load_training_data, train_model

# 加载数据
data = load_training_data('data/ml_training_data.json')

# 训练模型
predictor = train_model(data, model_type='random_forest')
```

### 3. 预测股票
```python
from ml.stock_predictor import StockPredictor, StockFeatures
from data.manager import DataManager

# 加载模型
predictor = StockPredictor()
predictor.load_model('models/stock_predictor.pkl')

# 获取股票数据
dm = DataManager()
stock = dm.get_stock_data('600036', '2024-01-01')

# 预测
features = StockFeatures.from_stock_data(stock)
result = predictor.predict(features)
print(f"上涨概率: {result.prob_up:.2%}, 置信度: {result.confidence:.2%}")
```

### 4. 测试特征提取
```python
python ml/test_features.py
```

## 注意事项

1. **特征一致性**: `build_training_data.py` 中直接构建特征字典，确保与 `StockFeatures` 字段一致
2. **持仓天数**: 动态计算，与回测逻辑保持一致
3. **标签定义**: 上涨 > 5%, 下跌 < -5%, 其他为持平
4. **数据缓存**: 使用 `DataManager` 的缓存机制，提高数据获取效率
5. **特征工程**: `features/` 模块提供完整功能，但 `build_training_data.py` 中简化使用

## 未来改进方向

1. 支持更多技术指标特征
2. 特征选择和重要性分析
3. 模型集成（Ensemble）
4. 在线学习（增量训练）
5. 特征重要性可视化
