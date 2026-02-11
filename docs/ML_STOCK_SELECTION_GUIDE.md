# ML 选股策略 — 完整流程与操作指南

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        完整流程                                  │
│                                                                  │
│  ① 季度数据构建          ② 模型训练            ③ 月度选股        │
│  batch_build_data.py  →  train.py           →  backtest.py      │
│                                                                  │
│  data/quarterly_data_*    models/*.pkl         选股结果输出       │
│                                                                  │
│  (每季度做一次)          (每季度做一次)        (每月初做一次)     │
└─────────────────────────────────────────────────────────────────┘
```

### 关键依赖关系

| 步骤 | 输入 | 输出 | 频率 |
|------|------|------|------|
| 数据构建 | Tushare API（成分股+行情+财务） | `data/quarterly_data_*/YYYY_QN_ml_training_data.json` | 每季度 |
| 模型训练 | 季度 JSON 数据 | `models/{index}_{model}_{year}q{quarter}.pkl` | 每季度 |
| 月度选股 | 训练好的模型 + 实时行情 | 控制台输出 Top N 股票 | 每月初 |

---

## 二、Step 1 — 季度训练数据构建

### 做什么

获取指数成分股的行情、基本面、技术指标、市场环境特征，计算每只股票的持有期收益，生成模型训练所需的 JSON 数据。

### 何时做

**每个季度开始后（1/4/7/10月初）**，等待上季度最后一个月的行情数据就绪后执行。
例如：2026Q2 开始（4月初），此时可构建 2026Q1 的数据（1-3月已有完整数据）。

### 命令

```bash

验证 python ml/batch_build_data.py --quarters 2025_Q4 --no-cache --output-dir data/test_data --index 000300

沪深300 — 构建指定季度
python ml/batch_build_data.py --quarters 2026_Q1 --index 000300 --output-dir data/
quarterly_data_v2

# 中证500 — 构建指定季度
python ml/batch_build_data.py --quarters 2026_Q1 --index 000905 --output-dir data/quarterly_data_csi500

# 批量构建（补历史数据）
python ml/batch_build_data.py --years 2024 2025 --index 000905 --output-dir data/quarterly_data_csi500
```

### 输出文件

```
data/quarterly_data_v2/2026_Q1_ml_training_data.json       # 沪深300
data/quarterly_data_csi500/2026_Q1_ml_training_data.json   # 中证500
```


---

## 三、Step 2 — 季度模型训练

### 做什么

用滑动窗口（默认前 8 个季度 = 2 年）的历史数据训练回归模型，预测每只股票的「相对收益」（跑赢/跑输基准的幅度 %）。

### 何时做

在 Step 1 完成后立即执行。每季度训练一个模型，该模型供本季度 3 个月的选股使用。

### 命令

```bash
# === 沪深300 模型 ===
# 训练单季度（推荐用于增量更新）
验证
python ml/train.py --output-dir new_models --window-quarters 8 --quarter 2025Q4 --feature-set full --model rf_200 --index 000300

python ml/train.py --output-dir models --window-quarters 8 --quarter 2026Q1 --feature-set full --model rf_200 --index 000300






# === 中证500 模型 ===
python ml/train.py --quarter 2026Q2 \
    --quarterly-data data/quarterly_data_csi500 \
    --output-dir models \
    --index 000905 \
    --model hgb_deep \
    --window-quarters 8

# 批量训练（补全历史模型）
python ml/train.py --train-quarterly \
    --quarterly-data data/quarterly_data_csi500 \
    --output-dir models \
    --start-year 2021 --end-year 2026 \
    --index 000905 --model hgb_deep
```

### 模型文件命名规则

```
models/{index_tag}_{model_tag}_{year}q{quarter}.pkl

示例:
  models/csi300_hgb_deep_2026q2.pkl
  models/csi500_hgb_deep_2026q2.pkl
```

### 模型内容（pickle）

| 字段 | 说明 |
|------|------|
| `model` | 训练好的模型对象（HGB/RF） |
| `scaler` | StandardScaler（特征标准化） |
| `feature_names` | 使用的特征列表 |
| `label_type` | `'relative'`（相对收益） |
| `threshold_up/down` | 3.0 / -3.0 |
| `model_type` | `'hgb_regressor'` / `'rf_regressor'` |

### 关键参数

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `--model` | `hgb_deep` | 模型类型，可选 `hgb_shallow/medium/deep`, `rf_100/200` |
| `--window-quarters` | `8` | 训练窗口（8 季度 = 2 年），可通过 strategy_optimizer 实验确定 |
| `--feature-set` | `full` | 特征组，可选 `base`, `momentum`, `full` |
| `--index` | `000905` | 指数代码 |

---

## 四、Step 3 — 月度选股

### 做什么

在每月初的交易日，加载当季度模型，对指数成分股进行预测打分，选出 Top N。

### 何时做

**每月第一个交易日**。模型在季度内不变，只是候选股票池和行情数据是当天的。

### 命令

```bash
# 中证500 选股（当前默认配置）
python backtest.py --select 20260401 --benchmark 000905

# 沪深300 选股
验证
python backtest.py --benchmark 000300 --select 20251201
python ml/quarterly_selector.py --top-n 5 --start-year 2012 --end-year 2025 --window-quarters 8 --features full --model rf_200 --index 000300 与logs/strategy_optimizer对应的文件对比summary_20260210——135007.txt
python ml/quarterly_selector.py --select 2025q4 --window-quarters 8 --features full --model rf_200 --index 000300

python backtest.py --select 20260401 --benchmark 000300

# 指定选股数量
python backtest.py --select 20260401 --benchmark 000905 --top-n 10
```

### 选股流程（内部）

```
1. DataManager._resolve_trading_date(date)  →  解析为实际交易日
2. get_index_constituents(index_code, date) →  获取成分股（约 300/500 只）
3. get_stock_data(code, trading_date)       →  获取每只股票行情
4. MLStrategy.set_current_date(date)        →  按 model_schedule 加载对应季度模型
5. MLStrategy.select(stocks, top_n)         →  提取特征 → 模型预测 → 排序 → 返回 Top N
```

### model_schedule 配置

在 `config/strategy_config.py` 中：

```python
'model_schedule': _build_model_schedule('csi500', 'hgb_deep', 2021, 2026)
```

自动生成映射：`2026-04` → `models/csi500_hgb_deep_2026q2.pkl`

**注意**：训练模型的 `--model` 和 `--index` 参数必须与此配置一致，否则会找不到模型文件。

---

## 五、策略参数优化

### 做什么

通过历史回测实验，确定最优的特征组合、模型类型、窗口大小、选股数量。

### 命令

```bash
# 运行完整优化实验（遍历特征组合 × 模型 × 窗口 × Top K）
python ml/strategy_optimizer.py --index 000905 --start-year 2012

# 报告输出到 logs/strategy_optimizer/
```

### 输出

- `logs/strategy_optimizer/report_*.json` — 完整实验数据
- `logs/strategy_optimizer/summary_*.txt` — 最优配置摘要

### 与 quarterly_selector 的关系

`ml/quarterly_selector.py` 使用与 strategy_optimizer 完全一致的逻辑（读同样的季度 JSON），可用于快速验证：

```bash
# 查看历史回测表现（应与 strategy_optimizer 结果一致）
python ml/quarterly_selector.py --backtest --index 000905 --model hgb_deep --window-quarters 8

# 按季度选股（基于历史数据，不依赖训练好的模型文件）
python ml/quarterly_selector.py --select 2025Q4 --index 000905
```

---

## 六、风险管控与验证建议

### 风险 1：模型参数不一致

**问题**：`train.py` 训练时的参数（模型类型、窗口大小、特征组）与 `strategy_config.py` 的 `model_schedule` 不一致。

**建议**：

1. **训练时保存元数据**：在模型 pickle 中已包含 `feature_names` 和 `model_type`。建议额外保存：
   - `window_quarters`：训练窗口
   - `index_code`：指数代码
   - `train_samples`：训练样本数
   - `train_quarters`：具体用了哪些季度
   - `train_timestamp`：训练时间

2. **选股时校验**：`MLStrategy` 加载模型后，校验模型中的 `feature_names` 是否与当前 `FeatureEngineer` 输出一致。

### 风险 2：模型一致性验证（MD5 方案分析）

**MD5 对比不可靠**，原因：
- HGB/RF 等树模型含随机性，即使数据完全相同，不同次训练的 pickle 二进制不同
- 不同 Python/sklearn 版本也会导致差异

**更好的替代方案 — 回测指标对比**：

```bash
# 训练完新模型后，用 quarterly_selector 对已知季度做回测
python ml/quarterly_selector.py --backtest \
    --index 000905 --model hgb_deep --window-quarters 8 \
    --start-year 2023 --end-year 2025

# 将年化 Alpha、Precision、Win Rate 与上次记录对比
# 偏差 < 1% → 正常（浮点/随机性）
# 偏差 > 5% → 需排查数据或参数变化
```

**推荐做法**：在每次训练后，自动运行一轮回测并将核心指标（Alpha、Precision、样本数）写入日志文件，形成历史基线。

### 风险 3：训练数据质量

**问题**：季度数据不完整或特征缺失。

**建议**：

```bash
# 每次构建数据后检查
python ml/batch_build_data.py --check

# 重点关注：
# - 每个季度的记录数是否合理（沪深300 约 900 条/季度，中证500 约 1500 条/季度）
# - 每个月是否都有数据（一个季度应有 3 个月）
# - 特征数是否与预期一致（full 约 27 个特征）
```

### 风险 4：选股时数据异常

**问题**：选股日行情数据未更新、非交易日等。

**已有防护**（本次重构）：
- `DataManager._resolve_trading_date` 通过交易日历自动解析非交易日
- `_get_daily_data` 精确匹配交易日，数据未就绪时返回 None + WARNING
- 缓存 key 使用实际交易日，不会产生脏数据

---

## 七、季度操作 Checklist

每个季度初（1/4/7/10月），按以下顺序操作：

### 1. 构建新季度数据

```bash
# 确认上季度数据是否可获取
# 例：4月初构建 Q1 数据

# 沪深300
python ml/batch_build_data.py --quarters 2026_Q1 --index 000300 --output-dir data/quarterly_data_v2
# 中证500
python ml/batch_build_data.py --quarters 2026_Q1 --index 000905 --output-dir data/quarterly_data_csi500

# 验证
python ml/batch_build_data.py --check
```

### 2. 训练新季度模型

```bash
# 沪深300 (如需)
python ml/train.py --quarter 2026Q2 --index 000300 --model hgb_deep \
    --quarterly-data data/quarterly_data_v2 --output-dir models

# 中证500
python ml/train.py --quarter 2026Q2 --index 000905 --model hgb_deep \
    --quarterly-data data/quarterly_data_csi500 --output-dir models
```

### 3. 验证模型（回测对比）

```bash
# 用 quarterly_selector 回测，确认指标在预期范围内
python ml/quarterly_selector.py --backtest --index 000905 --model hgb_deep \
    --start-year 2023 --end-year 2025

# 对比上一次的回测结果，Alpha/Precision 偏差 < 5% 为正常
```

### 4. 确认配置

检查 `config/strategy_config.py` 中 `model_schedule` 的参数：

```python
# 确保 index_tag 和 model_tag 与训练命令一致
_build_model_schedule('csi500', 'hgb_deep', 2021, 2026)
#                      ^^^^^^   ^^^^^^^^
#                      与 --index 000905 对应
#                      与 --model hgb_deep 对应
```

确保 `end_year` 覆盖了新训练的季度。

### 5. 月度选股

```bash
# 每月第一个交易日执行
python backtest.py --select 20260401 --benchmark 000905
```

---

## 八、文件目录总览

```
JYSstock_analyzer/
├── config/
│   └── strategy_config.py        # ML 策略配置（model_schedule）
├── data/
│   ├── quarterly_data_v2/        # 沪深300 季度训练数据
│   │   ├── 2020_Q1_ml_training_data.json
│   │   └── ...
│   └── quarterly_data_csi500/    # 中证500 季度训练数据
├── models/                       # 训练好的模型文件
│   ├── csi300_hgb_deep_2025q4.pkl
│   ├── csi500_hgb_deep_2025q4.pkl
│   └── ...
├── ml/
│   ├── batch_build_data.py       # Step 1: 批量构建训练数据
│   ├── build_training_data.py    # 单季度数据构建（被 batch 调用）
│   ├── train.py                  # Step 2: 模型训练
│   ├── quarterly_selector.py     # 回测验证 / 基于历史数据选股
│   ├── strategy_optimizer.py     # 参数优化实验
│   ├── models/                   # 模型定义（HGB/RF）
│   └── features/                 # 特征工程
├── strategy/
│   ├── ml_strategy.py            # MLStrategy 实现
│   └── registry.py               # 策略注册与创建
├── backtest.py                   # Step 3: 回测 / 选股入口
└── docs/
    └── ML_STOCK_SELECTION_GUIDE.md  # 本文档
```

---

## 九、常见问题

### Q: quarterly_selector 和 backtest --select 选出的股票不同？

这是正常的。两者的区别：

| | quarterly_selector --select | backtest --select |
|---|---|---|
| 数据来源 | 预构建的季度 JSON（历史数据） | 实时从 Tushare 获取当天行情 |
| 模型 | 每次从头训练（临时模型） | 加载已保存的 .pkl 模型 |
| 成分股 | JSON 中记录的历史成分股 | 当天实际的指数成分股 |
| 用途 | 回测验证、研究 | 实际操作选股 |

如果同一天选出的股票差异很大，检查：
1. 模型参数是否一致（model、window_quarters、feature_set）
2. 成分股列表是否一致（指数调仓可能导致差异）
3. 行情数据日期是否一致

### Q: 训练时报 "数据不足" 错误？

说明 `window_quarters` 要求的历史季度数据不完整。解决：
```bash
# 查看已有数据
python ml/batch_build_data.py --check

# 补充缺失季度
python ml/batch_build_data.py --quarters 2024_Q3 2024_Q4 --index 000905 --output-dir data/quarterly_data_csi500
```

### Q: 选股时报 "数据尚未更新"？

这是 DataManager 的保护机制：请求的交易日行情数据尚未被 Tushare 推送。通常在收盘后 1-2 小时内更新。建议晚间执行选股。
