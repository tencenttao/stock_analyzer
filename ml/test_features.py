# -*- coding: utf-8 -*-
"""
特征工程模块测试脚本

与 ml/features/engineer.py 对接：
- 使用 FeatureConfig 指定要提取的特征
- 使用 extract(stock, daily_data) 一次性按配置提取全部特征（推荐）
- 或使用 extract_with_history(stock, daily_data) 一步到位

运行方式:
    python ml/test_features.py          # 默认 INFO 级别
    python ml/test_features.py --debug  # DEBUG 级别
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志级别
log_level = logging.DEBUG if '--debug' in sys.argv else logging.INFO
log_level = logging.DEBUG
logging.basicConfig(
    level=log_level,
    format='%(levelname)s - %(name)s - %(message)s'
)

from data.manager import DataManager
from ml.features import (
    FeatureEngineer,
    FeatureConfig,
    FULL_FEATURE_CONFIG,
    DEFAULT_FEATURE_CONFIG,
)

print("=" * 50)
print("特征工程模块测试（与 engineer 适配）")
print("=" * 50)

dm = DataManager()
test_code = '600519'  # 招商银行
test_date = '2025-01-02'

# 1. 获取股票数据
print(f"\n1. 获取股票数据: {test_code} @ {test_date}")
stock = dm.get_stock_data(test_code, test_date)
if not stock:
    print("   错误: 无法获取股票数据")
    sys.exit(1)
print(f"   股票: {stock.name}, 价格: {stock.price:.2f}")

# 2. 按配置提取全部特征（推荐：一次调用）
print("\n2. 按 FeatureConfig 提取全部特征 [extract(stock, daily_data)]")
config = FULL_FEATURE_CONFIG
engineer = FeatureEngineer(config)
print(f"   配置特征: {engineer.get_feature_names()}")

daily = dm.get_daily_data(test_code, end_date=test_date)
if daily:
    print(f"   日线数据: {len(daily)} 条，将用于技术指标")
else:
    print("   日线数据: 无，技术指标使用默认值")

features = engineer.extract(stock, daily_data=daily)
print(f"   特征数量: {len(features)}")
for k in sorted(features.keys()):
    v = features[k]
    if isinstance(v, float):
        print(f"   - {k}: {v:.4f}")
    else:
        print(f"   - {k}: {v}")

# 3. 转换为数组
print("\n3. 转换为数组 [to_array]")
arr = engineer.to_array(features)
print(f"   形状: {arr.shape}, 类型: {arr.dtype}")
print(f"   数值特征名（前5）: {engineer.get_numeric_feature_names(features)[:5]}")

# 4. 一步到位 [extract_with_history]（等价于 FeatureEngineer(config).extract(stock, daily)）
print("\n4. 一步到位 [extract_with_history]")
features2 = FeatureEngineer.extract_with_history(stock, daily, config=config)
print(f"   特征数量: {len(features2)}")

# 5. 使用默认配置（仅基础+动量，无日线技术指标）
print("\n5. 默认配置 [DEFAULT_FEATURE_CONFIG]（无日线）")
eng_default = FeatureEngineer(DEFAULT_FEATURE_CONFIG)
features3 = eng_default.extract(stock, daily_data=None)
print(f"   特征数量: {len(features3)}")
print(f"   特征名: {eng_default.get_feature_names()}")

print("\n" + "=" * 50)
print("测试完成!")
print("=" * 50)
