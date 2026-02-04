# -*- coding: utf-8 -*-

"""
数据源配置 - 管理各数据源的连接参数

数据源对比:
    | 数据源  | 费用 | 历史数据 | 实时数据 | 稳定性 | 推荐场景 |
    |---------|------|----------|----------|--------|----------|
    | Tushare | 付费 | ✅ 完整   | ✅       | 高     | 历史回测 |
    | AkShare | 免费 | ✅ 部分   | ❌       | 中     | 免费回测 |
    | Tencent | 免费 | ❌        | ✅ 稳定   | 高     | 实时监控 |
"""

import os

# 默认数据源
DEFAULT_DATA_SOURCE = 'tushare'

# Tushare配置
TUSHARE_TOKEN = os.environ.get(
    'TUSHARE_TOKEN',
    'd1d61bfadf15055f02d893af6bb406f5dd3909dd6dc9a3b3c6bb5a72'
)

# 数据源配置
DATA_SOURCE_CONFIGS = {
    # ============================================
    # Tushare - 付费数据源（推荐）
    # ============================================
    'tushare': {
        'name': 'Tushare',
        'description': '专业金融数据接口，数据完整稳定',
        'token': TUSHARE_TOKEN,
        'enabled': True,
        'params': {
            'request_delay': 0.12,      # 请求间隔（秒），免费用户限制
            'batch_size': 100,          # 批量请求大小
            'retry_times': 3,           # 重试次数
            'timeout': 30,              # 超时时间（秒）
        },
        'features': {
            'historical_data': True,    # 支持历史数据
            'realtime_data': True,      # 支持实时数据
            'fundamental_data': True,   # 支持基本面数据
            'index_constituents': True, # 支持指数成分股
        }
    },
    
    # ============================================
    # AkShare - 免费数据源
    # ============================================
    'akshare': {
        'name': 'AkShare',
        'description': '开源金融数据接口，免费但不稳定',
        'enabled': True,
        'params': {
            'request_delay': 0.5,       # 请求间隔
            'retry_times': 3,
            'timeout': 60,
        },
        'features': {
            'historical_data': True,
            'realtime_data': False,     # 实时数据不稳定
            'fundamental_data': True,
            'index_constituents': True,
        }
    },
    
    # ============================================
    # Tencent - 实时数据源
    # ============================================
    'tencent': {
        'name': 'Tencent Finance',
        'description': '腾讯财经API，免费实时行情',
        'enabled': True,
        'params': {
            'batch_size': 80,           # 每次最多查询80只
            'request_delay': 0.1,
            'timeout': 10,
        },
        'features': {
            'historical_data': False,   # 不支持历史数据
            'realtime_data': True,
            'fundamental_data': True,   # 部分基本面数据
            'index_constituents': False,
        }
    },
}

# 数据源优先级（自动回退顺序）
DATA_SOURCE_PRIORITY = ['tushare', 'akshare', 'tencent']

# 功能到数据源的映射（推荐用法）
FEATURE_SOURCE_MAP = {
    'backtest': 'tushare',      # 回测推荐Tushare
    'realtime': 'tencent',      # 实时监控推荐腾讯
    'free_backtest': 'akshare', # 免费回测用AkShare
}
