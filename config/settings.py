# -*- coding: utf-8 -*-

"""
全局设置 - 项目级配置

所有模块的默认参数统一在此定义，代码中不应有硬编码的默认值。
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ===== 项目路径 =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
REPORT_DIR = os.path.join(LOG_DIR, 'backtest')

# 确保目录存在
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ===== 日志配置 =====
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_CONFIG = {
    'level': LOG_LEVEL,
    'format': LOG_FORMAT,
    'file': os.path.join(LOG_DIR, 'stock_analyzer.log')
}

# ===== 缓存配置 =====
CACHE_EXPIRE_DAYS = 700       # 缓存过期天数
USE_MEMORY_CACHE = True     # 启用内存缓存
USE_FILE_CACHE = True       # 启用文件缓存

# ===== 并发配置 =====
MAX_WORKERS = 4             # 最大并发数
REQUEST_DELAY = 0.1         # 请求间隔（秒）

# ===== 显示配置 =====
VERBOSE = True              # 详细输出
PROGRESS_BAR = True         # 显示进度条

# ===== 邮件配置（实盘通知用）=====
EMAIL_CONFIG = {
    'smtp_server': 'smtp.qq.com',
    'smtp_port': 587,
    'email': os.getenv('EMAIL_ADDRESS'),
    'password': os.getenv('EMAIL_PASSWORD'),
    'to_email': ['1120311927@qq.com', '18943656696@163.com', '1356163565@qq.com']
}

# ===== 调度配置（实盘定时任务）=====
SCHEDULE_CONFIG = {
    'analysis_time': '16:00',    # 盘后分析时间
    'email_time': '16:30',       # 收盘后发送邮件时间
    'weekdays_only': True,       # 仅工作日运行
    'immediate_email': True      # 分析完成后立即发送邮件
}

# ===== 数据源配置 =====
DATA_CONFIG = {
    'akshare_timeout': 30,
    'retry_times': 3,
    'cache_dir': CACHE_DIR
}

# ===== 回测筛选配置 =====
BACKTEST_FILTER_CONFIG = {
    'max_pe_ratio': 50,          # PE上限
    'min_turnover': 3000,        # 最小成交额（万元）
    'momentum_days': 20,         # 动量计算天数
    'min_price': 1.0,            # 最小股价
    'max_stocks': 10,            # 推荐数量
    'min_strength_score': 40     # 强势分数要求
}

# ===== 回测采样配置 =====
# 候选池已改为“基准指数全部成分股”，不再采样
BACKTEST_SAMPLE_CONFIG = {
    'random_seed': 42,
    'use_cache': True,
    'cache_expire_days': 7
}

# ===== 回测输出配置 =====
BACKTEST_OUTPUT_CONFIG = {
    'save_json': True,           # 保存JSON结果
    'save_report': True,         # 自动生成报告
    'verbose': True              # 详细日志输出
}

# ===== 回测主配置 =====
# 运行 python backtest.py 时使用这些配置
BACKTEST_CONFIG = {
    # 回测日期范围
    'start_date': '2021-01-01',     # 开始日期
    'end_date': '2026-01-15',       # 结束日期（覆盖完整 Q1）
    
    # 资金和基准
    'initial_capital': 100000,      # 初始资金
    'benchmark': '000905',          # 基准指数（000300=沪深300, 000905=中证500）
    
    # 选股参数（候选池=基准指数全部成分股）
    'top_n': 6,                     # 每月选股数量
    'random_seed': 42,              # 随机种子
    
    # 交易成本
    'commission_rate': 0.00025,     # 佣金率（万2.5）
    'stamp_tax_rate': 0.001,        # 印花税（千1）
    'slippage': 0.001,              # 滑点（0.1%）
    'transfer_fee_rate': 0.00001,   # 过户费（万0.1）
    'min_commission': 5.0,          # 最低佣金
    'enable_cost': False,           # 是否计算交易成本
    
    # 其他
    'save_report': True,            # 是否保存报告
}

# 为了向后兼容，保留别名
BACKTEST_DEFAULTS = BACKTEST_CONFIG

# ===== 选股配置 =====
SELECTION_CONFIG = {
    'index_code': '000300',         # 默认股票池（000300=沪深300, 000905=中证500）
    'min_score': 40,                # 最低评分阈值
}

# 向后兼容的别名
SELECTION_DEFAULTS = SELECTION_CONFIG
STOCK_FILTER_CONFIG = BACKTEST_FILTER_CONFIG  # 兼容旧引用
