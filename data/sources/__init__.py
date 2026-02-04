# -*- coding: utf-8 -*-
"""
数据源模块

提供多种数据源实现：
- TushareSource: Tushare数据源（推荐，付费稳定，支持历史数据）
- AkShareSource: AkShare数据源（免费，可能不稳定）
- TencentSource: 腾讯API数据源（实时数据，免费稳定）

数据源选择建议：
- 历史回测: 优先使用 TushareSource（数据完整，无前视偏差）
- 免费回测: 使用 AkShareSource（需要处理接口变动）
- 实时监控: 使用 TencentSource（实时行情稳定）
"""

from data.sources.tushare_source import TushareSource
from data.sources.akshare_source import AkShareSource
from data.sources.tencent_source import TencentSource

# 数据源注册表
AVAILABLE_SOURCES = {
    'tushare': TushareSource,
    'akshare': AkShareSource,
    'tencent': TencentSource,
}

# 默认数据源
DEFAULT_SOURCE = 'tushare'


def get_source(name: str):
    """
    获取数据源类
    
    Args:
        name: 数据源名称（tushare/akshare/tencent）
        
    Returns:
        数据源类
        
    Raises:
        ValueError: 未知的数据源名称
    """
    if name not in AVAILABLE_SOURCES:
        raise ValueError(f"未知数据源: {name}, 可用: {list(AVAILABLE_SOURCES.keys())}")
    return AVAILABLE_SOURCES[name]


def list_sources() -> list:
    """列出所有可用数据源"""
    return list(AVAILABLE_SOURCES.keys())


__all__ = [
    'TushareSource',
    'AkShareSource',
    'TencentSource',
    'AVAILABLE_SOURCES',
    'DEFAULT_SOURCE',
    'get_source',
    'list_sources',
]
