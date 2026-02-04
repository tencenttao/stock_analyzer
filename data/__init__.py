# -*- coding: utf-8 -*-
"""
数据层

提供统一的数据获取接口：
- DataManager: 数据管理器，统一数据入口
- 各数据源实现: TushareSource, AkShareSource 等
- CacheManager: 缓存管理
"""

from data.manager import DataManager
from data.cache import CacheManager

__all__ = [
    'DataManager',
    'CacheManager',
]
