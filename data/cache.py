# -*- coding: utf-8 -*-
"""
缓存管理器

提供数据缓存功能：
- 文件缓存（pickle）
- 内存缓存
- 缓存过期管理
"""

import os
import time
import pickle
import logging
from typing import Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CacheManager:
    """
    缓存管理器
    
    支持文件缓存和内存缓存，自动管理缓存过期。
    
    使用示例:
        cache = CacheManager(cache_dir='./cache', expire_days=7)
        
        # 保存数据
        cache.set('stock_600036_2024-01-01', stock_data)
        
        # 读取数据
        data = cache.get('stock_600036_2024-01-01')
    """
    
    def __init__(self, cache_dir: str = './cache', expire_days: int = 7):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
            expire_days: 缓存过期天数
        """
        self.cache_dir = cache_dir
        self.expire_days = expire_days
        self.expire_seconds = expire_days * 86400
        
        # 内存缓存
        self._memory_cache: dict = {}
        self._memory_cache_time: dict = {}
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_file_path(self, key: str) -> str:
        """获取缓存文件路径"""
        # 将key中的特殊字符替换为下划线
        safe_key = key.replace('/', '_').replace('\\', '_').replace(':', '_')
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存数据
        
        优先从内存缓存获取，其次从文件缓存获取。
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            缓存数据，不存在或过期返回default
        """
        # 1. 尝试内存缓存
        if key in self._memory_cache:
            cache_time = self._memory_cache_time.get(key, 0)
            if time.time() - cache_time < self.expire_seconds:
                return self._memory_cache[key]
            else:
                # 过期，清理内存缓存
                del self._memory_cache[key]
                del self._memory_cache_time[key]
        
        # 2. 尝试文件缓存
        file_path = self._get_file_path(key)
        if os.path.exists(file_path):
            try:
                # 检查文件是否过期
                file_mtime = os.path.getmtime(file_path)
                if time.time() - file_mtime < self.expire_seconds:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # 加载到内存缓存
                    self._memory_cache[key] = data
                    self._memory_cache_time[key] = file_mtime
                    
                    return data
                else:
                    # 过期，删除文件
                    os.remove(file_path)
            except Exception as e:
                logger.debug(f"读取缓存失败 {key}: {e}")
        
        return default
    
    def set(self, key: str, value: Any) -> bool:
        """
        设置缓存数据
        
        同时保存到内存和文件。
        
        Args:
            key: 缓存键
            value: 缓存值
            
        Returns:
            是否成功
        """
        try:
            # 1. 保存到内存
            self._memory_cache[key] = value
            self._memory_cache_time[key] = time.time()
            
            # 2. 保存到文件
            file_path = self._get_file_path(key)
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            
            return True
        except Exception as e:
            logger.warning(f"保存缓存失败 {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            是否成功
        """
        try:
            # 删除内存缓存
            if key in self._memory_cache:
                del self._memory_cache[key]
            if key in self._memory_cache_time:
                del self._memory_cache_time[key]
            
            # 删除文件缓存
            file_path = self._get_file_path(key)
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return True
        except Exception as e:
            logger.warning(f"删除缓存失败 {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        检查缓存是否存在且未过期
        
        Args:
            key: 缓存键
            
        Returns:
            是否存在
        """
        return self.get(key) is not None
    
    def clear(self) -> int:
        """
        清空所有缓存
        
        Returns:
            清理的缓存数量
        """
        count = 0
        
        # 清空内存缓存
        count += len(self._memory_cache)
        self._memory_cache.clear()
        self._memory_cache_time.clear()
        
        # 清空文件缓存
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
                    count += 1
        except Exception as e:
            logger.warning(f"清空缓存目录失败: {e}")
        
        logger.info(f"清空缓存: {count} 个")
        return count
    
    def clear_expired(self) -> int:
        """
        清理过期缓存
        
        Returns:
            清理的缓存数量
        """
        count = 0
        current_time = time.time()
        
        # 清理内存中过期的
        expired_keys = [
            k for k, t in self._memory_cache_time.items()
            if current_time - t >= self.expire_seconds
        ]
        for key in expired_keys:
            del self._memory_cache[key]
            del self._memory_cache_time[key]
            count += 1
        
        # 清理文件中过期的
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    if current_time - os.path.getmtime(file_path) >= self.expire_seconds:
                        os.remove(file_path)
                        count += 1
        except Exception as e:
            logger.warning(f"清理过期缓存失败: {e}")
        
        if count > 0:
            logger.info(f"清理过期缓存: {count} 个")
        return count
    
    def get_stats(self) -> dict:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        memory_count = len(self._memory_cache)
        
        file_count = 0
        total_size = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_count += 1
                    file_path = os.path.join(self.cache_dir, filename)
                    total_size += os.path.getsize(file_path)
        except:
            pass
        
        return {
            'memory_count': memory_count,
            'file_count': file_count,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': self.cache_dir,
            'expire_days': self.expire_days
        }
