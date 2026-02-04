# -*- coding: utf-8 -*-
"""
策略注册表

管理所有可用的选股策略，支持：
- 通过装饰器注册策略
- 通过名称获取策略
- 列出所有可用策略
"""

import logging
from typing import Dict, Type, List, Optional

from core.interfaces import Strategy

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    策略注册表
    
    管理所有已注册的选股策略。
    
    使用示例:
        # 注册策略（使用装饰器）
        @StrategyRegistry.register('my_strategy')
        class MyStrategy(Strategy):
            ...
        
        # 获取策略
        strategy_cls = StrategyRegistry.get('momentum_v2')
        strategy = strategy_cls(config={'top_n': 10})
        
        # 列出所有策略
        strategies = StrategyRegistry.list_all()
    """
    
    _strategies: Dict[str, Type[Strategy]] = {}
    _descriptions: Dict[str, str] = {}
    
    @classmethod
    def register(cls, name: str, description: str = ""):
        """
        装饰器：注册策略
        
        Args:
            name: 策略名称（唯一标识）
            description: 策略描述
            
        Returns:
            装饰器函数
            
        使用示例:
            @StrategyRegistry.register('momentum_v2', '动量优先策略V2')
            class MomentumV2Strategy(Strategy):
                ...
        """
        def decorator(strategy_cls: Type[Strategy]):
            if name in cls._strategies:
                logger.warning(f"策略 '{name}' 已存在，将被覆盖")
            
            cls._strategies[name] = strategy_cls
            cls._descriptions[name] = description or strategy_cls.__doc__ or ""
            
            logger.debug(f"注册策略: {name} -> {strategy_cls.__name__}")
            return strategy_cls
        
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[Strategy]:
        """
        获取策略类
        
        Args:
            name: 策略名称
            
        Returns:
            策略类
            
        Raises:
            ValueError: 未知的策略名称
        """
        if name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(f"未知策略: '{name}'，可用策略: {available}")
        
        return cls._strategies[name]
    
    @classmethod
    def create(cls, name: str, config: Dict = None) -> Strategy:
        """
        创建策略实例
        
        支持两种方式：
        1. 如果策略已注册（有对应的类），直接使用该类
        2. 如果策略在配置文件中但未注册，根据配置类型选择基础类
        
        Args:
            name: 策略名称
            config: 策略配置（会与配置文件中的配置合并）
            
        Returns:
            策略实例
        """
        # 从配置文件读取策略配置
        from config.strategy_config import STRATEGY_CONFIGS
        
        strategy_config = STRATEGY_CONFIGS.get(name, {})
        
        # 合并配置（传入的config优先）
        merged_config = {}
        
        # 添加配置文件中的参数
        if 'weights' in strategy_config:
            merged_config['weights'] = strategy_config['weights']
        if 'filters' in strategy_config:
            merged_config['filters'] = strategy_config['filters']
        if 'params' in strategy_config:
            merged_config.update(strategy_config['params'])
        
        # 传入的config可以覆盖
        if config:
            merged_config.update(config)
        
        # 如果策略已注册，直接使用
        if name in cls._strategies:
            strategy_cls = cls._strategies[name]
            return strategy_cls(config=merged_config)
        
        # 如果策略在配置文件中但未注册，根据类型选择基础类
        if name in STRATEGY_CONFIGS:
            # 评分类策略使用 momentum_v2 作为基础
            if 'weights' in strategy_config:
                if 'momentum_v2' in cls._strategies:
                    strategy_cls = cls._strategies['momentum_v2']
                    logger.info(f"策略 '{name}' 使用 momentum_v2 作为基类")
                    return strategy_cls(config=merged_config)
        
        # 都没找到，抛出异常
        available = list(cls._strategies.keys()) + [k for k in STRATEGY_CONFIGS.keys() if k not in cls._strategies]
        raise ValueError(f"未知策略: '{name}'，可用策略: {available}")
    
    @classmethod
    def list_all(cls) -> List[str]:
        """列出所有可用策略名称（包括配置文件中的）"""
        from config.strategy_config import STRATEGY_CONFIGS
        
        all_strategies = set(cls._strategies.keys())
        all_strategies.update(STRATEGY_CONFIGS.keys())
        return sorted(list(all_strategies))
    
    @classmethod
    def get_info(cls, name: str) -> Dict:
        """
        获取策略信息
        
        Args:
            name: 策略名称
            
        Returns:
            包含名称、类名、描述的字典
        """
        if name not in cls._strategies:
            raise ValueError(f"未知策略: '{name}'")
        
        strategy_cls = cls._strategies[name]
        return {
            'name': name,
            'class': strategy_cls.__name__,
            'description': cls._descriptions.get(name, ""),
            'module': strategy_cls.__module__,
        }
    
    @classmethod
    def list_all_info(cls) -> List[Dict]:
        """列出所有策略的详细信息"""
        return [cls.get_info(name) for name in cls._strategies.keys()]
    
    @classmethod
    def clear(cls):
        """清空注册表（主要用于测试）"""
        cls._strategies.clear()
        cls._descriptions.clear()


def register_strategy(name: str, description: str = ""):
    """
    便捷函数：注册策略装饰器
    
    等同于 StrategyRegistry.register()
    
    使用示例:
        @register_strategy('my_strategy', '我的策略')
        class MyStrategy(Strategy):
            ...
    """
    return StrategyRegistry.register(name, description)
