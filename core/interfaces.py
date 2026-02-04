# -*- coding: utf-8 -*-
"""
核心接口定义

定义系统的抽象基类，所有具体实现都应继承这些接口：
- DataSource: 数据源抽象
- Strategy: 选股策略抽象
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from core.types import StockData, IndexData, ScoreResult


class DataSource(ABC):
    """
    数据源抽象基类
    
    所有数据源（Tushare、AkShare、腾讯API等）都应实现此接口，
    确保数据层与业务层解耦。
    
    使用示例:
        class TushareSource(DataSource):
            @property
            def name(self) -> str:
                return "tushare"
            
            def get_stock_data(self, code: str, date: str) -> Optional[StockData]:
                # 实现数据获取逻辑
                pass
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        数据源名称
        
        Returns:
            数据源标识符，如 'tushare', 'akshare', 'tencent'
        """
        pass
    
    @abstractmethod
    def get_stock_data(self, code: str, date: str) -> Optional[StockData]:
        """
        获取单只股票在指定日期的数据
        
        Args:
            code: 股票代码（6位）
            date: 日期（YYYY-MM-DD）
            
        Returns:
            StockData对象，获取失败返回None
        """
        pass
    
    @abstractmethod
    def get_stock_list(self, date: str = None) -> List[str]:
        """
        获取所有可交易股票列表
        
        Args:
            date: 可选，指定日期
            
        Returns:
            股票代码列表
        """
        pass
    
    @abstractmethod
    def get_index_constituents(self, index_code: str, date: str = None) -> List[str]:
        """
        获取指数成分股
        
        Args:
            index_code: 指数代码（如 '000300' 沪深300）
            date: 可选，指定日期（获取历史成分股）
            
        Returns:
            成分股代码列表
        """
        pass
    
    @abstractmethod
    def get_index_data(self, index_code: str, start_date: str, end_date: str) -> Optional[IndexData]:
        """
        获取指数在期间的表现数据
        
        Args:
            index_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            IndexData对象，包含收益率等信息
        """
        pass
    
    def batch_get_stock_data(self, codes: List[str], date: str) -> List[StockData]:
        """
        批量获取股票数据（默认实现，子类可覆盖优化）
        
        Args:
            codes: 股票代码列表
            date: 日期
            
        Returns:
            StockData对象列表
        """
        results = []
        for code in codes:
            data = self.get_stock_data(code, date)
            if data and data.is_valid():
                results.append(data)
        return results
    
    def get_trading_calendar(self, start_date: str, end_date: str) -> List[str]:
        """
        获取交易日历
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            交易日列表
        """
        # 默认实现：返回工作日（子类应覆盖以获取真实交易日）
        from datetime import datetime, timedelta
        
        result = []
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current <= end:
            if current.weekday() < 5:  # 周一到周五
                result.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        return result
    
    def get_first_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """
        获取每月第一个交易日
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            每月第一个交易日列表
        """
        trading_days = self.get_trading_calendar(start_date, end_date)
        
        first_days = []
        current_month = None
        
        for day in trading_days:
            month_key = day[:7]  # YYYY-MM
            if month_key != current_month:
                first_days.append(day)
                current_month = month_key
        
        return first_days
    
    def get_index_return(self, index_code: str, start_date: str, end_date: str) -> float:
        """
        获取指数在期间的收益率
        
        Args:
            index_code: 指数代码（如 '000300'）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            收益率 (%)，如 5.0 表示涨了5%
        """
        # 默认实现：通过 get_index_data 计算
        index_data = self.get_index_data(index_code, start_date, end_date)
        if index_data and hasattr(index_data, 'return_pct'):
            return index_data.return_pct
        return 0.0


class Strategy(ABC):
    """
    选股策略抽象基类
    
    所有选股策略都应实现此接口。策略的核心职责：
    1. filter: 预筛选，排除不符合基本条件的股票
    2. score: 评分，给符合条件的股票打分
    3. select: 选股，从候选池中选出最终股票
    
    使用示例:
        class MomentumStrategy(Strategy):
            @property
            def name(self) -> str:
                return "momentum_v2"
            
            def score(self, stock: StockData) -> ScoreResult:
                # 实现评分逻辑
                pass
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化策略
        
        Args:
            config: 策略配置字典
        """
        self._config = config or {}
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        策略名称
        
        Returns:
            策略标识符，如 'momentum_v2', 'value_first'
        """
        pass
    
    @property
    def description(self) -> str:
        """策略描述"""
        return ""
    
    @property
    def config(self) -> Dict[str, Any]:
        """获取策略配置"""
        return self._config
    
    @abstractmethod
    def score(self, stock: StockData) -> ScoreResult:
        """
        计算股票评分
        
        Args:
            stock: 股票数据
            
        Returns:
            ScoreResult对象，包含总分、分项得分、评级
        """
        pass
    
    @abstractmethod
    def filter(self, stock: StockData) -> bool:
        """
        预筛选：判断股票是否满足基本条件
        
        用于快速排除明显不符合条件的股票，如：
        - 亏损股（PE < 0）
        - 停牌股票
        - 价格过低的股票
        
        Args:
            stock: 股票数据
            
        Returns:
            True表示通过筛选，False表示排除
        """
        pass
    
    def select(self, stocks: List[StockData], top_n: int = 10) -> List[StockData]:
        """
        从候选池中选出最终股票
        
        默认实现：
        1. 对所有股票进行filter筛选
        2. 对通过筛选的股票进行score评分
        3. 按评分排序，选出前top_n只
        
        子类可覆盖此方法实现更复杂的选股逻辑。
        
        Args:
            stocks: 候选股票列表
            top_n: 选择数量
            
        Returns:
            选中的股票列表（已排序）
        """
        # 1. 预筛选
        filtered = [s for s in stocks if self.filter(s)]
        
        # 2. 评分
        for stock in filtered:
            result = self.score(stock)
            stock.strength_score = result.total
            stock.strength_grade = result.grade
            stock.score_breakdown = result.breakdown
        
        # 3. 排序
        filtered.sort(key=lambda x: x.strength_score, reverse=True)
        
        # 4. 选择前N只
        selected = filtered[:top_n]
        
        # 5. 添加排名和选择理由
        for i, stock in enumerate(selected):
            stock.rank = i + 1
            stock.selection_reason = self._generate_reason(stock)
        
        return selected
    
    def _generate_reason(self, stock: StockData) -> str:
        """
        生成选择理由
        
        子类可覆盖此方法自定义理由生成逻辑。
        
        Args:
            stock: 股票数据
            
        Returns:
            选择理由文本
        """
        reasons = []
        
        # 根据分项得分生成理由
        if stock.score_breakdown:
            top_scores = sorted(
                stock.score_breakdown.items(),
                key=lambda x: x[1],
                reverse=True
            )[:2]
            for name, score in top_scores:
                if score > 0:
                    reasons.append(f"{name}={score:.0f}")
        
        reasons.append(f"总分{stock.strength_score:.0f}")
        
        return "；".join(reasons)
    
    def _get_grade(self, score: float) -> str:
        """
        根据分数获取评级
        
        Args:
            score: 总分
            
        Returns:
            评级字符串
        """
        if score >= 85:
            return 'A+'
        elif score >= 75:
            return 'A'
        elif score >= 65:
            return 'B+'
        elif score >= 55:
            return 'B'
        elif score >= 45:
            return 'C'
        else:
            return 'D'
