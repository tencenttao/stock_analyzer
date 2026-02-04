# -*- coding: utf-8 -*-
"""
技术指标计算模块

从日线数据（OHLCV）计算各种技术指标。

使用示例:
    from ml.features import TechnicalIndicators
    
    # 获取日线数据
    daily_data = data_manager.get_daily_data('600036', '2024-01-01', '2024-03-01')
    
    # 计算所有技术指标
    indicators = TechnicalIndicators.compute_all(daily_data)
    # {'rsi_14': 55.3, 'macd': 0.12, 'volatility_20d': 2.5, ...}
    
    # 单独计算某个指标
    rsi = TechnicalIndicators.calc_rsi(closes, period=14)
"""

from typing import List, Dict


class TechnicalIndicators:
    """
    技术指标计算器
    
    提供静态方法计算各种技术指标。
    
    支持的指标:
        - RSI (相对强弱指数)
        - MACD (指数平滑移动平均线)
        - 波动率 (Volatility)
        - 均线偏离 (MA Deviation)
        - 布林带位置 (Bollinger Position)
        - ATR (平均真实波幅)
        - 成交量比率 (Volume Ratio)
        - 动量 (Momentum)
    """
    
    @staticmethod
    def compute_all(daily_data: List[Dict], lookback: int = 60) -> Dict[str, float]:
        """
        计算所有技术指标
        
        Args:
            daily_data: 日线数据列表，每条包含:
                - trade_date: 交易日期
                - open, high, low, close: OHLC价格
                - vol/volume: 成交量
            lookback: 使用的历史天数（默认60天）
            
        Returns:
            技术指标字典
        """
        if not daily_data or len(daily_data) < 5:
            return {}
        
        # 确保数据按日期降序排列（最新在前）
        data = sorted(daily_data, key=lambda x: x.get('trade_date', ''), reverse=True)[:lookback]
        
        # 提取价格序列（按时间正序，便于计算）
        closes = [d.get('close', 0) for d in reversed(data)]
        highs = [d.get('high', 0) for d in reversed(data)]
        lows = [d.get('low', 0) for d in reversed(data)]
        volumes = [d.get('vol', 0) or d.get('volume', 0) for d in reversed(data)]
        
        indicators = {}
        
        # RSI
        if len(closes) >= 15:
            indicators['rsi_14'] = TechnicalIndicators.calc_rsi(closes, 14)
        
        # 波动率
        if len(closes) >= 21:
            indicators['volatility_20d'] = TechnicalIndicators.calc_volatility(closes, 20)
        
        # 均线偏离
        if len(closes) >= 21:
            indicators['ma_deviation_20'] = TechnicalIndicators.calc_ma_deviation(closes, 20)
        if len(closes) >= 61:
            indicators['ma_deviation_60'] = TechnicalIndicators.calc_ma_deviation(closes, 60)
        
        # MACD
        if len(closes) >= 35:
            macd_data = TechnicalIndicators.calc_macd(closes)
            indicators.update(macd_data)
        
        # 布林带位置
        if len(closes) >= 21:
            indicators['bollinger_position'] = TechnicalIndicators.calc_bollinger_position(closes, 20)
        
        # ATR（平均真实波幅）
        if len(closes) >= 15:
            indicators['atr_14'] = TechnicalIndicators.calc_atr(highs, lows, closes, 14)
        
        # 成交量变化
        if len(volumes) >= 6:
            indicators['volume_ratio_5d'] = TechnicalIndicators.calc_volume_ratio(volumes, 5)
        
        # 动量
        if len(closes) >= 21:
            indicators['momentum_20d_calc'] = (closes[-1] / closes[-21] - 1) * 100 if closes[-21] > 0 else 0
        if len(closes) >= 61:
            indicators['momentum_60d_calc'] = (closes[-1] / closes[-61] - 1) * 100 if closes[-61] > 0 else 0
        
        return indicators
    
    @staticmethod
    def calc_rsi(prices: List[float], period: int = 14) -> float:
        """
        计算RSI（相对强弱指数）
        
        RSI = 100 - 100 / (1 + RS)
        RS = 平均涨幅 / 平均跌幅
        
        Args:
            prices: 收盘价序列（按时间正序）
            period: 计算周期（默认14）
            
        Returns:
            RSI值（0-100）
        """
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)
    
    @staticmethod
    def calc_volatility(prices: List[float], period: int = 20) -> float:
        """
        计算波动率（标准差/均值 * 100%）
        
        Args:
            prices: 收盘价序列
            period: 计算周期（默认20）
            
        Returns:
            波动率百分比
        """
        if len(prices) < period:
            return 0.0
        
        recent = prices[-period:]
        mean = sum(recent) / len(recent)
        if mean == 0:
            return 0.0
        
        variance = sum((p - mean) ** 2 for p in recent) / len(recent)
        std = variance ** 0.5
        volatility = (std / mean) * 100
        return round(volatility, 2)
    
    @staticmethod
    def calc_ma_deviation(prices: List[float], period: int = 20) -> float:
        """
        计算价格偏离均线的百分比
        
        偏离度 = (当前价 - MA) / MA * 100%
        
        Args:
            prices: 收盘价序列
            period: 均线周期（默认20）
            
        Returns:
            偏离百分比（正值表示高于均线）
        """
        if len(prices) < period:
            return 0.0
        
        ma = sum(prices[-period:]) / period
        if ma == 0:
            return 0.0
        
        current = prices[-1]
        deviation = (current - ma) / ma * 100
        return round(deviation, 2)
    
    @staticmethod
    def calc_macd(prices: List[float]) -> Dict[str, float]:
        """
        计算MACD指标
        
        MACD = EMA(12) - EMA(26)
        Signal = EMA(MACD, 9)
        Histogram = MACD - Signal
        
        Args:
            prices: 收盘价序列
            
        Returns:
            包含 macd, macd_signal, macd_hist 的字典
        """
        if len(prices) < 35:
            return {'macd': 0, 'macd_signal': 0, 'macd_hist': 0}
        
        def ema(data: List[float], period: int) -> float:
            if len(data) < period:
                return data[-1] if data else 0
            multiplier = 2 / (period + 1)
            ema_val = sum(data[:period]) / period
            for price in data[period:]:
                ema_val = (price - ema_val) * multiplier + ema_val
            return ema_val
        
        ema12 = ema(prices, 12)
        ema26 = ema(prices, 26)
        macd = ema12 - ema26
        
        # 简化的 signal line
        signal = macd * 0.9
        hist = macd - signal
        
        return {
            'macd': round(macd, 4),
            'macd_signal': round(signal, 4),
            'macd_hist': round(hist, 4)
        }
    
    @staticmethod
    def calc_bollinger_position(prices: List[float], period: int = 20) -> float:
        """
        计算价格在布林带中的位置
        
        位置 = (当前价 - 下轨) / (上轨 - 下轨)
        上轨 = MA + 2 * STD
        下轨 = MA - 2 * STD
        
        Args:
            prices: 收盘价序列
            period: 计算周期（默认20）
            
        Returns:
            位置值（0-1，0.5表示在中轨）
        """
        if len(prices) < period:
            return 0.5
        
        recent = prices[-period:]
        ma = sum(recent) / period
        std = (sum((p - ma) ** 2 for p in recent) / period) ** 0.5
        
        if std == 0:
            return 0.5
        
        upper = ma + 2 * std
        lower = ma - 2 * std
        current = prices[-1]
        
        if upper == lower:
            return 0.5
        
        position = (current - lower) / (upper - lower)
        return round(max(0, min(1, position)), 2)
    
    @staticmethod
    def calc_atr(
        highs: List[float], 
        lows: List[float], 
        closes: List[float], 
        period: int = 14
    ) -> float:
        """
        计算ATR（平均真实波幅）
        
        TR = max(H-L, |H-C_prev|, |L-C_prev|)
        ATR = mean(TR, period)
        
        Args:
            highs: 最高价序列
            lows: 最低价序列
            closes: 收盘价序列
            period: 计算周期（默认14）
            
        Returns:
            ATR值
        """
        if len(closes) < period + 1:
            return 0.0
        
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        
        if len(trs) < period:
            return 0.0
        
        atr = sum(trs[-period:]) / period
        return round(atr, 4)
    
    @staticmethod
    def calc_volume_ratio(volumes: List[float], period: int = 5) -> float:
        """
        计算成交量比率（量比）
        
        量比 = 当日成交量 / 过去N日平均成交量
        
        Args:
            volumes: 成交量序列
            period: 对比周期（默认5）
            
        Returns:
            量比（>1 表示放量，<1 表示缩量）
        """
        if len(volumes) < period + 1:
            return 1.0
        
        current = volumes[-1]
        avg = sum(volumes[-(period+1):-1]) / period
        
        if avg == 0:
            return 1.0
        
        return round(current / avg, 2)
    
    @staticmethod
    def calc_momentum(prices: List[float], period: int = 20) -> float:
        """
        计算动量
        
        动量 = (当前价 / N日前价格 - 1) * 100%
        
        Args:
            prices: 收盘价序列
            period: 计算周期（默认20）
            
        Returns:
            动量百分比
        """
        if len(prices) < period + 1:
            return 0.0
        
        if prices[-(period+1)] == 0:
            return 0.0
        
        return round((prices[-1] / prices[-(period+1)] - 1) * 100, 2)
