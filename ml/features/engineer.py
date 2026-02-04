# -*- coding: utf-8 -*-
"""
特征工程器模块

核心职责：根据 FeatureConfig 提取所有特征。

扩展方式：新增特征时，只需定义特征函数并注册到 FEATURE_EXTRACTORS 即可。

使用示例:
    from ml.features import FeatureEngineer, FeatureConfig

    config = FeatureConfig()
    engineer = FeatureEngineer(config)
    features = engineer.extract(stock, daily_data=None)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, TYPE_CHECKING

import numpy as np

from .definitions import (
    BASIC_FEATURES,
    TECHNICAL_FEATURES,
    CATEGORICAL_FEATURES,
    MARKET_FEATURES,
    RELATIVE_FEATURES,
    DERIVED_FEATURES,
    ALL_FEATURES,
)
from .indicators import TechnicalIndicators

if TYPE_CHECKING:
    from core.types import StockData

logger = logging.getLogger(__name__)


# ============================================
# 特征提取函数（可扩展：新增特征 = 新函数 + 注册）
# ============================================

def _get_attr(stock: 'StockData', attr: str, default=0):
    """从 StockData 取属性，None 转为 default"""
    v = getattr(stock, attr, None)
    return default if v is None else v


def _basic_from_stock(stock: 'StockData', daily_data: Optional[List[Dict]] = None, *, source: str, default=0, transform=None):
    """通用：从 stock 取基础/技术字段"""
    v = getattr(stock, source, None)
    if v is None:
        return default
    if transform:
        return transform(v)
    return v


# ---------- 基础特征 ----------
def f_price(stock: 'StockData', daily_data=None):
    return _basic_from_stock(stock, daily_data, source='price', default=0)


def f_change_pct(stock: 'StockData', daily_data=None):
    return _basic_from_stock(stock, daily_data, source='change_pct', default=0)


def f_pe_ratio(stock: 'StockData', daily_data=None):
    # 保留负值（表示亏损），但限制范围避免极端值
    val = _basic_from_stock(stock, daily_data, source='pe_ratio', default=0)
    if val is None:
        return 0
    return max(-500, min(500, val))  # clip 到 [-500, 500]


def f_pb_ratio(stock: 'StockData', daily_data=None):
    return _basic_from_stock(stock, daily_data, source='pb_ratio', default=0, transform=lambda x: max(0, x) if x else 0)


def f_turnover_rate(stock: 'StockData', daily_data=None):
    return _basic_from_stock(stock, daily_data, source='turnover_rate', default=0)


def f_roe(stock: 'StockData', daily_data=None):
    return _basic_from_stock(stock, daily_data, source='roe', default=0)


def f_profit_growth(stock: 'StockData', daily_data=None):
    val = _basic_from_stock(stock, daily_data, source='profit_growth', default=0)
    if val is None:
        return 0
    return max(-500, min(500, val))  # clip 到 [-500%, 500%] 避免扭亏异常值


def f_revenue_growth(stock: 'StockData', daily_data=None):
    return _basic_from_stock(stock, daily_data, source='revenue_growth', default=0)


def f_dividend_yield(stock: 'StockData', daily_data=None):
    return _basic_from_stock(stock, daily_data, source='dividend_yield', default=0)


# ---------- 技术特征（从 StockData 直接取） ----------
def f_momentum_20d(stock: 'StockData', daily_data=None):
    return _basic_from_stock(stock, daily_data, source='momentum_20d', default=0)


def f_momentum_60d(stock: 'StockData', daily_data=None):
    return _basic_from_stock(stock, daily_data, source='momentum_60d', default=0)


# ---------- 技术特征（需日线数据） ----------
def _tech_from_daily(stock: 'StockData', daily_data: Optional[List[Dict]], key: str, default=0):
    if not daily_data:
        return default
    d = TechnicalIndicators.compute_all(daily_data)
    return d.get(key, default)


def f_volatility_20d(stock: 'StockData', daily_data=None):
    return _tech_from_daily(stock, daily_data, 'volatility_20d', 0)


def f_rsi_14(stock: 'StockData', daily_data=None):
    return _tech_from_daily(stock, daily_data, 'rsi_14', 50)


def f_ma_deviation_20(stock: 'StockData', daily_data=None):
    return _tech_from_daily(stock, daily_data, 'ma_deviation_20', 0)


# ---------- 分类特征 ----------
def f_industry(stock: 'StockData', daily_data=None):
    v = getattr(stock, 'industry', None)
    return str(v) if v else 'unknown'

def f_market(stock: 'StockData', daily_data=None):
    v = getattr(stock, 'market', None)
    return str(v) if v else 'unknown'


# ---------- 新增基础特征 ----------
def f_peg(stock: 'StockData', daily_data=None):
    val = _get_attr(stock, 'peg', 0)
    if val is None or val <= 0:
        return 0
    return min(val, 100)  # PEG > 100 通常无意义，限制上限

def f_turnover(stock: 'StockData', daily_data=None):
    return _get_attr(stock, 'turnover', 0)

def f_volume(stock: 'StockData', daily_data=None):
    return _get_attr(stock, 'volume', 0)

def f_volume_ratio(stock: 'StockData', daily_data=None):
    return _get_attr(stock, 'volume_ratio', 0)

def f_total_mv(stock: 'StockData', daily_data=None):
    return _get_attr(stock, 'total_mv', 0)

def f_circ_mv(stock: 'StockData', daily_data=None):
    return _get_attr(stock, 'circ_mv', 0)

def f_list_days(stock: 'StockData', daily_data=None):
    return _get_attr(stock, 'list_days', 0)


# ---------- 衍生特征（单样本） ----------
def f_value_score(stock: 'StockData', daily_data=None):
    pe = _get_attr(stock, 'pe_ratio', 0)
    div = _get_attr(stock, 'dividend_yield', 0)
    return (1.0 / pe if pe and pe > 0 else 0) + (div or 0)


def f_quality_score(stock: 'StockData', daily_data=None):
    roe = _get_attr(stock, 'roe', 0)
    growth = _get_attr(stock, 'profit_growth', 0)
    return (roe or 0) + (growth or 0) * 0.5


# ---------- 市场环境特征（需要 market_data） ----------
def f_market_momentum_20d(stock: 'StockData', daily_data=None, market_data: Dict = None):
    return (market_data or {}).get('market_momentum_20d', 0)

def f_market_momentum_60d(stock: 'StockData', daily_data=None, market_data: Dict = None):
    return (market_data or {}).get('market_momentum_60d', 0)

def f_market_volatility_20d(stock: 'StockData', daily_data=None, market_data: Dict = None):
    return (market_data or {}).get('market_volatility_20d', 0)

def f_market_trend(stock: 'StockData', daily_data=None, market_data: Dict = None):
    return (market_data or {}).get('market_trend', 0)


# ---------- 相对收益相关特征（需要 market_data） ----------
def f_relative_momentum_20d(stock: 'StockData', daily_data=None, market_data: Dict = None):
    m = market_data or {}
    stock_m = _get_attr(stock, 'momentum_20d', 0) or 0
    market_m = m.get('market_momentum_20d', 0) or 0
    return round(stock_m - market_m, 2)

def f_relative_momentum_60d(stock: 'StockData', daily_data=None, market_data: Dict = None):
    m = market_data or {}
    stock_m = _get_attr(stock, 'momentum_60d', 0) or 0
    market_m = m.get('market_momentum_60d', 0) or 0
    return round(stock_m - market_m, 2)

def f_volatility_ratio_20d(stock: 'StockData', daily_data=None, market_data: Dict = None):
    """个股20日年化波动率 / 大盘20日年化波动率（口径一致）"""
    from .market import calc_annualized_volatility_from_daily
    m = market_data or {}
    market_vol = m.get('market_volatility_20d', 0) or 0
    if market_vol <= 0:
        return 1.0
    if not daily_data:
        return 1.0
    stock_vol = calc_annualized_volatility_from_daily(daily_data, 20)
    if stock_vol <= 0:
        return 1.0
    return round(stock_vol / market_vol, 4)

def f_stock_market_correlation_20d(stock: 'StockData', daily_data=None, market_data: Dict = None):
    return (market_data or {}).get('stock_market_correlation_20d', 0)

def f_stock_beta_20d(stock: 'StockData', daily_data=None, market_data: Dict = None):
    return (market_data or {}).get('stock_beta_20d', 1.0)


# ============================================
# 特征提取器注册表（扩展点：注册新特征函数即可）
# ============================================

FEATURE_EXTRACTORS: Dict[str, Callable] = {
    # 基础
    'price': f_price,
    'change_pct': f_change_pct,
    'pe_ratio': f_pe_ratio,
    'pb_ratio': f_pb_ratio,
    'turnover_rate': f_turnover_rate,
    'roe': f_roe,
    'profit_growth': f_profit_growth,
    'revenue_growth': f_revenue_growth,
    'dividend_yield': f_dividend_yield,
    'peg': f_peg,
    'turnover': f_turnover,
    'volume': f_volume,
    'volume_ratio': f_volume_ratio,
    'total_mv': f_total_mv,
    'circ_mv': f_circ_mv,
    'list_days': f_list_days,
    # 技术（来自 StockData）
    'momentum_20d': f_momentum_20d,
    'momentum_60d': f_momentum_60d,
    # 技术（来自日线）
    'volatility_20d': f_volatility_20d,
    'rsi_14': f_rsi_14,
    'ma_deviation_20': f_ma_deviation_20,
    # 分类
    'industry': f_industry,
    'market': f_market,
    # 衍生（单样本）
    'value_score': f_value_score,
    'quality_score': f_quality_score,
    # 市场环境（需 market_data）
    'market_momentum_20d': f_market_momentum_20d,
    'market_momentum_60d': f_market_momentum_60d,
    'market_volatility_20d': f_market_volatility_20d,
    'market_trend': f_market_trend,
    # 相对收益相关（需 market_data）
    'relative_momentum_20d': f_relative_momentum_20d,
    'relative_momentum_60d': f_relative_momentum_60d,
    'volatility_ratio_20d': f_volatility_ratio_20d,
    'stock_market_correlation_20d': f_stock_market_correlation_20d,
    'stock_beta_20d': f_stock_beta_20d,
}


def register_feature(name: str, extractor: Callable) -> None:
    """
    注册特征提取函数，便于扩展新特征。

    使用示例:
        def my_custom_feature(stock, daily_data=None):
            return stock.pe_ratio * 0.1 if stock.pe_ratio else 0
        register_feature('my_custom', my_custom_feature)
    """
    FEATURE_EXTRACTORS[name] = extractor


# ============================================
# 特征配置
# ============================================

@dataclass
class FeatureConfig:
    """
    特征配置：指定要提取哪些特征。

    Attributes:
        basic_features: 基础特征名列表
        technical_features: 技术特征名列表
        categorical_features: 分类特征名列表
        market_features: 市场环境与相对收益特征名列表（需传入 market_data）
        derived_features: 衍生特征名列表（此处仅单样本可算的；需批量计算的由调用方单独处理）
    """
    basic_features: List[str] = field(default_factory=lambda: [
        'change_pct', 'pe_ratio', 'pb_ratio',
        'turnover_rate', 'roe', 'profit_growth', 'dividend_yield'
    ])
    technical_features: List[str] = field(default_factory=lambda: ['momentum_20d', 'momentum_60d'])
    categorical_features: List[str] = field(default_factory=list)
    market_features: List[str] = field(default_factory=list)
    derived_features: List[str] = field(default_factory=list)

    def get_all_feature_names(self) -> List[str]:
        """返回配置中要提取的全部特征名（去重、保序）"""
        seen = set()
        out = []
        for name in (
            self.basic_features
            + self.technical_features
            + self.categorical_features
            + self.market_features
            + self.derived_features
        ):
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out


# ============================================
# 预定义配置
# ============================================

DEFAULT_FEATURE_CONFIG = FeatureConfig()

FULL_FEATURE_CONFIG = FeatureConfig(
    basic_features=list(BASIC_FEATURES.keys()),
    technical_features=['momentum_20d', 'momentum_60d', 'rsi_14', 'volatility_20d', 'ma_deviation_20'],
    categorical_features=['industry', 'market'],
    market_features=[
        'market_momentum_20d', 'market_momentum_60d', 'market_volatility_20d', 'market_trend',
        'relative_momentum_20d', 'relative_momentum_60d', 'volatility_ratio_20d',
        'stock_market_correlation_20d', 'stock_beta_20d',
    ],
    derived_features=['value_score', 'quality_score'],
)

SIMPLE_FEATURE_CONFIG = FeatureConfig(
    basic_features=['change_pct', 'pe_ratio', 'turnover_rate'],
    technical_features=['momentum_20d'],
    categorical_features=[],
    derived_features=[],
)


def get_full_numeric_feature_names() -> List[str]:
    """
    返回 FULL_FEATURE_CONFIG 下的全部数值特征名（排除分类特征 industry/market）。
    供 train.py、quarterly_selector.py、strategy_optimizer 等统一使用，避免多处硬编码不一致。
    与 build_training_data 使用 FULL_FEATURE_CONFIG 时写入 JSON 的数值键一致（含 market_momentum_60d）。
    """
    names = FULL_FEATURE_CONFIG.get_all_feature_names()
    cat = set(FULL_FEATURE_CONFIG.categorical_features)
    return [n for n in names if n not in cat]


# ============================================
# 特征工程器（核心：按配置调用注册的提取函数）
# ============================================

class FeatureEngineer:
    """
    根据 FeatureConfig 提取所有特征。

    唯一核心方法：extract(stock, daily_data=None) -> Dict[str, Any]
    新增特征时，在 FEATURE_EXTRACTORS 中注册对应函数即可。
    """

    def __init__(self, config: FeatureConfig = None):
        self.config = config or DEFAULT_FEATURE_CONFIG
        self._names = self.config.get_all_feature_names()
        logger.debug(f"FeatureEngineer 初始化: {len(self._names)} 个特征")

    def get_feature_names(self) -> List[str]:
        """当前配置下的特征名列表"""
        return list(self._names)

    def extract(
        self,
        stock: 'StockData',
        daily_data: Optional[List[Dict]] = None,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        根据 FeatureConfig 提取该股票的全部特征。

        Args:
            stock: StockData 对象
            daily_data: 日线数据（部分技术指标需要，可为 None）
            market_data: 市场环境与相对特征（大盘动量/波动率、个股-大盘相关系数/Beta 等，可为 None）

        Returns:
            特征名字典
        """
        result = {}
        market_feature_names = set(self.config.market_features) if self.config.market_features else set()
        for name in self._names:
            if name in FEATURE_EXTRACTORS:
                try:
                    if name in market_feature_names:
                        result[name] = FEATURE_EXTRACTORS[name](stock, daily_data, market_data)
                    else:
                        result[name] = FEATURE_EXTRACTORS[name](stock, daily_data)
                except Exception as e:
                    logger.debug(f"特征 {name} 计算异常: {e}")
                    fd = ALL_FEATURES.get(name)
                    result[name] = fd.default if fd is not None else 0
            else:
                fd = ALL_FEATURES.get(name)
                result[name] = fd.default if fd is not None else 0
        return result

    def batch_extract(
        self,
        stocks: List['StockData'],
        daily_data_map: Optional[Dict[str, List[Dict]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量提取特征。若需「依赖整批数据」的衍生特征，请在调用后对返回的 list 做额外处理。
        """
        daily_data_map = daily_data_map or {}
        return [
            self.extract(s, daily_data_map.get(getattr(s, 'code', '')))
            for s in stocks
        ]

    def to_array(self, feature_dict: Dict[str, Any]) -> np.ndarray:
        """将特征字典转为数值数组（仅数值型特征，按 key 排序，与 get_numeric_feature_names 顺序一致）"""
        names = sorted(
            k for k, v in feature_dict.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        )
        values = [float(feature_dict.get(k, 0) or 0) for k in names]
        return np.array(values, dtype=np.float32)

    def get_numeric_feature_names(self, feature_dict: Dict[str, Any]) -> List[str]:
        """数值特征名列表（与 to_array 输出顺序一致）"""
        return sorted(
            k for k, v in feature_dict.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        )

    def enrich_with_daily_data(
        self,
        feature_dict: Dict[str, Any],
        daily_data: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """将日线计算出的技术指标并入特征字典（兼容旧用法；推荐直接使用 extract(stock, daily_data)）"""
        if not daily_data:
            return feature_dict
        tech = TechnicalIndicators.compute_all(daily_data)
        for k, v in tech.items():
            if k not in feature_dict or feature_dict[k] is None:
                feature_dict[k] = v
        return feature_dict

    @staticmethod
    def extract_with_history(
        stock: 'StockData',
        daily_data: Optional[List[Dict]] = None,
        config: Optional['FeatureConfig'] = None,
    ) -> Dict[str, Any]:
        """一步到位：按配置提取特征（含日线技术指标）。推荐入口。"""
        return FeatureEngineer(config or DEFAULT_FEATURE_CONFIG).extract(stock, daily_data)


# ============================================
# 批量衍生特征（依赖多样本，与 extract 解耦）
# ============================================

def compute_batch_derived(
    feature_dicts: List[Dict[str, Any]],
    derived_names: List[str],
) -> List[Dict[str, Any]]:
    """
    在已提取的特征列表上，计算依赖整批数据的衍生特征（如百分位、排名）。
    与 FeatureEngineer.extract 解耦，由调用方在需要时使用。
    """
    if not feature_dicts or not derived_names:
        return feature_dicts

    if 'pe_percentile' in derived_names:
        pe_list = [fd.get('pe_ratio', 0) or 0 for fd in feature_dicts]
        pe_valid = sorted([p for p in pe_list if p > 0])
        n = len(pe_valid)
        for fd in feature_dicts:
            p = fd.get('pe_ratio', 0) or 0
            fd['pe_percentile'] = (
                sum(1 for x in pe_valid if x <= p) / n if n else 0.5
            )

    if 'momentum_rank' in derived_names:
        mom = [(i, (fd.get('momentum_20d', 0) or 0)) for i, fd in enumerate(feature_dicts)]
        mom.sort(key=lambda x: x[1])
        n = len(mom)
        for r, (i, _) in enumerate(mom):
            feature_dicts[i]['momentum_rank'] = r / n if n else 0.5

    return feature_dicts


# ============================================
# 便捷函数
# ============================================

def extract_features(
    stock: 'StockData',
    config: FeatureConfig = None,
    daily_data: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """根据配置提取单只股票特征"""
    return FeatureEngineer(config or DEFAULT_FEATURE_CONFIG).extract(stock, daily_data)


def batch_extract_features(
    stocks: List['StockData'],
    config: FeatureConfig = None,
    daily_data_map: Optional[Dict[str, List[Dict]]] = None,
    add_batch_derived: bool = False,
) -> List[Dict[str, Any]]:
    """批量提取；若 add_batch_derived=True，会再算 pe_percentile、momentum_rank 等"""
    eng = FeatureEngineer(config or DEFAULT_FEATURE_CONFIG)
    out = eng.batch_extract(stocks, daily_data_map)
    if add_batch_derived and eng.config.derived_features:
        out = compute_batch_derived(out, eng.config.derived_features)
    return out
