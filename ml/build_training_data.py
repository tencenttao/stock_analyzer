# -*- coding: utf-8 -*-
"""
构建机器学习训练数据集

在给定日期范围内：
1. 按月取月初第一个交易日作为买入日、下月初第一个交易日作为卖出日
2. 用 FeatureEngineer 按 FeatureConfig 提取每只股票在买入日的特征
3. 计算实际持仓收益
4. 将「特征 + 收益 + 持仓天数」保存为训练数据
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.manager import DataManager
from ml.features import (
    FeatureEngineer,
    FeatureConfig,
    DEFAULT_FEATURE_CONFIG,
    FULL_FEATURE_CONFIG,
)
from ml.features.market import compute_market_features, compute_stock_market_relation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _count_trading_days(start_date: str, end_date: str) -> int:
    """
    计算两个日期之间的交易日数量（与 backtest 保持一致）
    简化计算：只计工作日（周一到周五），不考虑节假日
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    n = 0
    cur = start
    while cur < end:
        if cur.weekday() < 5:
            n += 1
        cur += timedelta(days=1)
    return n


def build_training_dataset(
    start_date: str,
    end_date: str,
    output_path: Optional[str] = None,
    feature_config: Optional[FeatureConfig] = None,
    use_cache: bool = False,
) -> List[Dict]:
    """
    在给定日期范围内，按月构建训练样本：买入日特征 + 实际收益 + 持仓天数。

    流程：
    1. 获取区间内每月第一个交易日（买入日/卖出日）
    2. 对每月：取沪深300成分股，对每只股票取买入日 StockData，若配置需要日线则取日线
    3. 用 FeatureEngineer.extract(stock, daily_data) 得到特征字典
    4. 用卖出日价格算收益率，记录持仓天数
    5. 每条样本包含：features（含 code/name/buy_date/sell_date/buy_price/sell_price + 配置特征）、return_pct、holding_days

    Args:
        start_date: 开始日期（含）
        end_date: 结束日期（含）
        output_path: 输出 JSON 路径，为 None 则不写文件
        feature_config: 特征配置，为 None 时使用 DEFAULT_FEATURE_CONFIG

    Returns:
        训练数据列表，每项为 {"features": dict, "return_pct": float, "holding_days": int}
    """
    config = feature_config or DEFAULT_FEATURE_CONFIG
    engineer = FeatureEngineer(config)
    # use_cache: 是否从缓存读取（加速，但可能使用旧数据）
    # write_cache=True: 总是写入缓存（供后续回测使用）
    data_source = DataManager(use_cache=use_cache, write_cache=True)

    # 需要日线时，在循环内按 code+buy_date 取日线
    need_daily = any(
        name in getattr(config, 'technical_features', [])
        for name in ('rsi_14', 'volatility_20d', 'ma_deviation_20')
    )
    need_market = bool(getattr(config, 'market_features', []))

    logger.info(f"开始构建训练数据: {start_date} ~ {end_date}")
    logger.info(f"特征配置: {engineer.get_feature_names()}")
    logger.info(f"需要日线数据: {'是' if need_daily else '否'}")
    logger.info(f"需要市场/相对特征: {'是' if need_market else '否'}")

    trading_days = data_source.get_first_trading_days(start_date, end_date)
    logger.info(f"月度调仓日: {trading_days[:3]}...{trading_days[-1] if len(trading_days) > 3 else ''}")

    training_data: List[Dict] = []
    total_stocks = 0
    valid_stocks = 0

    for i, buy_date in enumerate(trading_days):
        if i >= len(trading_days) - 1:
            break
        sell_date = trading_days[i + 1]
        holding_days = _count_trading_days(buy_date, sell_date)
        logger.info(f"\n月份 {i+1}/{len(trading_days)-1}: {buy_date} -> {sell_date} (持仓{holding_days}天)")

        # 获取指数收益（用于相对收益标签）
        try:
            index_return_pct = data_source.get_index_return('000300', buy_date, sell_date)
        except Exception:
            index_return_pct = 0
        logger.info(f"  沪深300收益: {index_return_pct:.2f}%")

        # 市场环境特征（月度一次）
        market_data_base = None
        index_daily_for_relation = None
        if need_market:
            market_data_base = compute_market_features(data_source, buy_date)
            logger.info(f"  市场特征: 20d动量={market_data_base.get('market_momentum_20d', 0):.1f}%, 趋势={market_data_base.get('market_trend', 0)}")
            # 指数日线用于个股-大盘相关系数/Beta（与日线区间一致）
            start_date_120 = (datetime.strptime(buy_date, '%Y-%m-%d') - timedelta(days=120)).strftime('%Y-%m-%d')
            index_daily_for_relation = data_source.get_index_daily('000300', start_date_120, buy_date)

        try:
            constituents = data_source.get_csi300_stocks(buy_date)
        except Exception as e:
            logger.error(f"  获取成分股失败: {e}")
            continue
        if not constituents:
            logger.warning(f"  {buy_date} 无成分股")
            continue
        logger.info(f"  成分股数量: {len(constituents)}")

        for j, code in enumerate(constituents):
            if (j + 1) % 50 == 0:
                logger.info(f"  进度: {j+1}/{len(constituents)}")
            total_stocks += 1

            try:
                buy_data = data_source.get_stock_data(code, buy_date)
                if buy_data is None:
                    continue
                sell_data = data_source.get_stock_data(code, sell_date)
                if sell_data is None:
                    continue
                if buy_data.price <= 0:
                    continue

                return_pct = (sell_data.price - buy_data.price) / buy_data.price * 100

                daily_data = None
                if need_daily or need_market:
                    daily_data = data_source.get_daily_data(code, end_date=buy_date, days=120)

                market_data = None
                if need_market and market_data_base is not None:
                    market_data = dict(market_data_base)
                    if index_daily_for_relation and daily_data:
                        corr, beta = compute_stock_market_relation(daily_data, index_daily_for_relation, 20)
                        market_data['stock_market_correlation_20d'] = corr
                        market_data['stock_beta_20d'] = beta

                features = engineer.extract(buy_data, daily_data=daily_data, market_data=market_data)
                # 元数据（非特征，用于回溯分析）
                features['code'] = code
                features['name'] = buy_data.name
                features['buy_date'] = buy_date
                features['sell_date'] = sell_date
                features['buy_price'] = buy_data.price
                features['sell_price'] = sell_data.price
                # 补充可能未在配置中的字段（确保完整性）
                if 'industry' not in features:
                    features['industry'] = buy_data.industry or 'unknown'
                if 'market' not in features:
                    features['market'] = buy_data.market or 'unknown'
                if 'list_days' not in features:
                    features['list_days'] = buy_data.list_days or 0
                if 'total_mv' not in features:
                    features['total_mv'] = buy_data.total_mv or 0
                if 'circ_mv' not in features:
                    features['circ_mv'] = buy_data.circ_mv or 0
                if 'volume_ratio' not in features:
                    features['volume_ratio'] = buy_data.volume_ratio or 0

                training_data.append({
                    'features': features,
                    'return_pct': return_pct,
                    'index_return_pct': index_return_pct,  # 新增：指数收益
                    'holding_days': holding_days,
                })
                valid_stocks += 1
            except Exception as e:
                logger.debug(f"  处理 {code} 失败: {e}")

    logger.info(f"\n数据收集完成: 总{total_stocks}, 有效{valid_stocks}, 有效率{valid_stocks/total_stocks:.1%}" if total_stocks else "\n无有效数据")

    if training_data:
        returns = [d['return_pct'] for d in training_data]
        logger.info(f"收益: 均值{sum(returns)/len(returns):.2f}%, 最小{min(returns):.2f}%, 最大{max(returns):.2f}%")
        n_up = sum(1 for r in returns if r > 5)
        n_down = sum(1 for r in returns if r < -5)
        n_neutral = len(returns) - n_up - n_down
        logger.info(f"标签(±5%): 上涨{n_up}, 下跌{n_down}, 持平{n_neutral}")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存: {output_path}")

    return training_data


def analyze_feature_effectiveness(training_data: List[Dict]) -> Dict:
    """按与收益的相关系数分析特征有效性。特征名从每条样本的 features 中的数值键推断。"""
    import numpy as np

    if not training_data:
        return {}

    sample = training_data[0]['features']
    meta_keys = {'code', 'name', 'buy_date', 'sell_date', 'buy_price', 'sell_price'}
    numeric_names = sorted(
        k for k, v in sample.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in meta_keys
    )
    if not numeric_names:
        numeric_names = sorted(
            k for k, v in sample.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        )

    returns = np.array([d['return_pct'] for d in training_data])
    results = {}

    for name in numeric_names:
        values = np.array([
            float(d['features'].get(name, 0) or 0) for d in training_data
        ])
        valid = ~np.isnan(values) & ~np.isnan(returns)
        if valid.sum() > 10:
            corr = np.corrcoef(values[valid], returns[valid])[0, 1]
        else:
            corr = 0
        results[name] = {
            'correlation': float(corr) if not np.isnan(corr) else 0,
            'mean': float(np.nanmean(values)),
            'std': float(np.nanstd(values)),
        }

    sorted_results = sorted(
        results.items(),
        key=lambda x: abs(x[1]['correlation']),
        reverse=True
    )
    logger.info("特征有效性（与收益相关系数）:")
    for name, st in sorted_results:
        c = st['correlation']
        logger.info(f"  {name}: {c:+.3f}")
    return dict(sorted_results)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='构建 ML 训练数据集（按 FeatureConfig 提取特征）')
    parser.add_argument('--start', default='2022-05-01', help='开始日期')
    parser.add_argument('--end', default='2022-06-10', help='结束日期')
    parser.add_argument('--output', default='data/2022_5_ml_training_data.json', help='输出 JSON 路径')
    parser.add_argument('--config', choices=['default', 'full'], default='full',
                        help='特征配置: default=基础+动量, full=含日线技术指标')
    parser.add_argument('--use-cache', action='store_true',
                        help='使用缓存数据（可加速，但可能使用旧数据）')
    parser.add_argument('--debug', action='store_true', help='开启 DEBUG 日志')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = FULL_FEATURE_CONFIG if args.config == 'full' else DEFAULT_FEATURE_CONFIG

    print("=" * 60)
    print("构建机器学习训练数据集")
    print("=" * 60)
    print(f"日期: {args.start} ~ {args.end}")
    print(f"特征配置: {args.config}")
    print(f"使用缓存: {'是' if args.use_cache else '否'}")
    print(f"输出: {args.output}")
    print("=" * 60)

    training_data = build_training_dataset(
        start_date=args.start,
        end_date=args.end,
        output_path=args.output,
        feature_config=config,
        use_cache=args.use_cache,
    )

    if training_data:
        analyze_feature_effectiveness(training_data)
        holding_days = sorted(set(d['holding_days'] for d in training_data))
        print(f"\n持仓天数分布: {holding_days}")
        print(f"\n完成，共 {len(training_data)} 条")
    else:
        print("\n无有效样本，未写入文件。")
