# -*- coding: utf-8 -*-
"""
季度选股系统

策略：每季度初用窗口期内历史数据训练模型，再用该模型对该季度所有月份选股。
（与 strategy_optimizer 逻辑一致，便于结果对比）

功能：
1. 运行回测：按季度迭代，每季度训练一次、对该季度3个月评估
2. 为指定季度选股：用该季度之前的数据训练，输出该季度3个月的选股结果

阈值说明：
    模型是回归模型，预测的是「相对收益」（即跑赢/跑输基准的幅度 %）。
    --min-threshold 参数用于过滤预测值：
    - 不设置（默认）: 直接选预测值最高的 Top N
    - 设置为 2: 只选预测相对收益 ≥ 2% 的股票，不足 Top N 则选少于 N 只
    - 设置为 0: 只选预测相对收益 ≥ 0% 的股票（预测跑赢基准）

使用方法:
    # 查看历史回测表现（默认从2021年起，8季度=2年滑动窗口）
    python ml/quarterly_selector.py --backtest
    
    # 指定起始年份和滑动窗口（按季度数，与 strategy_optimizer 一致）
    python ml/quarterly_selector.py --backtest --start-year 2022 --window-quarters 12
    
    # 使用全部历史数据（不限制窗口）
    python ml/quarterly_selector.py --backtest --window-quarters 0
    
    # 为指定季度选股
    python ml/quarterly_selector.py --select 2026Q1
    
    # 使用不同配置
    python ml/quarterly_selector.py --select 2026Q1 --top-n 20

    # 切换特征组：momentum | base | full
    python ml/quarterly_selector.py --backtest --features base
    python ml/quarterly_selector.py --backtest --features full

    # 切换模型配置：hgb_shallow | hgb_medium | hgb_deep
    python ml/quarterly_selector.py --backtest --model hgb_deep

    # 使用预测阈值过滤（只选预测跑赢基准 2%+ 的股票）
    python ml/quarterly_selector.py --backtest --min-threshold 2
    
    # 组合使用（full + hgb_shallow + 8季度 + Top10 + 阈值2%）
    python ml/quarterly_selector.py --backtest --features full --model hgb_shallow --window-quarters 8 --top-n 10 --min-threshold 2
"""

import os
import sys

# 直接运行本脚本时（如 python ml/quarterly_selector.py）把项目根加入 path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

import json
import glob
import argparse
import numpy as np
from collections import defaultdict
from ml.models import create_model, REGRESSOR_PRESETS


# 数据目录
DATA_DIR = './data/quarterly_data'

# 排除的特征（噪声或未来信息）
EXCLUDE_FEATURES = {'price', 'change_pct', 'turnover', 'volume', 'revenue_growth'}

# ============ 特征组合（与 strategy_optimizer 一致：full 来自 ml.features） ============
# 全量数值特征与 build_training_data(FULL_FEATURE_CONFIG) 一致，含 market_momentum_60d
try:
    from ml.features import get_full_numeric_feature_names
    _ALL_FEATURES = get_full_numeric_feature_names()
except Exception:
    _ALL_FEATURES = [
        'pe_ratio', 'pb_ratio', 'peg', 'roe', 'profit_growth', 'dividend_yield',
        'value_score', 'quality_score', 'momentum_20d', 'momentum_60d',
        'rsi_14', 'volatility_20d', 'ma_deviation_20', 'turnover_rate',
        'total_mv', 'circ_mv', 'list_days', 'volume_ratio',
        'market_momentum_20d', 'market_momentum_60d', 'market_volatility_20d', 'market_trend',
        'relative_momentum_20d', 'relative_momentum_60d', 'volatility_ratio_20d',
        'stock_market_correlation_20d', 'stock_beta_20d',
    ]

ALL_FEATURES = _ALL_FEATURES

# 特征子集（本脚本 CLI 用，与 strategy_optimizer 的 full 共用同一全量来源）
MOMENTUM_FEATURES = [
    'momentum_20d', 'momentum_60d', 'rsi_14', 'ma_deviation_20',
    'relative_momentum_20d', 'relative_momentum_60d',
]
BASE_FEATURES = [
    'pe_ratio', 'pb_ratio', 'peg', 'roe', 'profit_growth', 'dividend_yield',
    'value_score', 'quality_score', 'momentum_20d', 'momentum_60d',
]

FEATURE_SETS = {
    'momentum': MOMENTUM_FEATURES,
    'base': BASE_FEATURES,
    'full': ALL_FEATURES,
}
BEST_FEATURES = ALL_FEATURES  # 默认与 strategy_optimizer 一致用 full

# 模型配置：与 ml.models.REGRESSOR_PRESETS 统一，仅保留 HGB 预设
MODEL_CONFIGS = {k: v for k, v in REGRESSOR_PRESETS.items() if v.get('model') == 'hgb'}
# 默认模型配置（create_sklearn_regressor 接受 {model, params} 或纯 params）
MODEL_CONFIG = MODEL_CONFIGS['hgb_shallow']

# 策略配置（与 strategy_optimizer 一致：窗口按季度数）
DEFAULT_TOP_K = 10           # 默认选股数量（Top 5 精度最高，Top 10 Alpha 最高）
DEFAULT_WINDOW_QUARTERS = 8  # 默认训练窗口（8 季度 = 2 年）
MIN_PRED_THRESHOLD = 2.0     # 最小预测阈值（略有帮助）
MIN_TRAIN_MONTHS = 2         # 按月训练时至少需要的训练月数


def load_quarterly_data():
    """加载季度数据文件"""
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, '*_ml_training_data.json')))
    
    if not all_files:
        raise FileNotFoundError(f"未找到数据文件，请检查目录: {DATA_DIR}")
    
    quarterly_data = {}  # {(year, quarter): {month: [records]}}
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        # 解析文件名如 2020_Q1_ml_training_data.json
        parts = filename.split('_')
        year, quarter = int(parts[0]), int(parts[1][1])
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 按月份分组
        monthly = defaultdict(list)
        for record in data:
            buy_date = record.get('features', {}).get('buy_date', '')
            if buy_date:
                monthly[buy_date[:7]].append(record)
        
        quarterly_data[(year, quarter)] = dict(monthly)
    
    return quarterly_data


def get_features(data, use_best=True):
    """获取特征名列表
    
    Args:
        data: 数据列表
        use_best: 是否使用优化后的精选特征（默认True）
    """
    if use_best:
        return BEST_FEATURES
    
    if not data:
        return []
    sample = data[0].get('features', {})
    meta_keys = {'code', 'name', 'buy_date', 'sell_date', 'buy_price', 'sell_price', 'industry', 'market'}
    return [k for k, v in sample.items() 
            if isinstance(v, (int, float)) and k not in meta_keys and k not in EXCLUDE_FEATURES]


def extract_features(data, feature_names):
    """提取特征矩阵"""
    X = []
    for d in data:
        feat = d.get('features', {})
        row = [float(feat.get(f, 0) or 0) for f in feature_names]
        X.append(row)
    return np.nan_to_num(np.array(X), nan=0.0, posinf=0.0, neginf=0.0)


def get_relative_returns(data):
    """获取相对收益"""
    return np.array([(d.get('return_pct', 0) or 0) - (d.get('index_return_pct', 0) or 0) for d in data])


def get_labels(data, threshold=3.0):
    """获取标签（相对收益 > threshold 为上涨）"""
    return (get_relative_returns(data) > threshold).astype(int)


def _max_drawdown_pct(monthly_returns_pct):
    """从月度收益率序列计算最大回撤（%），与 strategy_optimizer 一致"""
    if not monthly_returns_pct:
        return 0.0
    cum = 1.0
    peak = 1.0
    dd = 0.0
    for r in monthly_returns_pct:
        cum *= (1 + r / 100)
        peak = max(peak, cum)
        dd = max(dd, (peak - cum) / peak * 100)
    return dd


def parse_quarter(quarter_str):
    """解析季度字符串，如 '2026Q1' -> (2026, 1)"""
    quarter_str = quarter_str.upper()
    if 'Q' in quarter_str:
        year, q = quarter_str.split('Q')
        return int(year), int(q)
    raise ValueError(f"无效的季度格式: {quarter_str}，应为如 '2026Q1'")


def build_monthly_index(quarterly_data):
    """从季度数据构建按时间排序的月份列表，便于按月回测。
    返回: [(month_str, (year, quarter)), ...]，按 month_str 升序
    """
    out = []
    for (y, q), monthly in quarterly_data.items():
        for month_str in monthly:
            if monthly[month_str]:  # 有数据才加入
                out.append((month_str, (y, q)))
    out.sort(key=lambda x: x[0])
    return out


def get_train_data_before_month(quarterly_data, monthly_index, month_str, window_quarters):
    """获取在 month_str 之前、在窗口期内的训练数据（按月训练用）。
    monthly_index: build_monthly_index 的返回值
    window_quarters: 滑动窗口季度数，0 表示全部历史；>0 表示取最近 N 个季度
    """
    train_entries = [(m, yq) for m, yq in monthly_index if m < month_str]
    if window_quarters > 0:
        # 按时间顺序的唯一季度，取最后 window_quarters 个
        seen = {}
        for m, yq in train_entries:
            if yq not in seen:
                seen[yq] = len(seen)
        order = sorted(seen.keys(), key=lambda yq: seen[yq])
        keep = set(order[-window_quarters:])
        train_entries = [(m, yq) for m, yq in train_entries if yq in keep]
    train_data = []
    for m, (y, q) in train_entries:
        train_data.extend(quarterly_data[(y, q)][m])
    return train_data


def evaluate_monthly(test_data, scores, threshold=3.0, top_k=10, min_pred_threshold=None):
    """按月评估：Precision@K 和收益率
    
    Args:
        test_data: 测试数据列表
        scores: 模型预测分数（预测的相对收益 %）
        threshold: 计算 Precision 的相对收益阈值（默认 3%，即跑赢基准 3% 算正确）
        top_k: 选股数量
        min_pred_threshold: 最小预测值阈值（%），低于此值不选入。
                           例如 min_pred_threshold=2 表示只选预测跑赢基准 2%+ 的股票。
                           None 表示不过滤。
    """
    y_true = get_labels(test_data, threshold)
    returns = np.array([d.get('return_pct', 0) or 0 for d in test_data])
    index_returns = np.array([d.get('index_return_pct', 0) or 0 for d in test_data])
    
    months = defaultdict(list)
    for i, d in enumerate(test_data):
        m = d.get('features', {}).get('buy_date', '')[:7]
        months[m].append(i)
    
    total_correct = 0
    total = 0
    precision_details = {}
    return_details = {}
    
    for month, indices in sorted(months.items()):
        indices = np.array(indices)
        month_scores = scores[indices]
        
        # 选股（支持阈值过滤，与 strategy_optimizer 一致）
        if min_pred_threshold is not None:
            # 只选预测值高于阈值的
            valid_mask = month_scores >= min_pred_threshold
            if valid_mask.sum() == 0:
                # 没有满足阈值的，选预测值最高的（至少选1只）
                k = min(top_k, len(indices))
                top_k_idx = np.argsort(month_scores)[::-1][:k]
            else:
                # 从满足阈值的中选 top_k
                valid_indices = np.where(valid_mask)[0]
                k = min(top_k, len(valid_indices))
                sorted_valid = valid_indices[np.argsort(month_scores[valid_indices])[::-1]]
                top_k_idx = sorted_valid[:k]
        else:
            k = min(top_k, len(indices))
            top_k_idx = np.argsort(month_scores)[::-1][:k]
        
        # Precision
        n_correct = y_true[indices][top_k_idx].sum()
        precision_details[month] = (n_correct, len(top_k_idx))
        total_correct += n_correct
        total += len(top_k_idx)
        
        # 收益率
        strategy_return = returns[indices][top_k_idx].mean() if len(top_k_idx) > 0 else 0
        benchmark_return = index_returns[indices][top_k_idx].mean() if len(top_k_idx) > 0 else 0
        return_details[month] = {
            'strategy': strategy_return,
            'benchmark': benchmark_return,
            'alpha': strategy_return - benchmark_return,
            'n_selected': len(top_k_idx),  # 实际选出的数量
        }
    
    precision = total_correct / total if total > 0 else 0
    return precision, precision_details, return_details


def run_backtest(start_year=2020, end_year=None, window_quarters=8, top_k=10, 
                 feature_name='full', model_name='hgb_shallow', min_pred_threshold=None):
    """运行历史回测（含收益计算）。
    策略：每季度初用窗口期内历史数据训练模型，再用该模型对该季度所有月份选股并评估。
    窗口按季度数：取测试季度之前的最近 N 个季度（与 strategy_optimizer 一致）。
    汇总输出格式与 strategy_optimizer 一致：Alpha(年化)/Prec/Win/Sharpe/VolR/AlphaVol/MaxDD。
    
    Args:
        min_pred_threshold: 最小预测值阈值（%），低于此值不选入。
                           例如 2 表示只选预测跑赢基准 2%+ 的股票。
                           None 表示不过滤（默认）。
    """
    print("=" * 70)
    print("加载数据...")
    quarterly_data = load_quarterly_data()
    all_quarters = sorted(quarterly_data.keys())
    if not all_quarters:
        print("无有效季度数据")
        return []
    
    # 获取特征名
    first_q = all_quarters[0]
    sample_month = list(quarterly_data[first_q].values())[0]
    feature_names = get_features(sample_month)
    print(f"数据范围: {all_quarters[0][0]}Q{all_quarters[0][1]} ~ {all_quarters[-1][0]}Q{all_quarters[-1][1]} ({len(all_quarters)} 个季度)")
    print(f"特征数量: {len(feature_names)}")
    
    window_str = f"最近 {window_quarters} 季度" if window_quarters > 0 else "全部历史"
    threshold_str = f">{min_pred_threshold}%" if min_pred_threshold is not None else "无"
    print(f"训练窗口: {window_str}（每季度初训练一次）")
    print(f"预测阈值: {threshold_str}（预测相对收益低于阈值不选入）")
    
    print()
    print("=" * 70)
    end_str = f" ~ {end_year}" if end_year else ""
    print(f"季度回测 ({start_year}{end_str})：每季度训练 → 该季度选股")
    print("=" * 70)
    
    results = []
    all_monthly_returns = []
    
    for test_year, test_quarter in all_quarters:
        if test_year < start_year:
            continue
        if end_year and test_year > end_year:
            continue
        
        # 滑动窗口训练数据：该季度之前的最近 N 个季度（与 strategy_optimizer 一致）
        train_quarters = [(y, q) for (y, q) in all_quarters if (y, q) < (test_year, test_quarter)]
        if window_quarters > 0:
            train_quarters = train_quarters[-window_quarters:]
        
        if len(train_quarters) < 2:
            continue
        
        # 合并训练数据
        train_data = []
        for yq in train_quarters:
            for month_data in quarterly_data[yq].values():
                train_data.extend(month_data)
        
        # 测试数据（该季度所有月份）
        test_data = []
        for month_data in quarterly_data[(test_year, test_quarter)].values():
            test_data.extend(month_data)
        
        if not test_data or len(train_data) < 100:
            continue
        
        # 每季度训练一次模型
        X_train = extract_features(train_data, feature_names)
        y_train = get_relative_returns(train_data)
        X_test = extract_features(test_data, feature_names)
        
        model = create_model(MODEL_CONFIG)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        # 按月评估（与 strategy_optimizer 一致）
        precision, prec_details, ret_details = evaluate_monthly(
            test_data, pred, threshold=3.0, top_k=top_k, min_pred_threshold=min_pred_threshold
        )
        
        # 计算季度收益
        q_strategy = np.mean([v['strategy'] for v in ret_details.values()])
        q_benchmark = np.mean([v['benchmark'] for v in ret_details.values()])
        q_alpha = q_strategy - q_benchmark
        q_precision = precision
        
        # 记录月度收益与精度（与 strategy_optimizer 一致，便于汇总 VolR/AlphaVol/MaxDD）
        for m, v in ret_details.items():
            n_correct, n_total = prec_details.get(m, (0, 0))
            all_monthly_returns.append({
                'month': m, 'strategy': v['strategy'], 'benchmark': v['benchmark'],
                'alpha': v['strategy'] - v['benchmark'], 'n_correct': n_correct, 'n_total': n_total,
            })
        
        status = "✅" if q_alpha > 0 else "❌"
        d_str = ' '.join([f"{m[-2:]}:{n}/{k}={n/k*100:.0f}%" for m, (n, k) in sorted(prec_details.items())])
        print(f"{test_year}Q{test_quarter}: 策略{q_strategy:+.1f}% 基准{q_benchmark:+.1f}% α={q_alpha:+.1f}%{status} | Prec={q_precision:.0%} [{d_str}]")
        
        results.append({
            'quarter': f'{test_year}Q{test_quarter}',
            'strategy_return': q_strategy,
            'benchmark_return': q_benchmark,
            'alpha': q_alpha,
            'precision': q_precision,
        })
    
    # 汇总
    print()
    print("=" * 70)
    print("汇总结果")
    print("=" * 70)
    
    if results:
        cum_strategy = 1.0
        cum_benchmark = 1.0
        for r in all_monthly_returns:
            cum_strategy *= (1 + r['strategy'] / 100)
            cum_benchmark *= (1 + r['benchmark'] / 100)
        total_strategy = (cum_strategy - 1) * 100
        total_benchmark = (cum_benchmark - 1) * 100
        total_alpha = total_strategy - total_benchmark
        n_months = len(all_monthly_returns)
        years = n_months / 12.0 if n_months else 1.0
        ann_strategy = ((cum_strategy ** (1 / years)) - 1) * 100 if years > 0 else 0.0
        ann_benchmark = ((cum_benchmark ** (1 / years)) - 1) * 100 if years > 0 else 0.0
        ann_alpha = ann_strategy - ann_benchmark
        total_correct = sum(r['n_correct'] for r in all_monthly_returns)
        total_samples = sum(r['n_total'] for r in all_monthly_returns)
        avg_precision = total_correct / total_samples if total_samples > 0 else 0.0
        win_rate = sum(1 for r in all_monthly_returns if r['alpha'] > 0) / n_months if n_months else 0.0
        monthly_alpha = [r['alpha'] for r in all_monthly_returns]
        monthly_strategy = [r['strategy'] for r in all_monthly_returns]
        monthly_benchmark = [r['benchmark'] for r in all_monthly_returns]
        vol_strategy = np.std(monthly_strategy)
        vol_benchmark = np.std(monthly_benchmark) or 1e-6
        vol_ratio = vol_strategy / vol_benchmark
        alpha_vol = np.std(monthly_alpha)
        max_dd_strategy = _max_drawdown_pct(monthly_strategy)
        sharpe = np.mean(monthly_alpha) / (np.std(monthly_alpha) + 1e-6)

        print(f"回测期间: {results[0]['quarter']} ~ {results[-1]['quarter']} ({len(results)}个季度, {n_months}月)")
        print(f"总月数: {len(all_monthly_returns)}（每季度训练一次）")
        print()
        print(f"📈 累计收益:")
        print(f"   策略: {total_strategy:+.1f}%")
        print(f"   基准: {total_benchmark:+.1f}%")
        print(f"   超额: {total_alpha:+.1f}%")
        print()
        print(f"📊 与 strategy_optimizer 一致汇总（年化 Alpha / Prec / Win / Sharpe / VolR / AlphaVol / MaxDD）:")
        print(f"   {feature_name:15} + {model_name:12} | Alpha={ann_alpha:+6.1f}%(年化) Prec={avg_precision:.1%} Win={win_rate:.0%} Sharpe={sharpe:.2f} | VolR={vol_ratio:.2f} AlphaVol={alpha_vol:.1f}% MaxDD={max_dd_strategy:.1f}%")
        print()
        print(f"📊 季度统计:")
        win_count = sum(1 for r in results if r['alpha'] > 0)
        print(f"   跑赢次数: {win_count}/{len(results)} ({win_count/len(results):.0%})")
        print(f"   平均季度Alpha: {np.mean([r['alpha'] for r in results]):+.1f}%")
        print(f"   平均Precision@10: {avg_precision:.1%}")
    
    return results


def select_stocks(quarter_str, top_n=10, window_quarters=None, min_pred_threshold=None):
    """为指定季度选股。策略：季度初用窗口期历史数据训练模型，再对该季度所有月份选股。
    窗口按季度数：取该季度之前的最近 N 个季度（与 strategy_optimizer 一致）。
    
    Args:
        quarter_str: 季度字符串，如 "2026Q1"
        top_n: 选股数量
        window_quarters: 训练窗口季度数，None 时使用 DEFAULT_WINDOW_QUARTERS，0 表示全部历史。
        min_pred_threshold: 最小预测值阈值（%），低于此值不选入。None 表示不过滤。
    """
    if window_quarters is None:
        window_quarters = DEFAULT_WINDOW_QUARTERS
    year, quarter = parse_quarter(quarter_str)
    
    print("=" * 70)
    print(f"为 {year}Q{quarter} 选股（季度初训练 → 该季度选股）")
    print("=" * 70)
    
    quarterly_data = load_quarterly_data()
    all_quarters = sorted(quarterly_data.keys())
    
    if (year, quarter) not in quarterly_data:
        print(f"错误: 未找到 {year}Q{quarter} 的数据")
        return []
    
    months_in_quarter = sorted(quarterly_data[(year, quarter)].keys())
    if not months_in_quarter:
        print(f"错误: {year}Q{quarter} 无月度数据")
        return []
    
    feature_names = get_features(quarterly_data[(year, quarter)][months_in_quarter[0]])
    print(f"特征数量: {len(feature_names)}")
    window_str = f"最近 {window_quarters} 季度" if window_quarters > 0 else "全部历史"
    threshold_str = f">{min_pred_threshold}%" if min_pred_threshold is not None else "无"
    print(f"训练窗口: {window_str}")
    print(f"预测阈值: {threshold_str}")
    
    # 滑动窗口训练数据：该季度之前的最近 N 个季度（与 strategy_optimizer 一致）
    train_quarters = [(y, q) for (y, q) in all_quarters if (y, q) < (year, quarter)]
    if window_quarters > 0:
        train_quarters = train_quarters[-window_quarters:]
    if len(train_quarters) < 2:
        print(f"错误: 训练数据不足（需要至少2个季度）")
        return []
    
    # 合并训练数据
    train_data = []
    for yq in train_quarters:
        for month_data in quarterly_data[yq].values():
            train_data.extend(month_data)
    
    print(f"训练数据: {len(train_quarters)} 个季度, {len(train_data)} 条记录")
    print()
    
    # 季度初训练一次模型（模型内部做 StandardScaler）
    X_train = extract_features(train_data, feature_names)
    y_train = get_relative_returns(train_data)
    model = create_model(MODEL_CONFIG)
    model.fit(X_train, y_train)
    
    # 对该季度每个月选股
    all_selections = {}
    for month_str in months_in_quarter:
        candidates = quarterly_data[(year, quarter)].get(month_str, [])
        if not candidates:
            continue
        X_candidates = extract_features(candidates, feature_names)
        pred_returns = model.predict(X_candidates)
        
        # 选股（支持阈值过滤，与 strategy_optimizer 一致）
        if min_pred_threshold is not None:
            # 只选预测值高于阈值的
            valid_mask = pred_returns >= min_pred_threshold
            if valid_mask.sum() == 0:
                # 没有满足阈值的，选预测值最高的（至少选1只）
                top_indices = np.argsort(pred_returns)[::-1][:top_n]
            else:
                # 从满足阈值的中选 top_n
                valid_indices = np.where(valid_mask)[0]
                k = min(top_n, len(valid_indices))
                sorted_valid = valid_indices[np.argsort(pred_returns[valid_indices])[::-1]]
                top_indices = sorted_valid[:k]
        else:
            top_indices = np.argsort(pred_returns)[::-1][:top_n]
        
        selected = []
        for i, idx in enumerate(top_indices):
            stock = candidates[idx]
            selected.append({
                'rank': i + 1,
                'code': stock['features'].get('code', ''),
                'name': stock['features'].get('name', ''),
                'predicted_return': float(pred_returns[idx])
            })
        all_selections[month_str] = selected
        
        n_selected = len(selected)
        threshold_info = f" (阈值过滤后)" if min_pred_threshold is not None else ""
        print(f"  {month_str}: 候选 {len(candidates)} 只，选出 {n_selected} 只{threshold_info}")
    
    # 打印并保存
    print()
    print("=" * 70)
    print(f"选股结果: Top {top_n}（按月份）")
    print("=" * 70)
    for month_str in sorted(all_selections.keys()):
        sel = all_selections[month_str]
        print(f"\n--- {month_str} ---")
        print(f"{'排名':<5} {'代码':<10} {'名称':<10} {'预测相对收益':<15}")
        print("-" * 45)
        for s in sel:
            print(f"{s['rank']:<5} {s['code']:<10} {s['name']:<10} {s['predicted_return']:+.2f}%")
    
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/selection_{year}Q{quarter}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'quarter': f'{year}Q{quarter}',
            'model_config': MODEL_CONFIG,
            'train_mode': 'monthly',
            'window_quarters': window_quarters,
            'monthly_selections': all_selections
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_file}")
    
    return list(all_selections.values())


def main():
    parser = argparse.ArgumentParser(description='季度选股系统')
    parser.add_argument('--backtest', action='store_true', help='运行历史回测')
    parser.add_argument('--select', type=str, help='为指定季度选股（如 2026Q1）')
    parser.add_argument('--top-n', type=int, default=DEFAULT_TOP_K, help=f'选股数量（默认{DEFAULT_TOP_K}）')
    parser.add_argument('--start-year', type=int, default=2020, help='回测起始年份（默认2012，与optimizer一致）')
    parser.add_argument('--end-year', type=int, default=2025, help='回测结束年份（可选，默认到最新）')
    parser.add_argument('--window-quarters', type=int, default=8,
                        help=f'滑动窗口季度数，0=全部历史（默认{DEFAULT_WINDOW_QUARTERS}=2年）')
    parser.add_argument('--features', type=str, choices=list(FEATURE_SETS.keys()), default='full',
                        help='特征组: momentum=动量聚焦, base=估值+质量+动量, full=全特征(默认)')
    parser.add_argument('--model', type=str, choices=list(MODEL_CONFIGS.keys()), default='hgb_shallow',
                        help='模型配置: hgb_shallow, hgb_medium, hgb_deep(默认)')
    parser.add_argument('--min-threshold', type=float, default=8,
                        help='最小预测阈值(%%)，低于此值不选入（如 2 表示只选预测跑赢基准2%%+的股票）')
    args = parser.parse_args()

    window_quarters = args.window_quarters if args.window_quarters is not None else DEFAULT_WINDOW_QUARTERS
    min_pred_threshold = args.min_threshold
    
    # 切换特征组和模型配置
    global BEST_FEATURES, MODEL_CONFIG
    BEST_FEATURES = FEATURE_SETS[args.features]
    MODEL_CONFIG = MODEL_CONFIGS[args.model]
    params = MODEL_CONFIG.get('params', MODEL_CONFIG)
    threshold_str = f" 阈值>{min_pred_threshold}%" if min_pred_threshold is not None else ""
    print(f"配置: 特征={args.features}({len(BEST_FEATURES)}个) 模型={args.model}(depth={params.get('max_depth', '')}) 窗口={window_quarters}季 Top{args.top_n}{threshold_str}")
    print("=" * 70)
    
    if args.backtest:
        run_backtest(start_year=args.start_year, end_year=args.end_year,
                     window_quarters=window_quarters, top_k=args.top_n,
                     feature_name=args.features, model_name=args.model,
                     min_pred_threshold=min_pred_threshold)
    elif args.select:
        select_stocks(args.select, args.top_n, window_quarters, min_pred_threshold=min_pred_threshold)
    else:
        # 默认运行回测
        run_backtest(start_year=args.start_year, end_year=args.end_year,
                     window_quarters=window_quarters, top_k=args.top_n,
                     feature_name=args.features, model_name=args.model,
                     min_pred_threshold=min_pred_threshold)


if __name__ == '__main__':
    main()
