# -*- coding: utf-8 -*-
"""
策略优化实验

逻辑概要:
  1. 数据: 季度 JSON（默认 data/quarterly_data_v2，可用 --data-dir 指定），按季度训练、按月评估。
  2. 训练窗口按季度: window_quarters=8 表示用测试季度前 8 个季度（2 年），不含更早季度。
  3. 实验1: 特征组合 x 模型（8 季度窗口）→ 按年化 Alpha 取最优。
  4. 实验2: 在最优特征+模型上扫窗口 8/12/16/20 季度 或 全部历史。
  5. 实验3: 扫选股数量 Top 5/10/15/20/30，按 Precision（且 Alpha>0）更新最优。
  6. 实验4/5: 样本加权、预测阈值，仅对比不更新最优。
  7. 输出: 最优配置 + 稳定性优先推荐（从实验1 + 最终最优配置中选 Alpha>0 且超额波动最小）。

数据流:
  1. 特征来源: 全部来自已生成的季度 JSON（默认 data/quarterly_data_v2/*_ml_training_data.json）。
     JSON 由 ml/batch_build_data.py 调用 ml/build_training_data.py 生成；
     build_training_data 使用 DataManager + ml/features 的 FeatureEngineer(FULL_FEATURE_CONFIG)
     + ml/features/market 的 compute_market_features、compute_stock_market_relation 计算特征。
  2. 本脚本不计算特征，只从每条记录的 record['features'] 中按 feature_names 读取数值。
  3. 流程: load_quarterly_data() → 按 (年,季) 索引 → 对每个测试季度用滑动窗口合并训练季度
     → extract_features(train_data, feature_names) / get_relative_returns(train_data)
     → 模型 fit(X_train, y_train) / predict(X_test)（模型内部 StandardScaler）→ 按月取 Top-K 算 Precision/收益/稳定性。

稳定性指标（相对基准指数）:
  VolR: 策略月收益标准差/指数月收益标准差，<1 表示策略波动小于指数。
  AlphaVol: 月度超额收益(Alpha)的标准差，越小超额越稳定。
  MaxDD: 策略最大回撤(%)，越小回撤越小。
"""

import os
import sys

# 直接运行本脚本时（如 python ml/strategy_optimizer.py）把项目根加入 path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

import json
import glob
import os
import numpy as np
from collections import defaultdict
from datetime import datetime
from ml.models import create_model, REGRESSOR_PRESETS

# 实验报告输出目录
REPORT_DIR = os.path.join(_root, 'logs', 'strategy_optimizer')

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

# 默认使用 v2 数据（含正确 revenue_growth、历史成分股、无错误 index_return 缓存）
DATA_DIRS = {
    '000300': './data/quarterly_data_v2',
    '000905': './data/quarterly_data_csi500',
}
DATA_DIR = DATA_DIRS['000300']

# 所有可用特征（优先与 ml/features 一致）
ALL_FEATURES = _ALL_FEATURES

# 特征组合方案
# 特征子集：full 与 ml/features 一致，其余为实验用子集，名称与含义固定即可，无需与别处统一
FEATURE_SETS = {
    'base_10': ['pe_ratio', 'pb_ratio', 'peg', 'roe', 'profit_growth', 'dividend_yield',
                'value_score', 'quality_score', 'momentum_20d', 'momentum_60d'],
    'value_only': ['pe_ratio', 'pb_ratio', 'peg', 'dividend_yield', 'value_score'],
    'momentum_focus': ['momentum_20d', 'momentum_60d', 'rsi_14', 'ma_deviation_20', 
                       'relative_momentum_20d', 'relative_momentum_60d'],
    'quality_focus': ['roe', 'profit_growth', 'quality_score', 'dividend_yield'],
    'market_aware': ['pe_ratio', 'pb_ratio', 'roe', 'momentum_20d', 'momentum_60d',
                     'market_momentum_20d', 'market_trend', 'stock_beta_20d'],
    'low_volatility': ['pe_ratio', 'pb_ratio', 'roe', 'dividend_yield', 'volatility_20d',
                       'stock_beta_20d', 'volatility_ratio_20d'],
    'full': ALL_FEATURES,
}

# 模型配置：与 ml.models.REGRESSOR_PRESETS 统一
MODEL_CONFIGS = REGRESSOR_PRESETS


def _to_serializable(obj):
    """将结果中的 numpy 类型转为 Python 原生类型，便于写入 JSON。"""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def _write_report(report_dir: str, start_year: int, data_dir: str,
                  experiments: list, best_config: tuple, best_result: dict, stable_best: dict = None):
    """将实验报告写入目录。"""
    os.makedirs(report_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    payload = {
        'start_year': start_year,
        'data_dir': data_dir,
        'experiments': _to_serializable(experiments),
        'best_config': list(best_config) if best_config else None,
        'best_result': _to_serializable(best_result) if best_result else None,
        'stable_best': _to_serializable(stable_best) if stable_best else None,
    }
    json_path = os.path.join(report_dir, f'report_{ts}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    lines = [
        "策略优化实验报告",
        "=" * 60,
        f"生成时间: {datetime.now().isoformat()}",
        f"起始年份: {start_year}  数据目录: {data_dir}",
        "",
    ]
    if best_config and best_result:
        wq = best_config[2]
        w_str = f"{wq}季度" if wq and wq > 0 else "全部历史"
        lines.extend([
            "最优配置",
            "-" * 40,
            f"特征组合: {best_config[0]}",
            f"模型配置: {best_config[1]}",
            f"训练窗口: {w_str}",
            f"选股数量: Top {best_config[3]}",
            f"年化Alpha: {best_result.get('ann_alpha', 0):+.1f}%  Precision: {best_result.get('avg_precision', 0):.1%}  月度胜率: {best_result.get('win_rate', 0):.0%}",
            f"VolR: {best_result.get('vol_ratio', 0):.2f}  AlphaVol: {best_result.get('alpha_vol', 0):.1f}%  MaxDD: {best_result.get('max_dd_strategy', 0):.1f}%",
            "",
        ])
    if stable_best:
        wq = stable_best.get('window')
        w_str = f"{wq}季度" if wq and wq > 0 else "全部历史"
        lines.extend([
            "稳定性优先推荐",
            "-" * 40,
            f"特征+模型: {stable_best.get('features')} + {stable_best.get('model')}  窗口={w_str}  Top{stable_best.get('top_k')}",
            f"Alpha: {stable_best.get('ann_alpha', 0):+.1f}%  Prec: {stable_best.get('avg_precision', 0):.1%}  Win: {stable_best.get('win_rate', 0):.0%}",
            f"VolR: {stable_best.get('vol_ratio', 0):.2f}  AlphaVol: {stable_best.get('alpha_vol', 0):.1f}%  MaxDD: {stable_best.get('max_dd_strategy', 0):.1f}%",
            "",
        ])
    lines.append(f"完整数据已保存: {json_path}")
    txt_path = os.path.join(report_dir, f'summary_{ts}.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n报告已写入: {json_path}")
    print(f"摘要已写入: {txt_path}")


def load_quarterly_data(data_dir=None):
    """加载季度数据。文件名约定: YYYY_Qn_ml_training_data.json（如 2023_Q1），parts[1][1] 取季度 1~4。
    data_dir: 数据目录，默认使用 DATA_DIR（quarterly_data_v2）。相对路径相对于项目根目录。
    """
    base = (data_dir or DATA_DIR).replace('/', os.sep)
    if not os.path.isabs(base):
        base = os.path.join(_root, base)
    all_files = sorted(glob.glob(os.path.join(base, '*_ml_training_data.json')))
    quarterly_data = {}
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        try:
            year, quarter = int(parts[0]), int(parts[1][1])
        except (IndexError, ValueError):
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        monthly = defaultdict(list)
        for record in data:
            buy_date = record.get('features', {}).get('buy_date', '')
            if buy_date:
                monthly[buy_date[:7]].append(record)
        
        quarterly_data[(year, quarter)] = dict(monthly)
    
    return quarterly_data


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


def _max_drawdown_pct(monthly_returns_pct):
    """从月度收益率序列计算最大回撤（%）"""
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




def evaluate_strategy(quarterly_data, feature_names, model_config, window_quarters=8, 
                      top_k=10, start_year=2019, use_sample_weights=False,
                      min_pred_threshold=None):
    """评估单个策略配置
    
    Args:
        window_quarters: 训练窗口季度数，如 8 表示用测试季度前 8 个季度；0 表示用全部历史。
        min_pred_threshold: 最小预测值阈值，低于此值不选入
    """
    all_quarters = sorted(quarterly_data.keys())
    all_monthly = []
    
    for test_year, test_quarter in all_quarters:
        if test_year < start_year:
            continue
        
        # 滑动窗口：仅用测试季度之前的最近 N 个季度
        train_quarters = [(y, q) for (y, q) in all_quarters if (y, q) < (test_year, test_quarter)]
        if window_quarters > 0:
            train_quarters = train_quarters[-window_quarters:]
        
        if len(train_quarters) < 2:
            continue
        
        # 合并数据
        train_data = [r for yq in train_quarters for r in quarterly_data[yq].values() for r in r]
        test_data = [r for month_data in quarterly_data[(test_year, test_quarter)].values() for r in month_data]
        
        if not test_data:
            continue
        
        # 准备数据
        X_train = extract_features(train_data, feature_names)
        y_train = get_relative_returns(train_data)
        X_test = extract_features(test_data, feature_names)
        
        # 样本权重（可选）
        sample_weights = None
        if use_sample_weights:
            # 近期数据权重更高
            weights = []
            for i, yq in enumerate(train_quarters):
                n_samples = len([r for r in quarterly_data[yq].values() for r in r])
                w = 1.0 + i * 0.1  # 越近的数据权重越高
                weights.extend([w] * n_samples)
            sample_weights = np.array(weights)
        
        # 训练（模型内部做 StandardScaler，传入原始 X）
        model = create_model(model_config)
        if sample_weights is not None:
            try:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            except Exception:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        
        # 按月评估
        months = defaultdict(list)
        for i, d in enumerate(test_data):
            m = d.get('features', {}).get('buy_date', '')[:7]
            months[m].append(i)
        
        for month, indices in sorted(months.items()):
            indices = np.array(indices)
            month_pred = pred[indices]
            month_returns = np.array([test_data[i].get('return_pct', 0) or 0 for i in indices])
            month_idx_returns = np.array([test_data[i].get('index_return_pct', 0) or 0 for i in indices])
            month_relative = month_returns - month_idx_returns
            
            # 选股（支持阈值过滤）
            if min_pred_threshold is not None:
                # 只选预测值高于阈值的
                valid_mask = month_pred >= min_pred_threshold
                if valid_mask.sum() == 0:
                    # 没有满足阈值的，选预测值最高的
                    k = min(top_k, len(indices))
                    top_idx = np.argsort(month_pred)[::-1][:k]
                else:
                    # 从满足阈值的中选top_k
                    valid_indices = np.where(valid_mask)[0]
                    k = min(top_k, len(valid_indices))
                    sorted_valid = valid_indices[np.argsort(month_pred[valid_indices])[::-1]]
                    top_idx = sorted_valid[:k]
            else:
                k = min(top_k, len(indices))
                top_idx = np.argsort(month_pred)[::-1][:k]
            
            # 计算指标
            n_correct = (month_relative[top_idx] > 3).sum()
            precision = n_correct / len(top_idx) if len(top_idx) > 0 else 0
            strategy_ret = month_returns[top_idx].mean()
            benchmark_ret = month_idx_returns[top_idx].mean()
            
            all_monthly.append({
                'month': month,
                'precision': precision,
                'n_correct': n_correct,
                'n_total': len(top_idx),
                'strategy': strategy_ret,
                'benchmark': benchmark_ret,
                'alpha': strategy_ret - benchmark_ret,
            })
    
    if not all_monthly:
        return None
    
    # 汇总
    cum_strategy = 1.0
    cum_benchmark = 1.0
    for r in all_monthly:
        cum_strategy *= (1 + r['strategy'] / 100)
        cum_benchmark *= (1 + r['benchmark'] / 100)
    
    total_correct = sum(r['n_correct'] for r in all_monthly)
    total_samples = sum(r['n_total'] for r in all_monthly)
    n_months = len(all_monthly)
    
    # 年化收益率（月度复利年化）: (1+R_cum)^(12/n_months) - 1
    years = n_months / 12.0 if n_months else 1.0
    ann_strategy = ((cum_strategy ** (1 / years)) - 1) * 100 if years > 0 else 0.0
    ann_benchmark = ((cum_benchmark ** (1 / years)) - 1) * 100 if years > 0 else 0.0
    ann_alpha = ann_strategy - ann_benchmark
    
    # 稳定性指标（相对基准指数）
    monthly_strategy = [r['strategy'] for r in all_monthly]
    monthly_benchmark = [r['benchmark'] for r in all_monthly]
    monthly_alpha = [r['alpha'] for r in all_monthly]
    vol_strategy = np.std(monthly_strategy)
    vol_benchmark = np.std(monthly_benchmark) or 1e-6
    vol_ratio = vol_strategy / vol_benchmark  # >1 表示策略波动大于指数
    alpha_vol = np.std(monthly_alpha)  # 超额收益波动，越小越稳定
    max_dd_strategy = _max_drawdown_pct(monthly_strategy)
    max_dd_benchmark = _max_drawdown_pct(monthly_benchmark)
    
    return {
        'total_strategy': (cum_strategy - 1) * 100,
        'total_benchmark': (cum_benchmark - 1) * 100,
        'total_alpha': (cum_strategy - 1) * 100 - (cum_benchmark - 1) * 100,
        'ann_strategy': ann_strategy,
        'ann_benchmark': ann_benchmark,
        'ann_alpha': ann_alpha,
        'n_years': years,
        'avg_precision': total_correct / total_samples if total_samples > 0 else 0,
        'win_rate': sum(1 for r in all_monthly if r['alpha'] > 0) / len(all_monthly),
        'n_months': n_months,
        'avg_monthly_alpha': np.mean(monthly_alpha),
        'sharpe': np.mean(monthly_alpha) / (np.std(monthly_alpha) + 1e-6),
        # 稳定性
        'vol_strategy': vol_strategy,
        'vol_benchmark': vol_benchmark,
        'vol_ratio': vol_ratio,
        'alpha_vol': alpha_vol,
        'max_dd_strategy': max_dd_strategy,
        'max_dd_benchmark': max_dd_benchmark,
    }


def run_optimization(start_year=2019, data_dir=None):
    """运行优化实验。
    data_dir: 季度数据目录，默认 DATA_DIR（data/quarterly_data_v2）。
    """
    print("=" * 80)
    print("策略优化实验")
    print("=" * 80)
    print()
    
    # 加载数据
    print("加载数据...")
    quarterly_data = load_quarterly_data(data_dir=data_dir)
    if not quarterly_data:
        print(f"无季度数据，请检查目录: {data_dir or DATA_DIR}")
        return [], None, None, None
    print(f"数据目录: {data_dir or DATA_DIR}")
    print(f"数据范围: {min(quarterly_data.keys())} ~ {max(quarterly_data.keys())}")
    print()
    
    # 实验配置
    experiments = []
    
    # 1. 特征组合 x 模型配置
    print("实验1: 特征组合 x 模型配置")
    print("-" * 80)
    
    best_result = None
    best_config = None
    
    for feat_name, features in FEATURE_SETS.items():
        for model_name, model_config in MODEL_CONFIGS.items():
            result = evaluate_strategy(
                quarterly_data, features, model_config,
                window_quarters=8, top_k=10, start_year=start_year
            )
            
            if result:
                experiments.append({
                    'features': feat_name,
                    'model': model_name,
                    'window': 8,
                    'top_k': 10,
                    **result
                })
                
                if best_result is None or result['ann_alpha'] > best_result['ann_alpha']:
                    best_result = result
                    best_config = (feat_name, model_name, 8, 10)
                
                # 稳定性: VolR=策略波动/指数波动, AlphaVol=超额波动(越小越稳), MaxDD=策略最大回撤；Alpha 为年化
                vol_r = result.get('vol_ratio', 0)
                a_vol = result.get('alpha_vol', 0)
                mdd = result.get('max_dd_strategy', 0)
                print(f"{feat_name:15} + {model_name:12} | Alpha={result['ann_alpha']:+6.1f}%(年化) Prec={result['avg_precision']:.1%} Win={result['win_rate']:.0%} Sharpe={result['sharpe']:.2f} | VolR={vol_r:.2f} AlphaVol={a_vol:.1f}% MaxDD={mdd:.1f}%")
    
    print()
    
    # 2. 不同窗口大小（按季度：8=2年，12=3年，16=4年，20=5年，0=全部）
    print("实验2: 训练窗口大小（使用最优特征+模型，按季度）")
    print("-" * 80)
    
    best_feat = best_config[0] if best_config else list(FEATURE_SETS.keys())[0]
    best_model = best_config[1] if best_config else list(MODEL_CONFIGS.keys())[0]
    
    for window_quarters in [8, 12, 16, 20, 0]:
        result = evaluate_strategy(
            quarterly_data, FEATURE_SETS[best_feat], MODEL_CONFIGS[best_model],
            window_quarters=window_quarters, top_k=10, start_year=start_year
        )
        
        if result:
            window_str = f"{window_quarters}季" if window_quarters > 0 else "全部"
            vr, av, mdd = result.get('vol_ratio', 0), result.get('alpha_vol', 0), result.get('max_dd_strategy', 0)
            print(f"窗口={window_str:4} | Alpha={result['ann_alpha']:+6.1f}%(年化) Prec={result['avg_precision']:.1%} Win={result['win_rate']:.0%} | VolR={vr:.2f} AlphaVol={av:.1f}% MaxDD={mdd:.1f}%")
            
            if best_result is None or result['ann_alpha'] > best_result['ann_alpha']:
                best_result = result
                best_config = (best_feat, best_model, window_quarters, 10)
    
    print()
    
    # 3. 不同选股数量
    print("实验3: 选股数量")
    print("-" * 80)
    
    best_window_q = best_config[2] if best_config else 8
    
    for top_k in [5, 10, 15, 20, 30]:
        result = evaluate_strategy(
            quarterly_data, FEATURE_SETS[best_feat], MODEL_CONFIGS[best_model],
            window_quarters=best_window_q, top_k=top_k, start_year=start_year
        )
        
        if result:
            vr, av, mdd = result.get('vol_ratio', 0), result.get('alpha_vol', 0), result.get('max_dd_strategy', 0)
            print(f"Top {top_k:2} | Alpha={result['ann_alpha']:+6.1f}%(年化) Prec={result['avg_precision']:.1%} Win={result['win_rate']:.0%} | VolR={vr:.2f} AlphaVol={av:.1f}% MaxDD={mdd:.1f}%")
            
            # 选更高精确率的配置（且年化超额为正）；实验2无结果时在此首次赋值
            if best_result is None:
                if result['ann_alpha'] > 0:
                    best_result, best_config = result, (best_feat, best_model, best_window_q, top_k)
            elif result['avg_precision'] > best_result['avg_precision'] and result['ann_alpha'] > 0:
                best_result, best_config = result, (best_feat, best_model, best_window_q, top_k)
    
    print()
    
    # 4. 样本加权
    print("实验4: 样本加权")
    print("-" * 80)
    
    best_top_k = best_config[3] if best_config else 10
    
    for use_weights in [False, True]:
        result = evaluate_strategy(
            quarterly_data, FEATURE_SETS[best_feat], MODEL_CONFIGS[best_model],
            window_quarters=best_window_q, top_k=best_top_k, start_year=start_year,
            use_sample_weights=use_weights
        )
        
        if result:
            w_str = "有权重" if use_weights else "无权重"
            print(f"{w_str:6} | Alpha={result['ann_alpha']:+6.1f}%(年化) Prec={result['avg_precision']:.1%} Win={result['win_rate']:.0%}")
    
    print()
    
    # 5. 预测值阈值
    print("实验5: 预测值阈值过滤")
    print("-" * 80)
    
    for threshold in [None, 0, 2, 5, 8]:
        result = evaluate_strategy(
            quarterly_data, FEATURE_SETS[best_feat], MODEL_CONFIGS[best_model],
            window_quarters=best_window_q, top_k=best_top_k, start_year=start_year,
            min_pred_threshold=threshold
        )
        
        if result:
            t_str = f">{threshold}%" if threshold is not None else "无"
            print(f"阈值={t_str:5} | Alpha={result['ann_alpha']:+6.1f}%(年化) Prec={result['avg_precision']:.1%} Win={result['win_rate']:.0%}")
    
    print()
    
    # 最优配置
    print("=" * 80)
    print("最优配置")
    print("=" * 80)
    
    if best_config:
        wq = best_config[2]
        w_str = f"{wq}季度" if wq and wq > 0 else "全部历史"
        print(f"特征组合: {best_config[0]}")
        print(f"模型配置: {best_config[1]}")
        print(f"训练窗口: {w_str}")
        print(f"选股数量: Top {best_config[3]}")
        print()
        print(f"预期表现（年化）:")
        print(f"  年化Alpha: {best_result['ann_alpha']:+.1f}%  (累计Alpha: {best_result['total_alpha']:+.1f}%)")
        print(f"  Precision: {best_result['avg_precision']:.1%}")
        print(f"  月度胜率: {best_result['win_rate']:.0%}")
        print(f"  Sharpe: {best_result['sharpe']:.2f}")
        print(f"稳定性（相对基准指数）:")
        print(f"  波动比 VolR: {best_result.get('vol_ratio', 0):.2f} (策略月收益标准差/指数，<1 更稳)")
        print(f"  超额波动 AlphaVol: {best_result.get('alpha_vol', 0):.1f}% (月度超额收益标准差，越小越稳)")
        print(f"  策略最大回撤 MaxDD: {best_result.get('max_dd_strategy', 0):.1f}%")
    
    # 稳定性优先：在 Alpha>0 的配置中选超额波动最小者（含实验1网格 + 最终最优配置）
    if experiments or best_config:
        candidates = list(experiments)
        if best_config and best_result:
            candidates.append({
                'features': best_config[0],
                'model': best_config[1],
                'window': best_config[2],
                'top_k': best_config[3],
                **best_result,
            })
        positive_alpha = [e for e in candidates if e.get('ann_alpha', 0) > 0]
        stable_best = None
        if positive_alpha:
            by_stable = sorted(positive_alpha, key=lambda e: (e.get('alpha_vol', 999), e.get('vol_ratio', 999)))
            stable_best = by_stable[0]
            print()
            print("=" * 80)
            print("稳定性优先推荐（Alpha>0 且超额波动最小、波动比尽量接近指数）")
            print("=" * 80)
            wq = stable_best.get('window')
            w_str = f"{wq}季度" if wq and wq > 0 else "全部历史"
            print(f"特征+模型: {stable_best.get('features')} + {stable_best.get('model')} 窗口={w_str} Top{stable_best.get('top_k')}")
            print(f"  Alpha={stable_best['ann_alpha']:+.1f}%(年化) Prec={stable_best['avg_precision']:.1%} Win={stable_best['win_rate']:.0%}")
            print(f"  VolR={stable_best.get('vol_ratio', 0):.2f} AlphaVol={stable_best.get('alpha_vol', 0):.1f}% MaxDD={stable_best.get('max_dd_strategy', 0):.1f}%")
    else:
        stable_best = None
    
    return experiments, best_config, best_result, stable_best


def run_ensemble_experiment(start_year=2019, data_dir=None):
    """集成策略实验。data_dir 与 run_optimization 一致，默认 DATA_DIR。"""
    print()
    print("=" * 80)
    print("集成策略实验（多模型投票）")
    print("=" * 80)
    
    quarterly_data = load_quarterly_data(data_dir=data_dir)
    all_quarters = sorted(quarterly_data.keys())
    
    # 使用多个不同的模型（仅使用当前已启用的配置）
    models_to_ensemble = [
        ('hgb_shallow', MODEL_CONFIGS['hgb_shallow']),
        ('hgb_medium', MODEL_CONFIGS['hgb_medium']),
        ('hgb_deep', MODEL_CONFIGS['hgb_deep']),
    ]
    
    features = FEATURE_SETS[list(FEATURE_SETS.keys())[0]]
    all_monthly = []
    
    for test_year, test_quarter in all_quarters:
        if test_year < start_year:
            continue
        
        # 滑动窗口：最近 8 个季度（2 年）
        train_quarters = [(y, q) for (y, q) in all_quarters if (y, q) < (test_year, test_quarter)]
        train_quarters = train_quarters[-8:] if len(train_quarters) >= 8 else train_quarters
        
        if len(train_quarters) < 2:
            continue
        
        train_data = [r for yq in train_quarters for r in quarterly_data[yq].values() for r in r]
        test_data = [r for month_data in quarterly_data[(test_year, test_quarter)].values() for r in month_data]
        
        if not test_data:
            continue
        
        X_train = extract_features(train_data, features)
        y_train = get_relative_returns(train_data)
        X_test = extract_features(test_data, features)
        
        # 训练多个模型并集成预测（模型内部做 StandardScaler）
        all_preds = []
        for name, config in models_to_ensemble:
            model = create_model(config)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            all_preds.append(pred)
        
        # 平均预测
        ensemble_pred = np.mean(all_preds, axis=0)
        
        # 按月评估
        months = defaultdict(list)
        for i, d in enumerate(test_data):
            m = d.get('features', {}).get('buy_date', '')[:7]
            months[m].append(i)
        
        for month, indices in sorted(months.items()):
            indices = np.array(indices)
            month_pred = ensemble_pred[indices]
            month_returns = np.array([test_data[i].get('return_pct', 0) or 0 for i in indices])
            month_idx_returns = np.array([test_data[i].get('index_return_pct', 0) or 0 for i in indices])
            month_relative = month_returns - month_idx_returns
            
            k = min(10, len(indices))
            top_idx = np.argsort(month_pred)[::-1][:k]
            
            n_correct = (month_relative[top_idx] > 3).sum()
            precision = n_correct / k
            strategy_ret = month_returns[top_idx].mean()
            benchmark_ret = month_idx_returns[top_idx].mean()
            
            all_monthly.append({
                'month': month,
                'precision': precision,
                'n_correct': n_correct,
                'strategy': strategy_ret,
                'benchmark': benchmark_ret,
                'alpha': strategy_ret - benchmark_ret,
            })
    
    # 汇总
    if not all_monthly:
        print("集成实验无月度数据")
        return 0.0, 0.0, 0.0
    cum_strategy = 1.0
    cum_benchmark = 1.0
    for r in all_monthly:
        cum_strategy *= (1 + r['strategy'] / 100)
        cum_benchmark *= (1 + r['benchmark'] / 100)
    
    total_alpha = (cum_strategy - 1) * 100 - (cum_benchmark - 1) * 100
    n_months = len(all_monthly)
    years = n_months / 12.0 if n_months else 1.0
    ann_strategy = ((cum_strategy ** (1 / years)) - 1) * 100 if years > 0 else 0.0
    ann_benchmark = ((cum_benchmark ** (1 / years)) - 1) * 100 if years > 0 else 0.0
    ann_alpha = ann_strategy - ann_benchmark
    avg_precision = np.mean([r['precision'] for r in all_monthly])
    win_rate = sum(1 for r in all_monthly if r['alpha'] > 0) / len(all_monthly)
    
    print(f"集成模型: {[m[0] for m in models_to_ensemble]}")
    print(f"年化Alpha: {ann_alpha:+.1f}%  (累计Alpha: {total_alpha:+.1f}%)")
    print(f"Precision: {avg_precision:.1%}")
    print(f"月度胜率: {win_rate:.0%}")
    
    return total_alpha, avg_precision, win_rate


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='策略优化实验')
    parser.add_argument('--start-year', type=int, default=2012, help='回测起始年份（多年评估）')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='季度数据目录，默认按 --index 自动选择')
    parser.add_argument('--index', default='000905', choices=['000300', '000905'],
                        help='指数代码: 000300=沪深300(默认), 000905=中证500')
    parser.add_argument('--ensemble', action='store_true', help='运行集成实验')
    parser.add_argument('--report-dir', type=str, default=None,
                        help='实验报告输出目录，默认 logs/strategy_optimizer')
    args = parser.parse_args()
    
    # 按指数自动选择数据目录（--data-dir 优先）
    data_dir = args.data_dir or DATA_DIRS.get(args.index, DATA_DIRS['000300'])
    
    experiments, best_config, best_result, stable_best = run_optimization(
        start_year=args.start_year,
        data_dir=data_dir
    )
    
    report_dir = args.report_dir or REPORT_DIR
    _write_report(
        report_dir,
        start_year=args.start_year,
        data_dir=data_dir,
        experiments=experiments,
        best_config=best_config or (),
        best_result=best_result or {},
        stable_best=stable_best,
    )
    
    if args.ensemble:
        run_ensemble_experiment(start_year=args.start_year, data_dir=data_dir)
