# -*- coding: utf-8 -*-
"""
训练脚本（默认配置与 quarterly_selector/strategy_optimizer 一致）

默认配置:
    - 特征: full（来自 ml.features.get_full_numeric_feature_names，与 optimizer 一致）
    - 模型: hgb_deep（来自 ml.models.REGRESSOR_PRESETS）
    - 上涨阈值: 3%

使用示例:
    # 使用默认配置（与 selector/optimizer 一致）
    python ml/train.py
    
    # 指定训练/测试数据
    python ml/train.py --data data/train_data --test-data data/test_data
    
    # 保存模型，用于 backtest.py
    python ml/train.py --save-model models/predictor.pkl
    
    # 季度模型：2012～2025 每季用前 8 季数据训练，保存为 predictor_2025q4 等
    python ml/train.py --train-quarterly
    python ml/train.py --train-quarterly --quarterly-data data/quarterly_data --output-dir models
    python ml/train.py --train-quarterly --start-year 2012 --end-year 2025 --window-quarters 8
    
    # 切换特征组: base, momentum, full(默认)
    python ml/train.py --feature-set base
    # 切换模型: hgb_shallow, hgb_medium, hgb_deep(默认)
    python ml/train.py --model hgb_shallow
"""

import os
import sys
import json
import logging
import argparse
import glob
import numpy as np

# 直接运行本脚本时把项目根加入 path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 特征组（与 strategy_optimizer/quarterly_selector 一致：full 来自 ml.features）
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
MOMENTUM_FEATURES = [
    'momentum_20d', 'momentum_60d', 'rsi_14', 'ma_deviation_20',
    'relative_momentum_20d', 'relative_momentum_60d',
]
BASE_FEATURES = [
    'pe_ratio', 'pb_ratio', 'peg', 'roe', 'profit_growth', 'dividend_yield',
    'value_score', 'quality_score', 'momentum_20d', 'momentum_60d',
]

FEATURE_SETS = {
    'base': BASE_FEATURES,
    'momentum': MOMENTUM_FEATURES,
    'full': ALL_FEATURES,
}


# 季度训练模式：数据目录
DATA_DIR_QUARTERLY = './data/quarterly_data'


def load_quarterly_data(data_dir: str = None):
    """加载 quarterly_data 目录下所有季度 JSON，返回 {(year, quarter): [records]}。
    文件名格式: 2020_Q1_ml_training_data.json
    """
    data_dir = data_dir or DATA_DIR_QUARTERLY
    all_files = sorted(glob.glob(os.path.join(data_dir, '*_ml_training_data.json')))
    if not all_files:
        raise FileNotFoundError(f"未找到季度数据文件: {data_dir}")
    quarterly = {}
    for filepath in all_files:
        name = os.path.basename(filepath)
        parts = name.split('_')
        year, quarter = int(parts[0]), int(parts[1][1])  # Q1 -> 1
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        quarterly[(year, quarter)] = data if isinstance(data, list) else []
    return quarterly


def load_training_data(path: str):
    """加载训练数据
    
    支持单个文件或文件夹路径：
    - 如果是文件，直接加载
    - 如果是文件夹，自动读取文件夹下所有 .json 文件并合并
    
    Args:
        path: 文件路径或文件夹路径
        
    Returns:
        训练数据列表
    """
    if os.path.isfile(path):
        # 单个文件
        logger.info(f"加载单个文件: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    elif os.path.isdir(path):
        # 文件夹 - 读取所有 JSON 文件
        json_files = sorted(glob.glob(os.path.join(path, '*.json')))
        
        if not json_files:
            raise ValueError(f"文件夹中没有找到 JSON 文件: {path}")
        
        logger.info(f"发现 {len(json_files)} 个 JSON 文件:")
        all_data = []
        
        for json_file in json_files:
            logger.info(f"  加载: {os.path.basename(json_file)}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                    logger.info(f"    -> {len(data)} 条记录")
                else:
                    logger.warning(f"    -> 跳过（非列表格式）")
        
        logger.info(f"合并完成，总计 {len(all_data)} 条记录")
        return all_data
    
    else:
        raise FileNotFoundError(f"路径不存在: {path}")


def extract_features(data, feature_names):
    """提取特征矩阵（与 selector/optimizer 一致）"""
    X = []
    for d in data:
        feat = d.get('features', {})
        row = [float(feat.get(f, 0) or 0) for f in feature_names]
        X.append(row)
    return np.nan_to_num(np.array(X), nan=0.0, posinf=0.0, neginf=0.0)


def get_relative_returns(data):
    """获取相对收益（与 selector/optimizer 一致）"""
    return np.array([(d.get('return_pct', 0) or 0) - (d.get('index_return_pct', 0) or 0) for d in data])


def get_labels(data, threshold=3.0):
    """获取二分类标签（相对收益 > threshold 为上涨）"""
    return (get_relative_returns(data) > threshold).astype(int)


def get_available_features(data, include_categorical=False):
    """从数据中获取可用特征
    
    Args:
        data: 训练数据
        include_categorical: 是否包含分类特征（industry, market）
        
    Returns:
        (numeric_features, categorical_features) if include_categorical else numeric_features
    """
    if not data:
        return ([], []) if include_categorical else []
    
    sample = data[0]['features']
    meta_keys = {'code', 'name', 'buy_date', 'sell_date', 'buy_price', 'sell_price'}
    categorical_keys = {'industry', 'market'}
    
    numeric = [k for k, v in sample.items() 
               if isinstance(v, (int, float)) and k not in meta_keys]
    
    if include_categorical:
        categorical = [k for k in categorical_keys if k in sample]
        return numeric, categorical
    return numeric


def validate_features(requested, available):
    """验证请求的特征是否都在数据中"""
    missing = set(requested) - set(available)
    if missing:
        raise ValueError(f"训练数据中缺少特征: {missing}\n可用特征: {available}")


def to_binary_labels(y, up_class=2):
    """将三分类标签转为二分类：上涨=1, 非上涨=0"""
    return (y == up_class).astype(int)


def evaluate_precision_at_k(y_true, y_proba, k_list=[20, 50, 100]):
    """评估 precision@K：按上涨概率排序，取前K个的精确率（全局排序）
    
    Args:
        y_true: 真实标签（1=上涨, 0=非上涨）
        y_proba: 上涨概率
        k_list: 要评估的K值列表
    """
    n = len(y_true)
    sorted_idx = np.argsort(y_proba)[::-1]  # 按概率降序
    
    results = {}
    for k in k_list:
        if k > n:
            k = n
        top_k_idx = sorted_idx[:k]
        precision = y_true[top_k_idx].sum() / k
        results[k] = precision
    return results


def evaluate_precision_by_month(test_data, y_true_binary, prob_up, top_k=10):
    """按月份分别计算 Precision@K（与回测逻辑一致）
    
    Args:
        test_data: 测试数据列表（需要包含 buy_date 字段）
        y_true_binary: 二分类真实标签
        prob_up: 上涨概率
        top_k: 每月选择数量
        
    Returns:
        (monthly_results, total_precision): 月度结果和总精确率
    """
    # 按月份分组
    months = {}
    for i, d in enumerate(test_data):
        buy_date = d.get('features', {}).get('buy_date', '')[:7]
        if not buy_date:
            buy_date = 'unknown'
        if buy_date not in months:
            months[buy_date] = []
        months[buy_date].append(i)
    
    monthly_results = {}
    total_correct = 0
    total_selected = 0
    
    for month, indices in sorted(months.items()):
        indices = np.array(indices)
        month_probs = prob_up[indices]
        month_labels = y_true_binary[indices]
        
        # 取该月 Top K
        k = min(top_k, len(indices))
        top_k_local = np.argsort(month_probs)[::-1][:k]
        n_correct = month_labels[top_k_local].sum()
        
        monthly_results[month] = (n_correct, k)
        total_correct += n_correct
        total_selected += k
    
    total_precision = total_correct / total_selected if total_selected > 0 else 0
    return monthly_results, total_precision


def train_quarterly_models(
    data_dir: str,
    output_dir: str,
    start_year: int = 2012,
    end_year: int = 2025,
    window_quarters: int = 8,
    feature_set: str = 'full',
    model_name: str = 'hgb_shallow',
):
    """
    按季度滑动窗口训练并保存模型：对 2012～2025 每个季度，用前 8 个季度数据训练，保存为 predictor_YYYYqN.pkl。
    数据来自 data_dir（quarterly_data），文件名格式 2020_Q1_ml_training_data.json。
    """
    from ml.models import create_model, REGRESSOR_PRESETS
    import pickle

    MODEL_CONFIGS = {k: v for k, v in REGRESSOR_PRESETS.items() if v.get('model') == 'hgb'}
    model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['hgb_shallow'])
    model = create_model(model_config)

    logger.info("加载季度数据: %s", data_dir)
    quarterly_data = load_quarterly_data(data_dir)
    all_quarters = sorted(quarterly_data.keys())
    if not all_quarters:
        raise ValueError("未找到任何季度数据")
    logger.info("季度范围: %s ~ %s", all_quarters[0], all_quarters[-1])

    # 特征名：从第一个季度样本推断
    first_records = list(quarterly_data.values())[0]
    numeric_features, _ = get_available_features(first_records, include_categorical=True)
    target_features = FEATURE_SETS.get(feature_set, FEATURE_SETS['full'])
    feature_names = [f for f in target_features if f in numeric_features]
    missing = set(target_features) - set(feature_names)
    if missing:
        logger.warning("特征组 %s 中缺失: %s", feature_set, missing)
    logger.info("特征数: %d (feature_set=%s)", len(feature_names), feature_set)

    os.makedirs(output_dir, exist_ok=True)
    saved = []

    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            yq = (year, quarter)
            if yq not in quarterly_data:
                logger.warning("跳过缺失季度: %dQ%d", year, quarter)
                continue
            # 训练集：该季度之前的最近 window_quarters 个季度
            train_quarters = [q for q in all_quarters if q < yq]
            if len(train_quarters) < window_quarters:
                logger.warning("跳过 %dQ%d：历史不足 %d 季（仅 %d）", year, quarter, window_quarters, len(train_quarters))
                continue
            train_quarters = train_quarters[-window_quarters:]
            train_data = []
            for tq in train_quarters:
                train_data.extend(quarterly_data[tq])

            X_train = extract_features(train_data, feature_names)
            y_train = get_relative_returns(train_data)
            if len(X_train) < 100:
                logger.warning("跳过 %dQ%d：训练样本过少 %d", year, quarter, len(X_train))
                continue

            model = create_model(model_config)
            model.fit(X_train, y_train)

            out_name = f"predictor_{year}q{quarter}.pkl"
            out_path = os.path.join(output_dir, out_name)
            kind = model_config.get('model', 'hgb')
            model_type = 'hgb_regressor' if kind == 'hgb' else 'rf_regressor'
            model_data = {
                'model': model.model,
                'scaler': model.scaler,
                'is_fitted': True,
                'feature_names': feature_names,
                'label_type': 'relative',
                'threshold_up': 3.0,
                'threshold_down': -3.0,
                'model_type': model_type,
            }
            with open(out_path, 'wb') as f:
                pickle.dump(model_data, f)
            saved.append(out_name)
            logger.info("已保存: %s (训练样本 %d)", out_path, len(X_train))

    logger.info("共保存 %d 个季度模型: %s", len(saved), output_dir)
    return saved


def main():
    from ml.models import create_model, REGRESSOR_PRESETS

    # 仅 HGB 预设供 train 使用（与 quarterly_selector 一致）
    MODEL_CONFIGS = {k: v for k, v in REGRESSOR_PRESETS.items() if v.get('model') == 'hgb'}

    parser = argparse.ArgumentParser(description='训练股票预测模型（与 quarterly_selector/strategy_optimizer 逻辑一致）')
    # 数据路径
    parser.add_argument('--data', default='./data/train_data',
                        help='训练数据路径（支持文件或文件夹）')
    parser.add_argument('--test-data', default='./data/test_data',
                        help='测试数据路径（支持文件或文件夹）；不指定则从 --data 按比例划分')
    # 季度训练模式：基于 quarterly_data 按季度滑动窗口训练并保存 predictor_YYYYqN.pkl
    parser.add_argument('--train-quarterly', action='store_true',
                        help='按季度训练：用前8季数据训练 2012～2025 每季模型，保存到 --output-dir')
    parser.add_argument('--quarterly-data', default='./data/quarterly_data',
                        help='季度数据目录（仅 --train-quarterly 时生效）')
    parser.add_argument('--output-dir', default='./models',
                        help='季度模型保存目录（仅 --train-quarterly 时生效）')
    parser.add_argument('--start-year', type=int, default=2012,
                        help='季度训练起始年（仅 --train-quarterly）')
    parser.add_argument('--end-year', type=int, default=2025,
                        help='季度训练结束年（仅 --train-quarterly）')
    parser.add_argument('--window-quarters', type=int, default=8,
                        help='每季训练使用前几季数据（仅 --train-quarterly，默认8）')
    # 特征配置（与 strategy_optimizer 一致，full 来自 ml.features）
    parser.add_argument('--feature-set', choices=['base', 'momentum', 'full'], default='full',
                        help='特征组: base, momentum, full(默认)')
    # 模型配置（与 strategy_optimizer 一致，来自 REGRESSOR_PRESETS）
    parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()), default='hgb_shallow',
                        help='模型预设: hgb_shallow, hgb_medium, hgb_deep(默认)')
    # 评估配置（仅在使用 --test-data 时生效）
    parser.add_argument('--threshold-up', type=float, default=3.0, help='上涨阈值%%(默认3.0)')
    # 保存模型
    parser.add_argument('--save-model', type=str, default='',
                        help='保存路径(如 models/predictor.pkl)；空则不保存；--train-quarterly 时忽略')
    args = parser.parse_args()

    # 季度训练模式：训练 2012～2025 每季模型并保存为 predictor_YYYYqN.pkl
    if args.train_quarterly:
        print("=" * 60)
        print("季度模型训练（前 N 季滑动窗口）")
        print("=" * 60)
        train_quarterly_models(
            data_dir=args.quarterly_data,
            output_dir=args.output_dir,
            start_year=args.start_year,
            end_year=args.end_year,
            window_quarters=args.window_quarters,
            feature_set=args.feature_set,
            model_name=args.model,
        )
        print("=" * 60)
        return

    print("=" * 60)
    print("股票预测模型训练")
    print("=" * 60)
    
    # 加载数据
    logger.info(f"加载数据: {args.data}")
    all_data = load_training_data(args.data)
    logger.info(f"总数据量: {len(all_data)} 条")
    
    # 获取可用特征
    numeric_features, _ = get_available_features(all_data, include_categorical=True)
    
    # 使用预定义特征组
    target_features = FEATURE_SETS[args.feature_set]
    feature_names = [f for f in target_features if f in numeric_features]
    missing = set(target_features) - set(feature_names)
    if missing:
        logger.warning(f"特征组 {args.feature_set} 中缺失: {missing}")
    
    logger.info(f"使用特征 ({len(feature_names)}): {feature_names}")
    
    # 训练/测试集：仅当指定 --test-data 时加载测试集并做评估
    train_data = all_data
    test_data = []
    if args.test_data:
        logger.info(f"加载测试数据: {args.test_data}")
        test_data = load_training_data(args.test_data)
        logger.info(f"训练集: {len(train_data)}, 测试集: {len(test_data)}")
    else:
        logger.info(f"训练集: {len(train_data)}（未指定 --test-data，仅训练不评估）")

    # 模型（与 strategy_optimizer/quarterly_selector 一致：create_model 内部含 StandardScaler）
    model_config = MODEL_CONFIGS[args.model]
    model = create_model(model_config)
    params = model_config.get('params', model_config)
    logger.info(f"模型: {args.model} (depth={params.get('max_depth', '')}, iter={params.get('max_iter', '')}, lr={params.get('learning_rate', '')})")

    # 提取特征和标签（原始 X，模型内部做标准化）
    X_train = extract_features(train_data, feature_names)
    y_train = get_relative_returns(train_data)

    # 训练
    logger.info("训练中...")
    model.fit(X_train, y_train)
    logger.info(f"模型训练完成, 样本数: {len(X_train)}")

    # 仅当提供了测试集时做评估
    if test_data:
        X_test = extract_features(test_data, feature_names)
        y_test_reg = get_relative_returns(test_data)
        pred_returns = model.predict(X_test)

        print("\n" + "=" * 60)
        print("测试集评估结果")
        print("=" * 60)
        print(f"预测相对收益: 均值={pred_returns.mean():.2f}%, 标准差={pred_returns.std():.2f}%")
        print(f"实际相对收益: 均值={y_test_reg.mean():.2f}%, 标准差={y_test_reg.std():.2f}%")
        corr = np.corrcoef(pred_returns, y_test_reg)[0, 1]
        print(f"预测与实际相关系数: {corr:.3f}")

        prob_up = pred_returns
        y_test_binary = get_labels(test_data, threshold=args.threshold_up)

        print("\n" + "=" * 60)
        print("Precision@K 评估（全局排序）")
        print("=" * 60)
        prec_at_k = evaluate_precision_at_k(y_test_binary, prob_up, k_list=[10, 20, 50, 100])
        for k, prec in prec_at_k.items():
            print(f"  Precision@{k}: {prec:.2%} (全局前{k}只)")

        print("\n" + "=" * 60)
        print("Precision@10 按月评估（与回测逻辑一致）")
        print("=" * 60)
        monthly_results, monthly_precision = evaluate_precision_by_month(
            test_data, y_test_binary, prob_up, top_k=10
        )
        for month, (n_correct, k) in monthly_results.items():
            print(f"  {month}: {n_correct}/{k} = {n_correct/k:.0%}")
        print(f"\n  ★ 按月总精确率: {monthly_precision:.2%} (每月选10只，共{sum(k for _, k in monthly_results.values())}只)")
        print(f"  随机基线: {y_test_binary.mean():.2%} (测试集上涨比例)")

    # 保存模型（与 predictor 兼容：保存包装器内部 model + scaler 的旧格式）
    if args.save_model:
        import pickle
        output_path = args.save_model
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        kind = model_config.get('model', 'hgb')
        model_type = 'hgb_regressor' if kind == 'hgb' else 'rf_regressor'
        model_data = {
            'model': model.model,
            'scaler': model.scaler,
            'is_fitted': True,
            'feature_names': feature_names,
            'label_type': 'relative',
            'threshold_up': args.threshold_up,
            'threshold_down': -args.threshold_up,
            'model_type': model_type,
        }
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n模型已保存: {output_path}")
        print(f"  可用于 backtest.py (ml 策略)")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
