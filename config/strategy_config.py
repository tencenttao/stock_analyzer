# -*- coding: utf-8 -*-

"""
策略配置 - 管理所有策略的参数配置

配置格式:
    {
        'name': 策略显示名称,
        'description': 策略描述,
        'weights': 各维度权重（评分类策略）,
        'filters': 硬性筛选条件,
        'params': 策略特定参数,
    }
"""

# 默认策略（推荐使用自适应策略）
DEFAULT_STRATEGY = 'ml'

# 策略配置字典
STRATEGY_CONFIGS = {
    # ============================================
    # 自适应策略（推荐！根据市场状态动态调整）
    # ============================================
    'adaptive': {
        'name': '自适应策略',
        'description': '根据市场状态（牛/熊/震荡）自动调整权重，牛市追涨、熊市防守',
        'weights_bull': {    # 牛市权重
            'momentum': 45,
            'growth': 25,
            'valuation': 15,
            'quality': 10,
            'safety': 5
        },
        'weights_bear': {    # 熊市权重
            'momentum': 15,
            'growth': 20,
            'valuation': 25,
            'quality': 20,
            'safety': 20
        },
        'weights_sideways': {  # 震荡市权重
            'momentum': 30,
            'growth': 25,
            'valuation': 20,
            'quality': 15,
            'safety': 10
        },
        'filters': {
            'max_pe': 50,
            'min_price': 1.0,
            'min_turnover': 30000000,
        },
        'params': {
            'top_n': 10,
            'min_score': 35,
            'momentum_days': 20,
        }
    },
    
    # ============================================
    # 动量优先策略 V2（固定权重版本）
    # ============================================
    'momentum_v2': {
        'name': '动量优先V2',
        'description': '基于动量/趋势的选股策略，固定权重，适合趋势市',
        'weights': {
            'momentum': 40,   # 动量/趋势（40%）
            'growth': 25,     # 成长性（25%）
            'valuation': 20,  # 估值（20%）
            'quality': 10,    # 质量（10%）
            'safety': 5       # 安全性（5%）
        },
        'filters': {
            'max_pe': 50,              # PE上限
            'min_price': 1.0,          # 最低股价
            'min_turnover': 30000000,  # 最小成交额（3000万）
        },
        'params': {
            'top_n': 10,               # 默认选股数量
            'min_score': 40,           # 最低评分阈值
            'momentum_days': 20,       # 动量计算天数
        }
    },
    
    # ============================================
    # 价值优先策略
    # ============================================
    'value_first': {
        'name': '价值优先',
        'description': '基于估值和安全边际的选股策略，适合震荡市',
        'weights': {
            'valuation': 40,  # 估值（40%）
            'safety': 25,     # 安全性（25%）
            'quality': 20,    # 质量（20%）
            'growth': 10,     # 成长性（10%）
            'momentum': 5     # 动量（5%）
        },
        'filters': {
            'max_pe': 20,              # PE上限（更严格）
            'max_pb': 3,               # PB上限
            'min_dividend_yield': 2.0, # 最低股息率
            'min_turnover': 30000000,
        },
        'params': {
            'top_n': 10,
            'min_score': 45,
        }
    },
    
    # ============================================
    # 平衡策略
    # ============================================
    'balanced': {
        'name': '平衡策略',
        'description': '各维度均衡考虑，追求稳健收益',
        'weights': {
            'momentum': 20,
            'growth': 20,
            'valuation': 20,
            'quality': 20,
            'safety': 20
        },
        'filters': {
            'max_pe': 35,
            'min_turnover': 30000000,
        },
        'params': {
            'top_n': 10,
            'min_score': 40,
        }
    },
    
    # ============================================
    # 稳健型策略（基于相对收益分析优化）
    # ============================================
    'stable': {
        'name': '稳健策略',
        'description': '基于相对收益分析，避免追高追涨，追求稳定跑赢大盘',
        'weights': {
            'momentum': 15,    # 降低动量权重
            'growth': 20,      # 适度重视成长
            'valuation': 25,   # 提高估值权重
            'quality': 20,     # 重视质量
            'safety': 20       # 重视安全性
        },
        'filters': {
            # 核心过滤：基于分析结果
            'max_momentum_20d': 15,     # 避免追高（跑赢组动量均值仅1.1%）
            'max_change_pct': 5,        # 避免追涨（跑赢组均值0.35%）
            'max_turnover_rate': 5,     # 筹码稳定（跑赢组均值1.2%）
            # 基本过滤
            'max_pe': 40,               # PE合理
            'min_price': 2.0,           # 排除仙股
        },
        'params': {
            'top_n': 10,
            'min_score': 30,   # 降低分数门槛，因为高分股往往是高动量
        }
    },
    
    # ============================================
    # 防守型策略（更保守）
    # ============================================
    'defensive': {
        'name': '防守策略',
        'description': '极度保守，重视股息和低波动，适合熊市',
        'weights': {
            'momentum': 5,     # 几乎不考虑动量
            'growth': 15,
            'valuation': 30,   # 重视估值
            'quality': 25,     # 重视质量
            'safety': 25       # 重视安全
        },
        'filters': {
            'max_momentum_20d': 10,     # 严格控制动量
            'max_change_pct': 3,        # 严格避免追涨
            'max_turnover_rate': 3,     # 严格筹码稳定
            'max_pe': 25,               # 严格估值
            'min_dividend_yield': 1.0,  # 要求有股息
            'min_price': 3.0,
        },
        'params': {
            'top_n': 10,
            'min_score': 25,
        }
    },
    
    # ============================================
    # 机器学习策略
    # ============================================
    'ml': {
        'name': '机器学习选股',
        'description': '基于ML模型预测上涨概率，选择概率最高的股票',
        'params': {
            # 单模型：固定使用该路径
            #'model_path': 'models/predictor.pkl',
            # 可选：按月份切换模型。key 为 "YYYY-MM" 或 "default"，value 为模型路径。
            # 回测/选股时按当前月份匹配，未匹配则用 model_path。
            # 示例：25年1月用模型1，2月用模型2，其余用默认
            'model_schedule': {
                # 2018
                '2018-01': 'models/predictor_2018q1.pkl', '2018-02': 'models/predictor_2018q1.pkl', '2018-03': 'models/predictor_2018q1.pkl',
                '2018-04': 'models/predictor_2018q2.pkl', '2018-05': 'models/predictor_2018q2.pkl', '2018-06': 'models/predictor_2018q2.pkl',
                '2018-07': 'models/predictor_2018q3.pkl', '2018-08': 'models/predictor_2018q3.pkl', '2018-09': 'models/predictor_2018q3.pkl',
                '2018-10': 'models/predictor_2018q4.pkl', '2018-11': 'models/predictor_2018q4.pkl', '2018-12': 'models/predictor_2018q4.pkl',
                # 2019
                '2019-01': 'models/predictor_2019q1.pkl', '2019-02': 'models/predictor_2019q1.pkl', '2019-03': 'models/predictor_2019q1.pkl',
                '2019-04': 'models/predictor_2019q2.pkl', '2019-05': 'models/predictor_2019q2.pkl', '2019-06': 'models/predictor_2019q2.pkl',
                '2019-07': 'models/predictor_2019q3.pkl', '2019-08': 'models/predictor_2019q3.pkl', '2019-09': 'models/predictor_2019q3.pkl',
                '2019-10': 'models/predictor_2019q4.pkl', '2019-11': 'models/predictor_2019q4.pkl', '2019-12': 'models/predictor_2019q4.pkl',
                # 2020
                '2020-01': 'models/predictor_2020q1.pkl', '2020-02': 'models/predictor_2020q1.pkl', '2020-03': 'models/predictor_2020q1.pkl',
                '2020-04': 'models/predictor_2020q2.pkl', '2020-05': 'models/predictor_2020q2.pkl', '2020-06': 'models/predictor_2020q2.pkl',
                '2020-07': 'models/predictor_2020q3.pkl', '2020-08': 'models/predictor_2020q3.pkl', '2020-09': 'models/predictor_2020q3.pkl',
                '2020-10': 'models/predictor_2020q4.pkl', '2020-11': 'models/predictor_2020q4.pkl', '2020-12': 'models/predictor_2020q4.pkl',
                # 2021
                '2021-01': 'models/predictor_2021q1.pkl', '2021-02': 'models/predictor_2021q1.pkl', '2021-03': 'models/predictor_2021q1.pkl',
                '2021-04': 'models/predictor_2021q2.pkl', '2021-05': 'models/predictor_2021q2.pkl', '2021-06': 'models/predictor_2021q2.pkl',
                '2021-07': 'models/predictor_2021q3.pkl', '2021-08': 'models/predictor_2021q3.pkl', '2021-09': 'models/predictor_2021q3.pkl',
                '2021-10': 'models/predictor_2021q4.pkl', '2021-11': 'models/predictor_2021q4.pkl', '2021-12': 'models/predictor_2021q4.pkl',
                # 2022
                '2022-01': 'models/predictor_2022q1.pkl', '2022-02': 'models/predictor_2022q1.pkl', '2022-03': 'models/predictor_2022q1.pkl',
                '2022-04': 'models/predictor_2022q2.pkl', '2022-05': 'models/predictor_2022q2.pkl', '2022-06': 'models/predictor_2022q2.pkl',
                '2022-07': 'models/predictor_2022q3.pkl', '2022-08': 'models/predictor_2022q3.pkl', '2022-09': 'models/predictor_2022q3.pkl',
                '2022-10': 'models/predictor_2022q4.pkl', '2022-11': 'models/predictor_2022q4.pkl', '2022-12': 'models/predictor_2022q4.pkl',
                # 2023
                '2023-01': 'models/predictor_2023q1.pkl', '2023-02': 'models/predictor_2023q1.pkl', '2023-03': 'models/predictor_2023q1.pkl',
                '2023-04': 'models/predictor_2023q2.pkl', '2023-05': 'models/predictor_2023q2.pkl', '2023-06': 'models/predictor_2023q2.pkl',
                '2023-07': 'models/predictor_2023q3.pkl', '2023-08': 'models/predictor_2023q3.pkl', '2023-09': 'models/predictor_2023q3.pkl',
                '2023-10': 'models/predictor_2023q4.pkl', '2023-11': 'models/predictor_2023q4.pkl', '2023-12': 'models/predictor_2023q4.pkl',
                # 2024
                '2024-01': 'models/predictor_2024q1.pkl', '2024-02': 'models/predictor_2024q1.pkl', '2024-03': 'models/predictor_2024q1.pkl',
                '2024-04': 'models/predictor_2024q2.pkl', '2024-05': 'models/predictor_2024q2.pkl', '2024-06': 'models/predictor_2024q2.pkl',
                '2024-07': 'models/predictor_2024q3.pkl', '2024-08': 'models/predictor_2024q3.pkl', '2024-09': 'models/predictor_2024q3.pkl',
                '2024-10': 'models/predictor_2024q4.pkl', '2024-11': 'models/predictor_2024q4.pkl', '2024-12': 'models/predictor_2024q4.pkl',
                # 2025
                '2025-01': 'models/predictor_2025q1.pkl', '2025-02': 'models/predictor_2025q1.pkl', '2025-03': 'models/predictor_2025q1.pkl',
                '2025-04': 'models/predictor_2025q2.pkl', '2025-05': 'models/predictor_2025q2.pkl', '2025-06': 'models/predictor_2025q2.pkl',
                '2025-07': 'models/predictor_2025q3.pkl', '2025-08': 'models/predictor_2025q3.pkl', '2025-09': 'models/predictor_2025q3.pkl',
                '2025-10': 'models/predictor_2025q4.pkl', '2025-11': 'models/predictor_2025q4.pkl', '2025-12': 'models/predictor_2025q4.pkl',
            },
            #'model_schedule': None,                # 不配置则全程使用 model_path
            'min_prob_up': 0.0,                    # 最低上涨概率阈值（0=不过滤）
            'min_pred_threshold': 8,            # 回归模型：最小预测相对收益(%)，低于此值不选入（如 2 表示只选预测跑赢基准2%+），None=不过滤
            'min_price': 2.0,                      # 最低股价
            'top_n': 10,
        }
    },
    
    # ============================================
    # 随机策略（基线对照）
    # ============================================
    'random': {
        'name': '随机选股',
        'description': '随机选择N只股票，用于对比策略有效性',
        'params': {
            'top_n': 10,
            'seed': 42,  # 固定随机种子，确保可重复
        }
    },
    
    # ============================================
    # 等权策略（全部持有）
    # ============================================
    'equal_weight': {
        'name': '等权持有',
        'description': '等权持有所有股票，代表市场平均水平',
        'params': {
            'max_stocks': 300,  # 最多持有数量
        }
    },
}

# 策略分类
STRATEGY_CATEGORIES = {
    'scoring': ['momentum_v2', 'value_first', 'balanced', 'stable', 'defensive'],  # 评分类策略
    'adaptive': ['adaptive'],                                 # 自适应策略
    'baseline': ['random', 'equal_weight'],                   # 基线策略
    'ml': ['ml'],                                             # 机器学习策略
}

# 策略推荐（按市场环境）
STRATEGY_RECOMMENDATIONS = {
    'bull_market': 'momentum_v2',   # 牛市推荐动量策略
    'bear_market': 'defensive',     # 熊市推荐防守策略
    'sideways': 'stable',           # 震荡市推荐稳健策略
    'unknown': 'stable',            # 不确定时推荐稳健策略
}
