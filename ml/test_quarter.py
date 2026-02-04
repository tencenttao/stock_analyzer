# -*- coding: utf-8 -*-
"""
一键测试指定季度

使用方法:
    python ml/test_quarter.py 2022 1   # 测试 2022Q1
    python ml/test_quarter.py 2023 2   # 测试 2023Q2
"""

import os
import sys
import json
import glob
from collections import defaultdict
import subprocess


def prepare_data(year, quarter):
    """准备训练和测试数据"""
    DATA_DIR = './data/all_data'
    
    q_months_map = {
        1: ['01', '02', '03'],
        2: ['04', '05', '06'],
        3: ['07', '08', '09'],
        4: ['10', '11', '12'],
    }
    test_months = [f'{year}-{m}' for m in q_months_map[quarter]]
    cutoff = test_months[0]
    
    # 加载数据
    monthly = defaultdict(list)
    for filepath in glob.glob(f'{DATA_DIR}/*.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            for r in json.load(f):
                m = r.get('features', {}).get('buy_date', '')[:7]
                if m:
                    monthly[m].append(r)
    
    # 去重
    for m in monthly:
        seen = set()
        unique = []
        for r in monthly[m]:
            key = f"{r['features'].get('code', '')}_{r['features'].get('buy_date', '')}"
            if key not in seen:
                seen.add(key)
                unique.append(r)
        monthly[m] = unique
    
    # 分割
    train_data = [r for m in sorted(monthly) if m < cutoff for r in monthly[m]]
    test_data = [r for m in test_months if m in monthly for r in monthly[m]]
    
    # 保存
    train_dir = f'./data/train_data_{year}q{quarter}'
    test_dir = f'./data/test_data_{year}q{quarter}'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    with open(f'{train_dir}/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)
    with open(f'{test_dir}/test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False)
    
    print(f'数据准备完成:')
    print(f'  训练数据: {len(train_data)} 条')
    print(f'  测试数据: {len(test_data)} 条')
    print(f'  测试月份: {test_months}')
    
    return train_dir, test_dir


def train_model(year, quarter, train_dir, test_dir):
    """训练模型"""
    output_path = f'models/predictor_{year}q{quarter}.pkl'
    
    cmd = [
        'python', 'ml/train.py',
        '--data', train_dir,
        '--test-data', test_dir,
        '--model', 'hgb_regressor',
        '--output', output_path,
    ]
    
    print(f'\n训练模型: {" ".join(cmd)}')
    subprocess.run(cmd)
    
    return output_path


def main():
    if len(sys.argv) < 3:
        print('用法: python ml/test_quarter.py <年份> <季度>')
        print('示例: python ml/test_quarter.py 2022 1')
        sys.exit(1)
    
    year = int(sys.argv[1])
    quarter = int(sys.argv[2])
    
    print('=' * 60)
    print(f'测试 {year}Q{quarter}')
    print('=' * 60)
    
    # 1. 准备数据
    train_dir, test_dir = prepare_data(year, quarter)
    
    # 2. 训练模型
    model_path = train_model(year, quarter, train_dir, test_dir)
    
    print('\n' + '=' * 60)
    print('完成!')
    print('=' * 60)
    print(f'模型已保存: {model_path}')
    print(f'\n如需运行回测，请:')
    print(f'  1. 修改 config/settings.py 中的日期范围')
    print(f'  2. cp {model_path} models/predictor.pkl')
    print(f'  3. python backtest.py')


if __name__ == '__main__':
    main()
