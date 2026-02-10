# -*- coding: utf-8 -*-
"""
批量生成季度训练数据

输出目录: data/quarterly_data

用法:
  # 生成 2010-2014 年所有季度数据（4个并行进程，默认沪深300）
  python ml/batch_build_data.py --years 2010 2011 2012 2013 2014 --parallel 4

  # 生成中证500训练数据
  python ml/batch_build_data.py --years 2015 2016 2017 --index 000905 --output-dir data/quarterly_data_csi500

  # 只生成特定季度
  python ml/batch_build_data.py --quarters 2015_Q1 2015_Q2

  # 检查数据完整性
  python ml/batch_build_data.py --check
"""

import os
import sys
import json
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, 'data', 'quarterly_data')

# 季度日期映射
# 注意: end_date 需要延后几天，确保能获取到下个月的首个交易日
# 因为某些月首日（如10-01国庆、01-01元旦、周末）不是交易日
QUARTER_DATES = {
    'Q1': ('01-01', '04-10'),   # 4月1日可能是周末
    'Q2': ('04-01', '07-10'),   # 7月1日可能是周末
    'Q3': ('07-01', '10-10'),   # 10月1日是国庆节
    'Q4': ('10-01', '01-10'),   # 1月1日是元旦
}


def get_quarter_date_range(year: int, quarter: str) -> tuple:
    """获取季度的开始和结束日期"""
    start_mmdd, end_mmdd = QUARTER_DATES[quarter]
    start_date = f"{year}-{start_mmdd}"
    if quarter == 'Q4':
        end_date = f"{year + 1}-{end_mmdd}"
    else:
        end_date = f"{year}-{end_mmdd}"
    return start_date, end_date


def build_quarter_data(year: int, quarter: str, use_cache: bool = True, output_dir: str = None, index_code: str = '000300', parallel: bool = False) -> dict:
    """生成单个季度的数据"""
    target_dir = output_dir or OUTPUT_DIR
    quarter_name = f"{year}_{quarter}"
    output_file = os.path.join(target_dir, f"{quarter_name}_ml_training_data.json")
    
    # 如果文件已存在，跳过
    if os.path.exists(output_file):
        print(f"[{quarter_name}] 文件已存在，跳过")
        return {'quarter': quarter_name, 'status': 'skipped', 'file': output_file}
    
    start_date, end_date = get_quarter_date_range(year, quarter)
    
    print(f"[{quarter_name}] 开始生成: {start_date} ~ {end_date} (use_cache={use_cache})")
    
    cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, 'ml', 'build_training_data.py'),
        '--start', start_date,
        '--end', end_date,
        '--output', output_file,
        '--config', 'full',
        '--index', index_code,
    ]
    
    if use_cache:
        cmd.append('--use-cache')
    
    try:
        # parallel=False 时实时输出日志；parallel=True 时由外层汇总进度
        result = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            capture_output=parallel,     # 并行时捕获避免交错，串行时实时输出
            text=parallel,
            timeout=1800,  # 30分钟超时
        )
        
        if result.returncode == 0 and os.path.exists(output_file):
            # 检查文件内容
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[{quarter_name}] 完成: {len(data)} 条记录")
            return {
                'quarter': quarter_name,
                'status': 'success',
                'records': len(data),
                'file': output_file
            }
        else:
            err_msg = ''
            if parallel and hasattr(result, 'stderr') and result.stderr:
                err_msg = result.stderr[:300]
            print(f"[{quarter_name}] 失败 (exit code={result.returncode}) {err_msg}")
            return {
                'quarter': quarter_name,
                'status': 'failed',
                'error': err_msg or f'exit code={result.returncode}'
            }
    except subprocess.TimeoutExpired:
        print(f"[{quarter_name}] 超时")
        return {'quarter': quarter_name, 'status': 'timeout'}
    except Exception as e:
        print(f"[{quarter_name}] 异常: {e}")
        return {'quarter': quarter_name, 'status': 'error', 'error': str(e)}


def build_quarter_wrapper(args):
    """包装函数，用于并行执行"""
    year, quarter, use_cache = args[:3]
    output_dir = args[3] if len(args) > 3 else None
    index_code = args[4] if len(args) > 4 else '000300'
    parallel = args[5] if len(args) > 5 else False
    return build_quarter_data(year, quarter, use_cache, output_dir, index_code, parallel)


def check_data_integrity():
    """检查数据完整性，与其他季度对比"""
    print("\n" + "=" * 60)
    print("数据完整性检查")
    print("=" * 60)
    
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')])
    
    results = []
    for filename in files:
        filepath = os.path.join(OUTPUT_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                results.append({
                    'file': filename,
                    'status': 'empty',
                    'records': 0
                })
                continue
            
            # 统计每月的记录数
            monthly_counts = {}
            for record in data:
                buy_date = record['features'].get('buy_date', '')
                if buy_date:
                    month = buy_date[:7]  # YYYY-MM
                    monthly_counts[month] = monthly_counts.get(month, 0) + 1
            
            # 检查特征完整性
            sample_features = set(data[0]['features'].keys())
            
            results.append({
                'file': filename,
                'status': 'ok',
                'records': len(data),
                'months': len(monthly_counts),
                'monthly_avg': sum(monthly_counts.values()) / len(monthly_counts) if monthly_counts else 0,
                'monthly_detail': monthly_counts,
                'feature_count': len(sample_features)
            })
        except Exception as e:
            results.append({
                'file': filename,
                'status': 'error',
                'error': str(e)
            })
    
    # 打印结果
    print(f"\n{'文件名':<40} {'状态':<8} {'记录数':<10} {'月份':<6} {'月均':<8} {'特征数'}")
    print("-" * 90)
    
    for r in results:
        if r['status'] == 'ok':
            print(f"{r['file']:<40} {r['status']:<8} {r['records']:<10} {r['months']:<6} {r['monthly_avg']:<8.0f} {r['feature_count']}")
        elif r['status'] == 'empty':
            print(f"{r['file']:<40} {'空文件':<8}")
        else:
            print(f"{r['file']:<40} {'错误':<8} {r.get('error', '')[:30]}")
    
    # 检查2015-2017是否缺失
    expected_quarters = []
    for year in range(2015, 2018):
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            expected_quarters.append(f"{year}_{q}_ml_training_data.json")
    
    existing = set(files)
    missing = [q for q in expected_quarters if q not in existing]
    
    if missing:
        print(f"\n缺失的 2015-2017 季度文件:")
        for m in missing:
            print(f"  - {m}")
    else:
        print(f"\n2015-2017 所有季度文件已存在")
    
    # 统计2015-2017的数据质量
    print("\n2015-2017 数据详情:")
    for r in results:
        if r['file'].startswith(('2015', '2016', '2017')) and r['status'] == 'ok':
            print(f"\n{r['file']}:")
            for month, count in sorted(r['monthly_detail'].items()):
                print(f"  {month}: {count} 条")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='批量生成季度训练数据')
    parser.add_argument('--years', nargs='+', type=int, default=[2010, 2011, 2012, 2013, 2014, 2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025],
                        help='要生成的年份列表，默认 2015 2016 2025')
    parser.add_argument('--quarters', nargs='+', type=str, default=None,
                        help='指定季度，如 2015_Q1 2015_Q2')
    parser.add_argument('--parallel', type=int, default=1,
                        help='并行进程数，默认 1（建议不超过4）')
    parser.add_argument('--check', action='store_true',
                        help='只检查数据完整性')
    parser.add_argument('--force', action='store_true',
                        help='强制重新生成（覆盖已存在的文件）')
    parser.add_argument('--use-cache', action='store_true', default=True,
                        help='使用缓存数据（默认启用，可加速）')
    parser.add_argument('--no-cache', action='store_true',
                        help='不使用缓存，强制从 API 获取')
    parser.add_argument('--output-dir', type=str, default='data/quarterly_data_csi500',
                        help='输出目录（默认 data/quarterly_data_v2）')
    parser.add_argument('--index', default='000905', choices=['000300', '000905'],
                        help='指数代码: 000300=沪深300(默认), 000905=中证500')
    
    args = parser.parse_args()
    
    # 处理缓存选项
    use_cache = args.use_cache and not args.no_cache
    
    # 处理输出目录
    output_dir = args.output_dir
    if output_dir and not os.path.isabs(output_dir):
        output_dir = os.path.join(ROOT_DIR, output_dir)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if args.check:
        check_data_integrity()
        return
    
    target_dir = output_dir or OUTPUT_DIR
    
    index_code = args.index
    
    is_parallel = args.parallel > 1
    
    # 确定要生成的季度
    if args.quarters:
        tasks = []
        for q in args.quarters:
            parts = q.split('_')
            if len(parts) == 2:
                year = int(parts[0])
                quarter = parts[1]
                tasks.append((year, quarter, use_cache, output_dir, index_code, is_parallel))
    else:
        tasks = []
        for year in args.years:
            for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                tasks.append((year, q, use_cache, output_dir, index_code, is_parallel))
    
    # 如果强制重新生成，删除已存在的文件
    if args.force:
        for year, quarter, *_ in tasks:
            output_file = os.path.join(target_dir, f"{year}_{quarter}_ml_training_data.json")
            if os.path.exists(output_file):
                os.remove(output_file)
                print(f"已删除: {output_file}")
    
    index_name = {'000300': '沪深300', '000905': '中证500'}.get(index_code, index_code)
    print("=" * 60)
    print(f"批量生成季度数据")
    print("=" * 60)
    print(f"指数: {index_name} ({index_code})")
    print(f"待生成季度: {len(tasks)}")
    print(f"并行进程数: {args.parallel}")
    print(f"使用缓存: {'是' if use_cache else '否（从API获取）'}")
    print(f"输出目录: {target_dir}")
    print("=" * 60)
    
    if args.parallel > 4 and not use_cache:
        print("\n⚠️  警告: 并行数 > 4 且不使用缓存，可能触发 tushare API 限流!")
        print("    建议: 使用 --use-cache 或降低并行数\n")
    
    results = []
    total = len(tasks)
    
    if args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(build_quarter_wrapper, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                done = len(results)
                status = result.get('status', '?')
                qname = result.get('quarter', '?')
                info = f"{result.get('records', 0)}条" if status == 'success' else status
                print(f"[进度 {done}/{total}] {qname}: {info}")
    else:
        for i, task in enumerate(tasks):
            result = build_quarter_wrapper(task)
            results.append(result)
            status = result.get('status', '?')
            qname = result.get('quarter', '?')
            info = f"{result.get('records', 0)}条" if status == 'success' else status
            print(f"[进度 {i+1}/{total}] {qname}: {info}")
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("生成结果汇总")
    print("=" * 60)
    
    success = [r for r in results if r.get('status') == 'success']
    skipped = [r for r in results if r.get('status') == 'skipped']
    failed = [r for r in results if r.get('status') not in ('success', 'skipped')]
    
    print(f"成功: {len(success)}")
    for r in success:
        print(f"  - {r['quarter']}: {r.get('records', 0)} 条")
    
    if skipped:
        print(f"跳过（已存在）: {len(skipped)}")
        for r in skipped:
            print(f"  - {r['quarter']}")
    
    if failed:
        print(f"失败: {len(failed)}")
        for r in failed:
            print(f"  - {r['quarter']}: {r.get('status', 'unknown')} - {r.get('error', '')[:50]}")
    
    # 自动检查完整性
    print("\n自动检查数据完整性...")
    check_data_integrity()


if __name__ == '__main__':
    main()
