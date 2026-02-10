# -*- coding: utf-8 -*-
"""
批量更新 stock_* 缓存文件中的 revenue_growth 字段

策略：
- 按股票代码分组（~789只），每只股票只调用一次 Tushare fina_indicator API
- 获取该股票所有报告期的 or_yoy（营收同比增长率）
- 对每个缓存文件，按 ann_date <= target_date 的逻辑匹配最新报告
- 更新 revenue_growth 字段并回写缓存

总 API 调用约 789 次（每只股票1次），预计 3-5 分钟。
"""

import os
import re
import sys
import time
import pickle
import logging
from collections import defaultdict
from datetime import datetime, timedelta

# 添加项目根目录到 path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    cache_dir = os.path.join(ROOT_DIR, 'cache')

    # 1. 扫描所有 stock_*.pkl 文件，按股票代码分组
    logger.info("扫描缓存文件...")
    code_files = defaultdict(list)  # code -> [(date, filepath), ...]
    pattern = re.compile(r'stock_(\d{6})_(\d{4}-\d{2}-\d{2})\.pkl')

    for fname in os.listdir(cache_dir):
        m = pattern.match(fname)
        if m:
            code = m.group(1)
            date = m.group(2)
            code_files[code].append((date, os.path.join(cache_dir, fname)))

    total_files = sum(len(v) for v in code_files.values())
    logger.info(f"共 {total_files} 个缓存文件，{len(code_files)} 只股票")

    # 2. 初始化 Tushare API
    from config.data_source_config import TUSHARE_TOKEN
    import tushare as ts
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    def to_ts_code(code: str) -> str:
        if code.startswith(('6', '5')):
            return f"{code}.SH"
        return f"{code}.SZ"

    # 3. 逐只股票处理
    updated_count = 0
    skipped_count = 0
    error_count = 0
    already_set_count = 0

    codes = sorted(code_files.keys())
    for i, code in enumerate(codes):
        files = code_files[code]
        dates = sorted([d for d, _ in files])

        if (i + 1) % 50 == 0 or i + 1 == len(codes):
            logger.info(f"进度: {i+1}/{len(codes)} 股票, "
                        f"已更新={updated_count}, 跳过(已有)={already_set_count}, "
                        f"跳过(无数据)={skipped_count}, 错误={error_count}")

        # 确定查询范围：最早文件日期 - 550 天 ~ 最晚文件日期
        earliest = datetime.strptime(dates[0], '%Y-%m-%d')
        latest = datetime.strptime(dates[-1], '%Y-%m-%d')
        start_date = (earliest - timedelta(days=550)).strftime('%Y%m%d')
        end_date = latest.strftime('%Y%m%d')

        # 调用 Tushare API
        ts_code = to_ts_code(code)
        try:
            time.sleep(0.15)  # 限流
            df = pro.fina_indicator(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,ann_date,end_date,or_yoy'
            )
        except Exception as e:
            logger.warning(f"{code}: API调用失败: {e}")
            error_count += len(files)
            continue

        if df is None or df.empty:
            skipped_count += len(files)
            continue

        # 按 end_date 降序排列（便于后续匹配）
        df = df.dropna(subset=['ann_date']).sort_values('end_date', ascending=False)

        # 4. 更新每个缓存文件
        for date_str, filepath in files:
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)

                # 如果已经有有效的 revenue_growth，跳过
                if isinstance(data, dict):
                    existing = data.get('revenue_growth')
                else:
                    existing = getattr(data, 'revenue_growth', None)

                if existing is not None:
                    already_set_count += 1
                    continue

                # 匹配：ann_date <= target_date 的最新报告
                target_dt_str = date_str.replace('-', '')
                matched = df[df['ann_date'] <= target_dt_str]
                if matched.empty:
                    skipped_count += 1
                    continue

                revenue_growth = matched.iloc[0]['or_yoy']
                # or_yoy 可能是 NaN
                if revenue_growth != revenue_growth:  # NaN check
                    revenue_growth = None

                # 更新并回写
                if isinstance(data, dict):
                    data['revenue_growth'] = revenue_growth
                else:
                    data.revenue_growth = revenue_growth

                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)

                updated_count += 1

            except Exception as e:
                logger.debug(f"{code}@{date_str}: 更新失败: {e}")
                error_count += 1

    logger.info("=" * 60)
    logger.info(f"完成！")
    logger.info(f"  已更新: {updated_count}")
    logger.info(f"  已有值跳过: {already_set_count}")
    logger.info(f"  无财务数据跳过: {skipped_count}")
    logger.info(f"  错误: {error_count}")
    logger.info(f"  总计: {total_files}")


if __name__ == '__main__':
    main()
