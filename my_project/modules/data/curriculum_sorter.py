# modules/data/curriculum_sorter.py
# 将分好句子的样本按长度 & PPL 指标均匀划分成十份

import random
from modules.utils.jsonl_handler import read_jsonl, save_results

def curriculum_sort(input_path, output_path, sort_keys=["word_count", "PPL"], partitions=10):
    """
    将数据按指定字段排序后划分成若干份，每份内部排序，打乱顺序保存

    参数：
    - input_path: 输入 jsonl 文件
    - output_path: 输出 jsonl 文件
    - sort_keys: 用于排序的字段列表（优先级从左到右）
    - partitions: 划分的份数
    """
    data = read_jsonl(input_path)
    print(f"读取 {len(data)} 条样本。开始排序与划分……")

    # 多级排序
    data.sort(key=lambda x: tuple(x[k] for k in sort_keys))

    # 平均划分为 partitions 份
    partitioned = [[] for _ in range(partitions)]
    for idx, item in enumerate(data):
        partitioned[idx % partitions].append(item)

    # 每份内部再按 word_count 升序排序（也可按多个字段）
    for group in partitioned:
        group.sort(key=lambda x: x["word_count"])

    # 拼接成一个大列表
    final_data = []
    for group in partitioned:
        final_data.extend(group)

    save_results(final_data, output_path)
    print(f"处理完成，已保存至 {output_path}")

if __name__ == "__main__":
    curriculum_sort(
        input_path="data/init/ieee-init_split.jsonl",
        output_path="data/init/ieee-init_curriculum_sorted.jsonl"
    )
