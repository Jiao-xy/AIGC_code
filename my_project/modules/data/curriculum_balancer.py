# python -m modules.data.curriculum_balancer
# 将数据按照困惑度、长度等进行筛选后，再按长度排序并均匀划分为十份（存入一个文件）

import os
import numpy as np
from modules.utils.jsonl_handler import read_jsonl, save_results
from tqdm import tqdm

def filter_and_chunk(
    input_path,
    output_path,
    min_len=6,
    max_len=50,
    max_ppl=300,
    num_chunks=10
):
    """
    筛选句子后均匀划分为 num_chunks 份（每份内部按 word_count 升序），再合并写入一个文件。
    """
    data = read_jsonl(input_path)
    print(f"原始数据量: {len(data)} 条")

    # 过滤合法句子
    filtered = []
    for item in data:
        if (
            isinstance(item.get("word_count"), int)
            and isinstance(item.get("PPL"), (float, int))
        ):
            if min_len <= item["word_count"] <= max_len and item["PPL"] <= max_ppl:
                filtered.append(item)

    print(f"筛选后剩余: {len(filtered)} 条")

    # 按长度排序后均匀划分
    filtered.sort(key=lambda x: x["word_count"])
    chunks = [[] for _ in range(num_chunks)]
    for i, item in enumerate(filtered):
        chunks[i % num_chunks].append(item)

    # 每块内部再排序
    for chunk in chunks:
        chunk.sort(key=lambda x: x["word_count"])

    # 合并写入
    merged = [sample for chunk in chunks for sample in chunk]
    save_results(merged, output_path)
    print(f"均匀划分结果已保存至: {output_path}")

if __name__ == "__main__":
    filter_and_chunk(
        input_path="data/ieee-merged.jsonl",
        output_path="data/ieee-merged-balanced.jsonl",
        min_len=6,
        max_len=50,
        max_ppl=300,
        num_chunks=100
    )
