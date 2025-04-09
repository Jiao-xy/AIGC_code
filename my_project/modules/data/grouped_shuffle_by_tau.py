# modules/data/grouped_shuffle_by_tau.py
# 从单个输入文件中读取数据（按 word_count 自然划分），每10份采用指定扰动策略生成统一训练对

import random
import time
import os
from tqdm import tqdm
from modules.utils.jsonl_handler import read_jsonl, save_results
from modules.analysis.aggressive_shuffle_analysis import aggressive_shuffle, evaluate_dissimilarity
from modules.data.shufflers.by_tau import shuffle_with_target_tau

def apply_strategy(sentence, strategy):
    """根据扰动策略生成打乱句子"""
    words = sentence.split()
    if strategy.startswith("tau_"):
        tau_value = float(strategy.split("_")[1])
        shuffled = shuffle_with_target_tau(words, target_tau=tau_value)
        return " ".join(shuffled), strategy
    elif strategy == "aggressive":
        best, _, _ = run_best_shuffle(sentence)
        return best, strategy
    elif strategy == "random":
        random.shuffle(words)
        return " ".join(words), strategy
    else:
        return sentence, "original"

def run_best_shuffle(text):
    """多轮扰动，选出最不相似的版本"""
    best_score = float("inf")
    best_output = text
    for _ in range(30):  # 降低尝试次数以加速
        shuffled = aggressive_shuffle(text)
        score = evaluate_dissimilarity(text, shuffled)["combined_score"]
        if score < best_score:
            best_score = score
            best_output = shuffled
    return best_output, best_score, None

def segment_chunks_by_wordcount(data, target_chunks=100):
    """根据 word_count 分段（单词数从小到大变化为新段）"""
    chunks = []
    current_chunk = []
    last_wc = -1
    for item in data:
        wc = item.get("word_count", 0)
        if last_wc != -1 and wc < last_wc and len(current_chunk) > 0:
            chunks.append(current_chunk)
            current_chunk = []
        current_chunk.append(item)
        last_wc = wc
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def generate_grouped_shuffle(input_path, output_path):
    """主函数：分组 + 分配策略 + 应用扰动 + 统一保存"""
    all_data = read_jsonl(input_path)
    grouped_data = segment_chunks_by_wordcount(all_data)

    assert len(grouped_data) >= 10, "分组不足 10 组，数据过少"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_results = []
    strategies_per_group = [
        ["tau_0.2"] * 2 + ["tau_0.5"] * 2 + ["tau_0.8"] * 2 + ["random"] * 2 + ["aggressive"] * 2
    ] * (len(grouped_data) // 10)

    for group_id in tqdm(range(len(strategies_per_group)), desc="Processing Groups"):
        group_start = time.time()
        group_results = []
        chunk_indices = list(range(group_id * 10, (group_id + 1) * 10))
        strategy_assignment = strategies_per_group[group_id]
        random.shuffle(strategy_assignment)

        for idx, chunk_idx in enumerate(chunk_indices):
            if chunk_idx >= len(grouped_data):
                continue
            strategy = strategy_assignment[idx]
            data_chunk = grouped_data[chunk_idx]

            for item in tqdm(data_chunk, desc=f"Group {group_id} | Chunk {chunk_idx} | {strategy}", leave=False):
                sid = item.get("sentence_id")
                orig = item.get("sentence", "").strip()
                shuffled, strategy_used = apply_strategy(orig, strategy)

                group_results.append({
                    "id": sid,
                    "original": orig,
                    "shuffled": shuffled,
                    "metadata": {
                        "group_id": group_id,
                        "strategy": strategy_used
                    }
                })

        total_results.extend(group_results)
        print(f"⏱️ Group {group_id} 完成，耗时 {time.time() - group_start:.2f} 秒")

        # ✅ 每处理完一组，追加保存一次中间结果
        intermediate_path = output_path.replace(".jsonl", f"_partial_g{group_id}.jsonl")
        save_results(group_results, intermediate_path)
        print(f"📁 已保存中间结果至 {intermediate_path}")

    save_results(total_results, output_path)
    print(f"✅ 共生成 {len(total_results)} 条训练对，保存至 {output_path}")

if __name__ == "__main__":
    generate_grouped_shuffle(
        input_path="data/train_pairs/ieee-merged-balanced.jsonl",
        output_path="data/train_pairs/grouped_shuffle_all.jsonl"
    )
