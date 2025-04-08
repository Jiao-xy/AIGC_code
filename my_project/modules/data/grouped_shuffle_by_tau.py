# modules/data/grouped_shuffle_by_tau.py
# 每10份分片中采用指定扰动策略分配（tau=0.2/0.5/0.8，最强扰动，随机扰动）

import os
import random
from modules.utils.jsonl_handler import read_jsonl, save_results
from modules.data.generate import shuffle_sentence
from modules.analysis.aggressive_shuffle_analysis import aggressive_shuffle, evaluate_dissimilarity
from modules.data.shufflers.by_tau import shuffle_with_target_tau
from tqdm import tqdm


def apply_strategy(sentence, strategy):
    if strategy.startswith("tau_"):
        tau_value = float(strategy.split("_")[1])
        words = sentence.split()
        shuffled = shuffle_with_target_tau(words, target_tau=tau_value)
        return " ".join(shuffled), strategy
    elif strategy == "aggressive":
        best, _, _ = run_best_shuffle(sentence)
        return best, strategy
    elif strategy == "random":
        words = sentence.split()
        random.shuffle(words)
        return " ".join(words), strategy
    else:
        return shuffle_sentence(sentence, strategy), strategy


def run_best_shuffle(text):
    best_score = float("inf")
    best_output = text
    for _ in range(50):  # 可调节次数
        shuffled = aggressive_shuffle(text)
        score = evaluate_dissimilarity(text, shuffled)["combined_score"]
        if score < best_score:
            best_score = score
            best_output = shuffled
    return best_output, best_score, None


def generate_grouped_shuffle(input_chunks_dir, output_path):
    total_results = []
    strategies_per_group = [
        ["tau_0.2"] * 2 + ["tau_0.5"] * 2 + ["tau_0.8"] * 2 + ["random"] * 2 + ["aggressive"] * 2
    ] * 10  # 10组

    for group_id in range(10):
        group_results = []
        chunk_indices = list(range(group_id * 10, (group_id + 1) * 10))
        strategy_assignment = strategies_per_group[group_id]
        random.shuffle(strategy_assignment)  # 打乱策略顺序

        for idx, chunk_idx in enumerate(chunk_indices):
            chunk_path = os.path.join(input_chunks_dir, f"chunk_{chunk_idx:02d}.jsonl")
            strategy = strategy_assignment[idx]

            data = read_jsonl(chunk_path)
            for item in tqdm(data, desc=f"Group {group_id} Chunk {chunk_idx} [{strategy}]"):
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

    save_results(total_results, output_path)
    print(f"✅ 共生成 {len(total_results)} 条训练对，保存至 {output_path}")


if __name__ == "__main__":
    generate_grouped_shuffle(
        input_chunks_dir="data/chunks",
        output_path="data/train_pairs/grouped_shuffle_all.jsonl"
    )
