#python -m modules.shuffle_sentences_by_tau
#基于目标 tau 打乱词序
import random
import numpy as np
from modules.jsonl_handler import read_jsonl, save_results
from scipy.stats import kendalltau

# 简化实现，尝试多次打乱直到接近目标 tau

def shuffle_with_target_tau(words, target_tau, max_trials=100):
    best = words[:]
    best_tau = -1
    for _ in range(max_trials):
        shuffled = words[:]
        random.shuffle(shuffled)
        tau, _ = kendalltau(range(len(words)), [shuffled.index(w) for w in words])
        if abs(tau - target_tau) < abs(best_tau - target_tau):
            best = shuffled
            best_tau = tau
        if abs(best_tau - target_tau) < 0.05:
            break
    return best

def process(file_path, taus=[0.3, 0.5, 0.7]):
    data = read_jsonl(file_path)

    for item in data:
        original = item.get("sentence", "").strip()
        words = original.split()
        for tau in taus:
            key = f"tau_{int(tau * 100):02d}_sentence"
            if len(words) <= 1:
                item[key] = original  # 不能打乱就保留原样
            else:
                shuffled = shuffle_with_target_tau(words, tau)
                item[key] = " ".join(shuffled)

    output_file = file_path.replace(".jsonl", "_tau.jsonl")
    save_results(data, output_path=output_file)
    print(f"打乱后数据已保存至 {output_file}")
    return data

if __name__ == "__main__":
    process("data/modules_test_data/test_split.jsonl")