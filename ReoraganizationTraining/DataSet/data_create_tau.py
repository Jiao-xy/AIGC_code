import json
import pandas as pd
import random
import nltk
from tqdm import tqdm
from scipy.stats import kendalltau, spearmanr
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize

# **1️⃣ 读取 JSONL 文件**
data_files = [
    "/home/jxy/Data/init/ieee-init.jsonl",
    "/home/jxy/Data/init/ieee-chatgpt-generation.jsonl"
]
data_pairs = []

#生成不同 Kendall’s Tau 级别的打乱句子
def shuffle_with_tau(sentence, tau=0.5):
    words = word_tokenize(sentence)  # 词分割
    num_swaps = int(len(words) * (1 - tau))  # 控制打乱程度
    shuffled = words[:]
    
    for _ in range(num_swaps):
        i, j = random.sample(range(len(words)), 2)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

    return " ".join(shuffled)

#计算 Kendall’s Tau、Spearman’s 相关性、Position Distance
def compute_metrics(original, shuffled):
    original_order = {word: i for i, word in enumerate(original.split())}
    shuffled_order = [original_order[word] for word in shuffled.split() if word in original_order]

    if len(shuffled_order) < 2:
        return 0, 0, 0  # 不能计算 Kendall’s Tau 或 Spearman’s 相关性

    tau_score, _ = kendalltau(list(range(len(shuffled_order))), shuffled_order)
    spearman_score, _ = spearmanr(list(range(len(shuffled_order))), shuffled_order)
    
    # 计算 Position Distance
    position_distances = [abs(i - shuffled_order[i]) for i in range(len(shuffled_order))]
    pos_dist_score = sum(position_distances) / len(position_distances)

    return tau_score, spearman_score, pos_dist_score


# **统计总行数（用于进度条）**
total_lines = sum(1 for file in data_files for _ in open(file, "r", encoding="utf-8"))
print(f"📄 发现 {total_lines} 条摘要数据，开始处理...\n")

# **2️⃣ 解析 JSONL 文件**
for file in data_files:
    with open(file, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="⏳ 读取 JSONL 数据", unit="条摘要"):
            try:
                entry = json.loads(line.strip())  
                abstract_text = entry.get("abstract", "").strip()

                if abstract_text:
                    sentences = sent_tokenize(abstract_text)  

                    for sentence in sentences:  # ✅ 直接遍历句子，不使用 tqdm
                        # 生成不同打乱级别的句子
                        shuffled_08 = shuffle_with_tau(sentence, tau=0.8)
                        shuffled_05 = shuffle_with_tau(sentence, tau=0.5)
                        shuffled_02 = shuffle_with_tau(sentence, tau=0.2)

                        # 计算打乱指标
                        tau_08, spearman_08, pos_dist_08 = compute_metrics(sentence, shuffled_08)
                        tau_05, spearman_05, pos_dist_05 = compute_metrics(sentence, shuffled_05)
                        tau_02, spearman_02, pos_dist_02 = compute_metrics(sentence, shuffled_02)

                        # **6️⃣ 添加到数据集**
                        data_pairs.append({
                            "原句": sentence,
                            "打乱句子_08": shuffled_08, "tau_08": tau_08, "spearman_08": spearman_08, "pos_dist_08": pos_dist_08,
                            "打乱句子_05": shuffled_05, "tau_05": tau_05, "spearman_05": spearman_05, "pos_dist_05": pos_dist_05,
                            "打乱句子_02": shuffled_02, "tau_02": tau_02, "spearman_02": spearman_02, "pos_dist_02": pos_dist_02,
                        })

            except json.JSONDecodeError:
                print(f"❌ JSON 解析失败: {line}")

# **7️⃣ 保存数据（带进度条）**
output_file = "sentence_shuffled_dataset.csv"
df = pd.DataFrame(data_pairs)

print("\n💾 正在保存数据...")
for _ in tqdm(range(100), desc="💾 写入 CSV 文件"):
    df.to_csv(output_file, index=False, encoding="utf-8")

print(f"✅ 训练数据已保存至: {output_file}")
