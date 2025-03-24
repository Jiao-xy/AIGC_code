import json
import random
import nltk
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt")

# **1️⃣ 设定文件路径**
data_files = [
    "/home/jxy/Data/ReoraganizationData/init/ieee-init.jsonl",
    "/home/jxy/Data/ReoraganizationData/init/ieee-chatgpt-generation.jsonl",
]

output_files = {
    "tau_08": "/home/jxy/Data/ReoraganizationData/init/sentence_shuffled_dataset_tau_08.jsonl",
    "tau_05": "/home/jxy/Data/ReoraganizationData/init/sentence_shuffled_dataset_tau_05.jsonl",
    "tau_02": "/home/jxy/Data/ReoraganizationData/init/sentence_shuffled_dataset_tau_02.jsonl",
}

# **2️⃣ 生成不同 Kendall’s Tau 级别的打乱句子**
def shuffle_with_tau(sentence, tau=0.5):
    words = word_tokenize(sentence)
    num_swaps = int(len(words) * (1 - tau))  # 计算交换次数
    shuffled = words[:]
    
    for _ in range(num_swaps):
        i, j = random.sample(range(len(words)), 2)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

    return " ".join(shuffled)

# **3️⃣ 统计总行数（用于进度条）**
total_lines = sum(1 for file in data_files for _ in open(file, "r", encoding="utf-8"))
print(f"📄 发现 {total_lines} 条摘要数据，开始处理...\n")

# **4️⃣ 解析 JSONL 文件并生成 3 个 JSONL 数据集**
file_handles = {tau: open(path, "w", encoding="utf-8") for tau, path in output_files.items()}

for file in data_files:
    with open(file, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="⏳ 处理中", unit="条摘要"):
            try:
                entry = json.loads(line.strip())  
                abstract_text = entry.get("abstract", "").strip()

                if abstract_text:
                    sentences = sent_tokenize(abstract_text)  

                    for sentence in sentences:
                        shuffled_sentences = {
                            "tau_08": shuffle_with_tau(sentence, tau=0.8),
                            "tau_05": shuffle_with_tau(sentence, tau=0.5),
                            "tau_02": shuffle_with_tau(sentence, tau=0.2),
                        }

                        for tau, shuffled in shuffled_sentences.items():
                            json_entry = {
                                "shuffled_sentence": shuffled,
                                "original_sentence": sentence,
                            }
                            file_handles[tau].write(json.dumps(json_entry, ensure_ascii=False) + "\n")

            except json.JSONDecodeError:
                print(f"❌ JSON 解析失败，跳过该行: {line}")

# **5️⃣ 关闭 JSONL 文件**
for f in file_handles.values():
    f.close()

print("\n✅ 3 个 JSONL 数据集已保存：")
for tau, path in output_files.items():
    print(f"📂 {tau}: {path}")
