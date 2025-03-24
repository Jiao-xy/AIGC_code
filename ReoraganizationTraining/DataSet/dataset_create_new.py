import json
import random
import nltk
from tqdm import tqdm  # 进度条库

nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize

# **1️⃣ 指定多个 JSONL 文件**
jsonl_files = [
    "/home/jxy/Data/ReoraganizationData/init/ieee-init.jsonl", 
    "/home/jxy/Data/ReoraganizationData/init/ieee-chatgpt-polish.jsonl", 
]

output_file = "/home/jxy/Data/ReoraganizationData/sentence_reorder_dataset.jsonl"  # **目标存储路径**

# **统计所有 JSONL 文件的总行数**
total_lines = 0
for file in jsonl_files:
    with open(file, "r", encoding="utf-8") as f:
        total_lines += sum(1 for _ in f)

print(f"📄 发现 {total_lines} 条摘要数据，开始处理...\n")

# **2️⃣ 逐步解析 JSONL 文件并写入新的 JSONL**
with open(output_file, "w", encoding="utf-8") as f_out:

    for file in jsonl_files:
        with open(file, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=total_lines, desc="⏳ 处理中", unit="条摘要"):
                try:
                    entry = json.loads(line.strip())  # 解析 JSON
                    abstract_text = entry.get("abstract", "").strip()

                    if abstract_text:  # 确保摘要存在
                        sentences = sent_tokenize(abstract_text)  # 句子拆分
                        for sentence in sentences:
                            words = word_tokenize(sentence)  # 词分割
                            shuffled_words = words[:]
                            random.shuffle(shuffled_words)  # 打乱顺序
                            shuffled_sentence = " ".join(shuffled_words)

                            # **直接存储为 JSONL 格式**
                            json_entry = {
                                "shuffled_sentence": shuffled_sentence,
                                "original_sentence": sentence
                            }
                            f_out.write(json.dumps(json_entry, ensure_ascii=False) + "\n")  # 写入 JSONL

                except json.JSONDecodeError:
                    print(f"❌ JSON 解析失败，跳过该行: {line}")

print(f"✅ 训练数据已保存至: {output_file}")
