import json
import pandas as pd
import random
import nltk
from tqdm import tqdm  # 进度条库

nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize

# **1️⃣ 指定多个 JSONL 文件**
jsonl_files = [
     "/home/jxy/Data/init/ieee-init.jsonl", 
    "/home/jxy/Data/init/ieee-chatgpt-polish.jsonl", 
    "/home/jxy/Data/init/ieee-chatgpt-fusion.jsonl", 
    "/home/jxy/Data/init/ieee-chatgpt-generation.jsonl",
]

output_file = "sentence_reorder_dataset.csv"

# **统计所有 JSONL 文件的总行数**
total_lines = 0
for file in jsonl_files:
    with open(file, "r", encoding="utf-8") as f:
        total_lines += sum(1 for _ in f)

print(f"📄 发现 {total_lines} 条摘要数据，开始处理...\n")

# **2️⃣ 逐步解析 JSONL 文件并写入 CSV**
data_pairs = []
batch_size = 10000  # 每 10K 行写入一次，防止内存占用过大

with open(output_file, "w", encoding="utf-8") as f_out:
    f_out.write("乱序句子,正确句子\n")  # 写入 CSV 头部

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

                            data_pairs.append(f'"{shuffled_sentence}","{sentence}"')

                            # **每 batch_size 行写入一次**
                            if len(data_pairs) >= batch_size:
                                f_out.write("\n".join(data_pairs) + "\n")
                                data_pairs = []  # 清空缓冲区

                except json.JSONDecodeError:
                    print(f"❌ JSON 解析失败: {line}")

    # **写入最后剩余的数据**
    if data_pairs:
        f_out.write("\n".join(data_pairs) + "\n")

print(f"✅ 训练数据已保存至: {output_file}")
