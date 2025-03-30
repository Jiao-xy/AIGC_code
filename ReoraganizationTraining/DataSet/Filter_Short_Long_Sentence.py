import json
import os
import matplotlib.pyplot as plt

# 定义输入和输出目录
input_dir = "/home/jxy/Data/ReoraganizationData/init/split/"
output_dir = "/home/jxy/Data/ReoraganizationData/init/filtered/"
os.makedirs(output_dir, exist_ok=True)

# 过滤标准
min_words = 10
max_words = 50

# 统计数据存储
word_count_distributions = {}

# 遍历文件夹中的所有句子文件，并处理
for filename in os.listdir(input_dir):
    if filename.endswith("-split.jsonl"):
        file_path = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename.replace("-split.jsonl", "-filtered.jsonl"))
        
        filtered_sentences = []
        word_counts = []
        
        with open(file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                record = json.loads(line)
                word_count = record["word_count"]
                if min_words <= word_count <= max_words:
                    filtered_sentences.append(record)
                    word_counts.append(word_count)
        
        # 保存过滤后的句子
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for record in filtered_sentences:
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"Filtered sentences saved to: {output_file}")
        
        # 记录统计数据
        word_count_distributions[filename] = word_counts

# 生成合并的统计图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (filename, word_counts) in enumerate(word_count_distributions.items()):
    axes[i].hist(word_counts, bins=range(min_words, max_words + 2), edgecolor='black', alpha=0.7)
    axes[i].set_xlabel("Word Count per Sentence")
    axes[i].set_ylabel("Number of Sentences")
    axes[i].set_title(f"Distribution in {filename}")
    axes[i].grid(axis="y")

# 调整布局并保存合并的统计图
plt.tight_layout()
output_chart = os.path.join(output_dir, "filtered_sentence_distribution.svg")
plt.savefig(output_chart, format="svg")
plt.close()

print(f"Combined filtered sentence distribution chart saved to: {output_chart}")
