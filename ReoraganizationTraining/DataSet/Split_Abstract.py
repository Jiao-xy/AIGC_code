import json
import os
import re
import matplotlib.pyplot as plt
""" import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize """
import spacy
nlp = spacy.load("en_core_web_sm")
input_files = {
    "Fusion": "/home/jxy/Data/ReoraganizationData/init/ieee-chatgpt-fusion.jsonl",
    "Generation": "/home/jxy/Data/ReoraganizationData/init/ieee-chatgpt-generation.jsonl",
    "Polish": "/home/jxy/Data/ReoraganizationData/init/ieee-chatgpt-polish.jsonl",
    "Init": "/home/jxy/Data/ReoraganizationData/init/ieee-init.jsonl"
}
output_dir = "/home/jxy/Data/ReoraganizationData/init/split/"
os.makedirs(output_dir, exist_ok=True)

def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def count_words(sentence):
    return len(sentence.split())

for key, input_file in input_files.items():
    output_file = os.path.join(output_dir, os.path.basename(input_file).replace(".jsonl", "-split.jsonl"))
    sentences_data = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            record = json.loads(line)
            abstract = record.get("abstract", "")
            sentences = split_sentences(abstract)
            
            for sentence in sentences:
                sentences_data.append({
                    "id": record["id"],
                    "sentence": sentence,
                    "word_count": count_words(sentence)
                })
    
    # 按照单词数量从少到多排序
    sentences_data.sort(key=lambda x: x["word_count"])
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for data in sentences_data:
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    print(f"Processing complete. Output saved to: {output_file}")

# 统计单词数分布
word_counts = {key: [] for key in input_files.keys()}

for key, file_path in input_files.items():
    split_file = os.path.join(output_dir, os.path.basename(file_path).replace(".jsonl", "-split.jsonl"))
    if os.path.exists(split_file):
        with open(split_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                record = json.loads(line)
                word_counts[key].append(record["word_count"])

# 统计单词数的分布
word_count_distribution = {key: {} for key in input_files.keys()}

for key, counts in word_counts.items():
    for count in counts:
        if count in word_count_distribution[key]:
            word_count_distribution[key][count] += 1
        else:
            word_count_distribution[key][count] = 1

# 创建4个子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (key, distribution) in enumerate(word_count_distribution.items()):
    sorted_counts = sorted(distribution.keys())
    frequencies = [distribution[count] for count in sorted_counts]

    axes[i].bar(sorted_counts, frequencies)
    axes[i].set_xlabel("Word Count per Sentence")
    axes[i].set_ylabel("Number of Sentences")
    axes[i].set_title(f"Distribution of Sentence Word Count in {key} File")
    axes[i].grid(axis="y")

# 调整布局并保存为 SVG 文件
plt.tight_layout()
output_svg = "/home/jxy/Data/ReoraganizationData/init/split/sentence_word_count_distribution.svg"
plt.savefig(output_svg, format="svg")
plt.show()

print(f"SVG file saved: {output_svg}")
