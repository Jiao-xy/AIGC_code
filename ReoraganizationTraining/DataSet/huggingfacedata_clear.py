from datasets import Dataset
import json
import re

# 修改为你自己的文件路径
arrow_file = "/home/jxy/.cache/huggingface/datasets/scientific_papers/arxiv/1.1.1/306757013fb6f37089b6a75469e6638a553bd9f009484938d8f75a4c5e84206f/scientific_papers-train-00000-of-00014.arrow"
output_file = "arxiv_cleaned_100.jsonl"
max_samples = 100

def clean_text(text):
    """清除 arXiv 中的数学符号、LaTeX、引用等标记"""
    text = re.sub(r"@xmath\d*", "", text)                     # @xmath 占位符
    text = re.sub(r"@xcite", "", text)                        # @xcite 引用
    text = re.sub(r"\[[^\]]*\]", "", text)                    # [table1] 这类引用
    text = re.sub(r"\$[^$]*\$", "", text)                     # $...$ 数学公式
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", text)            # \mbox{...} 类LaTeX
    text = re.sub(r"\s*[_]+\s*", " ", text)  # 移除孤立下划线
    text = re.sub(r"\s+", " ", text)                          # 多余空格
    text = re.sub(r"\(\s*\)", "", text)
    return text.strip()

# 加载 arrow 数据集
print("📥 Loading .arrow file...")
dataset = Dataset.from_file(arrow_file)

# 清洗并保存前 N 条数据为 JSONL
print(f"🚀 Cleaning and saving first {max_samples} samples to {output_file}...")
with open(output_file, "w", encoding="utf-8") as fout:
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
        cleaned_article = clean_text(example["article"])
        cleaned_abstract = clean_text(example["abstract"])
        if cleaned_article and cleaned_abstract:
            fout.write(json.dumps({
                "article": cleaned_article,
                "summary": cleaned_abstract
            }, ensure_ascii=False) + "\n")

print(f"✅ 清洗并保存完成，文件位置：{output_file}")
