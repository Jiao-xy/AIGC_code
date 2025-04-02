import json
import re
from datasets import Dataset

# ===== 配置部分 =====
ARROW_FILE = "/home/jxy/.cache/huggingface/datasets/scientific_papers/arxiv/1.1.1/306757013fb6f37089b6a75469e6638a553bd9f009484938d8f75a4c5e84206f/scientific_papers-train-00000-of-00014.arrow"
OUTPUT_JSONL = "arxiv_cleaned_enhanced_100.jsonl"
MAX_SAMPLES = 100
MIN_LEN = 100     # 最小词数
MAX_LEN = 4096    # 最大词数（防止模型截断）

# ===== 清洗函数 =====
def clean_text(text):
    text = re.sub(r"@xmath\d*", "", text)
    text = re.sub(r"@xcite", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)                     # 移除 [table] [figure] 等
    text = re.sub(r"\$[^$]*\$", "", text)                      # LaTeX公式
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", text)             # \mbox{}
    text = re.sub(r"\(\s*\)", "", text)                        # 空括号
    text = re.sub(r"\s*[_]+\s*", " ", text)                    # 孤立下划线
    text = re.sub(r"\b\d+\s*(gev|mev|tev|ns|ps|s)\b", "", text, flags=re.IGNORECASE)  # 单位
    text = re.sub(r"\b(gev|mev|tev|sec|ns|ps)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwith( [a-z]{1,3})?( and)? (decays|measurements|data)?", "", text, flags=re.IGNORECASE)

    # 去除参考文献 / 致谢
    text = re.split(r"(we gratefully acknowledge|this work was supported by|references\b|phys\.\s+rev\.)", text, flags=re.IGNORECASE)[0]

    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ===== 载入并处理 =====
dataset = Dataset.from_file(ARROW_FILE)
count = 0

with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
    for example in dataset:
        if count >= MAX_SAMPLES:
            break

        article = clean_text(example.get("article", ""))
        summary = clean_text(example.get("abstract", ""))

        # 过滤空值和长度不合适的
        if not article or not summary:
            continue
        if not (MIN_LEN <= len(article.split()) <= MAX_LEN):
            continue

        fout.write(json.dumps({
            "article": article,
            "summary": summary
        }, ensure_ascii=False) + "\n")
        count += 1

print(f"✅ 共清洗并保存 {count} 条样本 ➜ {OUTPUT_JSONL}")
