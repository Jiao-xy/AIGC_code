# python -m modules.data.splitter
# 使用 spaCy 将摘要拆分为句子，并避免小数点被错误拆分

import spacy
import re
from modules.utils.jsonl_handler import read_jsonl, save_results
from spacy.language import Language  # ✅ 新增这一行

# ✅ 加载 spaCy 并添加小数保护规则
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def prevent_split_on_decimal(doc):
    """
    自定义 spaCy 分句规则：防止小数点之间被当作句子边界
    """
    for i, token in enumerate(doc[:-2]):
        if (
            token.text == '.' and
            token.nbor(-1).like_num and
            token.nbor(1).like_num
        ):
            doc[i + 1].is_sent_start = False
    return doc

# ✅ 注册自定义分句规则
nlp.add_pipe(prevent_split_on_decimal, before="parser")

def split_into_sentences(text):
    doc = nlp(text.strip())
    sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return merge_reference_prefix(sents)

def merge_reference_prefix(sentences):
    """
    合并 '[1], ...' 或 '[3], and so on.' 这类不完整句子
    """
    merged = []
    i = 0
    while i < len(sentences):
        current = sentences[i]
        # 检查是否是类似 "[1], something" 或单纯的 "[1]," 独立句
        if re.fullmatch(r"\[\d+\],?", current) or re.match(r"^\[\d+\],?\s*$", current):
            # 跳过或拼接下一句
            if i + 1 < len(sentences):
                merged.append(current + " " + sentences[i + 1])
                i += 2
            else:
                i += 1
        elif re.match(r"^\[\d+\],", current) and len(current.split()) <= 4:
            # 很短的 "[1], weighted average" 这种句子，尝试合并下一句
            if i + 1 < len(sentences):
                merged.append(current + " " + sentences[i + 1])
                i += 2
            else:
                merged.append(current)
                i += 1
        else:
            merged.append(current)
            i += 1
    return merged

def process(file_path):
    data = read_jsonl(file_path)
    results = []

    for item in data:
        doc_id = item.get("id")
        text = item.get("abstract", "")
        sentences = split_into_sentences(text)
        for i, sent in enumerate(sentences):
            word_count = len(sent.split())
            results.append({
                "id": doc_id,
                "sentence_id": f"{doc_id}_{i}",
                "sentence": sent,
                "word_count": word_count
            })

    output_file = file_path.replace(".jsonl", "_split.jsonl")
    save_results(results, output_file)
    print(f"共拆分出 {len(results)} 句，保存至 {output_file}")
    return results

if __name__ == "__main__":
    process("data/modules_test_data/test.jsonl")
