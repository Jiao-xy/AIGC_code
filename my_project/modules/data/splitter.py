# python -m modules.data.splitter
# 使用 spaCy 拆句，并防止在小数点或 [1], 被误分割

import spacy
import re
from spacy.language import Language
from modules.utils.jsonl_handler import read_jsonl, save_results

# 加载 spaCy 英文模型
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ✅ 注册自定义断句规则组件
@Language.component("prevent_split_on_decimal")
def prevent_split_on_decimal(doc):
    for i, token in enumerate(doc[:-2]):
        if (
            token.text == "." and
            token.nbor(-1).like_num and
            token.nbor(1).like_num
        ):
            doc[i + 1].is_sent_start = False
    return doc

# ✅ 正确地添加到 spaCy pipeline（字符串名而不是函数对象）
if "prevent_split_on_decimal" not in nlp.pipe_names:
    nlp.add_pipe("prevent_split_on_decimal", before="parser")

def merge_reference_prefix(sentences):
    """
    合并 '[1], xxx' 或 '[3],' 等引用型前缀与后句
    """
    merged = []
    i = 0
    while i < len(sentences):
        curr = sentences[i]
        # 情况1：单独一个 "[1]," 没有其他内容
        if re.fullmatch(r"\[\d+\],?", curr):
            if i + 1 < len(sentences):
                merged.append(curr + " " + sentences[i + 1])
                i += 2
            else:
                i += 1
        # 情况2：开头是 "[3]," 且句子太短（可能是短语）
        elif re.match(r"^\[\d+\],", curr) and len(curr.split()) <= 4:
            if i + 1 < len(sentences):
                merged.append(curr + " " + sentences[i + 1])
                i += 2
            else:
                merged.append(curr)
                i += 1
        else:
            merged.append(curr)
            i += 1
    return merged

def split_into_sentences(text):
    doc = nlp(text.strip())
    sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return merge_reference_prefix(sents)

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
    #process("data/modules_test_data/test.jsonl")
    process("data/init/ieee-init.jsonl")
    process("data/init/ieee-chatgpt-generation.jsonl")