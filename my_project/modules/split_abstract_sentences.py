# python -m modules.split_abstract_sentences
# 使用 spaCy 将摘要分割成句子

import spacy
from modules.jsonl_handler import JSONLHandler

# 加载 spaCy 英文模型
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def split_into_sentences(text):
    doc = nlp(text.strip())
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def process(file_path, handler=None):
    if handler is None:
        handler = JSONLHandler()  # 默认读取全部
    data = handler.read_jsonl(file_path)
    for item in data:
        text = item.get("abstract", "")
        item["sentences"] = split_into_sentences(text)
    handler.save_results(data, file_path)
    return data

if __name__ == "__main__":
    custom_handler = JSONLHandler(max_records=5)
    process("../data/init/ieee-init.jsonl", handler=custom_handler)
