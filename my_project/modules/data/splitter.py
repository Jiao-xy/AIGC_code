#python -m modules.data.splitter
#使用spacy将摘要分割成句子
import spacy
from modules.utils.jsonl_handler import read_jsonl, save_results

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

def process(file_path):
    data = read_jsonl(file_path)
    results = []

    for item in data:
        doc_id = item.get("id")
        text = item.get("abstract", "")
        sentences = split_into_sentences(text)
        for i, sent in enumerate(sentences):
            word_count = len(sent.split())  # 用空格分词，统计词数
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
