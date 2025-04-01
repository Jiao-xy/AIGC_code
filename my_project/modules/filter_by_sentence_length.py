#python -m modules.filter_by_sentence_length
#过滤句子长度不合规项
from modules.jsonl_handler import read_jsonl, save_results

def process(file_path, min_words=8, max_words=50):
    data = read_jsonl(file_path)
    filtered = []
    for item in data:
        new_sentences = []
        for s in item.get("sentences", []):
            words = s.split()
            if min_words <= len(words) <= max_words:
                new_sentences.append(s)
        if new_sentences:
            item["sentences"] = new_sentences
            filtered.append(item)
    save_results(filtered, file_path)
    return filtered

if __name__ == "__main__":
    process("data/raw/example.jsonl")