#python -m modules.shuffle_sentences_basic
#将句子内词随机打乱
import random
from modules.jsonl_handler import read_jsonl, save_results

def process(file_path):
    data = read_jsonl(file_path)
    for item in data:
        shuffled = []
        for s in item.get("sentences", []):
            words = s.strip().split()
            if len(words) > 1:
                random.shuffle(words)
            shuffled.append(" ".join(words))
        item["shuffled_sentences"] = shuffled
    output_file = file_path.replace(".jsonl", "_shuffled.jsonl")
    save_results(data, output_file)

    return data

if __name__ == "__main__":
    process("data/modules_test_data/test_split.jsonl")