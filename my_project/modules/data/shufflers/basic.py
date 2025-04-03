#python -m modules.shufflers.basic
#将句子内词随机打乱
import random
from modules.utils.jsonl_handler import read_jsonl, save_results

def process(file_path):
    data = read_jsonl(file_path)

    for item in data:
        original = item.get("sentence", "").strip()
        words = original.split()
        if len(words) > 1:
            random.shuffle(words)
        item["shuffled_sentence"] = " ".join(words)

    output_file = file_path.replace(".jsonl", "_shuffled.jsonl")
    save_results(data, output_path=output_file)
    print(f"已打乱 {len(data)} 条句子，结果保存至 {output_file}")
    return data

if __name__ == "__main__":
    process("data/modules_test_data/test_split.jsonl")