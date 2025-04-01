#python -m modules.check_shuffled_dataset
#检查字段完整性和打乱字段
from modules.jsonl_handler import read_jsonl

def process(file_path):
    data = read_jsonl(file_path)
    missing = []
    for i, item in enumerate(data):
        if "sentences" not in item:
            missing.append((i, "sentences"))
        elif not any(k.endswith("sentences") and k != "sentences" for k in item):
            missing.append((i, "shuffled"))
    print(f"共检查 {len(data)} 条记录，缺失信息：{len(missing)} 条")
    for idx, issue in missing[:5]:
        print(f"第 {idx} 条缺失 {issue}")
    return missing

if __name__ == "__main__":
    process("data/modules_test_data/example.jsonl")
