import json
import os
import random

def shuffle_abstract_sentences(file_path):
    # 读取JSONL文件，每行是一个独立的JSON对象
    with open(file_path, 'r', encoding='utf-8') as f:
        data_list = [json.loads(line) for line in f]  # 逐行解析JSON对象
    
    # 处理每个JSON对象的摘要
    for data in data_list:
        abstract = data.get("abstract", "")
        sentences = abstract.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        random.shuffle(sentences)
        shuffled_abstract = '. '.join(sentences) + '.' if sentences else ""
        data["abstract"] = shuffled_abstract  # 更新摘要

    # 生成新的文件名
    base, ext = os.path.splitext(file_path)
    new_file_path = f"{base}_random{ext}"

    # 保存为新的 JSONL 文件
    with open(new_file_path, 'w', encoding='utf-8') as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')  # 每行写入一个 JSON 对象
    
    print(f"Processed file saved as: {new_file_path}")

# 处理多个JSONL文件
json_files = [
    "/home/jxy/Data/ieee-init.jsonl", 
    "/home/jxy/Data/ieee-chatgpt-generation.jsonl"
]

for json_file in json_files:
    shuffle_abstract_sentences(json_file)
