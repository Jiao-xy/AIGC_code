#python -m modules.jsonl_handler
# 处理 JSONL 文件的类

import json
import os

class JSONLHandler:
    def __init__(self, max_records=None):
        self.max_records = max_records

    def read_jsonl(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if self.max_records is not None and i >= self.max_records:
                    break
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return data

    def save_results(self, data, output_path):
        """
        保存数据到指定 JSONL 路径（output_path 为必填）
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"结果已保存至: {output_path}")

# ✅ 提供函数式封装（默认不限制记录条数）
def read_jsonl(file_path, max_records=None):
    return JSONLHandler(max_records=max_records).read_jsonl(file_path)

def save_results(data, output_path):
    return JSONLHandler().save_results(data, output_path)