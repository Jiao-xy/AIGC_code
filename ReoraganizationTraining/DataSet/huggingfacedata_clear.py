from datasets import Dataset
import os

# 路径指向任意一个分片文件
arrow_file = "/home/jxy/.cache/huggingface/datasets/scientific_papers/arxiv/1.1.1/306757013fb6f37089b6a75469e6638a553bd9f009484938d8f75a4c5e84206f/scientific_papers-train-00000-of-00014.arrow"

# 加载 arrow 文件为 Dataset 对象
dataset = Dataset.from_file(arrow_file)

# 输出字段名称
print("字段名（columns）:")
print(dataset.column_names)

# 输出第一个样本的所有字段
print("\n第一个样本示例:")
for key, value in dataset[0].items():
    print(f"{key}:\n{value if isinstance(value, str) else value}\n{'-'*50}")
