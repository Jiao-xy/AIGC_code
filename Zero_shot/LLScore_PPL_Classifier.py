import json
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载 GPT-2 预训练模型
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

tokenizer.pad_token = tokenizer.eos_token  # 设置 PAD token

# 文件路径（请修改为实际路径）
json_files = {
    "init": "/home/jxy/Data/Zero_shot/llscore_ppl/ieee-init_llscore_ppl.jsonl",
    "generation": "/home/jxy/Data/Zero_shot/llscore_ppl/ieee-chatgpt-generation_llscore_ppl.jsonl",
    "init_random": "/home/jxy/Data/Zero_shot/llscore_ppl_random/ieee-init_random_llscore_ppl.jsonl",
    "generation_random": "/home/jxy/Data/Zero_shot/llscore_ppl_random/ieee-chatgpt-generation_random_llscore_ppl.jsonl"
}
""" init_file_path = "/home/jxy/Data/Zero_shot/llscore_ppl/ieee-init_llscore_ppl.jsonl"
generation_file_path = "/home/jxy/Data/Zero_shot/llscore_ppl/ieee-chatgpt-generation_llscore_ppl.jsonl"
init_random_file_path = "/home/jxy/Data/Zero_shot/llscore_ppl_random/ieee-init_random_llscore_ppl.jsonl"
generation_random_file_path = "/home/jxy/Data/Zero_shot/llscore_ppl_random/ieee-chatgpt-generation_random_llscore_ppl.jsonl" """

# 读取 JSONL 文件
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 读取数据
data = []
labels = []

for label, file_path in json_files.items():
    dataset = load_jsonl(file_path)
    for entry in dataset:
        data.append([entry["LLScore"], entry["PPL"]])
        labels.append(0 if "init" in label else 1)  # 0: init, 1: generation

# 转换为 NumPy 数组
data = np.array(data)
labels = np.array(labels)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 训练分类器（随机森林）
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 预测测试集
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("分类准确率:", accuracy)
print("分类报告:")
print(classification_report(y_test, y_pred))

# 绘制分类结果可视化
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap="coolwarm", alpha=0.7)
plt.xlabel("LLScore")
plt.ylabel("PPL")
plt.title("Classification of Init vs. Generation based on LLScore & PPL")
plt.colorbar(label="Predicted Label (0: Init, 1: Generation)")
plt.savefig("classification_result.svg")
plt.close()
