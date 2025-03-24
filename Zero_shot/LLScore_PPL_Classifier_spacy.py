import json 
import torch
import numpy as np
import os
import random
import spacy
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

# 加载 SpaCy 依存分析模型
nlp = spacy.load("en_core_web_sm")

# JSONL 文件路径
json_files = {
    "init": "/home/jxy/Data/ReoraganizationData/init/ieee-init.jsonl",
    "generation": "/home/jxy/Data/ReoraganizationData/init/ieee-chatgpt-generation.jsonl"
}

# 计算 LLScore（对数似然）和 PPL（困惑度）
def compute_llscore_ppl(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    
    log_likelihood = -outputs.loss.item() * inputs.input_ids.shape[1]
    perplexity = torch.exp(outputs.loss).item()
    
    return log_likelihood, perplexity

# 读取 JSONL 文件并计算 LLScore 和 PPL
def load_and_process_jsonl(file_path, label):
    data = []
    labels = []
    
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                entry = json.loads(line.strip())
                abstract = entry.get("abstract", "").strip()
                if abstract:
                    reordered_abstract = reorder_sentences_using_dependencies(abstract)
                    llscore, ppl = compute_llscore_ppl(reordered_abstract)
                    data.append([llscore, ppl])
                    labels.append(label)
            except json.JSONDecodeError:
                print(f"JSON 解码错误，跳过 {file_path} 的某一行。")
    
    return data, labels

# 分句
def split_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

# 依存分析
def analyze_dependencies(sentences):
    parsed_sentences = []
    for sent in sentences:
        doc = nlp(sent)
        root_count = sum(1 for token in doc if token.dep_ == 'ROOT')
        subject_count = sum(1 for token in doc if token.dep_ in ['nsubj', 'nsubjpass'])
        object_count = sum(1 for token in doc if token.dep_ in ['dobj', 'pobj'])
        tree_depth = max(token.i for token in doc) - min(token.i for token in doc) if len(doc) > 1 else 1
        parsed_sentences.append((sent, root_count, subject_count, object_count, tree_depth))
    return parsed_sentences

# 逻辑重排
def reorder_sentences_using_dependencies(text):
    sentences = split_sentences(text)
    dependency_parsed = analyze_dependencies(sentences)
    reordered = sorted(
        dependency_parsed,
        key=lambda x: (x[1] + x[2] + x[3], -x[4]), 
        reverse=True
    )
    return " ".join([sent[0] for sent in reordered])

# 读取所有数据
all_data = []
all_labels = []

for label, file_path in json_files.items():
    label_val = 0 if "init" in label else 1  # 0: init, 1: generation
    data, labels = load_and_process_jsonl(file_path, label_val)
    all_data.extend(data)
    all_labels.extend(labels)

# 转换为 NumPy 数组
data = np.array(all_data)
labels = np.array(all_labels)

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
