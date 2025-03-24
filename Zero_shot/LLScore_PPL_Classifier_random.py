import json 
import torch
import numpy as np
import os
import random
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
def load_and_process_jsonl(file_path):
    data = {}
    
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                entry = json.loads(line.strip())
                abstract = entry.get("abstract", "").strip()
                doc_id = entry.get("id", None)
                if abstract and doc_id:
                    llscore, ppl = compute_llscore_ppl(abstract)
                    data[doc_id] = (llscore, ppl)
            except json.JSONDecodeError:
                print(f"JSON 解码错误，跳过 {file_path} 的某一行。")
    
    return data

# 处理文本，打乱摘要中的句子顺序
def shuffle_abstract_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data_list = [json.loads(line) for line in f]
    
    for data in data_list:
        abstract = data.get("abstract", "")
        sentences = abstract.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        random.shuffle(sentences)
        shuffled_abstract = '. '.join(sentences) + '.' if sentences else ""
        data["abstract"] = shuffled_abstract
    
    base, ext = os.path.splitext(file_path)
    new_file_path = f"{base}_random{ext}"
    
    with open(new_file_path, 'w', encoding='utf-8') as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Processed file saved as: {new_file_path}")
    return new_file_path

# 读取所有数据
data = []
labels = []

for label, file_path in json_files.items():
    label_val = 0 if "init" in label else 1  # 0: init, 1: generation
    shuffled_file_path = shuffle_abstract_sentences(file_path)
    original_data = load_and_process_jsonl(file_path)
    shuffled_data = load_and_process_jsonl(shuffled_file_path)
    
    for doc_id in original_data:
        if doc_id in shuffled_data:
            llscore_orig, ppl_orig = original_data[doc_id]
            llscore_shuf, ppl_shuf = shuffled_data[doc_id]
            llscore_diff = llscore_shuf - llscore_orig
            ppl_diff = ppl_shuf - ppl_orig
            data.append([llscore_orig, ppl_orig, llscore_shuf, ppl_shuf, llscore_diff, ppl_diff])
            labels.append(label_val)

# 转换为 NumPy 数组
data = np.array(data)
labels = np.array(labels)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

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
plt.scatter(X_test[:, 4], X_test[:, 5], c=y_pred, cmap="coolwarm", alpha=0.7)
plt.xlabel("LLScore Change")
plt.ylabel("PPL Change")
plt.title("Classification of Init vs. Generation based on LLScore & PPL Changes")
plt.colorbar(label="Predicted Label (0: Init, 1: Generation)")
plt.savefig("classification_result.svg")
plt.close()
