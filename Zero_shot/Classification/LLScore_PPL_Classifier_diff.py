import json 
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载 GPT-2 预训练模型
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

tokenizer.pad_token = tokenizer.eos_token  # 设置 PAD token

# 创建输出文件夹
output_dir = "classification_results_diff"
os.makedirs(output_dir, exist_ok=True)

# 文件路径（请修改为实际路径）
json_files = {
    "init": "/home/jxy/Data/Zero_shot/llscore_ppl/ieee-init_llscore_ppl.jsonl",
    "generation": "/home/jxy/Data/Zero_shot/llscore_ppl/ieee-chatgpt-generation_llscore_ppl.jsonl",
    "init_random": "/home/jxy/Data/Zero_shot/llscore_ppl_random/ieee-init_random_llscore_ppl.jsonl",
    "generation_random": "/home/jxy/Data/Zero_shot/llscore_ppl_random/ieee-chatgpt-generation_random_llscore_ppl.jsonl"
}

# 读取 JSONL 文件并构建数据
def load_jsonl(file_path):
    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            data[entry["id"]] = [entry["LLScore"], entry["PPL"]]
    return data

# 加载数据
init_data = load_jsonl(json_files["init"])
generation_data = load_jsonl(json_files["generation"])
init_random_data = load_jsonl(json_files["init_random"])
generation_random_data = load_jsonl(json_files["generation_random"])

# 计算 PPL 和 LLScore 变化值
data = []
labels = []

for key in init_data.keys():
    if key in init_random_data:
        llscore_diff = init_random_data[key][0] - init_data[key][0]
        ppl_diff = init_random_data[key][1] - init_data[key][1]
        data.append([llscore_diff, ppl_diff])
        labels.append(0)  # 0: init

for key in generation_data.keys():
    if key in generation_random_data:
        llscore_diff = generation_random_data[key][0] - generation_data[key][0]
        ppl_diff = generation_random_data[key][1] - generation_data[key][1]
        data.append([llscore_diff, ppl_diff])
        labels.append(1)  # 1: generation

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
y_prob = clf.predict_proba(X_test)[:, 1]  # 获取概率分数

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("分类准确率:", accuracy)

# 计算 ROC-AUC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print("AUROC:", roc_auc)

# 计算分类报告
print("分类报告:")
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# 生成分类指标可视化
def plot_classification_report(report):
    categories = list(report.keys())[:-3]  # 过滤掉 'accuracy', 'macro avg', 'weighted avg'
    precision = [report[c]["precision"] for c in categories]
    recall = [report[c]["recall"] for c in categories]
    f1_score = [report[c]["f1-score"] for c in categories]
    
    x = np.arange(len(categories))
    width = 0.25
    
    plt.figure(figsize=(8, 6))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1_score, width, label="F1-score")
    
    plt.xlabel("Categories")
    plt.ylabel("Score")
    plt.title("Classification Report Metrics")
    plt.xticks(ticks=x, labels=categories)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "classification_report_metrics.svg"))
    plt.close()

plot_classification_report(report)

# 生成混淆矩阵图像
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Init', 'Generation'], yticklabels=['Init', 'Generation'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(output_dir, "confusion_matrix.svg"))
plt.close()

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_dir, "roc_curve.svg"))
plt.close()

# 绘制分类结果可视化
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap="coolwarm", alpha=0.7)
plt.xlabel("LLScore Change")
plt.ylabel("PPL Change")
plt.title("Classification based on LLScore & PPL Changes")
plt.colorbar(label="Predicted Label (0: Init, 1: Generation)")
plt.savefig(os.path.join(output_dir, "classification_result.svg"))
plt.close()