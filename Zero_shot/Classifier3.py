import json
import os
import numpy as np
import spacy
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.model_selection import train_test_split

# **设置 GPU 设备**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# **创建结果保存目录**
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# **加载 GPT-2 预训练模型（用于计算 LLScore）**
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()
tokenizer.pad_token = tokenizer.eos_token

# **加载 SpaCy 和 BERT（用于计算 RScore）**
nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

# **文件路径**
file_path_human = "/home/jxy/Data/ReoraganizationData/init/ieee-init.jsonl"
file_path_generated = "/home/jxy/Data/ReoraganizationData/init/ieee-chatgpt-generation.jsonl"

# **文本分割**
def split_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

# **计算 LLScore（在 GPU 运行）**
def compute_llscore(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return -outputs.loss.cpu().item() * inputs.input_ids.shape[1]

# **计算 BERT 余弦相似度（在 GPU 运行）**
def compute_bert_similarity(original, reordered):
    vec_orig = bert_model.encode([original], convert_to_tensor=True, device=device)
    vec_reorder = bert_model.encode([reordered], convert_to_tensor=True, device=device)
    return cosine_similarity(vec_orig.cpu().numpy(), vec_reorder.cpu().numpy())[0, 0]

# **加载数据（仅取十分之一数据进行测试）**
def load_jsonl(file_path, limit=None):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line.strip()) for i, line in enumerate(file) if limit is None or i < limit]

limit = 10000  # 只处理前 500 条数据
print(f"仅处理前 {limit} 条数据进行测试...")
data_human = load_jsonl(file_path_human, limit=limit)
data_generated = load_jsonl(file_path_generated, limit=limit)

# **处理数据**
all_texts = [item["abstract"] for item in data_human + data_generated]
print("计算 LLScore 和 RScore...")
llscores = [compute_llscore(text) for text in tqdm(all_texts, desc="Computing LLScore")]
rscores = [compute_bert_similarity(text, " ".join(split_sentences(text)[:5])) for text in tqdm(all_texts, desc="Computing RScore")]

# **存储数据**
data_samples = np.column_stack((llscores, rscores))
labels = np.array([1] * len(data_human) + [0] * len(data_generated))

# **划分训练集和测试集**
X_train, X_test, y_train, y_test = train_test_split(data_samples, labels, test_size=0.2, random_state=42)
print("数据划分完成，训练集大小: {}，测试集大小: {}".format(len(X_train), len(X_test)))

# **训练逻辑回归模型**
clf = LogisticRegression()
clf.fit(X_train, y_train)

# **测试集分类预测**
y_pred_test = clf.predict(X_test)

# **计算分类性能指标**
test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='binary')
print(f"测试集分类准确率: {test_accuracy:.4f}")
print(f"测试集精确率: {test_precision:.4f}, 召回率: {test_recall:.4f}, F1-score: {test_f1:.4f}")

# **绘制并保存 ROC 曲线**
def plot_and_save_roc(y_true, scores, filename, title):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, filename))
    print(f"ROC 曲线已保存至 {output_dir}/{filename}")
    plt.show()

plot_and_save_roc(y_test, clf.predict_proba(X_test)[:, 1], "roc_curve_test.png", "ROC Curve (Test)")

print("所有任务已完成！")