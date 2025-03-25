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
import joblib

# **设置 GPU 设备**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# **创建结果保存目录**
output_dir = "Reorder_Classifier_result"
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

# **依存分析并重排序文本**
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

def reorder_sentences(dependency_parsed):
    reordered = sorted(
        dependency_parsed,
        key=lambda x: (x[1] + x[2] + x[3], -x[4]), 
        reverse=True
    )
    return " ".join([sent[0] for sent in reordered])

# **加载数据（仅取十分之一数据进行测试）**
def load_jsonl(file_path, limit=None):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line.strip()) for i, line in enumerate(file) if limit is None or i < limit]

limit = 3000  # 只处理前 10000 条数据
print(f"仅处理前 {limit} 条数据进行测试...")
data_human = load_jsonl(file_path_human, limit=limit)
data_generated = load_jsonl(file_path_generated, limit=limit)

# **处理数据**
all_texts = [item["abstract"] for item in data_human + data_generated]
print("计算 LLScore 和 RScore...")
llscores = [compute_llscore(text) for text in tqdm(all_texts, desc="Computing LLScore")]

# **计算重排序相似度**
rscores = []
r_reorder_scores = []
for text in tqdm(all_texts, desc="Computing RScore & Reordered RScore"):
    sentences = split_sentences(text)
    dependency_parsed = analyze_dependencies(sentences)
    reordered_text = reorder_sentences(dependency_parsed)
    rscore = compute_bert_similarity(text, " ".join(sentences[:5]))
    r_reorder_score = compute_bert_similarity(text, reordered_text)
    rscores.append(rscore)
    r_reorder_scores.append(r_reorder_score)

# **存储数据**
data_samples = np.column_stack((llscores, rscores, r_reorder_scores))
labels = np.array([1] * len(data_human) + [0] * len(data_generated))
np.save(os.path.join(output_dir, "data_samples.npy"), data_samples)  # 存储特征数据
np.save(os.path.join(output_dir, "labels.npy"), labels)  # 存储标签数据

# **划分训练集和测试集**
X_train, X_test, y_train, y_test = train_test_split(data_samples, labels, test_size=0.2, random_state=42)
np.save(os.path.join(output_dir, "X_train.npy"), X_train)  # 训练特征
np.save(os.path.join(output_dir, "X_test.npy"), X_test)  # 测试特征
np.save(os.path.join(output_dir, "y_train.npy"), y_train)  # 训练标签
np.save(os.path.join(output_dir, "y_test.npy"), y_test)  # 测试标签

# **训练逻辑回归模型**
clf = LogisticRegression()
clf.fit(X_train, y_train)

# **保存模型**
joblib.dump(clf, os.path.join(output_dir, "logistic_regression_model.pkl"))  # 逻辑回归模型

# **测试集分类预测**
y_pred_test = clf.predict(X_test)
np.save(os.path.join(output_dir, "y_pred_test.npy"), y_pred_test)  # 预测结果

# **计算分类性能指标并存储**
test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='binary')

evaluation_results = {
    "accuracy": test_accuracy,
    "precision": test_precision,
    "recall": test_recall,
    "f1_score": test_f1
}
with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
    json.dump(evaluation_results, f, indent=4)  # 存储分类指标

print("所有任务已完成！结果已保存至 Reorder_Classifier_result 目录。")
