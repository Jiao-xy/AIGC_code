import json
import os
import numpy as np
import spacy
import torch
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.model_selection import train_test_split

# **创建结果保存目录**
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# **加载 GPT-2 预训练模型**
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()
tokenizer.pad_token = tokenizer.eos_token  # 处理 padding token

# **加载 SpaCy 依存分析模型和 BERT 句向量模型**
nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# **文件路径**
file_path_human = "/home/jxy/Data/ReoraganizationData/init/ieee-init.jsonl"
file_path_generated = "/home/jxy/Data/ReoraganizationData/init/ieee-chatgpt-generation.jsonl"

print("开始处理数据...")

# **文本分割**
def split_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

# **依存分析与逻辑重构**
def reorder_sentences(sentences):
    parsed_sentences = []
    for sent in sentences:
        doc = nlp(sent)
        root_count = sum(1 for token in doc if token.dep_ == 'ROOT')
        subject_count = sum(1 for token in doc if token.dep_ in ['nsubj', 'nsubjpass'])
        object_count = sum(1 for token in doc if token.dep_ in ['dobj', 'pobj'])
        tree_depth = max(token.i for token in doc) - min(token.i for token in doc) if len(doc) > 1 else 1
        parsed_sentences.append((sent, root_count, subject_count, object_count, tree_depth))
    reordered = sorted(parsed_sentences, key=lambda x: (x[1] + x[2] + x[3], -x[4]), reverse=True)
    return " ".join([sent[0] for sent in reordered])

# **计算 LLScore**
def compute_llscore(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return -outputs.loss.item() * inputs.input_ids.shape[1]

# **计算 BERT 余弦相似度**
def compute_bert_similarity(original, reordered):
    vec_orig = bert_model.encode([original], convert_to_tensor=True)
    vec_reorder = bert_model.encode([reordered], convert_to_tensor=True)
    return cosine_similarity(vec_orig.cpu().numpy(), vec_reorder.cpu().numpy())[0, 0]

# **处理数据**
data_samples = []
labels = []
processed_data = []
total_lines = sum(1 for _ in open(file_path_human, "r", encoding="utf-8")) + sum(1 for _ in open(file_path_generated, "r", encoding="utf-8"))

for file_path, label in [(file_path_human, 1), (file_path_generated, 0)]:
    with open(file_path, "r", encoding="utf-8") as file:
        for line in tqdm(file, total=total_lines, desc=f"Processing {file_path}"):
            data = json.loads(line.strip())
            abstract = data.get("abstract", "").strip()
            if abstract:
                sentences = split_sentences(abstract)
                reordered_text = reorder_sentences(sentences)
                llscore = compute_llscore(abstract)
                similarity = compute_bert_similarity(abstract, reordered_text)
                data_samples.append((llscore, similarity))
                labels.append(label)
                processed_data.append({"original": abstract, "reordered": reordered_text, "LLScore": llscore, "RScore": similarity})

# **保存中间数据**
with open(os.path.join(output_dir, "processed_data.json"), "w", encoding="utf-8") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=4)
print("中间数据已保存至 results/processed_data.json")

print("数据处理完成，开始分类分析...")

# **转换数据格式**
data_samples = np.array(data_samples)
labels = np.array(labels)

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
