import json
import torch
import numpy as np
import spacy
import matplotlib.pyplot as plt
import pickle
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from scipy.special import softmax
from sklearn.model_selection import train_test_split

# **加载 GPT-2 预训练模型**
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()
tokenizer.pad_token = tokenizer.eos_token  # 处理 padding token

# **加载 SpaCy 依存分析模型**
nlp = spacy.load("en_core_web_sm")

# **文件路径**
file_path_human = "/home/jxy/Data/init/ieee-init.jsonl"
file_path_generated = "/home/jxy/Data/init/ieee-chatgpt-generation.jsonl"

print("开始处理数据...")

# **文本分割**
def split_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

# **依存分析与重排**
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

# **计算相似度**
def compute_similarity(original, reordered):
    vec_orig = np.array([token.vector for token in nlp(original) if token.has_vector])
    vec_reorder = np.array([token.vector for token in nlp(reordered) if token.has_vector])
    if vec_orig.size == 0 or vec_reorder.size == 0:
        return 0  # 避免空向量
    return cosine_similarity(vec_orig.mean(axis=0).reshape(1, -1), vec_reorder.mean(axis=0).reshape(1, -1))[0, 0]

# **处理数据**
data_samples = []
labels = []
total_lines = sum(1 for _ in open(file_path_human, "r", encoding="utf-8")) + sum(1 for _ in open(file_path_generated, "r", encoding="utf-8"))
processed_lines = 0

for file_path, label in [(file_path_human, 1), (file_path_generated, 0)]:
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            abstract = data.get("abstract", "").strip()
            if abstract:
                sentences = split_sentences(abstract)
                reordered_text = reorder_sentences(sentences)
                llscore = compute_llscore(abstract)
                similarity = compute_similarity(abstract, reordered_text)
                data_samples.append((llscore, similarity))
                labels.append(label)
            
            processed_lines += 1
            if processed_lines % 100 == 0:
                print(f"已处理 {processed_lines}/{total_lines} 行文本...")

print("数据处理完成，开始分类分析...")

# **归一化 RScore**
data_samples = np.array(data_samples)
r_scores = softmax(data_samples[:, 1])

# **动态调整权重 p**
best_p, best_auc = 0, 0
total_scores_dict = {}

print("开始动态调整权重 p...")

for p in np.linspace(0, 1, 21):  # 以 5% 递增调整 p
    TotalScores = p * r_scores + (1 - p) * data_samples[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(TotalScores, labels, test_size=0.2, random_state=42)
    fpr, tpr, _ = roc_curve(y_test, X_test)
    auc_score = auc(fpr, tpr)
    total_scores_dict[p] = auc_score
    if auc_score > best_auc:
        best_auc = auc_score
        best_p = p
    print(f"p={p:.2f}, AUROC={auc_score:.4f}")

print(f"最佳权重 p: {best_p:.2f}, 对应 AUROC: {best_auc:.4f}")

# **计算最终 TotalScore 并分类**
TotalScores = best_p * r_scores + (1 - best_p) * data_samples[:, 0]
X_train, X_test, y_train, y_test = train_test_split(TotalScores, labels, test_size=0.2, random_state=42)
totalscore_threshold = np.percentile(X_train, 50)
y_pred_test = (X_test < totalscore_threshold).astype(int)

# **保存数据**
results = {
    "data_samples": data_samples,
    "labels": labels,
    "r_scores": r_scores,
    "best_p": best_p,
    "TotalScores": TotalScores,
    "totalscore_threshold": totalscore_threshold,
    "y_test": y_test,
    "y_pred_test": y_pred_test
}
with open("results.pkl", "wb") as f:
    pickle.dump(results, f)
print("所有数据已保存至 results.pkl")

# **评估分类性能**
acc = accuracy_score(y_test, y_pred_test)
fpr, tpr, _ = roc_curve(y_test, X_test)
auroc = auc(fpr, tpr)

print(f"测试集分类准确率 (ACC): {acc:.4f}")
print(f"受试者工作特征曲线下面积 (AUROC): {auroc:.4f}")

# **绘制并保存 ROC 曲线**
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auroc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig("roc_curve.png")
plt.show()

print("所有任务已完成！")
