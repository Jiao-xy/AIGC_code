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

# 设置设备为CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# 创建结果保存目录
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# 加载模型（必须在函数定义前）
print("加载模型中...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # 禁用不需要的组件加速处理
bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device="cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
model.eval()
tokenizer.pad_token = tokenizer.eos_token

# 依存分析相关函数
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

# 辅助函数
def count_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def convert_floats(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().item()
    elif isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats(elem) for elem in obj]
    return obj

# 核心处理函数
def compute_llscore(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return float(-outputs.loss.item() * inputs.input_ids.shape[1])

def compute_bert_similarity(original, reordered):
    vec_orig = bert_model.encode([original], convert_to_tensor=True)
    vec_reorder = bert_model.encode([reordered], convert_to_tensor=True)
    return float(cosine_similarity(vec_orig.cpu().numpy(), vec_reorder.cpu().numpy())[0, 0])

def split_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

# 文件路径（保持原拼写）
file_path_human = "/home/jxy/Data/ReoraganizationData/init/ieee-init.jsonl"
file_path_generated = "/home/jxy/Data/ReoraganizationData/init/ieee-chatgpt-generation.jsonl"

# 主处理流程
print("开始处理数据...")
data_samples = []
labels = []
processed_data = []

for file_path, label in [(file_path_human, 1), (file_path_generated, 0)]:
    line_count = count_lines(file_path)
    with open(file_path, "r", encoding="utf-8") as file:
        for line in tqdm(file, total=line_count, desc=f"Processing {os.path.basename(file_path)}"):
            try:
                data = json.loads(line.strip())
                abstract = data.get("abstract", "").strip()
                if not abstract:
                    continue
                    
                sentences = split_sentences(abstract)
                dependency_parsed = analyze_dependencies(sentences)
                reordered_text = reorder_sentences(dependency_parsed)
                
                llscore = compute_llscore(abstract)
                similarity = compute_bert_similarity(abstract, reordered_text)
                
                data_samples.append((llscore, similarity))
                labels.append(label)
                processed_data.append({
                    "original": abstract,
                    "reordered": reordered_text,
                    "LLScore": llscore,
                    "RScore": similarity
                })
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                continue

# 保存中间结果
with open(os.path.join(output_dir, "processed_data.json"), "w", encoding="utf-8") as f:
    json.dump(convert_floats(processed_data), f, ensure_ascii=False, indent=4)

# 分类分析
print("开始分类分析...")
X = np.array(data_samples)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

# 保存模型信息
weight_details = {
    "feature_names": ["LLScore", "RScore"],
    "coefficients": clf.coef_.tolist()[0],
    "intercept": float(clf.intercept_[0])
}
with open(os.path.join(output_dir, "logistic_regression_weights.json"), "w", encoding="utf-8") as f:
    json.dump(convert_floats(weight_details), f, ensure_ascii=False, indent=4)

# 评估模型
y_pred = clf.predict(X_test)
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_recall_fscore_support(y_test, y_pred, average='binary')[0],
    "recall": precision_recall_fscore_support(y_test, y_pred, average='binary')[1],
    "f1": precision_recall_fscore_support(y_test, y_pred, average='binary')[2]
}
with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    json.dump(convert_floats(metrics), f, indent=4)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig(os.path.join(output_dir, "roc_curve.png"))
plt.close()

print("所有处理完成！结果保存在", output_dir)