import json
import os
import numpy as np
import spacy
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
from scipy.optimize import minimize

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

# **Softmax 归一化**
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

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

# **加载数据**
def load_jsonl(file_path, limit=None):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line.strip()) for i, line in enumerate(file) if limit is None or i < limit]

limit = 10000  # 只处理前 10000 条数据
print(f"仅处理前 {limit} 条数据进行测试...")
data_human = load_jsonl(file_path_human, limit=limit)
data_generated = load_jsonl(file_path_generated, limit=limit)

# **处理数据**
all_texts = [item["abstract"] for item in data_human + data_generated]
print("计算 LLScore 和 RScore...")
llscores = [compute_llscore(text) for text in tqdm(all_texts, desc="Computing LLScore")]

# **计算重排序相似度**
r_reorder_scores = []
for text in tqdm(all_texts, desc="Computing Reordered RScore"):
    sentences = split_sentences(text)
    dependency_parsed = analyze_dependencies(sentences)
    reordered_text = reorder_sentences(dependency_parsed)
    r_reorder_score = compute_bert_similarity(text, reordered_text)
    r_reorder_scores.append(r_reorder_score)

# **Softmax 归一化 RScore**
r_reorder_scores = softmax(np.array(r_reorder_scores))

# **优化权重 p**
def optimize_p(p):
    total_scores = p * r_reorder_scores + (1 - p) * np.array(llscores)
    epsilon = np.percentile(total_scores, 50)
    predictions = (total_scores > epsilon).astype(int)
    return -np.mean(predictions == np.array([1] * len(data_human) + [0] * len(data_generated)))

result = minimize(optimize_p, x0=0.5, bounds=[(0, 1)], method='L-BFGS-B')
p_optimized = result.x[0]
print(f"优化后的 p 值: {p_optimized}")

# **计算 TotalScore**
total_scores = p_optimized * r_reorder_scores + (1 - p_optimized) * np.array(llscores)

# **确定分类阈值**
epsilon = np.percentile(total_scores, 50)
y_pred = (total_scores > epsilon).astype(int)
y_true = np.array([1] * len(data_human) + [0] * len(data_generated))

# **计算 ROC 曲线**
fpr, tpr, _ = roc_curve(y_true, total_scores)
roc_auc = auc(fpr, tpr)

# **绘制并保存 ROC 曲线**
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig(os.path.join(output_dir, "roc_curve.svg"), format='svg')
plt.close()

print("所有任务已完成！ROC 曲线已保存为 SVG 矢量图。")
