import random
import numpy as np
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from scipy.stats import entropy

# 加载 RoBERTa 预训练模型
model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 输入文本
text = "人工智能和深度学习是当今科技领域的热门话题，特别是在自然语言处理领域。"

def get_token_embeddings(input_text):
    """ 获取文本的 Token 级别词向量 """
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state.squeeze(0).numpy()  # (seq_length, hidden_dim)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  # 获取 Token
    
    # 过滤掉特殊 Token
    valid_tokens, valid_embeddings = [], []
    for token, embedding in zip(tokens, token_embeddings):
        if token not in ["[CLS]", "[SEP]", "<s>", "</s>"]:  
            valid_tokens.append(token)
            valid_embeddings.append(embedding)
    
    return valid_tokens, np.array(valid_embeddings)

def compute_kmeans_entropy(embeddings, num_clusters=6):
    """ 计算 KMeans 聚类后类别分布的熵值 """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # 统计类别数量
    label_counts = Counter(labels)
    total_tokens = len(labels)
    probabilities = np.array([count / total_tokens for count in label_counts.values()])

    # 计算熵值
    return entropy(probabilities, base=2), labels

def shuffle_text(text):
    """ 按字符随机打乱文本 """
    return "".join(random.sample(text, len(text)))

# 计算原始文本的 Token 词向量
original_tokens, original_embeddings = get_token_embeddings(text)

# 计算原始文本的 KMeans 熵值
original_entropy, original_labels = compute_kmeans_entropy(original_embeddings)
print(f"原始文本的聚类熵值: {original_entropy:.4f}")

# 生成随机打乱的文本
shuffled_text = shuffle_text(text)
print("\n打乱后的文本:", shuffled_text)

# 计算打乱文本的 Token 词向量
shuffled_tokens, shuffled_embeddings = get_token_embeddings(shuffled_text)

# 计算打乱文本的 KMeans 熵值
shuffled_entropy, shuffled_labels = compute_kmeans_entropy(shuffled_embeddings)
print(f"打乱文本的聚类熵值: {shuffled_entropy:.4f}")

# 输出 Token 归类信息
def print_cluster_info(tokens, labels, title):
    """ 按类别输出 Token """
    print(f"\n{title}")
    cluster_dict = {i: [] for i in set(labels)}
    for token, label in zip(tokens, labels):
        cluster_dict[label].append(token)
    
    for cluster, tokens in cluster_dict.items():
        print(f"类别 {cluster}: {tokens}")

# 输出聚类结果
print_cluster_info(original_tokens, original_labels, "原始文本的 Token 聚类")
print_cluster_info(shuffled_tokens, shuffled_labels, "打乱文本的 Token 聚类")
