from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import torch
import jieba
import numpy as np
from collections import defaultdict

# 1. 加载 RoBERTa 中文预训练模型
model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. 处理输入文本
text = "人工智能和深度学习是当今科技领域的热门话题，特别是在自然语言处理领域。"
words = list(jieba.cut(text))  # 先进行分词
print("分词结果:", words)

# 3. 计算 Token 级别词向量
word_embeddings = []
for word in words:
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    word_vec = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # 取平均作为词向量
    word_embeddings.append(word_vec)

# 4. 转换为 NumPy 数组
word_embeddings = np.array(word_embeddings)

# 5. 使用 K-Means 进行聚类
num_clusters = 5  # 设定聚类数目
kmeans = KMeans(n_clusters=num_clusters, random_state=42,n_init=10)#
labels = kmeans.fit_predict(word_embeddings)

# 6. 输出聚类结果

for word, label in zip(words, labels):
    print(f"词: {word} -> 聚类类别: {label}")

# 8. 将 Token 按类别归类
cluster_dict = defaultdict(list)
for token, label in zip(words, labels):
    cluster_dict[label].append(token)

# 9. 输出聚类结果
for cluster, tokens in cluster_dict.items():
    print(f"类别 {cluster}: {tokens}")