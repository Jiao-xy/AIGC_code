from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import torch
import spacy
import numpy as np
import math
from collections import defaultdict, Counter

# 1. 加载 BERT 英文预训练模型
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. 加载 spaCy 进行英文分词
nlp = spacy.load("en_core_web_sm")

def tokenize(text):
    """ 使用 spaCy 进行英文分词 """
    return [token.text for token in nlp(text) if token.is_alpha]  # 仅保留字母单词

def calculate_entropy(probabilities):
    """计算熵 H(X) = -sum(p * log2(p))"""
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

# 3. 处理输入文本
text = """Artificial intelligence and deep learning are hot topics in today's technology field, 
especially in the field of natural language processing. Researchers are constantly improving 
machine learning models to achieve better accuracy and efficiency in NLP tasks."""
text="""processing Researchersmachinelearning  Artificialtodayhot,in  field deeptasks languageconstantlytopicslearningbetter to
  .   improvingareachieve
 NLP      efficiencyespecially  technologyin' infieldareintelligencemodels
  snaturaloftheand   . accuracyand
"""
# 4. 分词
words = tokenize(text)
print("分词结果:", words)

# 5. 计算 Token 级别的词向量
word_embeddings = []
for word in words:
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    word_vec = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # 取平均作为词向量
    word_embeddings.append(word_vec)

# 6. 转换为 NumPy 数组
word_embeddings = np.array(word_embeddings)

# 7. 使用 K-Means 进行聚类
num_clusters = 5  # 设定聚类数目
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(word_embeddings)

# 8. 输出聚类结果
cluster_dict = defaultdict(list)
for word, label in zip(words, labels):
    cluster_dict[label].append(word)

# 9. 计算聚类熵
cluster_sizes = Counter(labels)  # 统计每个聚类的大小
total_words = len(words)
cluster_probabilities = [count / total_words for count in cluster_sizes.values()]
cluster_entropy = calculate_entropy(cluster_probabilities)

# 10. 输出结果
print("\n===== 聚类结果 =====")
for cluster, tokens in cluster_dict.items():
    print(f"类别 {cluster}: {tokens}")

print("\n===== 统计信息 =====")
print(f"总词数: {total_words}")
print(f"聚类熵（Cluster Entropy）: {cluster_entropy:.4f}")
