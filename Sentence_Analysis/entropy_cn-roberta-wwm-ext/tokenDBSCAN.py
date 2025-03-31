from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import DBSCAN
import torch
import numpy as np

# 1. 加载 RoBERTa 预训练模型
model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. 定义输入文本并按字分割
text = "人工智能和深度学习是当今科技领域的热门话题，特别是在自然语言处理领域。"
characters = list(text)  # 按字拆分
print("拆分后的字符:", characters)

# 3. 计算 Token 级别词向量
char_embeddings = []
for char in characters:
    inputs = tokenizer(char, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    char_vec = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # 取平均作为词向量
    char_embeddings.append(char_vec)

# 4. 转换为 NumPy 数组
char_embeddings = np.array(char_embeddings)

# 5. 使用 DBSCAN 进行自动聚类
dbscan = DBSCAN(eps=0.5, min_samples=2, metric='cosine')  # eps 取决于向量相似度范围
labels = dbscan.fit_predict(char_embeddings)

# 6. 输出聚类结果
for char, label in zip(characters, labels):
    print(f"字符: {char} -> 聚类类别: {label}")
