import random
import numpy as np
from sentence_transformers import SentenceTransformer, util

# 1. 加载英文 Sentence Transformer 模型
model = SentenceTransformer("sentence-transformers/all-roberta-large-v1")  # 英文句嵌入模型

# 2. 定义文本
#original_text = "To solve the problems of the data reliability for NAND flash storages, a variable-node-based belief-propagation with message pre-processing (VNBP-MP) decoding algorithm for binary low-density parity-check (LDPC) codes is proposed."
original_text="In this paper, we propose a simple yet effective black-box zero-shot detection approach based on the observation that, from the perspective of LLMs, human-written texts typically contain more grammatical errors than LLM-generated texts."
# 3. 定义随机打乱函数
def shuffle_text(text):
    """
    随机打乱文本，使其相似度尽可能降低
    - 确保单词之间仍然有空格，不会粘连
    """
    words = text.split()  # 分割单词
    random.shuffle(words)  # 随机打乱顺序
    return " ".join(words)  # 重新组合单词，保持空格

# 4. 进行 N 次打乱实验，并计算相似度
N = 200  # 设定实验次数
min_similarity = float("inf")
best_shuffled_text = None

original_embedding = model.encode(original_text, convert_to_tensor=True)  # 原文本向量

for _ in range(N):
    shuffled_text = shuffle_text(original_text)
    shuffled_embedding = model.encode(shuffled_text, convert_to_tensor=True)

    # 计算余弦相似度（越低越不同）
    similarity = util.pytorch_cos_sim(original_embedding, shuffled_embedding).item()

    # 记录最小相似度的文本
    if similarity < min_similarity:
        min_similarity = similarity
        best_shuffled_text = shuffled_text

# 5. 输出结果
print(f"Original Text: {original_text}")
print(f"Shuffled Text with Lowest Similarity: {best_shuffled_text}")
print(f"Lowest Similarity Score: {min_similarity:.4f}")
