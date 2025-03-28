import random
import numpy as np
from collections import Counter
from nltk import ngrams
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import torch

# 初始化模型（使用对词序更敏感的模型）
model = SentenceTransformer("hkunlp/instructor-large", device="cuda" if torch.cuda.is_available() else "cpu")

# 示例文本
original_text = "In this paper, we propose a simple yet effective black-box zero-shot detection approach based on the observation that, from the perspective of LLMs, human-written texts typically contain more grammatical errors than LLM-generated texts."

# 增强版打乱函数
def aggressive_shuffle(text, p_mask=0.3, p_char=0.2):
    """
    混合打乱策略：
    - 随机打乱单词顺序
    - 按概率p_mask替换单词为[MASK]
    - 按概率p_char打乱单词内部字符
    """
    words = text.split()
    
    # 词汇级干扰
    for i in range(len(words)):
        if random.random() < p_mask:
            words[i] = "[MASK]"
        elif random.random() < p_char and len(words[i]) > 3:
            words[i] = ''.join(random.sample(words[i], len(words[i])))
    
    random.shuffle(words)
    return ' '.join(words)

# 多维度相似度评估
def evaluate_dissimilarity(original, shuffled):
    # 语义相似度（BERT类模型）
    emb_orig = model.encode(original, convert_to_tensor=True)
    emb_shuf = model.encode(shuffled, convert_to_tensor=True)
    semantic_sim = util.pytorch_cos_sim(emb_orig, emb_shuf).item()
    
    # N-gram重叠度（捕捉词序）
    def ngram_overlap(a, b, n=2):
        a_ngrams = Counter(ngrams(a.split(), n))
        b_ngrams = Counter(ngrams(b.split(), n))
        intersection = sum((a_ngrams & b_ngrams).values())
        return intersection / max(len(a_ngrams), 1)
    
    bigram_sim = ngram_overlap(original, shuffled)
    
    # 编辑距离归一化
    def norm_edit_distance(a, b):
        m, n = len(a.split()), len(b.split())
        max_len = max(m, n)
        if max_len == 0: return 0
        return edit_distance(a.split(), b.split()) / max_len
    
    # 综合评分（可调整权重）
    return {
        "semantic_similarity": semantic_sim,
        "bigram_similarity": bigram_sim,
        "combined_score": 0.6*(1-semantic_sim) + 0.3*(1-bigram_sim) + 0.1*norm_edit_distance(original, shuffled)
    }

# 实验运行
def run_experiment(text, n_trials=200):
    results = []
    min_combined = float('inf')
    best_shuffled = None
    
    for _ in range(n_trials):
        shuffled = aggressive_shuffle(text)
        metrics = evaluate_dissimilarity(text, shuffled)
        results.append(metrics)
        
        if metrics["combined_score"] < min_combined:
            min_combined = metrics["combined_score"]
            best_shuffled = shuffled
    
    # 可视化
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist([r["semantic_similarity"] for r in results], bins=20)
    ax[0].set_title("Semantic Similarity")
    
    ax[1].hist([r["bigram_similarity"] for r in results], bins=20)
    ax[1].set_title("Bigram Overlap")
    
    ax[2].hist([r["combined_score"] for r in results], bins=20)
    ax[2].set_title("Combined Dissimilarity")
    plt.show()
    
    return best_shuffled, min_combined, results

# 执行实验
best_shuffled, min_score, all_results = run_experiment(original_text)

# 打印最佳结果
print("Original:", original_text)
print("Best Shuffled:", best_shuffled)
print(f"Min Combined Score: {min_score:.4f}")

# 分析统计
avg_semantic = np.mean([r["semantic_similarity"] for r in all_results])
avg_bigram = np.mean([r["bigram_similarity"] for r in all_results])
print(f"\nAverage Scores Across {len(all_results)} Trials:")
print(f"Semantic Similarity: {avg_semantic:.4f}")
print(f"Bigram Similarity: {avg_bigram:.4f}")