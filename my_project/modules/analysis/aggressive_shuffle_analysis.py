# modules/analysis/aggressive_shuffle_analysis.py
# 用于实验性地生成句子打乱版本并评估其语义与结构变化

import random
import numpy as np
from collections import Counter
from nltk import ngrams
from nltk.metrics import edit_distance
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import torch

# ✅ 初始化嵌入模型（对词序更敏感的模型）
model = SentenceTransformer("hkunlp/instructor-large", device="cuda" if torch.cuda.is_available() else "cpu")

# ✅ 改写后的打乱函数：不再使用 [MASK] 替换

def aggressive_shuffle(text, p_char=0.2):
    """
    混合打乱策略（无 mask）：
    - 随机打乱单词顺序
    - 按概率 p_char 打乱词内部字符（长度>3）
    """
    words = text.split()

    for i in range(len(words)):
        if random.random() < p_char and len(words[i]) > 3:
            words[i] = ''.join(random.sample(words[i], len(words[i])))

    random.shuffle(words)
    return ' '.join(words)


def evaluate_dissimilarity(original, shuffled):
    """计算三个维度的相似度/差异度：语义、bigram、编辑距离"""
    # 嵌入语义相似度
    emb_orig = model.encode(original, convert_to_tensor=True)
    emb_shuf = model.encode(shuffled, convert_to_tensor=True)
    semantic_sim = util.pytorch_cos_sim(emb_orig, emb_shuf).item()

    # bigram 相似度
    def ngram_overlap(a, b, n=2):
        a_ngrams = Counter(ngrams(a.split(), n))
        b_ngrams = Counter(ngrams(b.split(), n))
        intersection = sum((a_ngrams & b_ngrams).values())
        return intersection / max(len(a_ngrams), 1)

    bigram_sim = ngram_overlap(original, shuffled)

    # 编辑距离（归一化）
    def norm_edit_distance(a, b):
        m, n = len(a.split()), len(b.split())
        max_len = max(m, n)
        return edit_distance(a.split(), b.split()) / max_len if max_len > 0 else 0

    # 组合差异分数（越大越不相似）
    return {
        "semantic_similarity": semantic_sim,
        "bigram_similarity": bigram_sim,
        "combined_score": 0.6 * (1 - semantic_sim) + 0.3 * (1 - bigram_sim) + 0.1 * norm_edit_distance(original, shuffled)
    }


def run_experiment(text, n_trials=200):
    results = []
    min_score = float('inf')
    best_version = None

    for _ in range(n_trials):
        shuffled = aggressive_shuffle(text)
        scores = evaluate_dissimilarity(text, shuffled)
        results.append(scores)

        if scores["combined_score"] < min_score:
            min_score = scores["combined_score"]
            best_version = shuffled

    # 可视化分布
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist([r["semantic_similarity"] for r in results], bins=20)
    ax[0].set_title("Semantic Similarity")

    ax[1].hist([r["bigram_similarity"] for r in results], bins=20)
    ax[1].set_title("Bigram Overlap")

    ax[2].hist([r["combined_score"] for r in results], bins=20)
    ax[2].set_title("Combined Dissimilarity")

    plt.tight_layout()
    plt.show()

    return best_version, min_score, results


if __name__ == "__main__":
    original_text = "In this paper, we propose a simple yet effective black-box zero-shot detection approach based on the observation that, from the perspective of LLMs, human-written texts typically contain more grammatical errors than LLM-generated texts."

    best_shuffled, min_score, results = run_experiment(original_text)

    print("\n✅ Original:", original_text)
    print("🔀 Best Shuffled:", best_shuffled)
    print(f"📉 Min Combined Score: {min_score:.4f}")

    # 简要统计
    avg_sem = np.mean([r["semantic_similarity"] for r in results])
    avg_bi = np.mean([r["bigram_similarity"] for r in results])
    print("\n📊 Average Similarity (200 Trials):")
    print(f"Semantic: {avg_sem:.4f}, Bigram: {avg_bi:.4f}")