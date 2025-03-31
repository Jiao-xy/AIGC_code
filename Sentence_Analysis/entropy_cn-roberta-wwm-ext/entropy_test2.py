import nltk
import numpy as np
import math
import re
from collections import Counter
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

def tokenize(text):
    """使用 TreebankWordTokenizer 进行英文分词，避免 nltk.word_tokenize 的 punkt 依赖问题"""
    return tokenizer.tokenize(text)

def calculate_entropy(probabilities):
    """计算熵 H(X) = -sum(p * log2(p))"""
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def global_entropy(tokens):
    """计算全局词频熵"""
    counter = Counter(tokens)
    total = sum(counter.values())
    probabilities = [count / total for count in counter.values()]
    return calculate_entropy(probabilities)

def window_entropy(tokens, window_size=10):
    """计算滑动窗口熵，默认窗口大小为10"""
    num_windows = max(1, len(tokens) - window_size + 1)
    entropies = []
    
    for i in range(num_windows):
        window = tokens[i:i + window_size]
        counter = Counter(window)
        probabilities = [count / len(window) for count in counter.values()]
        entropies.append(calculate_entropy(probabilities))
    
    return entropies, np.mean(entropies), np.std(entropies)

def sentence_entropy(text):
    """计算按句子分割的熵"""
    sentences = re.split(r'[.!?]', text)  # 以标点符号分割句子
    sentences = [s.strip() for s in sentences if s.strip()]
    entropies = []
    
    for sentence in sentences:
        tokens = tokenize(sentence)
        if tokens:
            counter = Counter(tokens)
            probabilities = [count / len(tokens) for count in counter.values()]
            entropies.append(calculate_entropy(probabilities))
    
    return entropies, np.mean(entropies), np.var(entropies)

def main():
    # 示例文本
    text = """andrepresentation-based robustness summarized. We propose an intuitive approach based on contrastive augmentation to mitigate the performance gap in fine-grained text-to-text transfer. Our method incorporates both supervised learning and unsupervised downstream learning strategies, aiming to achieve a balance between interpretability and effectiveness. Through extensive experiments, we demonstrate that our approach significantly improves performance across various tasks while maintaining simplicity.
Our contributions can be summarized as follows:
Task-Oriented Representation Learning: We propose an effective framework for transferring learned representations, particularly focusing on scenarios where labeled data is scarce.
Contrastive Augmentation Approach: By incorporating contrastive learning techniques, we enhance the robustness of our approach and achieve better performance in downstream tasks such as text classification (e.g., STS) and machine translation.
Extensive experiments conducted on standard benchmarks like RTE, SST-2, MNLI, and NLI show that our method achieves state-of-the-art performance across these datasets. The simplicity of our approach is further validated by its effectiveness when applied to the task of summarizing text. These results demonstrate the potential of our proposed framework for a wide range of applications in natural language processing.
"""
    text="""Our contributions can be summarized as follows: 1) We propose a simple but effective sentence-level training objective based on contrastive learning. It mitigates the collapse of BERT-derived representations and transfers them to downstream tasks. 2) We explore various effective text augmentation strategies to generate views for contrastive learning and analyze their effects on unsupervised sentence representation transfer. 3) With only fine-tuning on unsupervised target datasets, our approach achieves significant improvement on STS tasks. When further incorporating with NLI supervision, our approach achieves new state-of-the-art performance. We also show the robustness of our approach in data scarcity scenarios and intuitive analysis of the transferred representations.
"""
    # 1. 分词
    tokens = tokenize(text)

    # 2. 计算全局熵
    global_H = global_entropy(tokens)

    # 3. 计算窗口熵
    window_H_list, window_mean, window_std = window_entropy(tokens, window_size=10)

    # 4. 计算句子熵
    sentence_H_list, sentence_mean, sentence_var = sentence_entropy(text)

    # 5. 输出结果
    print(f"Global entropy: {global_H:.4f}")
    print(f"Window entropy mean: {window_mean:.4f}, standard deviation: {window_std:.4f}")
    print(f"Sentence entropy mean: {sentence_mean:.4f}, variance: {sentence_var:.4f}")

if __name__ == "__main__":
    main()
