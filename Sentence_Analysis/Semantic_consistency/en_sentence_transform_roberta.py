import random
import re
from sentence_transformers import SentenceTransformer, util

# 1. ✅ 加载适用于英文的 RoBERTa 语义相似性模型
model = SentenceTransformer("sentence-transformers/all-roberta-large-v1")

# 2. ✅ 定义英文原始文本
original_text = """A man is playing the piano while the audience listens quietly.
The music is soothing and the atmosphere is peaceful.
Suddenly, a dog runs onto the stage, surprising everyone.
The pianist stops playing and looks at the dog curiously.
People in the audience start laughing, enjoying the unexpected event.
"""

# 3. ✅ **完全打乱单词顺序**
def shuffle_words(text):
    words = text.split()  # 分割单词
    random.shuffle(words)  # 打乱单词顺序
    return " ".join(words)  # 重新组合单词，保持空格

shuffled_text_by_words = shuffle_words(original_text)

# 4. ✅ **保持句子顺序，但打乱句子内部的单词**
def shuffle_words_within_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)  # 按标点符号拆分句子
    shuffled_sentences = []

    for sentence in sentences:
        words = sentence.split()  # 分割单词
        random.shuffle(words)  # 只打乱当前句子的单词
        shuffled_sentences.append(" ".join(words))  # 重新组合单词

    return " ".join(shuffled_sentences)  # 保持句子顺序

shuffled_text_within_sentences = shuffle_words_within_sentences(original_text)

# 5. ✅ **计算文本嵌入**
embedding_original = model.encode(original_text, convert_to_tensor=True)
embedding_shuffled_words = model.encode(shuffled_text_by_words, convert_to_tensor=True)
embedding_shuffled_sentences = model.encode(shuffled_text_within_sentences, convert_to_tensor=True)

# 6. ✅ **计算余弦相似度**
similarity_words = util.pytorch_cos_sim(embedding_original, embedding_shuffled_words).item()
similarity_sentences = util.pytorch_cos_sim(embedding_original, embedding_shuffled_sentences).item()

# 7. ✅ **输出结果**
print(f"🔹 Original Text:\n{original_text}")

print(f"\n🔸 Fully Shuffled Words:\n{shuffled_text_by_words}")
print(f"➡ Fully Shuffled Words Similarity: {similarity_words:.4f}")

print(f"\n🔸 Sentence-Preserved Shuffling:\n{shuffled_text_within_sentences}")
print(f"➡ Sentence-Preserved Similarity: {similarity_sentences:.4f}")
