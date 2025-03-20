import random
import jieba
import re
from sentence_transformers import SentenceTransformer, util

# 1. ✅ 选择更适合中文的 Sentence Transformer 模型
model = SentenceTransformer("shibing624/text2vec-base-chinese")  # 中文相似度模型

# 2. ✅ 定义原始中文文本
original_text = """男子骑无牌助力车被拦撞伤交警(图)
昨天中午12点多，记者在南京新街口洪武路口等红灯时，目睹了一名骑无牌助力车的男子为了躲避交警执法，竟然将交警撞倒的全过程。记者随后从警方了解到，当天在新街口地区，有两名交警在对无牌助力车进行检查时被撞伤，所幸伤势并不严重。
"""

# 3. ✅ **按字打乱（保留标点）**
def shuffle_by_char(text):
    chars = list(re.sub(r"\s+", "", text))  # 去除多余空格
    random.shuffle(chars)
    return "".join(chars)

shuffled_text_by_char = shuffle_by_char(original_text)

# 4. ✅ **按词打乱（保持空格）**
def shuffle_by_word(text):
    words = list(jieba.cut(text))  # 结巴分词
    random.shuffle(words)           # 打乱词序
    return " ".join(words)          # 重新拼接（确保单词间有空格）

shuffled_text_by_word = shuffle_by_word(original_text)

# 5. ✅ **计算文本嵌入**
embedding_original = model.encode(original_text, convert_to_tensor=True)
embedding_char_shuffled = model.encode(shuffled_text_by_char, convert_to_tensor=True)
embedding_word_shuffled = model.encode(shuffled_text_by_word, convert_to_tensor=True)

# 6. ✅ **计算余弦相似度**
similarity_char = util.pytorch_cos_sim(embedding_original, embedding_char_shuffled).item()
similarity_word = util.pytorch_cos_sim(embedding_original, embedding_word_shuffled).item()

# 7. ✅ **输出结果**
print(f"🔹 原始文本: {original_text}")

print(f"\n🔸 按字打乱: {shuffled_text_by_char}")
print(f"➡ 按字打乱相似度: {similarity_char:.4f}")

print(f"\n🔸 按词打乱: {shuffled_text_by_word}")
print(f"➡ 按词打乱相似度: {similarity_word:.4f}")
