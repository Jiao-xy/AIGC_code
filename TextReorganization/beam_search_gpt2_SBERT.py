import torch
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import word_tokenize

# 下载 NLTK 词库（仅需运行一次）
nltk.download('punkt')

# 加载 GPT-2 预训练模型
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# 加载 SBERT 预训练模型
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# 原始乱序句子
shuffled_sentence = ". variable - node - based decoding algorithm for LDPC binary MP the problems ) low proposed a of ( , message pre-processing (VNBP-MP density VNBP propagation the data reliability parity is codes pre belief algorithm To processing check with solve NAND flash storages"

# 预处理函数：去除多余标点，修复单词顺序
def preprocess_text(text):
    # 1. 删除多余标点符号
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text)  # 只保留字母、数字、空格和连字符
    text = text.replace("-", " ")  # 处理连字符
    text = re.sub(r"\s+", " ", text).strip()  # 移除多余空格

    # 2. 进行基本分词
    words = word_tokenize(text)

    # 3. 重新排列，使得句子更自然（简单基于词性调整）
    words = sorted(words, key=lambda x: len(x))  # 仅作为简单示例，你可以用 NLP 规则优化
    return " ".join(words)

# 预处理输入文本
input_text = preprocess_text(shuffled_sentence)

# 生成候选句子（束搜索）
input_ids = tokenizer.encode(input_text, return_tensors="pt")
beam_outputs = model.generate(
    input_ids, 
    max_length=100,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True,
    num_return_sequences=5
)

# 解码生成的句子
generated_sentences = [tokenizer.decode(output, skip_special_tokens=True) for output in beam_outputs]

# 计算语言模型得分（LM Score）
def compute_lm_score(sentence):
    """计算 GPT-2 生成的句子的概率得分"""
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss
    return -loss.item()  # 取负值，越高越好

# 计算语义相似度（Semantic Coherence）
original_embedding = sbert_model.encode(input_text, convert_to_tensor=True)
def compute_semantic_score(sentence):
    """计算句子的语义相似度"""
    generated_embedding = sbert_model.encode(sentence, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(original_embedding, generated_embedding)
    return similarity.item()  # 越高越好

# 计算综合评分
sentence_scores = []
for sentence in generated_sentences:
    lm_score = compute_lm_score(sentence)
    semantic_score = compute_semantic_score(sentence)
    total_score = lm_score + semantic_score
    sentence_scores.append((sentence, total_score))

# 选择最佳句子
best_sentence = max(sentence_scores, key=lambda x: x[1])[0]

# 输出结果
print("【原始乱序句子】")
print(shuffled_sentence)
print("\n【束搜索+连贯性评分优化的句子】")
print(best_sentence)
