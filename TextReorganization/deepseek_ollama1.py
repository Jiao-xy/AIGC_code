import ollama
import re
import spacy
from collections import Counter
from difflib import SequenceMatcher

# 加载 spaCy 英语模型（用于语法分析）
nlp = spacy.load("en_core_web_sm")

# 需要重组的文本
input_text = "storages, problems low-density message belief-propagation pre-processing for flash To solve the with a algorithm codes is NAND (LDPC) parity-check the decoding binary of for proposed. data (VNBP-MP) reliability variable-node-based"

# 计算原始文本的单词数
def word_count(text):
    return len(text.split())

original_word_count = word_count(input_text)

# 改进后的 Prompt，确保不改变长度，不增加或删除单词
prompt = f"Reorganize the following text while keeping the original sentence length and structure intact. Do not add or remove words. Maintain all technical terms and logical flow exactly as in the original. Only adjust word order for better readability.: {input_text}"

# 使用 System Prompt 禁止 DeepSeek 解释
messages = [
    {"role": "system", "content": "You must only return the reorganized text. Do not explain, do not analyze, do not add extra words."},
    {"role": "user", "content": prompt}
]

# 目标：至少收集 10 个符合单词数要求的句子
num_target_sentences = 10
generated_sentences = []
number=0
while len(generated_sentences) < num_target_sentences:
    number+=1
    print(f"第{number}次循环")
    response = ollama.chat(model='deepseek-r1:7b', messages=messages)
    generated_text = response['message']['content']
    
    # 去掉 <think>...</think> 之间的内容
    cleaned_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()
    
    # 仅保留单行文本
    cleaned_text = cleaned_text.replace("\n", " ").strip()
    
    # **在生成时检测单词数是否变化太大**
    if cleaned_text and abs(word_count(cleaned_text) - original_word_count) <= 10:  # 允许最多 ±3 词的变化
        generated_sentences.append(cleaned_text)
    else:
        print("字数不符合")
        print(f"字数变化{word_count(cleaned_text) - original_word_count}，\n生成文本内容：{cleaned_text}")

# ================= 过滤掉明显错误的句子（基于 spaCy 语法分析） =================
def is_valid_sentence(sentence):
    """使用 spaCy 检测句子是否语法完整"""
    doc = nlp(sentence)
    
    # 如果句子太短或者不是完整的主谓结构，则剔除
    if len(doc) < 5 or not any(token.dep_ == "ROOT" for token in doc):
        return False
    
    # 计算单词频率，避免无意义重复
    word_freq = Counter(word.text.lower() for word in doc if word.is_alpha)
    if max(word_freq.values()) > len(doc) * 0.5:  # 如果某个单词占比超过 50%，可能是无意义的句子
        return False

    return True

filtered_sentences = [sentence for sentence in generated_sentences if is_valid_sentence(sentence)]

# 确保至少有 3 个候选句子可用
if len(filtered_sentences) < 3:
    print("⚠️ 语法检测后剩余的句子数量过少，请调整 prompt 或降低筛选标准。")
    print(generated_sentences)
    exit()

# ================= 选择相似度最高的前 3 个句子 =================
def sentence_similarity(s1, s2):
    """计算两个句子的相似度"""
    return SequenceMatcher(None, s1, s2).ratio()

# 按与原句的相似度排序，选择最好的 3 个句子
filtered_sentences = sorted(filtered_sentences, key=lambda x: sentence_similarity(input_text, x), reverse=True)[:3]

# ================= 输出最终结果 =================
print("🔹 原文：", input_text)
print("\n✅ 生成的最佳 3 个重组版本：")
for i, sentence in enumerate(filtered_sentences, 1):
    print(f"{i}. {sentence}")
