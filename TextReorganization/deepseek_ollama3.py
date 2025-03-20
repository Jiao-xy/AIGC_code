import ollama
import re
import spacy
from collections import Counter

# 加载 spaCy 英语模型（用于语法分析）
nlp = spacy.load("en_core_web_sm")

# 需要重组的文本
input_text = "storages, problems low-density message belief-propagation pre-processing for flash To solve the with a algorithm codes is NAND (LDPC) parity-check the decoding binary of for proposed. data (VNBP-MP) reliability variable-node-based"
input_text="the the A playing listens audience man piano quietly."
# 计算原始文本的单词数
def word_count(text):
    return len(text.split())

original_word_count = word_count(input_text)

# 改进后的 Prompt，确保不改变长度，不增加或删除单词
prompt = f"""
Reorganize the following text while keeping the original sentence length and structure intact. 
Ensure grammatical correctness, improve readability, and maintain all technical terms and logical flow exactly as in the original.
Do not add or remove words excessively, and avoid unnecessary rewording.
Only adjust word order for better readability.

Text: {input_text}
"""

# 使用 System Prompt 禁止 DeepSeek 解释
messages = [
    {"role": "system", "content": "You must only return the reorganized text. Do not explain, do not analyze, do not add extra words."},
    {"role": "user", "content": prompt}
]

# 目标：至少收集 10 个符合所有要求的句子
num_target_sentences = 10
collected_sentences = []
iteration_count = 0

while len(collected_sentences) < num_target_sentences:
    iteration_count += 1
    print(f"第 {iteration_count} 次尝试生成句子...")

    response = ollama.chat(model='deepseek-r1:7b', messages=messages)
    generated_text = response['message']['content']

    # 去掉 <think>...</think> 之间的内容
    cleaned_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()
    
    # 仅保留单行文本
    cleaned_text = cleaned_text.replace("\n", " ").strip()
    
    # **1️⃣ 在生成时检测字数是否符合要求**
    word_diff = word_count(cleaned_text) - original_word_count
    if not (-2 <= word_diff <= 3):  # 允许单词数变化范围：-5 到 +10
        print(f"❌ 字数不符合要求 ({word_diff})，跳过")
        continue
    
    # **2️⃣ 语法完整性检测**
    def is_valid_sentence(sentence):
        """使用 spaCy 进行更严格的语法检测"""
        doc = nlp(sentence)

        # 确保句子长度适中
        if len(doc) < 5:
            print("❌ 语法错误: 句子过短")
            return False

        # 确保存在动词（谓语）
        if not any(token.dep_ == "ROOT" for token in doc):
            print("❌ 语法错误: 缺少主谓结构")
            return False
        
        # 计算单词频率，避免无意义重复
        word_freq = Counter(word.text.lower() for word in doc if word.is_alpha)
        if max(word_freq.values()) > len(doc) * 0.5:  # 如果某个单词占比超过 50%，可能是无意义的句子
            print("❌ 语法错误: 句子中某个单词重复过多")
            return False

        return True
    
    if not is_valid_sentence(cleaned_text):
        continue  # 语法不完整，跳过
    
    # **3️⃣ 可读性评估**
    def readability_score(sentence):
        """计算句子的可读性评分（更简洁、更有逻辑的句子得分更高）"""
        doc = nlp(sentence)
        avg_word_length = sum(len(token.text) for token in doc) / len(doc)
        num_commas = sum(1 for token in doc if token.text == ',')
        score = 1.0 - (num_commas / len(doc)) - (avg_word_length / 10)  # 适当平衡标点和单词长度
        
        # 确保句子流畅，不要太短或冗长
        if len(doc) < 8 or len(doc) > 40:
            print("❌ 可读性太低: 句子过短或过长")
            return 0  # 直接剔除
        
        return score
    
    read_score = readability_score(cleaned_text)
    
    if read_score < 0.3:  # 如果可读性太差，跳过
        print("❌ 可读性太差，跳过")
        continue

    # **4️⃣ 通过所有筛选，加入候选集**
    collected_sentences.append((cleaned_text, read_score))
    print(f"✅ 句子通过 (评分: {read_score:.2f}): {cleaned_text}")

# ================= 按可读性排序 =================
collected_sentences.sort(key=lambda x: x[1], reverse=True)

# ================= 输出最终结果 =================
print("\n🔹 原文：", input_text)
print("\n✅ 生成的最佳句子（按可读性排序）：")
for i, (sentence, score) in enumerate(collected_sentences, 1):
    print(f"{i}. (评分: {score:.2f}) {sentence}")
