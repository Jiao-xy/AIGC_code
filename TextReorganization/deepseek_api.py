import openai
import re
import spacy
from collections import Counter

# 加载 spaCy 英语模型（用于语法分析）
nlp = spacy.load("en_core_web_sm")

# 需要重组的文本
input_text = "storages, problems low-density message belief-propagation pre-processing for flash To solve the with a algorithm codes is NAND (LDPC) parity-check the decoding binary of for proposed. data (VNBP-MP) reliability variable-node-based"

# 计算原始文本的单词数
def word_count(text):
    return len(text.split())

original_word_count = word_count(input_text)

# DeepSeek API 设置
api_key = ""  # 请替换为你的 API Key
with open("/home/jxy/Data/deepseek_api_key.txt", "r") as file:
    api_key = file.readline().strip()  # 读取第一行并去除首尾空格和换行符
base_url = "https://api.deepseek.com"

# 推荐模型：deepseek-chat（更经济）
model = "deepseek-chat"

# 提示词
prompt = f"""
Reorganize the following text while keeping the original sentence length and structure intact.
Ensure grammatical correctness, improve readability, and maintain all technical terms.
Do not add or remove words excessively. Only adjust word order for better clarity.

Text: {input_text}
"""

# OpenAI 兼容的 API 客户端
client = openai.OpenAI(api_key=api_key, base_url=base_url)

# 目标：至少收集 10 个符合所有要求的句子
num_target_sentences = 3
collected_sentences = []
iteration_count = 0

while len(collected_sentences) < num_target_sentences:
    iteration_count += 1
    print(f"第 {iteration_count} 次尝试生成句子...")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an AI assistant specialized in text restructuring."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,  # 控制随机性
        max_tokens=150,  # 限制输出 Token 数
        stream=False
    )

    # 解析 API 响应
    if response.choices:
        generated_text = response.choices[0].message.content
    else:
        print("❌ API 未返回有效数据，跳过")
        continue

    # 直接使用 API 返回的内容
    cleaned_text = generated_text.strip()

    # **1️⃣ 在生成时检测字数是否符合要求**
    word_diff = word_count(cleaned_text) - original_word_count
    if not (-5 <= word_diff <= 10):  # 允许单词数变化范围：-5 到 +10
        print(f"❌ 字数不符合要求 ({word_diff})，跳过")
        continue

    # **2️⃣ 语法完整性检测**
    def is_valid_sentence(sentence):
        """使用 spaCy 进行更严格的语法检测"""
        doc = nlp(sentence)

        # 确保句子长度适中（适当放宽）
        if len(doc) < 5 or len(doc) > 60:  # 以前是 8-40，现在改为 5-60
            print(f"❌ 语法错误: 句子长度 ({len(doc)}) 不符合范围")
            return False

        # 确保存在动词（谓语）
        if not any(token.dep_ == "ROOT" for token in doc):
            print("❌ 语法错误: 缺少主谓结构")
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
        
        # 适当放宽可读性评分的标准
        score = 1.0 - (num_commas / len(doc)) - (avg_word_length / 15)  # 以前是 /10，现在改为 /15
        
        return score
    
    read_score = readability_score(cleaned_text)
    
    if read_score < 0.3:  # 以前是 0.5，现在降低到 0.3
        print(f"❌ 可读性太差 ({read_score:.2f})，跳过")
        continue

    # **4️⃣ 通过所有筛选，加入候选集**
    collected_sentences.append((cleaned_text, read_score))
    print(f"✅ 句子通过 (评分: {read_score:.2f}): {cleaned_text}")

# ================= 按可读性排序 =================
collected_sentences.sort(key=lambda x: x[1], reverse=True)

# ================= 输出最终结果 =================
print("\n🔹 原文：", input_text)
print("\n✅ 生成的最佳句子（按可读性排序）:")
for i, (sentence, score) in enumerate(collected_sentences, 1):
    print(f"{i}. (评分: {score:.2f}) {sentence}")
