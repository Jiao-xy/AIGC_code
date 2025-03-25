import openai
import re
import spacy
from collections import Counter

# 加载 spaCy 英语模型（用于语法分析）
nlp = spacy.load("en_core_web_sm")

# 需要重组的文本
test_sentences = [ 
    "The rapid advancement of language large (LLM) model particularly technology, ChatGPT, the distinguishing emergence between of and texts models advanced like human-written LLM-generated has increasingly challenging. become",
    "This phenomenon unprecedented challenges presents academic authenticity, to integrity and making of detection a LLM-generated pressing research. concern in scientific",
    "To effectively and detect accurately generated LLMs, by this constructs study a comprehensive dataset of medical paper introductions, both encompassing human-written and LLM-generated content.",
    "Based dataset, on this simple and an efficient black-box, detection zero-shot method proposed. is",
    "The method builds upon that hypothesis differences fundamental exist in linguistic logical between ordering human-written and texts. LLMgenerated",
    "Specifically, reorders this original method text using dependency trees, parse calculates the similarity (Rscore) score between reordered the text and original, the integrates and log-likelihood as features metrics. auxiliary",
    "The approach reordered synthesizes similarity log-likelihood and scores derive to composite a establishing metric, effective classification an for threshold discriminating between human-written and texts. LLM-generated",
    "The results experimental our show approach that not effectively only detects but texts LLMgenerated also identifies LLM-polished abstracts, state-of-the-art outperforming current zero-shot detection methods (SOTA)."
]

# DeepSeek API 设置
api_key = ""  # 请替换为你的 API Key
with open("/home/jxy/Data/deepseek_api_key.txt", "r") as file:
    api_key = file.readline().strip()  # 读取第一行并去除首尾空格和换行符
base_url = "https://api.deepseek.com"

# OpenAI 兼容的 API 客户端
client = openai.OpenAI(api_key=api_key, base_url=base_url)
model = "deepseek-chat"

# 目标版本数
num_versions_per_sentence = 3
collected_sentences = {}

for input_text in test_sentences:
    collected_sentences[input_text] = []
    iteration_count = 0
    
    while len(collected_sentences[input_text]) < num_versions_per_sentence:
        iteration_count += 1
        print(f"\n🔄 尝试重组句子 {iteration_count}: {input_text[:50]}...")

        prompt = f"""
        Reorder the words in this sentence to make it grammatically correct and readable, without adding or removing words.
        
        Text: {input_text}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Only return the reordered sentence. No explanations or extra words."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=150,
            stream=False
        )

        if response.choices:
            generated_text = response.choices[0].message.content.strip()
        else:
            print("❌ API 未返回有效数据，跳过")
            continue

        def is_valid_sentence(sentence):
            doc = nlp(sentence)
            if len(doc) < 5 or not any(token.dep_ == "ROOT" for token in doc):
                print("❌ 语法错误: 句子结构不完整")
                return False

            word_freq = Counter(word.text.lower() for word in doc if word.is_alpha)
            if max(word_freq.values(), default=0) > len(doc) * 0.5:
                print("❌ 语法错误: 句子中某个单词重复过多")
                return False
            
            return True

        if not is_valid_sentence(generated_text):
            continue
        
        collected_sentences[input_text].append(generated_text)
        print(f"✅ 句子通过: {generated_text}")

print("\n🔹 最终生成的句子：")
for original, variations in collected_sentences.items():
    print(f"\n🔸 原句: {original}")
    for i, sentence in enumerate(variations, 1):
        print(f"{i}. {sentence}")
