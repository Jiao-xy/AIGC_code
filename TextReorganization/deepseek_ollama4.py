import ollama
import re
import spacy
import textstat
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

# 计算单词数
def word_count(text):
    doc = nlp(text)
    return sum(1 for token in doc if token.is_alpha)

# 目标句子数
num_target_sentences = 3
collected_sentences = []
max_iterations = 20  # 最大尝试次数
iteration_count = 0

for input_text in test_sentences:
    if len(collected_sentences) >= num_target_sentences:
        break
    
    iteration_count += 1
    print(f"\n🔄 尝试重组句子 {iteration_count}: {input_text[:50]}...")

    original_word_count = word_count(input_text)
    prompt = f"""
    Reorganize the following text while keeping the original sentence length and structure intact. 
    Ensure grammatical correctness, improve readability, and maintain all technical terms and logical flow exactly as in the original.
    Do not add or remove words excessively, and avoid unnecessary rewording.
    Only adjust word order for better readability.
    
    Text: {input_text}
    """
    
    messages = [
        {"role": "system", "content": "You must only return the reorganized text. Do not explain, do not analyze, do not add extra words."},
        {"role": "user", "content": prompt}
    ]
    
    response = ollama.chat(model='deepseek-r1:7b', messages=messages)
    generated_text = response.get('message', {}).get('content', '').strip()
    cleaned_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()
    cleaned_text = cleaned_text.replace("\n", " ")
    
    word_diff = word_count(cleaned_text) - original_word_count
    if not (-2 <= word_diff <= 3):
        print(f"❌ 字数不符合要求 ({word_diff})，跳过")
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

    if not is_valid_sentence(cleaned_text):
        continue
    
    def readability_score(sentence):
        score = textstat.flesch_reading_ease(sentence)
        return max(0, min(1, score / 100))

    read_score = readability_score(cleaned_text)
    if read_score < 0.3:
        print("❌ 可读性太低，跳过")
        continue
    
    collected_sentences.append(cleaned_text)
    print(f"✅ 句子通过: {cleaned_text}")

print("\n🔹 最终生成的句子：")
for i, sentence in enumerate(collected_sentences, 1):
    print(f"{i}. {sentence}")
