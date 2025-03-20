import re
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. 选择 BART 或 DeBERTa（尝试不同模型）
model_name = "facebook/bart-large-mnli"  # 或 "microsoft/deberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 2. 计算文本蕴含分数
def entailment_score(sentence1, sentence2):
    """
    计算两个句子之间的文本蕴含分数，返回 Entailment 预测概率
    """
    inputs = tokenizer(sentence1, sentence2, return_tensors="pt", truncation=True, padding=True, max_length=256)

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze()

    return probs[2].item()  # 直接返回 Entailment 分数

# 3. 句子拆分
def split_sentences(text):
    """
    拆分文本成句子，避免过短的句子
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)  # 按 .!? 分割
    return [s.strip() for s in sentences if len(s.strip()) > 5]  # 过滤过短的句子

# 4. 计算文本逻辑连贯性
def evaluate_text_coherence(text):
    """
    计算文本逻辑连贯性：
    - 逐句计算相邻句子的 Entailment 分数。
    - 取平均值作为文本的连贯性评分。
    """
    sentences = split_sentences(text)
    scores = []

    for i in range(len(sentences) - 1):
        score = entailment_score(sentences[i], sentences[i + 1])
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0  # 计算平均分

# 5. **测试文本**
text = """A man is playing the piano. The audience listens quietly. Suddenly, a dog runs onto the stage. The pianist stops playing and looks at the dog. People in the audience start laughing. The dog sits next to the pianist and watches him. The pianist resumes playing, and the dog stays still, listening to the music."""
shuffled_text = """A dog runs onto the stage. A man is playing the piano. The audience listens quietly. The pianist stops playing and looks at the dog. The dog sits next to the pianist and watches him. People in the audience start laughing. The pianist resumes playing, and the dog stays still, listening to the music."""

# **计算逻辑连贯性**
print(f"Original Text Coherence Score: {evaluate_text_coherence(text):.4f}")
print(f"Shuffled Text Coherence Score: {evaluate_text_coherence(shuffled_text):.4f}")
