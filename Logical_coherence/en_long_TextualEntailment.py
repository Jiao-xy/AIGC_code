import re
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. 加载 RoBERTa MNLI 预训练模型
model_name = "roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 2. 计算文本蕴含分数
def entailment_score(sentence1, sentence2):
    """
    计算两个句子之间的文本蕴含分数。
    - 高分（接近1）：蕴含（Entailment）。
    - 低分（接近-1）：矛盾（Contradiction）。
    - 接近0：无关（Neutral）。
    """
    inputs = tokenizer(sentence1, sentence2, return_tensors="pt", truncation=True, padding=True)

    # 计算 logits 并获取类别概率
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze()

    # 获取三类分数
    contradiction_score = probs[0].item()  # 矛盾
    neutral_score = probs[1].item()        # 中立
    entailment_score = probs[2].item()     # 蕴含

    return entailment_score - contradiction_score  # 计算蕴含度（正数：蕴含，负数：矛盾）

# 3. 句子拆分
def split_sentences(text):
    """
    使用标点符号拆分文本成句子：
    - 句号（.）、问号（?）、感叹号（!）等分隔。
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)  # 按标点符号分句
    return [s.strip() for s in sentences if s.strip()]  # 去除空白

# 4. 计算整段文本的逻辑连贯性分数
def evaluate_text_coherence(text):
    """
    计算文本整体的逻辑连贯性：
    - 逐句计算相邻句子的 Textual Entailment 得分。
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
