from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 加载 Fine-tuned 的中文 RoBERTa NLI 模型
model_name = "clue/roberta_chinese_pair_large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 计算文本蕴含分数
def entailment_score(sentence1, sentence2):
    inputs = tokenizer(sentence1, sentence2, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

    # 取 "Entailment" 类别的概率
    entailment_prob = probs[0][0].item()
    contradiction_prob = probs[0][1].item()
    neutral_prob = probs[0][2].item()

    if entailment_prob > max(contradiction_prob, neutral_prob):
        return entailment_prob  # 0-1，越高表示句子间连贯性越强
    elif contradiction_prob > entailment_prob:
        return -contradiction_prob  # 负数表示矛盾
    else:
        return 0  # Neutral（无关）时返回 0

# 示例句子
sentence1 = "男子骑无牌助力车被拦撞伤交警。"
sentence2 = "记者随后从警方了解到，当天在新街口地区，有两名交警在对无牌助力车进行检查时被撞伤，所幸伤势并不严重。"

print(f"文本蕴含得分: {entailment_score(sentence1, sentence2)}")
