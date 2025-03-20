from transformers import pipeline

# 加载 RoBERTa MNLI（Multi-Genre Natural Language Inference）模型
entailment_model = pipeline("text-classification", model="roberta-large-mnli")

# 计算两个句子的文本蕴含分数
def entailment_score(sentence1, sentence2):
    prediction = entailment_model(f"{sentence1} {sentence2}")
    label = prediction[0]['label']
    score = prediction[0]['score']
    
    if label == 'ENTAILMENT':
        return score  # 0-1 之间，越高表示句子之间连贯性越强
    elif label == 'CONTRADICTION':
        return -score  # 负数表示矛盾
    else:
        return 0  # Neutral 中性关系，表示前后句无关

# 示例：原文本（逻辑连贯）
sentence1 = "男子骑无牌助力车被拦撞伤交警。"
sentence2 = "记者随后从警方了解到，当天在新街口地区，有两名交警在对无牌助力车进行检查时被撞伤，所幸伤势并不严重。"
print(f"原文本句子间因果得分: {entailment_score(sentence1, sentence2)}")

# 示例：打乱文本（逻辑被破坏）
shuffled_sentence1 = "记者随后从警方了解到，当天在新街口地区，有两名交警在对无牌助力车进行检查时被撞伤，所幸伤势并不严重。"
shuffled_sentence2 = "男子骑无牌助力车被拦撞伤交警。"
print(f"打乱文本句子间因果得分: {entailment_score(shuffled_sentence1, shuffled_sentence2)}")
