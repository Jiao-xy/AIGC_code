
import torch
import numpy as np
from transformers import RobertaModel, RobertaTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# 加载 RoBERTa 预训练模型和分词器
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

# 设为评估模式
model.eval()

# 示例文本（原始文本 & 重排文本）
original_text = "The quick brown fox jumps over the lazy dog. A machine learning model can predict the future. RoBERTa embeddings capture deep semantic meanings."
reordered_text = "Over the lazy dog, the quick brown fox jumps. Predicting the future is possible with machine learning. Semantic meanings are captured deeply by RoBERTa embeddings."
text1="""Excessive or prolonged psychological stress compromises several physiological systems,which might increase susceptibility to disease. Strong evidence from animal models and human studies suggests a considerable modulation of the hypothalamic-pituitary-adrenal axis in response to stress with altered biological functions such as compromised immunity (eg,impaired humoral and cell mediated immunity)and increased inflammatory reactivity.Correspondingly,people exposed to psychological stress have been reported to have a higher risk of respiratory virus infections paralleled with reduced immune responses to several antiviral and antibacterial vaccines."""
text2="""Strong evidence from animal models and human studies suggests a considerable modulation of the hypothalamic-pituitary-adrenal axis in response to stress,with altered biological functions such as compromised immunity (eg,impaired humoral and cell mediated immunity)and increased inflammatory reactivity.Excessive or prolonged psychological stress compromises several physiological systems,which might increase susceptibility to disease.Correspondingly,people exposed to psychological stress have been reported to have a higher risk of respiratory virus infections paralleled with reduced immune responses to several antiviral and antibacterial vaccines."""
original_text=text1
reordered_text=text2
# **计算文本的 RoBERTa 向量**
def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 取最后一层隐藏状态的均值作为文本向量
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# 计算文本向量
original_vector = get_text_embedding(original_text)
reordered_vector = get_text_embedding(reordered_text)

# 计算余弦相似度
similarity = cosine_similarity([original_vector], [reordered_vector])[0, 0]

# 输出结果
print("余弦相似度:", similarity)
