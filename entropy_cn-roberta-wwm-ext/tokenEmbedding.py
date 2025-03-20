from transformers import AutoTokenizer, AutoModel
import torch

# 选择适合中文的 RoBERTa 预训练模型
model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 输入文本
text = "深度学习是人工智能的一个分支。"
inputs = tokenizer(text, return_tensors="pt")

# 获取词向量
with torch.no_grad():
    outputs = model(**inputs)

# 提取 Token 级别词向量
token_embeddings = outputs.last_hidden_state  # (batch_size, seq_length, hidden_dim)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# 输出 Token 及其对应的词向量
for token, vec in zip(tokens, token_embeddings[0]):
    print(f"{token}: {vec[:5]}...")  # 仅显示部分向量

