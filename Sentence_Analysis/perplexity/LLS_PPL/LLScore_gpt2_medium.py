import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载 GPT-2 Medium 预训练模型
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# **修复 GPT-2 padding token 问题**
tokenizer.pad_token = tokenizer.eos_token  # 设置 PAD token

# 示例文本（原始文本 & 重排文本）
original_text = "The quick brown fox jumps over the lazy dog. A machine learning model can predict the future. RoBERTa embeddings capture deep semantic meanings."
reordered_text = "Over the lazy dog, the quick brown fox jumps. Predicting the future is possible with machine learning. Semantic meanings are captured deeply by RoBERTa embeddings."

# **计算 LLScore（对数似然）和 PPL（困惑度）**
def compute_llscore_ppl(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    
    # 计算对数似然得分（负的 Loss 乘以 token 长度）
    log_likelihood = -outputs.loss.item() * inputs.input_ids.shape[1]

    # 计算困惑度（PPL）
    perplexity = torch.exp(outputs.loss).item()
    
    return log_likelihood, perplexity

# 计算 LLScore 和 PPL
llscore_original, ppl_original = compute_llscore_ppl(original_text)
llscore_reordered, ppl_reordered = compute_llscore_ppl(reordered_text)

# 输出结果
print("原始文本 LLScore:", llscore_original, "PPL:", ppl_original)
print("重排文本 LLScore:", llscore_reordered, "PPL:", ppl_reordered)
