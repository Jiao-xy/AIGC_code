import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# **加载 GPT-2 Medium 预训练模型**
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# **修复 GPT-2 padding token 问题**
tokenizer.pad_token = tokenizer.eos_token  # 设置 PAD token

# **文件路径（请修改为你的 JSON 文件路径）**
file_path = "/home/jxy/code/data/ieee-chatgpt-generation.jsonl"  # 替换为你的 JSON 文件路径
num_texts_to_read = 1000  # 控制读取的 abstract 数量

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

# **读取 JSON 文件并提取 abstract**
llscores = []
ppls = []
abstracts = []

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        try:
            data = json.loads(line.strip())  # 解析 JSON
            abstract = data.get("abstract", "").strip()  # 获取 abstract
            if abstract:
                abstracts.append(abstract)
                """ if len(abstracts) >= num_texts_to_read:
                    break  # 只读取指定数量的文本 """
        except json.JSONDecodeError:
            print("JSON 解码错误，跳过该行。")
num=0
# **计算所有 abstract 的 LLScore 和 PPL**
for abstract in abstracts:
    llscore, ppl = compute_llscore_ppl(abstract)
    llscores.append(llscore)
    ppls.append(ppl)
    num+=1
    if num%100==0:
        print(f"已经处理{num}份文本")

# **绘制 LLScore 和 PPL 统计图**
plt.figure(figsize=(12, 5))

# **LLScore 分布**
plt.subplot(1, 2, 1)
plt.hist(llscores, bins=10, color="blue", alpha=0.7, label="LLScore")
plt.xlabel("LLScore")
plt.ylabel("Frequency")
plt.title("Distribution of LLScore")
plt.legend()

# **PPL 分布**
plt.subplot(1, 2, 2)
plt.hist(ppls, bins=10, color="red", alpha=0.7, label="PPL")
plt.xlabel("Perplexity (PPL)")
plt.ylabel("Frequency")
plt.title("Distribution of Perplexity (PPL)")
plt.legend()

# **保存图像**
image_path = "LLScore_PPL_Distribution_gpt.png"
plt.savefig(image_path)


print(f"统计图已保存至 {image_path}")
