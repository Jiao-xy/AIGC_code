import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# **加载 GPT-2 Medium 预训练模型**
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# **修复 GPT-2 padding token 问题**
tokenizer.pad_token = tokenizer.eos_token  # 设置 PAD token

# **文件路径列表（请修改为你的 JSONL 文件路径）**
json_files = [
    "/home/jxy/code/data/ieee-init.jsonl", 
    "/home/jxy/code/data/ieee-chatgpt-polish.jsonl", 
    "/home/jxy/code/data/ieee-chatgpt-fusion.jsonl", 
    "/home/jxy/code/data/ieee-chatgpt-generation.jsonl",
    "/home/jxy/code/data/ieee-init_random.jsonl", 
    "/home/jxy/code/data/ieee-chatgpt-polish_random.jsonl", 
    "/home/jxy/code/data/ieee-chatgpt-fusion_random.jsonl", 
    "/home/jxy/code/data/ieee-chatgpt-generation_random.jsonl"
]

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

# **处理多个 JSONL 文件**
for file_path in json_files:
    results = []
    abstracts = []
    
    # 读取 JSONL 文件
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line.strip())  # 解析 JSON
                abstract = data.get("abstract", "").strip()  # 获取 abstract
                doc_id = data.get("id", None)  # 获取 id
                if abstract and doc_id:
                    abstracts.append((doc_id, abstract))
            except json.JSONDecodeError:
                print(f"JSON 解码错误，跳过文件 {file_path} 的某一行。")
    
    num = 0
    
    # **计算所有 abstract 的 LLScore 和 PPL**
    for doc_id, abstract in abstracts:
        llscore, ppl = compute_llscore_ppl(abstract)
        results.append({"id": doc_id, "LLScore": llscore, "PPL": ppl})
        num += 1
        if num % 100 == 0:
            print(f"文件 {file_path} 已处理 {num} 份文本")
    
    # **生成新的输出文件名**
    base, ext = os.path.splitext(file_path)
    output_file_path = f"{base}_llscore_ppl{ext}"
    
    # **保存 LLScore 和 PPL 结果**
    with open(output_file_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"计算结果已保存至 {output_file_path}")
    
    # **绘制 LLScore 和 PPL 统计图**
    llscores = [res["LLScore"] for res in results]
    ppls = [res["PPL"] for res in results]
    
    plt.figure(figsize=(12, 5))
    
    # **LLScore 分布**
    plt.subplot(1, 2, 1)
    bins_ll, edges_ll, _ = plt.hist(llscores, bins=50, color="blue", alpha=0.7, label="LLScore")
    plt.xlabel("LLScore")
    plt.ylabel("Frequency")
    plt.xticks(np.arange(min(llscores), max(llscores) + 1, 50))  # 以 50 为刻度
    plt.title(f"Distribution of LLScore ({os.path.basename(file_path)})")
    plt.legend()
    
    for i in range(len(bins_ll)):
        plt.text(edges_ll[i], bins_ll[i], str(int(bins_ll[i])), ha='center', va='bottom')  # 标注具体数量
    
    # **PPL 分布**
    plt.subplot(1, 2, 2)
    bins_ppl, edges_ppl, _ = plt.hist(ppls, bins=50, color="red", alpha=0.7, label="PPL")
    plt.xlabel("Perplexity (PPL)")
    plt.ylabel("Frequency")
    plt.xticks(np.arange(0, np.percentile(ppls, 99) + 1, 5))  # 以 5 为刻度
    plt.title(f"Distribution of Perplexity (PPL) ({os.path.basename(file_path)})")
    plt.legend()
    
    for i in range(len(bins_ppl)):
        plt.text(edges_ppl[i], bins_ppl[i], str(int(bins_ppl[i])), ha='center', va='bottom')  # 标注具体数量
    
    # **保存矢量图**
    image_path = f"{base}_llscore_ppl.svg"  # 使用 SVG 矢量格式
    plt.savefig(image_path, format='svg')
    
    print(f"统计图已保存至 {image_path}")
