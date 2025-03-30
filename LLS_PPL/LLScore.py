import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载 GPT-2 语言模型
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def compute_log_likelihood(text):
    # 对文本进行 tokenization
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    
    # 提取 loss（对数似然的负数）
    log_likelihood = -outputs.loss.item() * inputs.input_ids.shape[1]
    return log_likelihood

# 示例文本
text = """Extensive experiments conducted on standard benchmarks like RTE, SST-2, MNLI, and NLI show that our method achieves state-of-the-art performance across these datasets. Contrastive Augmentation Approach: By incorporating contrastive learning techniques, we enhance the robustness of our approach and achieve better performance in downstream tasks such as text classification (e.g., STS) and machine translation. These results demonstrate the potential of our proposed framework for a wide range of applications in natural language processing. We propose an effective framework for transferring learned representations, particularly focusing on scenarios where labeled data is scarce. The simplicity of our approach is further validated by its effectiveness when applied to the task of summarizing text.
"""
# 计算对数似然得分
ll_score = compute_log_likelihood(text)
print(f"Log-Likelihood Score: {ll_score}")
