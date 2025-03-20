import ollama
import re

# 需要重组的文本
input_text = "storages, problems low-density message belief-propagation pre-processing for flash To solve the with a algorithm codes is NAND (LDPC) parity-check the decoding binary of for proposed. data (VNBP-MP) reliability variable-node-based"

# 改进后的 Prompt，明确要求仅输出结果
prompt = f"Reorganize the following text while keeping the original sentence length and structure intact. Do not add or remove words. Maintain all technical terms and logical flow exactly as in the original. Only adjust word order for better readability.: {input_text}"

# 使用 System Prompt 禁止 DeepSeek 解释
messages = [
    {"role": "system", "content": "You must only return the reorganized text. Do not explain, do not analyze, do not add extra words."},
    {"role": "user", "content": prompt}
]

# 调用 Ollama 本地部署的 DeepSeek 7B
response = ollama.chat(model='deepseek-r1:7b', messages=messages)

# 获取 AI 生成的文本
generated_text = response['message']['content']
#print(generated_text)
# 使用正则表达式去掉 <think>...</think> 之间的内容
cleaned_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()

# 输出最终重组的文本
print(cleaned_text)
