import openai

# DeepSeek API 设置
api_key = ""  # 请替换成你的 API Key
base_url = "https://api.deepseek.com"

# 选择模型（推荐使用 deepseek-chat）
model = "deepseek-chat"

# 测试用的输入文本
input_text = "storages, problems low-density message belief-propagation pre-processing for flash To solve the with a algorithm codes is NAND (LDPC) parity-check the decoding binary of for proposed. data (VNBP-MP) reliability variable-node-based"

# 提示词
prompt = f"""
Reorganize the following text while keeping the original sentence length and structure intact.
Ensure grammatical correctness, improve readability, and maintain all technical terms.
Do not add or remove words excessively. Only adjust word order for better clarity.

Text: {input_text}
"""

# OpenAI 兼容的 API 客户端
client = openai.OpenAI(api_key=api_key, base_url=base_url)

# 调用 API
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are an AI assistant specialized in text restructuring."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.5,  # 控制随机性
    max_tokens=150,  # 限制输出 Token 数
    stream=False
)

# **解析 API 响应**
print("\n🔹 **完整 API 返回内容:**")
print(response)

# **检查输出格式**
if response.choices:
    generated_text = response.choices[0].message.content
    print("\n✅ **提取出的返回文本:**")
    print(generated_text)

    # **检查是否包含 `<think>...</think>` 结构**
    if "<think>" in generated_text and "</think>" in generated_text:
        print("\n⚠️ **警告: 输出内容包含 `<think>...</think>`，需要清理！**")
    else:
        print("\n✅ **输出内容结构正常，无需清理 `<think>` 相关内容。**")
else:
    print("\n❌ **API 未返回有效数据。**")
