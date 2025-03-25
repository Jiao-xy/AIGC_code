import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

# **1️⃣ 设置模型目录**
models_dir = "/home/jxy/models"  # 训练模型存放路径

# **2️⃣ 选择模型**
model_name = "t5-small_tau_08_100"  # 这里直接指定某个模型
model_path = os.path.join(models_dir, model_name)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"⚠️ 模型 {model_name} 未找到，请检查路径是否正确！")

# **3️⃣ 需要重组的句子**
test_sentences = [
    #"The rapid advancement of language large (LLM) model particularly technology, ChatGPT, the distinguishing emergence between of and texts models advanced like human-written LLM-generated has increasingly challenging. become",
    "This phenomenon unprecedented challenges presents academic authenticity, to integrity and making of detection a LLM-generated pressing research. concern in scientific",
    #"To effectively and detect accurately generated LLMs, by this constructs study a comprehensive dataset of medical paper introductions, both encompassing human-written and LLM-generated content.",
    "Based dataset, on this simple and an efficient black-box, detection zero-shot method proposed. is",
    "The method builds upon that hypothesis differences fundamental exist in linguistic logical between ordering human-written and texts. LLMgenerated",
    #"Specifically, reorders this original method text using dependency trees, parse calculates the similarity (Rscore) score between reordered the text and original, the integrates and log-likelihood as features metrics. auxiliary",
    #"The approach reordered synthesizes similarity log-likelihood and scores derive to composite a establishing metric, effective classification an for threshold discriminating between human-written and texts. LLM-generated",
    #"The results experimental our show approach that not effectively only detects but texts LLMgenerated also identifies LLM-polished abstracts, state-of-the-art outperforming current zero-shot detection methods (SOTA)."
]

# **4️⃣ 句子重组函数**
def reorder_sentences(model, tokenizer, sentences):
    inputs = [f"reorder: {sentence}" for sentence in sentences]
    
    inputs_encodings = tokenizer(
        inputs, return_tensors="pt", padding=True, truncation=True, max_length=128
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs_encodings.input_ids,
            attention_mask=inputs_encodings.attention_mask,
            max_length=128,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )

    reordered_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return reordered_sentences

# **5️⃣ 加载模型**
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🚀 加载模型: {model_name}")
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

# **6️⃣ 执行句子重组**
reordered_results = reorder_sentences(model, tokenizer, test_sentences)

# **7️⃣ 输出结果**
print(f"\n🔹 模型: {model_name}")
for i, (original, reordered) in enumerate(zip(test_sentences, reordered_results)):
    """ print(f"【句子 {i+1}】")
    print(f"   🔹 原始句子: {original}") """
    print(f" 【句子 {i+1}  ✅ 重组后: {reordered}\n")

# **8️⃣ 释放显存**
del model
torch.cuda.empty_cache()