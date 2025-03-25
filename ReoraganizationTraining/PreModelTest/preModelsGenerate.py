import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

# **1️⃣ 设置模型目录**
models_dir = "/home/jxy/models"  # 训练模型存放路径

# **2️⃣ 读取所有已训练的模型**
def get_trained_models(models_dir):
    model_versions = []
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path) and (model_name.startswith("t5-small") or model_name.startswith("t5-base")):
            model_versions.append(model_name)
    return sorted(model_versions)

# **3️⃣ 获取所有可用的模型**
model_versions = get_trained_models(models_dir)

# **4️⃣ 需要重组的句子**
test_sentences = [
    "The rapid advancement of language large (LLM) model particularly technology, ChatGPT, the distinguishing emergence between of and texts models advanced like human-written LLM-generated has increasingly challenging. become",
    "This phenomenon unprecedented challenges presents academic authenticity, to integrity and making of detection a LLM-generated pressing research. concern in scientific",
    "To effectively and detect accurately generated LLMs, by this constructs study a comprehensive dataset of medical paper introductions, both encompassing human-written and LLM-generated content.",
    "Based dataset, on this simple and an efficient black-box, detection zero-shot method proposed. is",
    "The method builds upon that hypothesis differences fundamental exist in linguistic logical between ordering human-written and texts. LLMgenerated",
    "Specifically, reorders this original method text using dependency trees, parse calculates the similarity (Rscore) score between reordered the text and original, the integrates and log-likelihood as features metrics. auxiliary",
    "The approach reordered synthesizes similarity log-likelihood and scores derive to composite a establishing metric, effective classification an for threshold discriminating between human-written and texts. LLM-generated",
    "The results experimental our show approach that not effectively only detects but texts LLMgenerated also identifies LLM-polished abstracts, state-of-the-art outperforming current zero-shot detection methods (SOTA)."
]

# **5️⃣ 句子重组函数（支持三种模式）**
def reorder_sentence(model, tokenizer, sentence, mode="beam_search"):
    input_text = f"reorder: {sentence}"
    inputs_encodings = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=128
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        if mode == "beam_search":
            output_ids = model.generate(
                input_ids=inputs_encodings.input_ids,
                attention_mask=inputs_encodings.attention_mask,
                max_length=128,
                num_beams=8,
                repetition_penalty=1.5,
                length_penalty=0.8,
                early_stopping=True,
            )
        elif mode == "sampling":
            output_ids = model.generate(
                input_ids=inputs_encodings.input_ids,
                attention_mask=inputs_encodings.attention_mask,
                max_length=128,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )
        elif mode == "beam_sampling":
            output_ids = model.generate(
                input_ids=inputs_encodings.input_ids,
                attention_mask=inputs_encodings.attention_mask,
                max_length=128,
                num_beams=5,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.5,
                length_penalty=0.8,
            )
        else:
            raise ValueError("Invalid mode. Choose from 'beam_search', 'sampling', or 'beam_sampling'")

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# **6️⃣ 遍历多个模型并重组句子**
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n📌 共找到 {len(model_versions)} 个训练好的模型:\n")
for model_version in model_versions:
    print(f" - {model_version}")

modes = ["beam_search", "sampling", "beam_sampling"]

for model_version in model_versions:
    model_path = os.path.join(models_dir, model_version)

    if not os.path.exists(model_path):
        print(f"⚠️ 模型 {model_version} 未找到，跳过！")
        continue

    print(f"\n🚀 加载模型: {model_version}")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

    print(f"\n🔹 模型: {model_version}")
    for mode in modes:
        print(f"\n🔹 生成模式: {mode}")
        for i, original in enumerate(test_sentences):
            reordered = reorder_sentence(model, tokenizer, original, mode=mode)
            print(f"【句子 {i+1}】✅重组后: {reordered}\n")

    del model
    torch.cuda.empty_cache()


"""
With the rapid advancement of large language model (LLM) technology, particularly with the emergence of advanced models like ChatGPT, distinguishing between LLM-generated and human-written texts has become increasingly challenging. 
This phenomenon presents unprecedented challenges to academic integrity and authenticity, making the detection of LLM-generated content a pressing concern in scientific research. 
To effectively and accurately detect texts generated by LLMs, this study constructs a comprehensive dataset of medical paper introductions, encompassing both human-written and LLM-generated content. 
Based on this dataset, a simple and efficient black-box, zero-shot detection method is proposed. 
The method builds upon the hypothesis that fundamental differences exist in the linguistic logical ordering between human-written and LLMgenerated texts. 
Specifically, this method reorders the original text using dependency parse trees, calculates the similarity score (Rscore) between the reordered text and the original, and integrates log-likelihood features as auxiliary metrics. 
The approach synthesizes the reordered similarity and log-likelihood scores to derive a composite metric, establishing an effective classification threshold for discriminating between human-written and LLM-generated texts. 
The experimental results show that our approach not only effectively detects LLMgenerated texts but also identifies LLM-polished abstracts, outperforming current state-of-the-art zero-shot detection methods (SOTA).
"""