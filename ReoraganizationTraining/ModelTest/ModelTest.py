import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# **1️⃣ 设置模型名称（已预下载）**
model_names = ["t5-small", "t5-base", "facebook/bart-large"]

# **2️⃣ 需要重组的句子**
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

# **3️⃣ 句子重组函数（支持三种模式）**
def reorder_sentence(model, tokenizer, sentence, mode="beam_search"):
    input_text = f"reorder: {sentence}"
    inputs_encodings = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=128
    ).to(device)

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

# **4️⃣ 自动加载 Tokenizer 和 Model**
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model

# **5️⃣ 直接加载已下载的模型并执行重组**
device = "cuda" if torch.cuda.is_available() else "cpu"
modes = ["beam_search", "sampling", "beam_sampling"]

for model_name in model_names:
    print(f"\n🚀 加载模型: {model_name}")
    tokenizer, model = load_model_and_tokenizer(model_name)

    print(f"\n🔹 模型: {model_name}")
    for mode in modes:
        print(f"\n🔹 生成模式: {mode}")
        for i, original in enumerate(test_sentences):
            reordered = reorder_sentence(model, tokenizer, original, mode=mode)
            print(f"【句子 {i+1}】✅重组后: {reordered}\n")

    del model
    torch.cuda.empty_cache()
