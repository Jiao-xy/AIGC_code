import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

# **1️⃣ 设置模型目录**
models_dir = "/home/jxy/models"  # 训练模型存放路径
input_file = "input_sentences.txt"  # 输入文件
output_file = "output_results.txt"  # 输出文件

# **2️⃣ 读取所有已训练的模型**
def get_trained_models(models_dir):
    model_versions = []
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path) and (model_name.startswith("t5-small") or model_name.startswith("t5-base")):
            model_versions.append(model_name)
    return sorted(model_versions)

# **3️⃣ 从文件中读取需要重组的句子**
def read_sentences_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ 文件 {file_path} 不存在！")
        return []
    with open(file_path, "r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file.readlines() if line.strip()]
    return sentences

# **4️⃣ 句子重组函数（支持三种模式）**
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

# **5️⃣ 遍历多个模型并重组句子**
device = "cuda" if torch.cuda.is_available() else "cpu"
model_versions = get_trained_models(models_dir)
test_sentences = read_sentences_from_file(input_file)

if not test_sentences:
    print("⚠️ 没有找到有效的输入句子，程序终止。")
else:
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write(f"📌 共找到 {len(model_versions)} 个训练好的模型:\n")
        for model_version in model_versions:
            out_f.write(f" - {model_version}\n")
        
        modes = ["beam_search", "sampling", "beam_sampling"]
        
        for model_version in model_versions:
            model_path = os.path.join(models_dir, model_version)
            if not os.path.exists(model_path):
                out_f.write(f"⚠️ 模型 {model_version} 未找到，跳过！\n")
                continue
            
            print(f"\n🚀 加载模型: {model_version}")
            tokenizer = T5Tokenizer.from_pretrained(model_path)
            model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
            
            out_f.write(f"\n🔹 模型: {model_version}\n")
            for mode in modes:
                out_f.write(f"\n🔹 生成模式: {mode}\n")
                for i, original in enumerate(test_sentences):
                    reordered = reorder_sentence(model, tokenizer, original, mode=mode)
                    out_f.write(f"【句子 {i+1}】✅ 重组后: {reordered}\n")
            
            del model
            torch.cuda.empty_cache()

    print(f"✅ 处理完成，结果已保存至 {output_file}")
