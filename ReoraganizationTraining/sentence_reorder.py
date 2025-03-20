import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm  # 进度条库

# **1️⃣ 选择要测试的模型**
models = {
    "T5-Base": "t5-base",
    "BART-Large": "facebook/bart-large",
    "UL2": "google/ul2"
}

# **2️⃣ 需要测试的乱序句子**
sentences = [
    "NAND solve codes , the of data reliability for To flash algorithm problems a variable-node-based belief-propagation with message pre-processing ( VNBP-MP ) decoding storages for binary LDPC parity-check ( low-density ) the is proposed .",
    "The major that is effectively , . making use of the pre-processing of the NAND feature channel , propagation proposed algorithm performs the message characteristics ( MP ) scheme reliable flash prevent the propagation of unreliable the and speed up messages by of to messages the",
    "To ) speed up the decoding convergence , the further for oscillating variable nodes ( treatment VNs being considered after . MP scheme is employed the",
    "noticeable has show that . proposed VNBP-MP algorithm improvement a Simulation results in convergence speed without compromising the error-correction performance , compared with the existing algorithms the",
    "To solve , simultaneous one and mapping ( SLAM ) problem , many have PF been proposed effective and the Particle Filter ( techniques ) is localization of ways the .",
    "However , the PF of approximate a large number algorithm samples to needs the posterior probability . of the system , which the makes algorithm complex density",
        
]

# **3️⃣ 推理函数**
def reorder_sentence(model, tokenizer, sentence):
    input_text = "reorder: " + sentence  # **T5 风格输入**
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        input_ids,
        max_length=128,
        do_sample=True,  # 采样生成
        top_k=50,  # 选择概率最高的 50 个 token
        top_p=0.95,  # 仅保留累积概率 95% 的 token
        temperature=0.7,  # 控制采样的随机性
    )

    reordered_sentence = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return reordered_sentence

# **4️⃣ 按模型逐个加载，减少显存占用**
for model_name, model_path in models.items():
    print(f"\n🚀 加载模型: {model_name} ...")
    
    # **只加载当前模型**
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"✅ {model_name} 加载完成，开始推理...")
    
    # **推理句子**
    for shuffled_sentence in tqdm(sentences, desc=f"⏳ 处理 {model_name}", unit="句"):
        reordered = reorder_sentence(model, tokenizer, shuffled_sentence)
        print(f"🔄 乱序句子: {shuffled_sentence}")
        print(f"✅ {model_name} 重组句子: {reordered}\n")
    
    # **释放显存，减少占用**
    del model
    torch.cuda.empty_cache()  # 清理显存

print("🎉 所有模型推理完成！")

