from transformers import T5Tokenizer, T5ForConditionalGeneration

# **1️⃣ 加载本地训练好的模型**
model_name = "/home/jxy/models/t5_reorder_test"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# **2️⃣ 设定推理函数**
def reorder_sentence(sentence):
    input_text = "reorder: " + sentence  # 输入格式
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    #output_ids = model.generate(input_ids, max_length=128, num_beams=5)  # 🚀 进行推理
    output_ids = model.generate(
    input_ids,
    max_length=128,
    do_sample=True,  # ✅ 启用采样，让输出有随机性
    top_k=50,  # ✅ 仅从概率最高的 50 个 token 采样
    top_p=0.95,  # ✅ Nucleus Sampling，只保留累计概率 95% 的 token
    temperature=0.7,  # ✅ 控制采样的随机程度（越高越随机）
    )

    reordered_sentence = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return reordered_sentence

# **3️⃣ 测试推理**
shuffled_sentence = "algorithm decoding variable - node - based for LDPC codes"
shuffled_sentence="a the of for algorithm ( is , data with proposed NAND LDPC . VNBP-MP belief-propagation variable-node-based problems ) decoding the parity-check To ) message binary codes pre-processing reliability storages flash for solve low-density ("
#"To solve the problems of the data reliability for NAND flash storages, a variable-node-based belief-propagation with message pre-processing (VNBP-MP) decoding algorithm for binary low-density parity-check (LDPC) codes is proposed."
#"To solve the reliability problems of low-density data storages, a variable-node-based parity-check decoding (LDPC) algorithm is proposed for the NAND (VNBP-MP) message pre-processing with NAND binary codes for belief-propagation."
shuffled_sentence="texts than human-written LLMs, black-box texts. paper, a simple the contain that, effective from on zero-shot more propose perspective detection observation yet typically LLM-generated this based In grammatical the we errors approach of"
#原句子："In this paper, we propose a simple yet effective black-box zero-shot detection approach based on the observation that, from the perspective of LLMs, human-written texts typically contain more grammatical errors than LLM-generated texts."
#重组后："In this paper, we propose a simple, yet effective LLM-generated detection approach based on the observation of grammatical errors that typically contain zero-shot black-box texts from the perspective of human-written LLMs."
reordered = reorder_sentence(shuffled_sentence)

print("🔄 乱序句子:", shuffled_sentence)
print("✅ 重组句子:", reordered)
