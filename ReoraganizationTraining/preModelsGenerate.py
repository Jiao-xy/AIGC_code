import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

# **1️⃣ 设置模型目录**
models_dir = "/home/jxy/models"  # 训练模型存放路径

# **2️⃣ 读取所有已训练的模型**
def get_trained_models(models_dir):
    """
    读取 /home/jxy/models/ 目录，获取所有已训练的模型子目录
    - 仅保留以 t5-small_* 或 t5-base_* 开头的模型
    - 忽略 JSON 评估结果文件
    - 若存在 `t5-small_tau_08/` 这样的目录，优先加载 `_100` 版本
    """
    model_versions = []
    
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        
        # 只选取 T5 相关的模型目录，排除评估文件
        if os.path.isdir(model_path) and (model_name.startswith("t5-small") or model_name.startswith("t5-base")):
            model_versions.append(model_name)

    # 处理 `_100` 版本优先
    final_model_list = []
    for model_name in model_versions:
        base_name = model_name.rstrip("_100").rstrip("_30").rstrip("_50").rstrip("_80")
        if base_name in model_versions and f"{base_name}_100" in model_versions:
            continue  # 如果 `_100` 版本存在，则跳过非 `_100` 版本
        final_model_list.append(model_name)
    
    return sorted(final_model_list)  # 排序，保证加载顺序一致

# **3️⃣ 获取所有可用的模型**
model_versions = get_trained_models(models_dir)

# **4️⃣ 需要重组的句子**
test_sentences = shuffled_sentences = [
    "and the challenges facing the research area. by summarizing key implications, Finally, we conclude future research directions,",
    "The last decade has seen due to the unprecedented success of deep learning. in this area a surge of research",
    "datasets, and evaluation metrics have been proposed in the literature, Numerous methods, raising the need for a comprehensive and updated survey.",
    "support tests of predictions. A comprehensive comparison between different techniques, as well as identifying the pros and cons of various evaluation metrics are also provided in this survey.",
    "This paper fills the gap by reviewing the state-of-the-art approaches focusing on models from traditional models to deep learning. from 1961 to 2021,",
    "natural language processing. Text classification is the most fundamental and essential task in",
    "for text classification according to the text involved and the models used for feature extraction and classification. We create a taxonomy",
    "We then discuss each of these categories in detail, dealing with both the technical developments and benchmark datasets that"
]

# **5️⃣ 句子重组函数**
def reorder_sentences(model, tokenizer, sentences):
    """
    使用指定的 T5 模型对多个句子进行重组
    """
    inputs = [f"reorder: {sentence}" for sentence in sentences]
    
    # **Tokenize**
    inputs_encodings = tokenizer(
        inputs, return_tensors="pt", padding=True, truncation=True, max_length=128
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # **模型预测**
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs_encodings.input_ids,
            attention_mask=inputs_encodings.attention_mask,
            max_length=128,
            num_return_sequences=1,  # 只生成一个候选结果
            temperature=0.7,  # 控制采样多样性
            top_k=50,  # 选取概率最高的前50个单词
            top_p=0.95,  # 过滤掉低概率词
        )

    # **解码生成结果**
    reordered_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return reordered_sentences

# **6️⃣ 遍历多个模型并重组句子**
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n📌 共找到 {len(model_versions)} 个训练好的模型:\n")
for model_version in model_versions:
    print(f" - {model_version}")

for model_version in model_versions:
    model_path = os.path.join(models_dir, model_version)

    if not os.path.exists(model_path):
        print(f"⚠️ 模型 {model_version} 未找到，跳过！")
        continue

    print(f"\n🚀 加载模型: {model_version}")
    
    # **加载 Tokenizer 和 Model**
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

    # **执行句子重组**
    reordered_results = reorder_sentences(model, tokenizer, test_sentences)

    # **输出结果**
    print(f"\n🔹 模型: {model_version}")
    for i, (original, reordered) in enumerate(zip(test_sentences, reordered_results)):
        print(f"【句子 {i+1}】")
        print(f"   🔹 原始句子: {original}")
        print(f"   ✅ 重组后: {reordered}\n")

    # **释放显存**
    del model
    torch.cuda.empty_cache()
