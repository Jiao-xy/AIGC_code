from transformers import T5Tokenizer
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os


# **1️⃣ 设置模型目录**
models_dir = "/home/jxy/models"  # 训练模型存放路径

def get_trained_models(models_dir):
    """
    读取 /home/jxy/models/ 目录，获取所有已训练的模型子目录
    - 仅保留以 t5-small_* 或 t5-base_* 开头的模型
    - 忽略 JSON 评估结果文件
    - 若存在 `t5-small_tau_08/` 这样的目录，优先加载 `_100` 版本
    """
    model_versions = []
    
    for model_name in os.listdir(models_dir):
        #print(f"model_name:{model_name}")
        model_path = os.path.join(models_dir, model_name)
        
        # 只选取 T5 相关的模型目录，排除评估文件
        if os.path.isdir(model_path) and (model_name.startswith("t5-small") or model_name.startswith("t5-base")):
            model_versions.append(model_name)
    
    # 处理 `_100` 版本优先
    final_model_list = []
    for model_name in model_versions:
        print(model_name)
        base_name = model_name.rstrip("_100").rstrip("_30").rstrip("_50").rstrip("_80")
        """ if base_name in model_versions and f"{base_name}_100" in model_versions:
            continue  # 如果 `_100` 版本存在，则跳过非 `_100` 版本 """
        final_model_list.append(model_name)
    
    return sorted(final_model_list)  # 排序，保证加载顺序一致

model_versions = get_trained_models(models_dir)

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

"""     model_name = model_path
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_name)  # ✅ 重新生成 tokenizer.json

print("✅ tokenizer.json 重新生成完成！")
 """