# python -m modules.training.T5_base_Training.py
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import os
from tqdm import tqdm
import json

from modules.models.manager import ModelManager  # ✅ 加载模型管理器
from modules.utils.jsonl_handler import read_jsonl  # ✅ 加载 JSONL 文件
from transformers import T5Tokenizer, T5ForConditionalGeneration

# **1️⃣ 模型与数据路径设定，仅训练 t5-base + reorder 数据集**
model_name = "t5-base"
dataset_path = "data/train_pairs/grouped_shuffle_all.jsonl"
output_dir = "models"
version_name = "t5-base_reorder"

# **2️⃣ 加载 Tokenizer 与模型（使用 ModelManager）**
if ModelManager.is_model_loaded(model_name):
    model, tokenizer = ModelManager.get_model(model_name)
else:
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    ModelManager.register_model(model_name, model, tokenizer)

# **3️⃣ 读取并预处理数据**
data = read_jsonl(dataset_path)
dataset = Dataset.from_list(data)

def preprocess_function(examples):
    inputs = ["reorder: " + s for s in examples["shuffled"]]
    targets = examples["original"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess_function, batched=True)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"✅ 数据预处理完成，共 {len(train_dataset)} 条训练数据，{len(eval_dataset)} 条验证数据")

# **4️⃣ 设置训练参数**
training_args = TrainingArguments(
    output_dir=os.path.join(output_dir, version_name),
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=True,
    num_train_epochs=3,
    save_total_limit=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=500,
    eval_steps=2000,
    report_to="none",
    dataloader_pin_memory=True,
    dataloader_num_workers=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# **5️⃣ 分阶段训练与评估（每10%保存一次）**
epochs = training_args.num_train_epochs
eval_results = {}
save_ratios = [i / 10 for i in range(1, 11)]

if __name__ == "__main__":
    print("🚀 正在训练 T5-base 模型...")

    for epoch in tqdm(range(epochs), desc=f"Training {version_name}"):
        trainer.train()

        for progress in save_ratios:
            if epoch == int(epochs * progress) - 1:
                tag = f"{int(progress * 100)}"
                save_path = os.path.join(output_dir, f"{version_name}_{tag}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"✅ {tag}% 模型保存至 {save_path}")
                eval_metrics = trainer.evaluate()
                eval_results[tag] = eval_metrics
                print(f"📊 {tag}% 评估结果: {eval_metrics}")

    # 最终保存
    final_path = os.path.join(output_dir, f"{version_name}_100")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✅ 最终模型保存至 {final_path}")

    # 保存评估记录
    with open(os.path.join(output_dir, f"{version_name}_eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)
    print("📁 所有阶段评估结果已写入 JSON。")
