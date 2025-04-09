# python -m modules.training.T5_base_Training.py
# modules/training/t5_base_training.py
# 使用 t5-base 模型培训打乱文本恢复任务，逐随进度保存模型

import os
import json
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset

from modules.utils.jsonl_handler import read_jsonl
from modules.models.manager import ModelManager

# 【基本设置】
model_name = "t5-base"
dataset_path = "data/train_pairs/grouped_shuffle_all.jsonl"
output_dir = "models"
version_name = "t5-base_reorder"

# 【加载 Tokenizer 和模型（使用 ModelManager 防止重复加载）】
if ModelManager.is_model_loaded(model_name):
    model, tokenizer = ModelManager.get_model(model_name)
else:
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    ModelManager.register_model(model_name, model, tokenizer)

# 【读取数据，进行预处理】
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

# 【设置培训参数】
training_args = TrainingArguments(
    output_dir=os.path.join(output_dir, version_name),
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=True,
    num_train_epochs=10,
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

# 【递增进度保存模型 + 分段评估】
epochs = training_args.num_train_epochs
eval_results = {}
save_ratios = [i / 10 for i in range(1, 11)]  # 10%为单位

for epoch in tqdm(range(epochs), desc=f"Training {version_name}"):
    trainer.train()
    for progress in save_ratios:
        if epoch == int(epochs * progress) - 1:
            tag = f"{int(progress * 100)}"
            save_path = os.path.join(output_dir, f"{version_name}_{tag}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"✅ {tag}% 训练完成，模型已保存至 {save_path}")
            eval_metrics = trainer.evaluate()
            eval_results[tag] = eval_metrics
            print(f"📊 {tag}% 评估结果: {eval_metrics}")

# 【最终保存完模型 + 评估结果】
save_path_final = os.path.join(output_dir, f"{version_name}_100")
model.save_pretrained(save_path_final)
tokenizer.save_pretrained(save_path_final)
eval_metrics = trainer.evaluate()
eval_results["100"] = eval_metrics
print(f"✅ 训练完成，最终模型保存至 {save_path_final}")
print(f"📊 最终评估结果: {eval_metrics}")

# 【保存评估结果为 JSON】
eval_results_path = os.path.join(output_dir, f"{version_name}_eval_results.json")
with open(eval_results_path, "w") as f:
    json.dump(eval_results, f, indent=4)
print(f"📁 所有阶段评估结果已保存至 {eval_results_path}")
