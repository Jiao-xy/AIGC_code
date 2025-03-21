import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import json

# **1️⃣ 设定模型及数据集参数**
model_names = {"t5-small": "t5-small", "t5-base": "t5-base"}
dataset_files = {
    "tau_08": "/home/jxy/Data/ReoraganizationData/sentence_shuffled_dataset_tau_08.csv",
    "full": "/home/jxy/Data/ReoraganizationData/sentence_reorder_dataset.csv"
}
output_dir = "/home/jxy/models"

# **2️⃣ 预加载和预处理数据**
tokenizer = T5Tokenizer.from_pretrained("t5-small")  # 预加载 tokenizer
processed_datasets = {}

for dataset_key, dataset_path in dataset_files.items():
    print(f"\n📥 加载数据集: {dataset_path}")
    df = pd.read_csv(dataset_path, on_bad_lines="skip")
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 转换为 Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    # 预处理函数
    def preprocess_function(examples):
        inputs = ["reorder: " + s for s in examples["乱序句子"]]
        targets = examples["正确句子"]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # 应用预处理
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)
    
    # 缓存数据集
    processed_datasets[dataset_key] = {"train": train_dataset, "eval": eval_dataset}

print("✅ 数据预处理完成，已缓存！")

# **3️⃣ 训练多个版本的模型**
def train_model(model_name, dataset_key, version_name):
    print(f"\n🚀 开始训练 {model_name} - {version_name}...")

    # 取已处理好的数据
    train_dataset = processed_datasets[dataset_key]["train"]
    eval_dataset = processed_datasets[dataset_key]["eval"]

    # 加载模型
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # 训练参数
    if model_name == "t5-small":
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, version_name),
            per_device_train_batch_size=12,
            per_device_eval_batch_size=12,
            gradient_accumulation_steps=1,
            fp16=True,
            num_train_epochs=5,
            save_total_limit=3,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=500,
            eval_steps=2000,
            report_to="none",
            dataloader_pin_memory=True,
            dataloader_num_workers=1,
        )
    else:  # T5-Base
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, version_name),
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            fp16=True,
            num_train_epochs=5,
            save_total_limit=3,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=500,
            eval_steps=2000,
            report_to="none",
            dataloader_pin_memory=True,
            dataloader_num_workers=1,
        )
    
    # 训练
    torch.backends.cudnn.benchmark = True
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    epochs = training_args.num_train_epochs
    eval_results = {}
    
    for epoch in tqdm(range(epochs), desc=f"Training {version_name}"):
        trainer.train()

        # 在 30%、50%、80% 进度点进行评估并保存模型
        for progress, tag in zip([0.3, 0.5, 0.8], ["30", "50", "80"]):
            if epoch == int(epochs * progress) - 1:
                save_path = os.path.join(output_dir, f"{version_name}_{tag}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"✅ {tag}% 训练完成，模型已保存至 {save_path}")
                eval_metrics = trainer.evaluate()
                eval_results[tag] = eval_metrics
                print(f"📊 {tag}% 评估结果: {eval_metrics}")
    
    # 100% 训练完成后评估并保存
    save_path_final = os.path.join(output_dir, f"{version_name}_100")
    model.save_pretrained(save_path_final)
    tokenizer.save_pretrained(save_path_final)
    eval_metrics = trainer.evaluate()
    eval_results["100"] = eval_metrics
    print(f"✅ 训练完成，完整模型已保存至 {save_path_final}！")
    print(f"📊 100% 评估结果: {eval_metrics}")
    
    # 保存评估结果到 JSON 文件
    eval_results_path = os.path.join(output_dir, f"{version_name}_eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"📁 评估结果已保存至 {eval_results_path}")

# **4️⃣ 训练所有版本的模型**
for model_key, model_name in model_names.items():
    for dataset_key in dataset_files.keys():
        version_name = f"{model_key}_{dataset_key}"
        train_model(model_name, dataset_key, version_name)
