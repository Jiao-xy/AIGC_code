# modules/training/t5_base_group_training.py
# 分组训练 T5 模型：每个 group_id 单独提取并训练，保存模型与 loss 曲线图

import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset

from modules.utils.jsonl_handler import read_jsonl
from modules.models.manager import ModelManager

# 【参数设置】
model_name = "t5-base"  # 使用的预训练模型

# 输入数据文件，内容应包含 grouped shuffle 的每条元素（含 group_id 信息）
dataset_path = "data/train_pairs/grouped_shuffle_all.jsonl"

output_root = "models/grouped"  # 模型输出根目录
version_prefix = "t5-base_group"  # 模型保存文件前缀

target_groups = [0, 1]  # 指定要训练的 group_id 
num_epochs = 3  # 训练轮数

# 【加载 Tokenizer 和模型，使用 ModelManager 防止重复载入】
if ModelManager.is_model_loaded(model_name):
    model, tokenizer = ModelManager.get_model(model_name)
else:
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    ModelManager.register_model(model_name, model, tokenizer)

# 【分组数据加载 + 处理】
def load_group_data(group_id, all_data):
    """根据 group_id 过滤数据，并进行分类输入和标签编码。返回 train/test 分割
    """
    group_data = [item for item in all_data if item.get("metadata", {}).get("group_id") == group_id]
    dataset = Dataset.from_list(group_data)

    def preprocess_function(examples):
        inputs = ["reorder: " + s for s in examples["shuffled"]]  # prompt prefix
        targets = examples["original"]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = dataset.map(preprocess_function, batched=True)
    return dataset.train_test_split(test_size=0.2, seed=42)["train"], dataset.train_test_split(test_size=0.2, seed=42)["test"]

# 【保存 loss 曲线图】
def plot_and_save_loss(logs, save_path):
    steps = list(range(1, len(logs) + 1))
    losses = [entry["loss"] for entry in logs]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, label="Train Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 【训练单个 group_id 模型】
def train_model_on_group(group_id, all_data):
    print(f"\n🚀 正在训练 Group {group_id}...")
    train_dataset, eval_dataset = load_group_data(group_id, all_data)

    group_tag = f"{version_prefix}_{group_id}"
    output_dir = os.path.join(output_root, group_tag)
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        fp16=True,
        num_train_epochs=num_epochs,
        save_total_limit=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        eval_steps=500,
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

    # ✅ 记录 loss
    loss_log = []
    def log_callback(info):
        if "loss" in info:
            loss_log.append({"step": info.get("step", len(loss_log)), "loss": info["loss"]})

    trainer.add_callback(
        type("LossLogger", (), {
            "on_log": lambda self, args, state, control, logs=None, **kwargs: log_callback(logs or {})
        })()
    )

    trainer.train()

    # ✅ 保存模型
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ 模型已保存至 {output_dir}")

    # ✅ 保存 loss 图
    plot_path = os.path.join(output_dir, "loss_curve.png")
    plot_and_save_loss(loss_log, plot_path)
    print(f"📈 Loss 图已保存至 {plot_path}")

    # ✅ 保存评估结果
    eval_result = trainer.evaluate()
    eval_path = os.path.join(output_dir, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_result, f, indent=4)
    print(f"📊 评估结果已保存至 {eval_path}")

# 【正式执行训练】
if __name__ == "__main__":
    all_data = read_jsonl(dataset_path)
    for gid in target_groups:
        train_model_on_group(gid, all_data)