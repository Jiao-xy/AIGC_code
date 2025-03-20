import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# **1️⃣ 加载 T5-Small**
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# **2️⃣ 用 pandas 读取 CSV**
#df = pd.read_csv("sentence_reorder_dataset.csv")
df = pd.read_csv("sentence_reorder_dataset.csv", on_bad_lines="skip")


# **3️⃣ 划分训练集 & 评估集（80% 训练，20% 评估）**
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# **4️⃣ pandas 转换为 Hugging Face Dataset**
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# **5️⃣ 预处理**
def preprocess_function(examples):
    inputs = ["reorder: " + s for s in examples["乱序句子"]]
    targets = examples["正确句子"]
    model_inputs = tokenizer(
        inputs, max_length=128, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        targets, max_length=128, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# **6️⃣ 训练参数（适配 RTX 2060，优化 GPU 利用率）**
training_args = TrainingArguments(
    output_dir="./t5_reorder_test_v2",
    per_device_train_batch_size=6,  # ✅ 增大 batch_size，提高 GPU 计算利用率
    per_device_eval_batch_size=6,  # ✅ 评估 batch_size 也增大
    gradient_accumulation_steps=2,  # ✅ 累积梯度，减小显存占用
    fp16=True,  # ✅ 启用混合精度训练，减少显存占用
    num_train_epochs=5,  # ✅ 增加训练轮数，让电脑长时间跑
    save_total_limit=2,
    eval_strategy="epoch",
    save_strategy="epoch",  # ✅ 只在 epoch 结束时保存，避免频繁写入磁盘
    logging_steps=1000,  # ✅ 降低日志频率，减少 CPU 负载
    eval_steps=2000,  # ✅ 降低评估频率，提高训练效率
    report_to="none",
    dataloader_pin_memory=True,  # ✅ 启用 pin_memory，提高数据加载速度
    dataloader_num_workers=2,  # ✅ 启用多线程数据加载，提高效率
)

# **7️⃣ 训练**
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()

# **8️⃣ 保存本地模型**
model.save_pretrained("./t5_reorder_test")
tokenizer.save_pretrained("./t5_reorder_test")
print("✅ 训练完成，模型已保存！")
