import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# **1️⃣ 加载 T5-Small**
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# **2️⃣ 读取 CSV**
df = pd.read_csv("sentence_reorder_dataset.csv", on_bad_lines="skip")

# **3️⃣ 划分训练集 & 评估集**
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# **4️⃣ 预处理**
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

# **5️⃣ 训练参数（优化训练强度）**
training_args = TrainingArguments(
    output_dir="./t5_reorder_test_v3",
    per_device_train_batch_size=8,  # ✅ 提高 batch_size，利用更多显存
    per_device_eval_batch_size=8,  # ✅ 评估 batch_size 也提高
    gradient_accumulation_steps=1,  # ✅ 降低累积步长，加快更新频率，提高 GPU 计算负载
    fp16=True,  # ✅ 混合精度训练，减少显存占用
    num_train_epochs=5,  # ✅ 训练更久，提高质量
    save_total_limit=2,
    eval_strategy="epoch",
    save_strategy="epoch",  # ✅ 只在 epoch 结束时保存，减少写入磁盘开销
    logging_steps=500,  # ✅ 记录间隔适当
    eval_steps=2000,  # ✅ 评估间隔适当
    report_to="none",
    dataloader_pin_memory=True,  # ✅ 启用 pin_memory，提高数据传输效率
    dataloader_num_workers=1,  # ✅ 减少 CPU 线程，避免数据加载成为瓶颈
)

# **6️⃣ 训练**
torch.backends.cudnn.benchmark = True  # ✅ 让 cuDNN 自动优化计算
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()

# **7️⃣ 保存本地模型**
model.save_pretrained("./t5_reorder_test")
tokenizer.save_pretrained("./t5_reorder_test")
print("✅ 训练完成，模型已保存！")
