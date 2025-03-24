import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# **1️⃣ 加载 T5-Small**
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# **2️⃣ 用 pandas 读取 CSV**
df = pd.read_csv("sentence_reorder_dataset.csv")

# **3️⃣ pandas 转换为 Hugging Face Dataset**
dataset = Dataset.from_pandas(df)

# **4️⃣ 预处理**
def preprocess_function(examples):
    inputs = ["reorder: " + s for s in examples["乱序句子"]]
    targets = examples["正确句子"]
    model_inputs = tokenizer(
        inputs, max_length=128, truncation=True, padding="max_length"
    )  # 🚀 max_length 变长
    labels = tokenizer(
        targets, max_length=128, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess_function, batched=True)

# **5️⃣ 训练参数（适用于 RTX 2060）**
training_args = TrainingArguments(
    output_dir="./t5_reorder_test",
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=True,
    num_train_epochs=2, 
    save_total_limit=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=500,
)

# **6️⃣ 训练**
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

# **7️⃣ 保存本地模型**
model.save_pretrained("./t5_reorder_test")
tokenizer.save_pretrained("./t5_reorder_test")
print("✅ 本地调试完成，模型已保存")
