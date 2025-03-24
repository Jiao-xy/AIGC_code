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
df = pd.read_csv("sentence_reorder_dataset.csv")

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
    )  # 🚀 max_length 适应长句子
    labels = tokenizer(
        targets, max_length=128, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# **6️⃣ 训练参数（适用于 RTX 2060）**
training_args = TrainingArguments(
    output_dir="./t5_reorder_test",
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=True,
    num_train_epochs=2, 
    save_total_limit=2,
    evaluation_strategy="epoch",  # ✅ 现在有 eval_dataset，可以开启评估
    save_strategy="epoch",
    logging_steps=500,
    eval_steps=500,  # 每 500 step 评估一次
    #evaluation_strategy="epoch",
    report_to="none",  # 禁止上传日志到 wandb
)

# **7️⃣ 训练**
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # ✅ 传入 eval_dataset
)
trainer.train()

# **8️⃣ 保存本地模型**
model.save_pretrained("./t5_reorder_test")
tokenizer.save_pretrained("./t5_reorder_test")
print("✅ 训练完成，模型已保存！")
