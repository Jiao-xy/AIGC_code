import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# **1️⃣ 加载 T5-Base**
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# **2️⃣ 读取 CSV**
df = pd.read_csv("sentence_reorder_dataset.csv", on_bad_lines="skip")

# **3️⃣ 仅选取部分数据用于训练（50%）**
df = df.sample(frac=0.8, random_state=42)  # ✅ 仅使用 50% 数据集
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# **4️⃣ 转换为 Hugging Face Dataset**
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# **5️⃣ 预处理函数**
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

# **6️⃣ 预处理数据**
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# **7️⃣ 训练参数（适配 T5-Base）**
training_args = TrainingArguments(
    output_dir="/home/jxy/models/t5_reorder_base_v2",  # ✅ 训练中间结果存放位置
    per_device_train_batch_size=4,  # ✅ T5-Base 显存占用高，batch_size 需降低
    per_device_eval_batch_size=4,  
    gradient_accumulation_steps=4,  # ✅ 使用梯度累积，等效 batch_size=16
    fp16=True,  # ✅ 开启混合精度训练，减少显存占用
    num_train_epochs=5,  
    save_total_limit=2,
    eval_strategy="epoch",
    save_strategy="epoch",  
    logging_steps=500,  
    eval_steps=2000,  
    report_to="none",
    dataloader_pin_memory=True,  
    dataloader_num_workers=1,  
)

# **8️⃣ 训练**
torch.backends.cudnn.benchmark = True  # ✅ 让 cuDNN 自动优化计算
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()

# **9️⃣ 保存最终模型**
save_path = "/home/jxy/models/t5_reorder_base_final_v2"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ 训练完成，模型已保存至 {save_path} ！")
