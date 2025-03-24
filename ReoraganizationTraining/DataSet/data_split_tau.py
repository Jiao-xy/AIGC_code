import pandas as pd
from tqdm import tqdm  # 进度条库

# **1️⃣ 读取原始数据**
input_file = "sentence_shuffled_dataset.csv"
df = pd.read_csv(input_file, on_bad_lines="skip")

# **2️⃣ 填充 NaN**
df.fillna("", inplace=True)

# **3️⃣ 生成不同 tau 的数据集**
tau_levels = [0.8, 0.5, 0.2]

for tau in tqdm(tau_levels, desc="⏳ 处理不同 tau 数据集"):
    tau_col = f"打乱句子_0{int(tau * 10)}"
    if tau_col in df.columns:
        # **4️⃣ 逐行处理数据，显示进度**
        subset_data = []
        for idx in tqdm(range(len(df)), desc=f"📄 处理 tau={tau}", leave=False):
            shuffled_sentence = str(df[tau_col].iloc[idx]).strip()
            correct_sentence = str(df["原句"].iloc[idx]).strip()
            
            if shuffled_sentence and correct_sentence:
                subset_data.append({"乱序句子": shuffled_sentence, "正确句子": correct_sentence})

        # **5️⃣ 转换为 DataFrame 并保存**
        subset_df = pd.DataFrame(subset_data)

        if len(subset_df) == 0:
            print(f"⚠️ tau={tau} 没有有效数据，跳过保存！")
            continue

        output_file = f"sentence_shuffled_dataset_tau_0{int(tau * 10)}.csv"
        print(f"📂 正在保存 {output_file}，数据行数: {len(subset_df)}")
        subset_df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"✅ {output_file} 保存完成！")
    else:
        print("啥也没有")
print("🎉 所有 `tau` 数据集拆分完成！")
