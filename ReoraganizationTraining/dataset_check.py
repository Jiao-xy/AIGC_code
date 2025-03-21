import pandas as pd
import csv

def load_and_clean_dataset(file_path, expected_columns, max_length=512):
    """ 读取并清理数据集，并统计过滤的数据量 """
    print(f"\n📥 加载数据集: {file_path}")
    
    try:
        # 读取 CSV，并跳过格式错误的行
        df = pd.read_csv(file_path, on_bad_lines="skip", quoting=csv.QUOTE_NONE)
        total_rows = len(df)  # 读取后总行数
        
        # 1️⃣ 检查是否包含预期的列
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            print(f"⚠️ 警告：数据集缺少列 {missing_cols}，跳过该数据集。")
            return None

        # 2️⃣ 删除 NaN 值
        before_drop = len(df)
        df = df.dropna()
        after_drop = len(df)
        dropped_nan = before_drop - after_drop
        
        # 3️⃣ 过滤超长文本（> 512）
        before_filter = len(df)
        df["input_length"] = df[expected_columns[0]].astype(str).apply(len)
        df["target_length"] = df[expected_columns[1]].astype(str).apply(len)
        df = df[(df["input_length"] <= max_length) & (df["target_length"] <= max_length)]
        after_filter = len(df)
        dropped_long_text = before_filter - after_filter
        
        # 统计过滤比例
        print(f"🔹 总数据量: {total_rows}")
        print(f"⚠️ 过滤 NaN 数据: {dropped_nan} 行")
        print(f"⚠️ 过滤超长文本 (>512 tokens): {dropped_long_text} 行")
        print(f"✅ 过滤后剩余数据: {after_filter} 行")
        
        return df  # 返回清理后的数据集

    except Exception as e:
        print(f"❌ 读取数据集 {file_path} 失败: {e}")
        return None


# **运行数据集检查**
dataset_files = {
    "tau_08": "/home/jxy/Data/ReoraganizationData/sentence_shuffled_dataset_tau_08.csv",
    "full": "/home/jxy/Data/ReoraganizationData/sentence_reorder_dataset.csv"
}

expected_columns = ["乱序句子", "正确句子"]

filtered_datasets = {}

for name, path in dataset_files.items():
    print(f"\n🔍 正在检查数据集：{name}")
    filtered_datasets[name] = load_and_clean_dataset(path, expected_columns)

print("\n📌 数据集检测完成，若无错误，则可用于训练！")
