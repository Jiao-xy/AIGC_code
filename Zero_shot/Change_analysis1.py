import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

# 读取 JSONL 文件
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# 文件路径
init_file_path = "/home/jxy/Data/Zero_shot/llscore_ppl/ieee-init_llscore_ppl.jsonl"
generation_file_path = "/home/jxy/Data/Zero_shot/llscore_ppl/ieee-chatgpt-generation_llscore_ppl.jsonl"
init_random_file_path = "/home/jxy/Data/Zero_shot/llscore_ppl_random/ieee-init_random_llscore_ppl.jsonl"
generation_random_file_path = "/home/jxy/Data/Zero_shot/llscore_ppl_random/ieee-chatgpt-generation_random_llscore_ppl.jsonl"

# 加载数据
init_df = load_jsonl(init_file_path)
generation_df = load_jsonl(generation_file_path)
init_random_df = load_jsonl(init_random_file_path)
generation_random_df = load_jsonl(generation_random_file_path)

# 确保数据按照相同 ID 排序
for df in [init_df, generation_df, init_random_df, generation_random_df]:
    df.sort_values(by="id", inplace=True)

# 计算波动情况
def compute_fluctuation(df):
    df["LLScore_Diff"] = df["LLScore"].diff().abs()
    df["PPL_Diff"] = df["PPL"].diff().abs()
    return df

init_df = compute_fluctuation(init_df)
generation_df = compute_fluctuation(generation_df)
init_random_df = compute_fluctuation(init_random_df)
generation_random_df = compute_fluctuation(generation_random_df)

# 计算均值
fluctuation_data = {
    "Metric": ["LLScore Fluctuation", "PPL Fluctuation"],
    "Init": [init_df["LLScore_Diff"].mean(), init_df["PPL_Diff"].mean()],
    "Generation": [generation_df["LLScore_Diff"].mean(), generation_df["PPL_Diff"].mean()],
    "Init Random": [init_random_df["LLScore_Diff"].mean(), init_random_df["PPL_Diff"].mean()],
    "Generation Random": [generation_random_df["LLScore_Diff"].mean(), generation_random_df["PPL_Diff"].mean()],
}

fluctuation_df = pd.DataFrame(fluctuation_data)
print(fluctuation_df)

# 生成 LLScore 波动柱状图并保存
plt.figure(figsize=(8, 6))
x_labels = ["LLScore Fluctuation"]
values = [fluctuation_data["Init"][0], fluctuation_data["Generation"][0], fluctuation_data["Init Random"][0], fluctuation_data["Generation Random"][0]]
labels = ["Init", "Generation", "Init Random", "Generation Random"]

x = np.arange(len(x_labels))
width = 0.2

for i, (label, val) in enumerate(zip(labels, values)):
    plt.bar(x + (i - 1.5) * width, [val], width, label=label)

plt.xlabel("Metric")
plt.ylabel("Fluctuation (Mean Difference)")
plt.title("Comparison of LLScore Fluctuations")
plt.xticks(x, x_labels)
plt.legend()
plt.savefig("llscore_fluctuation.svg")
plt.close()

# 生成 PPL 波动柱状图并保存
plt.figure(figsize=(8, 6))
x_labels = ["PPL Fluctuation"]
values = [fluctuation_data["Init"][1], fluctuation_data["Generation"][1], fluctuation_data["Init Random"][1], fluctuation_data["Generation Random"][1]]

x = np.arange(len(x_labels))

for i, (label, val) in enumerate(zip(labels, values)):
    plt.bar(x + (i - 1.5) * width, [val], width, label=label)

plt.xlabel("Metric")
plt.ylabel("Fluctuation (Mean Difference)")
plt.title("Comparison of PPL Fluctuations")
plt.xticks(x, x_labels)
plt.legend()
plt.savefig("ppl_fluctuation.svg")
plt.close()

# 生成 LLScore 变化曲线图并保存
plt.figure(figsize=(10, 5))
plt.plot(init_df["id"], init_df["LLScore"], label="Init LLScore", marker="o", linestyle="-")
plt.plot(generation_df["id"], generation_df["LLScore"], label="Generation LLScore", marker="s", linestyle="--")
plt.plot(init_random_df["id"], init_random_df["LLScore"], label="Init Random LLScore", marker="^", linestyle=":")
plt.plot(generation_random_df["id"], generation_random_df["LLScore"], label="Generation Random LLScore", marker="x", linestyle="-." )
plt.xlabel("ID")
plt.ylabel("LLScore")
plt.title("LLScore Variation Comparison")
plt.legend()
plt.savefig("llscore_variation.svg")
plt.close()

# 生成 PPL 变化曲线图并保存
plt.figure(figsize=(10, 5))
plt.plot(init_df["id"], init_df["PPL"], label="Init PPL", marker="o", linestyle="-")
plt.plot(generation_df["id"], generation_df["PPL"], label="Generation PPL", marker="s", linestyle="--")
plt.plot(init_random_df["id"], init_random_df["PPL"], label="Init Random PPL", marker="^", linestyle=":")
plt.plot(generation_random_df["id"], generation_random_df["PPL"], label="Generation Random PPL", marker="x", linestyle="-." )
plt.xlabel("ID")
plt.ylabel("PPL")
plt.title("PPL Variation Comparison")
plt.legend()
plt.savefig("ppl_variation.svg")
plt.close()
