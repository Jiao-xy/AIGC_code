import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# 计算横轴范围（剔除极端值）
def get_trimmed_limits(data, trim_percent=0.01):
    lower = np.percentile(data, trim_percent * 100)
    upper = np.percentile(data, (1 - trim_percent) * 100)
    return lower, upper

llscore_lower, llscore_upper = get_trimmed_limits(pd.concat([init_df["LLScore"], generation_df["LLScore"]]))
ppl_lower, ppl_upper = get_trimmed_limits(pd.concat([init_df["PPL"], generation_df["PPL"]]))

# 生成 Init 和 Generation 直接对比图（调整横轴范围）
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Init vs Generation Comparison")

sns.histplot(init_df["LLScore"], kde=True, label="Init LLScore", color="blue", ax=axes[0])
sns.histplot(generation_df["LLScore"], kde=True, label="Generation LLScore", color="red", ax=axes[0])
axes[0].set_title("Init vs Generation LLScore")
axes[0].set_xlabel("LLScore")
axes[0].set_ylabel("Frequency")
axes[0].set_xlim(llscore_lower, llscore_upper)

sns.histplot(init_df["PPL"], kde=True, label="Init PPL", color="blue", ax=axes[1])
sns.histplot(generation_df["PPL"], kde=True, label="Generation PPL", color="red", ax=axes[1])
axes[1].set_title("Init vs Generation PPL")
axes[1].set_xlabel("PPL")
axes[1].set_xlim(ppl_lower, ppl_upper)

for ax in axes.flat:
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("init_vs_generation_trimmed.svg")
plt.close()