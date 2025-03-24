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

# 生成直方图和密度曲线
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Log Probability Change (Perturbation Discrepancy)")

hist_params = {"bins": 50, "alpha": 0.7, "density": True}

sns.histplot(init_df["LLScore"], kde=True, label="Init", color="blue", ax=axes[0, 0])
sns.histplot(init_random_df["LLScore"], kde=True, label="Init Random", color="orange", ax=axes[0, 0])
axes[0, 0].set_title("Init vs Init Random")
axes[0, 0].set_ylabel("Frequency")

sns.histplot(generation_df["LLScore"], kde=True, label="Generation", color="blue", ax=axes[0, 1])
sns.histplot(generation_random_df["LLScore"], kde=True, label="Generation Random", color="orange", ax=axes[0, 1])
axes[0, 1].set_title("Generation vs Generation Random")

sns.histplot(init_df["PPL"], kde=True, label="Init", color="blue", ax=axes[1, 0])
sns.histplot(init_random_df["PPL"], kde=True, label="Init Random", color="orange", ax=axes[1, 0])
axes[1, 0].set_title("Init PPL vs Init Random PPL")
axes[1, 0].set_xlabel("Log Probability Change")
axes[1, 0].set_ylabel("Frequency")

sns.histplot(generation_df["PPL"], kde=True, label="Generation", color="blue", ax=axes[1, 1])
sns.histplot(generation_random_df["PPL"], kde=True, label="Generation Random", color="orange", ax=axes[1, 1])
axes[1, 1].set_title("Generation PPL vs Generation Random PPL")
axes[1, 1].set_xlabel("Log Probability Change")

for ax in axes.flat:
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("perturbation_discrepancy.svg")
plt.close()
