import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.manifold import TSNE

# 读取 JSONL 文件
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# 文件路径（修改为实际路径）
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

# 计算原始数据和带 random 后缀的数据的差值波动
init_diff_llscore = (init_df["LLScore"] - init_random_df["LLScore"]).abs()
generation_diff_llscore = (generation_df["LLScore"] - generation_random_df["LLScore"]).abs()

init_diff_ppl = (init_df["PPL"] - init_random_df["PPL"]).abs()
generation_diff_ppl = (generation_df["PPL"] - generation_random_df["PPL"]).abs()

# 方法 1: Box-Cox 变换
init_diff_llscore_boxcox, _ = stats.boxcox(init_diff_llscore + 1)
generation_diff_llscore_boxcox, _ = stats.boxcox(generation_diff_llscore + 1)
init_diff_ppl_boxcox, _ = stats.boxcox(init_diff_ppl + 1)
generation_diff_ppl_boxcox, _ = stats.boxcox(generation_diff_ppl + 1)

# 方法 2: 数值映射 (Rescale)
def rescale(data, new_min, new_max):
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * (new_max - new_min) + new_min

init_diff_llscore_rescaled = rescale(init_diff_llscore, 0, 50)
generation_diff_llscore_rescaled = rescale(generation_diff_llscore, 60, 110)
init_diff_ppl_rescaled = rescale(init_diff_ppl, 0, 10)
generation_diff_ppl_rescaled = rescale(generation_diff_ppl, 15, 25)

# 方法 3: t-SNE 降维
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_data = np.vstack([
    np.column_stack((init_diff_llscore, init_diff_ppl)),
    np.column_stack((generation_diff_llscore, generation_diff_ppl))
])
labels = np.array([0] * len(init_diff_llscore) + [1] * len(generation_diff_llscore))
tsne_result = tsne.fit_transform(tsne_data)

# 绘制图像
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Comparison of Different Transformations")

# Box-Cox KDE
sns.kdeplot(init_diff_llscore_boxcox, label="Init vs Init Random", color="blue", ax=axes[0])
sns.kdeplot(generation_diff_llscore_boxcox, label="Generation vs Generation Random", color="red", ax=axes[0])
axes[0].set_title("Box-Cox Transformed Fluctuation KDE")
axes[0].set_xlabel("Transformed Fluctuation")
axes[0].legend()

# Rescaled KDE
sns.kdeplot(init_diff_llscore_rescaled, label="Init vs Init Random", color="blue", ax=axes[1])
sns.kdeplot(generation_diff_llscore_rescaled, label="Generation vs Generation Random", color="red", ax=axes[1])
axes[1].set_title("Rescaled Fluctuation KDE")
axes[1].set_xlabel("Rescaled Fluctuation")
axes[1].legend()

# t-SNE Scatter Plot
axes[2].scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap="coolwarm", alpha=0.7)
axes[2].set_title("t-SNE Visualization")
axes[2].set_xlabel("t-SNE Dim1")
axes[2].set_ylabel("t-SNE Dim2")

plt.tight_layout()
plt.savefig("comparison_transformations.svg")
plt.close()
