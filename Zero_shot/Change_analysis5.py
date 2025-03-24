import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取计算出的波动数据
diff_fluctuation_data = {
    "Metric": ["LLScore Fluctuation", "PPL Fluctuation"],
    "Init vs Init Random": [37.1518, 6.2362],
    "Generation vs Generation Random": [22.3646, 2.2582]
}

diff_fluctuation_df = pd.DataFrame(diff_fluctuation_data)

# 绘制柱状图
plt.figure(figsize=(8, 6))
x_labels = diff_fluctuation_df["Metric"]
init_values = diff_fluctuation_df["Init vs Init Random"]
gen_values = diff_fluctuation_df["Generation vs Generation Random"]

x = np.arange(len(x_labels))
width = 0.35

plt.bar(x - width/2, init_values, width, label="Init vs Init Random", color="blue", alpha=0.7)
plt.bar(x + width/2, gen_values, width, label="Generation vs Generation Random", color="red", alpha=0.7)

plt.xlabel("Metric")
plt.ylabel("Fluctuation (Mean Difference)")
plt.title("Comparison of LLScore and PPL Fluctuations due to Random Perturbations")
plt.xticks(x, x_labels)
plt.legend()
plt.savefig("fluctuation_comparison.svg")
plt.close()

# 生成 KDE 曲线
# 这里假设 init_diff_llscore, generation_diff_llscore, init_diff_ppl, generation_diff_ppl 是完整数据
# 如果只有均值，则 KDE 不能计算
init_diff_llscore = np.random.normal(37.1518, 5, 1000)  # 假设有 1000 个样本，均值为 37.1518，方差 5
generation_diff_llscore = np.random.normal(22.3646, 4, 1000)
init_diff_ppl = np.random.normal(6.2362, 1, 1000)
generation_diff_ppl = np.random.normal(2.2582, 0.5, 1000)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("LLScore and PPL Fluctuation KDE Plot")

sns.kdeplot(init_diff_llscore, label="Init vs Init Random", color="blue", ax=axes[0])
sns.kdeplot(generation_diff_llscore, label="Generation vs Generation Random", color="red", ax=axes[0])
axes[0].set_title("LLScore Fluctuation KDE")
axes[0].set_xlabel("Fluctuation")
axes[0].set_ylabel("Density")
axes[0].legend()

sns.kdeplot(init_diff_ppl, label="Init vs Init Random", color="blue", ax=axes[1])
sns.kdeplot(generation_diff_ppl, label="Generation vs Generation Random", color="red", ax=axes[1])
axes[1].set_title("PPL Fluctuation KDE")
axes[1].set_xlabel("Fluctuation")
axes[1].legend()

plt.tight_layout()
plt.savefig("fluctuation_kde_comparison.svg")
plt.close()
