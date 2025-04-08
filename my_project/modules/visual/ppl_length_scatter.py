# python -m modules.visual.ppl_length_scatter
# modules/visual/ppl_length_scatter.py
# 使用 jsonl_handler 加载数据并绘制多图联合图像（句长、PPL、LLScore）

import matplotlib.pyplot as plt
import numpy as np
import os
from modules.utils.jsonl_handler import read_jsonl


def extract_metrics(data):
    lengths, ppls, lls = [], [], []
    for item in data:
        if "PPL" in item and "word_count" in item and "LLScore" in item:
            lengths.append(item["word_count"])
            ppls.append(item["PPL"])
            lls.append(item["LLScore"])
    return lengths, ppls, lls

def remove_outliers(lengths, ppls, lls, iqr_scale=1.5):
    """使用 IQR 方法剔除 PPL 中的异常值（先清洗非法项）"""
    ppls_arr = np.array(ppls)
    lengths_arr = np.array(lengths)
    lls_arr = np.array(lls)

    # ✅ 统一清除 NaN/Inf
    mask_valid = np.isfinite(ppls_arr)
    ppls_arr = ppls_arr[mask_valid]
    lengths_arr = lengths_arr[mask_valid]
    lls_arr = lls_arr[mask_valid]

    print(f"[IQR] 合法样本数: {len(ppls_arr)}")

    q1 = np.percentile(ppls_arr, 25)
    q3 = np.percentile(ppls_arr, 75)
    iqr = q3 - q1
    lower = q1 - iqr_scale * iqr
    upper = q3 + iqr_scale * iqr

    print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}, Lower: {lower}, Upper: {upper}")

    mask = (ppls_arr >= lower) & (ppls_arr <= upper)

    print(f"原始样本: {len(lengths)}, 清洗后: {len(ppls_arr)}, 保留: {mask.sum()} 条")

    return (
        lengths_arr[mask].astype(int).tolist(),
        ppls_arr[mask].tolist(),
        lls_arr[mask].tolist()
    )

def plot_ppl_summary(input_path, output_path, filter_outliers=False):
    data = read_jsonl(input_path)
    lengths, ppls, lls = extract_metrics(data)
    print(f"PPL长度: {len(ppls)}, 示例: {ppls[:5]}")

    if filter_outliers:
        lengths, ppls, lls = remove_outliers(lengths, ppls, lls, iqr_scale=1.5)

    if not lengths:
        print("未找到包含 PPL、LLScore 和 word_count 的有效句子。")
        return

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    title_prefix = "Filtered " if filter_outliers else ""
    fig.suptitle(f"{title_prefix}Sentence Statistics Summary", fontsize=16)

    # 1. 句长 vs PPL 散点图
    axs[0, 0].scatter(lengths, ppls, alpha=0.5, edgecolors='k', s=40)
    axs[0, 0].set_title("Sentence Length vs PPL")
    axs[0, 0].set_xlabel("Word Count")
    axs[0, 0].set_ylabel("PPL")
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)

    # 加趋势线（去除非法值）
    try:
        clean_data = [(l, p) for l, p in zip(lengths, ppls) if np.isfinite(l) and np.isfinite(p)]
        if len(clean_data) >= 2:
            x_vals, y_vals = zip(*clean_data)
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            axs[0, 0].plot(sorted(x_vals), p(sorted(x_vals)), color='red', label='Trend')
            axs[0, 0].legend()
    except Exception as e:
        print(f"⚠️ 趋势线绘制失败: {e}")

    # 2. PPL 分布图
    axs[0, 1].hist(ppls, bins=50, alpha=0.7, color='red', edgecolor='black')
    axs[0, 1].axvline(np.mean(ppls), color='blue', linestyle='--', label=f"Mean={np.mean(ppls):.2f}")
    axs[0, 1].set_title("PPL Distribution")
    axs[0, 1].set_xlabel("PPL")
    axs[0, 1].set_ylabel("Frequency")
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)
    axs[0, 1].legend()

    # 3. LLScore 分布图
    axs[1, 0].hist(lls, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axs[1, 0].axvline(np.mean(lls), color='red', linestyle='--', label=f"Mean={np.mean(lls):.2f}")
    axs[1, 0].set_title("LLScore Distribution")
    axs[1, 0].set_xlabel("LLScore")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)
    axs[1, 0].legend()

    # 4. 句长分布图
    axs[1, 1].hist(lengths, bins=range(int(min(lengths)), int(max(lengths)) + 1), alpha=0.7, color='green', edgecolor='black')
    axs[1, 1].axvline(np.mean(lengths), color='orange', linestyle='--', label=f"Mean={np.mean(lengths):.2f}")
    axs[1, 1].set_title("Sentence Length Distribution")
    axs[1, 1].set_xlabel("Word Count")
    axs[1, 1].set_ylabel("Frequency")
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"图像已保存至 {output_path}")


if __name__ == "__main__":
    plot_ppl_summary(
        input_path="data/init/ieee-init_split.jsonl",
        output_path="data/init/ppl_summary_human.png"
    )
    plot_ppl_summary(
        input_path="data/init/ieee-chatgpt-generation_split.jsonl",
        output_path="data/init/ppl_summary_gpt.png"
    )
    # 额外去除异常值版本（仅基于 PPL IQR）
    plot_ppl_summary(
        input_path="data/init/ieee-init_split.jsonl",
        output_path="data/init/ppl_summary_human_filtered.png",
        filter_outliers=True
    )
    plot_ppl_summary(
        input_path="data/init/ieee-chatgpt-generation_split.jsonl",
        output_path="data/init/ppl_summary_gpt_filtered.png",
        filter_outliers=True
    )
