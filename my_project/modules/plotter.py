#python -m modules.plotter
import numpy as np
import matplotlib.pyplot as plt
import os

class Plotter:
    def __init__(self):
        """初始化绘图工具"""

    def plot_llscore_ppl(self, results, file_path):
        """
        统计并绘制 LLScore 和 PPL 分布
        """
        llscores = [res["LLScore"] for res in results]
        ppls = [res["PPL"] for res in results]

        plt.figure(figsize=(12, 5))

        # **LLScore 分布**
        plt.subplot(1, 2, 1)
        plt.hist(llscores, bins=50, color="blue", alpha=0.7, label="LLScore")
        plt.xlabel("LLScore")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of LLScore ({os.path.basename(file_path)})")
        plt.legend()

        # **PPL 分布**
        plt.subplot(1, 2, 2)
        plt.hist(ppls, bins=50, color="red", alpha=0.7, label="PPL")
        plt.xlabel("Perplexity (PPL)")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Perplexity (PPL) ({os.path.basename(file_path)})")
        plt.legend()

        # **保存图片**
        image_path = f"data/tmp/{os.path.basename(os.path.splitext(file_path)[0])}_llscore_ppl.png"
        os.makedirs("results", exist_ok=True)
        plt.savefig(image_path, format='png')
        print(f"统计图已保存至 {image_path}")
