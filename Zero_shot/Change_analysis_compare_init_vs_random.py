import json
import pandas as pd
import os
import matplotlib.pyplot as plt

# **文件路径**
original_file = "/home/jxy/Data/Zero_shot/llscore_ppl/ieee-init_llscore_ppl.jsonl"
randomized_file = "/home/jxy/Data/Zero_shot/llscore_ppl_random/ieee-init_random_llscore_ppl.jsonl"

# **读取 JSONL 文件并转换为字典**
def read_jsonl(file_path):
    data_dict = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line.strip())
            data_dict[record["id"]] = record  # 以 id 作为键
    return data_dict

# **读取数据**
original_data = read_jsonl(original_file)
randomized_data = read_jsonl(randomized_file)

# **匹配数据**
comparison = []
for doc_id in original_data.keys():
    if doc_id in randomized_data:
        comparison.append({
            "id": doc_id,
            "LLScore_Original": original_data[doc_id]["LLScore"],
            "PPL_Original": original_data[doc_id]["PPL"],
            "LLScore_Randomized": randomized_data[doc_id]["LLScore"],
            "PPL_Randomized": randomized_data[doc_id]["PPL"],
            "LLScore_Change": randomized_data[doc_id]["LLScore"] - original_data[doc_id]["LLScore"],
            "PPL_Change": randomized_data[doc_id]["PPL"] - original_data[doc_id]["PPL"]
        })

# **转换为 DataFrame 并保存**
df = pd.DataFrame(comparison)
df.to_csv("llscore_ppl_comparison.csv", index=False)
print("数据对比结果已保存到 llscore_ppl_comparison.csv")

# **绘制 LLScore 和 PPL 的变化分布图**
plt.figure(figsize=(12, 5))

# **LLScore 变化**
plt.subplot(1, 2, 1)
plt.hist(df["LLScore_Change"], bins=50, color="blue", alpha=0.7, label="LLScore Change")
plt.xlabel("Change in LLScore")
plt.ylabel("Frequency")
plt.title("Distribution of LLScore Changes")
plt.legend()

# **PPL 变化**
plt.subplot(1, 2, 2)
plt.hist(df["PPL_Change"], bins=50, color="red", alpha=0.7, label="PPL Change")
plt.xlabel("Change in Perplexity (PPL)")
plt.ylabel("Frequency")
plt.title("Distribution of Perplexity (PPL) Changes")
plt.legend()

# **保存并显示图像**
image_path = "llscore_ppl_change_analysis.svg"
plt.savefig(image_path, format='svg')
plt.show()
print(f"统计图已保存至 {image_path}")
