import pandas as pd
import json
import numpy as np

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

# 计算均值
diff_fluctuation_data = {
    "Metric": ["LLScore Fluctuation", "PPL Fluctuation"],
    "Init vs Init Random": [init_diff_llscore.mean(), init_diff_ppl.mean()],
    "Generation vs Generation Random": [generation_diff_llscore.mean(), generation_diff_ppl.mean()]
}

diff_fluctuation_df = pd.DataFrame(diff_fluctuation_data)

# 输出结果
print(diff_fluctuation_df)

# 保存为 CSV 以便查看
diff_fluctuation_df.to_csv("random_difference_fluctuation.csv", index=False)
