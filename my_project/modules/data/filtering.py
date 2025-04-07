#python -m modules.data.filtering
#过滤句子长度不合规项
import matplotlib.pyplot as plt
import numpy as np
from modules.utils.jsonl_handler import read_jsonl, save_results

def process(file_path, min_words=4, max_words=80):
    data = read_jsonl(file_path)
    filtered = []

    for item in data:
        sentence = item.get("sentence", "").strip()
        word_count = len(sentence.split())
        if min_words <= word_count <= max_words:
            item["word_count"] = word_count  # 添加词数字段
            filtered.append(item)

    # 按照词数从少到多排序
    filtered.sort(key=lambda x: x["word_count"])

    # 保存结果
    output_file = file_path.replace(".jsonl", "-filtered.jsonl")
    save_results(filtered, output_path=output_file)
    print(f"过滤后保留 {len(filtered)} 条句子，已保存至 {output_file}")

    # 生成统计图
    word_counts = [item["word_count"] for item in filtered]
    plt.figure(figsize=(10, 5))
    plt.hist(word_counts, bins=range(min_words, max_words + 2), edgecolor='black', alpha=0.75)
    plt.title(f"Sentence Word Count Distribution ({len(filtered)} sentences)")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.xticks(np.arange(min_words, max_words + 1, 2))
    plt.grid(True, linestyle='--', alpha=0.5)

    image_file = output_file.replace(".jsonl", "_wordcount_hist.png")
    plt.savefig(image_file)
    plt.close()
    print(f"词数统计图已保存至 {image_file}")

    return filtered


if __name__ == "__main__":
    # process("data/modules_test_data/test_split.jsonl")
    """ path=("data/init/split/ieee-init-split.jsonl",
          "data/init/split/ieee-chatgpt-generation-split.jsonl",
          "data/init/split/ieee-chatgpt-fusion-split.jsonl",
          "data/init/split/ieee-chatgpt-polish-split.jsonl")
    for i in path:
        process(i)
        print(f"处理完成：{i}") """
    process("data/ieee-merged.jsonl")