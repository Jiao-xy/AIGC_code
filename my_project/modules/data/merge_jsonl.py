# python -m modules.data.merge_jsonl
# 将多个 JSONL 文件合并为一个，并添加 "source" 字段标记来源

from modules.utils.jsonl_handler import read_jsonl, save_results

def merge_jsonl_with_source(file_paths, source_labels, output_path):
    """
    合并多个 JSONL 文件，并为每条记录添加 source 字段。

    参数：
    - file_paths: List[str]，每个 JSONL 文件的路径
    - source_labels: List[str]，与 file_paths 一一对应，用于标记来源
    - output_path: str，合并后输出的文件路径
    """
    assert len(file_paths) == len(source_labels), "文件路径与来源标签数量不一致"

    all_data = []
    for path, label in zip(file_paths, source_labels):
        data = read_jsonl(path)
        for item in data:
            item["source"] = label
        all_data.extend(data)
        print(f"读取 {path} 共 {len(data)} 条记录，标记来源为: {label}")

    save_results(all_data, output_path)
    print(f"已合并总计 {len(all_data)} 条记录，保存至 {output_path}")
    return all_data

# ✅ 示例运行方式
if __name__ == "__main__":
    file_paths = [
        # "data/init/split/ieee-init-split.jsonl",
        # "data/init/split/ieee-chatgpt-generation-split.jsonl",
        "data/init/ieee-init_split.jsonl",
        "data/init/ieee-chatgpt-generation_split.jsonl"
          
    ]
    source_labels = [
        "init",
        "generation",

    ]
    output_path = "data/ieee-merged.jsonl"

    merge_jsonl_with_source(file_paths, source_labels, output_path)
