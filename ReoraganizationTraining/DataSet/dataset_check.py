import json

jsonl_files = {
    "tau_08": "/home/jxy/Data/ReoraganizationData/sentence_shuffled_dataset_tau_08.jsonl",
    "reorder": "/home/jxy/Data/ReoraganizationData/sentence_reorder_dataset.jsonl",
}

for name, file in jsonl_files.items():
    print(f"\n🔍 正在检查文件: {file}")

    with open(file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())  # 解析 JSON
                if "shuffled_sentence" not in data or "original_sentence" not in data:
                    print(f"❌ 第 {i+1} 行缺少关键字段: {line}")
                if not data["original_sentence"].strip():
                    print(f"⚠️ 第 {i+1} 行原句为空: {line}")
                if not data["shuffled_sentence"].strip():
                    print(f"⚠️ 第 {i+1} 行打乱句为空: {line}")

            except json.JSONDecodeError:
                print(f"❌ 第 {i+1} 行 JSON 解析失败: {line}")

print("\n✅ `tau_08` 和 `reorder` 数据集检查完成！")

import random


# 读取数据
datasets = {}
for name, file in jsonl_files.items():
    with open(file, "r", encoding="utf-8") as f:
        datasets[name] = [json.loads(line.strip()) for line in f.readlines()]

# 随机抽样 10 组对比
samples = random.sample(datasets["tau_08"], min(10, len(datasets["tau_08"])))

print("\n🔍 对比 `tau_08` 和 `reorder` 的句子:")
for i, sample in enumerate(samples):
    original = sample["original_sentence"]
    shuffled = sample["shuffled_sentence"]
    
    # 找到 `reorder` 数据集中相同 `original_sentence`
    reorder_match = next((x for x in datasets["reorder"] if x["original_sentence"] == original), None)

    print(f"\n【样本 {i+1} 】")
    print(f"原句 (Reorder): {original}")
    print(f"打乱句 (tau_08): {shuffled}")
    print(f"✅ 匹配情况: {'匹配成功 ✅' if reorder_match else '❌ 原句在 `reorder` 中未找到'}")
