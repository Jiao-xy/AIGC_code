# modules/data/splitter.py
# 使用 SentenceSegmenter 批量分句并保存 LLScore 与 PPL

from modules.utils.jsonl_handler import read_jsonl, save_results
from modules.data.sentence_segmenter import SentenceSegmenter
from tqdm import tqdm

def process(file_path):
    # 初始化分句器：启用引用合并、PPL + LLScore 合并、自动阈值估算
    segmenter = SentenceSegmenter(
        enable_reference_merge=True,
        enable_ppl_merge=True,
        auto_threshold=True,               # ✅ 启用自动估算
        threshold_strategy="percentile",   # 可选 "robust" 或 "std"
        max_short_len=6
    )

    data = read_jsonl(file_path)
    results = []

    for item in tqdm(data, desc=f"Splitting {file_path}"):
        doc_id = item.get("id")
        text = item.get("abstract", "")
        segmented = segmenter.segment(text)

        for i, (sent, (ll, ppl)) in enumerate(segmented):
            results.append({
                "id": doc_id,
                "sentence_id": f"{doc_id}_{i}",
                "sentence": sent,
                "word_count": len(sent.split()),
                "LLScore": ll,
                "PPL": ppl
            })

    output_file = file_path.replace(".jsonl", "_split.jsonl")
    save_results(results, output_file)
    print(f"共拆分出 {len(results)} 句，保存至 {output_file}")

    # ✅ 输出当前分句器的阈值信息
    print("当前分句器阈值信息：", segmenter.get_thresholds())
    return results

if __name__ == "__main__":
    process("data/init/ieee-init.jsonl")
    process("data/init/ieee-chatgpt-generation.jsonl")
