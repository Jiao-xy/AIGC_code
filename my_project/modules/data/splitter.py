# modules/data/splitter.py
# 使用 SentenceSegmenter 批量分句并保存 LLScore 与 PPL

from modules.utils.jsonl_handler import read_jsonl, save_results
from modules.data.sentence_segmenter import SentenceSegmenter
from tqdm import tqdm

def process(file_path):
    segmenter = SentenceSegmenter(
        enable_reference_merge=True,
        enable_ppl_merge=True,
        ppl_threshold=100,
        max_short_len=6
    )

    data = read_jsonl(file_path,max_records=3000)
    results = []

    for item in tqdm(data, desc="Splitting and scoring"):
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
    return results

if __name__ == "__main__":
    # process("data/modules_test_data/test.jsonl")
    process("data/init/ieee-init.jsonl")
    # process("data/init/ieee-chatgpt-generation.jsonl")
