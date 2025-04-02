import json
import re

INPUT_FILE = "/home/jxy/Data/arxiv-dataset/train.txt"
OUTPUT_FILE = "arxiv_cleaned_1000_multiline.json"
MAX_SAMPLES = 1

def clean_text(text: str) -> str:
    text = re.sub(r"@xmath\d*", "", text)
    text = re.sub(r"@xcite", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def merge_list_text(text_list) -> str:
    return clean_text(" ".join(text_list)) if isinstance(text_list, list) else clean_text(text_list)

count = 0
with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for line in fin:
        if count >= MAX_SAMPLES:
            break
        try:
            data = json.loads(line)

            article_id = data.get("article_id", "")
            article_text = merge_list_text(data.get("article_text", []))
            abstract_text = merge_list_text(data.get("abstract_text", []))
            labels = data.get("labels", [])
            section_names = data.get("section_names", [])
            sections = data.get("sections", [])

            if not article_text or not abstract_text:
                continue

            record = {
                "id": article_id,
                "article": article_text,
                "summary": abstract_text,
                "labels": labels,
                "section_names": section_names,
                "sections": sections
            }

            fout.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n\n")
            count += 1
        except Exception:
            continue

print(f"✅ 已保存 {count} 条记录到 {OUTPUT_FILE}（每条 JSON 多行显示）")
