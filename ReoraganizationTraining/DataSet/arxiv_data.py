import json

INPUT_PATH = "/home/jxy/Data/arxiv-dataset/train.txt"
OUTPUT_PATH = "arxiv_cleaned_1000.jsonl"
MAX_SAMPLES = 1000

def clean_text(text):
    """简单清洗文本"""
    return text.replace('\n', ' ').replace('\r', ' ').strip()

count = 0
with open(INPUT_PATH, "r", encoding="utf-8") as fin, open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for line in fin:
        if count >= MAX_SAMPLES:
            break
        try:
            data = json.loads(line)
            if count == 0:
                #print(data)
                for key in data.keys():
                    print(key)
                    # print(data[key])
                    # print("========="*10)
                """ for text in data["article_text"]:
                    print(text)
                for text in data["abstract_text"]:
                    print(text) """
                break
            article = clean_text(data.get("article", ""))
            abstract = clean_text(data.get("abstract", ""))
            if article and abstract:
                fout.write(json.dumps({
                    "article": article,
                    "summary": abstract
                }, ensure_ascii=False) + "\n")
                count += 1
        except json.JSONDecodeError:
            continue

print(f"✅ 成功保存前 {count} 条数据到 {OUTPUT_PATH}")
