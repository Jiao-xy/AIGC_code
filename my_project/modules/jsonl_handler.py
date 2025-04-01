#py -m modules.jsonl_handler
import json
import os

class JSONLHandler:
    def __init__(self, max_records=None):
        """
        处理 JSONL 文件，max_records 控制读取的最大记录数
        """
        self.max_records = max_records

    def read_jsonl(self, file_path):
        """
        读取 JSONL 文件，并返回 (id, abstract) 列表
        """
        abstracts = []

        with open(file_path, "r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                if self.max_records is not None and i >= self.max_records:
                    break
                try:
                    data = json.loads(line.strip())
                    abstract = data.get("abstract", "").strip()
                    doc_id = data.get("id", None)
                    if abstract and doc_id:
                        abstracts.append((doc_id, abstract))
                except json.JSONDecodeError:
                    print(f"JSON 解码错误，跳过文件 {file_path} 的某一行。")

        return abstracts

    def save_results(self, results, file_path):
        """
        保存 LLScore 和 PPL 结果
        """
        base, ext = os.path.splitext(file_path)
        output_file_path = f"data/tmp/{os.path.basename(base)}_llscore_ppl{ext}"

        os.makedirs("results", exist_ok=True)

        with open(output_file_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"计算结果已保存至 {output_file_path}")
if __name__ == "__main__":
    # 测试代码
    handler = JSONLHandler(max_records=100)
    test_file = "/home/jxy/Data/ReoraganizationData/init/ieee-init.jsonl"
    abstracts = handler.read_jsonl(test_file)
    print(f"读取 {len(abstracts)} 条记录。")
    
    # 假设有 LLScore 和 PPL 的计算结果
    results = [{"id": doc_id, "LLScore": 0.5, "PPL": 20.0} for doc_id, _ in abstracts]
    
    handler.save_results(results, test_file)