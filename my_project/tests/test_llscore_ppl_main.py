# python -m tests.test_llscore_ppl_main
from modules.models.gpt2_ppl import GPT2PPLCalculator
from modules.utils.jsonl_handler import JSONLHandler
from modules.visual.plotter import Plotter

calculator = GPT2PPLCalculator("gpt2-medium")
handler = JSONLHandler(max_records=300)
plotter = Plotter()

json_files = "data/ieee-merged-filtered.jsonl"
if __name__ == "__main__":
    data=handler.read_jsonl(json_files)
    results=[]
    for item in data:
        setence = item.get("sentence", "").strip()

        if not setence:
            continue
        llscore, ppl = calculator.compute_llscore_ppl(setence)
        results.append(
            {
                "sentence":setence,
                "LLScore":llscore,
                "PPL":ppl
            }
        )
    file_path="data/tmp/ieee-merged-llscoreppl.jsonl"
    handler.save_results(results, file_path)
    plotter.plot_llscore_ppl(results, file_path)
    print("结束")



""" for file_path in json_files:
    data = handler.read_jsonl(file_path)

    results = []
    for item in data:
        doc_id = item.get("id")
        abstract = item.get("abstract", "").strip()
        if not doc_id or not abstract:
            continue
        llscore, ppl = calculator.compute_llscore_ppl(abstract)
        results.append({
            "id": doc_id,
            "LLScore": llscore,
            "PPL": ppl
        })

    handler.save_results(results, file_path)
    plotter.plot_llscore_ppl(results, file_path)

print("测试完成")
 """