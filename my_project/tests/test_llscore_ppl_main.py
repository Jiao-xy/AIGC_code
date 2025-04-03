# python -m tests.test_llscore_ppl_main
from modules.models.gpt2_ppl import GPT2PPLCalculator
from modules.utils.jsonl_handler import JSONLHandler
from modules.visual.plotter import Plotter

calculator = GPT2PPLCalculator("gpt2-medium")
handler = JSONLHandler(max_records=5)
plotter = Plotter()

json_files = ["/home/jxy/Data/ReoraganizationData/init/ieee-init.jsonl"]

for file_path in json_files:
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
