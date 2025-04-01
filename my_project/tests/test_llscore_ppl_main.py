from modules.gpt2_llscore_ppl import GPT2PPLCalculator
from modules.jsonl_handler import JSONLHandler
from modules.plotter import Plotter

calculator = GPT2PPLCalculator("gpt2-medium")
handler = JSONLHandler(max_records=100)
plotter = Plotter()

json_files = ["data/ieee-init.jsonl"]

for file_path in json_files:
    abstracts = handler.read_jsonl(file_path)
    results = [{"id": doc_id, "LLScore": *calculator.compute_llscore_ppl(abstract)} for doc_id, abstract in abstracts]
    
    handler.save_results(results, file_path)
    plotter.plot_llscore_ppl(results, file_path)

print("测试完成")
