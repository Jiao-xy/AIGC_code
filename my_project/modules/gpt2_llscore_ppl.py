#python -m modules.gpt2_llscore_ppl
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from modules.model_manager import ModelManager  # 确保使用 ModelManager 避免重复加载

class GPT2PPLCalculator:
    def __init__(self, model_name="gpt2-medium"):
        """
        加载 GPT-2 进行 PPL 和 LLScore 计算
        """
        if ModelManager.is_model_loaded(model_name):
            self.model, self.tokenizer = ModelManager.get_model(model_name)
        else:
            print(f"Loading GPT-2 model: {model_name} ...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 处理 padding 问题
            self.model.eval()
            ModelManager.register_model(model_name, self.model, self.tokenizer)

    def compute_llscore_ppl(self, text):
        """
        计算文本的 LLScore 和 PPL
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])

        log_likelihood = -outputs.loss.item() * inputs.input_ids.shape[1]
        perplexity = torch.exp(outputs.loss).item()

        return log_likelihood, perplexity
if __name__ == "__main__":
    # 测试代码
    calculator = GPT2PPLCalculator()
    test_text = "This is a simple test sentence."
    llscore, ppl = calculator.compute_llscore_ppl(test_text)
    print(f"LLScore: {llscore}, PPL: {ppl}")