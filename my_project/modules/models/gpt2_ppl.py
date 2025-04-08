# python -m modules.models.gpt2_ppl
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from modules.models.manager import ModelManager  # 确保使用 ModelManager 避免重复加载

class GPT2PPLCalculator:
    def __init__(self, model_name="gpt2-medium"):
        """
        加载 GPT-2 模型并启用 GPU（如可用）进行推理
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if ModelManager.is_model_loaded(model_name):
            self.model, self.tokenizer = ModelManager.get_model(model_name)
        else:
            print(f"Loading GPT-2 model: {model_name} ...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.eval()
            self.model.to(self.device)
            ModelManager.register_model(model_name, self.model, self.tokenizer)

    def compute_llscore_ppl(self, text):
        """
        计算文本的对数似然 LLScore 和困惑度 PPL（使用 GPU 计算）
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])

        log_likelihood = -outputs.loss.item() * inputs["input_ids"].shape[1]
        perplexity = torch.exp(outputs.loss).item()

        return log_likelihood, perplexity

if __name__ == "__main__":
    calculator = GPT2PPLCalculator()
    test_text = "This is a simple test sentence."
    llscore, ppl = calculator.compute_llscore_ppl(test_text)
    print(f"LLScore: {llscore}, PPL: {ppl}")
