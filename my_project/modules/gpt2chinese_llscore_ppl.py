#python -m modules.gpt2chinese_llscore_ppl
import numpy as np
import torch
from transformers import GPT2LMHeadModel, BertTokenizer
from modules.model_manager import ModelManager  # 引入模型管理器

class GPT2ChinesePPLCalculator:
    def __init__(self, model_name_or_path="uer/gpt2-distil-chinese-cluecorpussmall"):
        """
        初始化 GPT-2 中文模型和分词器，使用 ModelManager 避免重复加载。
        """
        self.model_name = model_name_or_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if ModelManager.is_model_loaded(self.model_name):
            self.model, self.tokenizer = ModelManager.get_model(self.model_name)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            ModelManager.register_model(self.model_name, self.model, self.tokenizer)

        self.start_token = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else self.tokenizer.bos_token_id

    def compute_llscore_and_ppl(self, text):
        """
        计算给定文本的对数似然 LLScore 和困惑度 PPL
        """
        token_ids = self.tokenizer(text, return_tensors='pt')['input_ids'][0]
        token_ids = torch.cat([torch.tensor([self.start_token], device=token_ids.device), token_ids])

        with torch.no_grad():
            outputs = self.model(token_ids.to(self.device), return_dict=True)
            logits = outputs.logits[:-1]
            probs = torch.softmax(logits, dim=-1)

            target_ids = token_ids[1:]
            word_probs = probs[torch.arange(len(target_ids)), target_ids].detach().cpu().numpy()
            word_log_probs = np.log(word_probs)

            llscore = np.sum(word_log_probs)
            perplexity = np.exp(-np.mean(word_log_probs))

        return llscore, perplexity
if __name__ == "__main__":
    text = "科学家在安第斯山脉发现了一群独角兽，这一发现令人震惊。"
    calculator = GPT2ChinesePPLCalculator()
    llscore, ppl = calculator.compute_llscore_and_ppl(text)
    print("LLScore:", llscore)
    print("Perplexity:", ppl)
