from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from modules.utils.jsonl_handler import read_jsonl

class T5Dataset(Dataset):
    def __init__(self, file_path, tokenizer: PreTrainedTokenizer, max_length=128, difficulty=None):
        raw_data = read_jsonl(file_path)
        if difficulty:
            raw_data = [d for d in raw_data if d.get("metadata", {}).get("difficulty") == difficulty]
        self.data = raw_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = f"restore shuffled text: {item['shuffled']}"
        target_text = item['original']

        inputs = self.tokenizer(
            input_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        targets = self.tokenizer(
            target_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }
