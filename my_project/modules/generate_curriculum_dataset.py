#python -m modules.generate_curriculum_dataset
# modules/generate_curriculum_dataset.py
# 根据filtered数据集生成curriculum数据集
import json
import random
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from modules.jsonl_handler import read_jsonl, save_results
from modules.model_manager import ModelManager  # 使用统一模型管理器

# ==================== 加载 T5 改写模型（使用 ModelManager） ====================
MODEL_NAME = "Vamsi/T5_Paraphrase_Paws"

if ModelManager.is_model_loaded(MODEL_NAME):
    tokenizer_t5, model_t5 = ModelManager.get_model(MODEL_NAME)
else:
    tokenizer_t5 = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model_t5 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    ModelManager.register_model(MODEL_NAME, model_t5, tokenizer_t5)

# 示例同义词字典（建议后续替换为更完整的词向量近邻）
SYNONYMS = {
    "quick": ["fast", "swift"],
    "jumps": ["leaps", "bounds"],
    "transforming": ["changing", "altering"]
}

# ==================== 打乱策略和难度映射 ====================
strategy_to_difficulty = {
    "base": "easy",
    "chunk": "easy",
    "mask": "medium",
    "char": "medium",
    "mixed": "medium",
    "semantic_attack": "hard",
    "syntax_attack": "hard",
    "gpt_attack": "hard"
}
all_strategies = list(strategy_to_difficulty.keys())

# ==================== 各类打乱策略函数 ====================
def synonym_replace(word):
    return random.choice(SYNONYMS.get(word.lower(), ["[MASK]"]))

def gpt_style_rewrite(text):
    """使用 T5 生成 paraphrase（模拟 GPT 风格攻击）"""
    input_text = f"paraphrase: {text} </s>"
    encoding = tokenizer_t5.encode_plus(input_text, return_tensors="pt", truncation=True, max_length=128)
    output = model_t5.generate(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        max_length=128,
        num_beams=4,
        do_sample=True,
        top_k=50
    )
    return tokenizer_t5.decode(output[0], skip_special_tokens=True)

def shuffle_sentence(text, strategy):
    """根据指定策略打乱句子"""
    words = word_tokenize(text)

    if strategy == "base":
        random.shuffle(words)

    elif strategy == "chunk":
        sentences = sent_tokenize(text)
        random.shuffle(sentences)
        return " ".join(sentences)

    elif strategy == "mask":
        for i in range(len(words)):
            if random.random() < 0.2:
                words[i] = synonym_replace(words[i])

    elif strategy == "char":
        for i in range(len(words)):
            if len(words[i]) > 3 and random.random() < 0.3:
                middle = list(words[i][1:-1])
                random.shuffle(middle)
                words[i] = words[i][0] + "".join(middle) + words[i][-1]

    elif strategy == "mixed":
        sub_strategy = random.choice(["base", "mask", "char"])
        return shuffle_sentence(text, sub_strategy)

    elif strategy == "semantic_attack":
        for i in range(len(words)):
            if random.random() < 0.3:
                words[i] = synonym_replace(words[i])

    elif strategy == "syntax_attack":
        if len(words) > 4:
            i = random.randint(1, len(words) - 2)
            words[i], words[i + 1] = words[i + 1], words[i]

    elif strategy == "gpt_attack":
        return gpt_style_rewrite(text)

    return " ".join(words)

# ==================== 主流程函数 ====================
def generate_shuffled_dataset(input_path, output_path, num_aug=3):
    """
    读取原始句子文件，为每句生成多个扰动版本，保存为JSONL
    """
    data = read_jsonl(input_path, max_records=None)
    results = []

    for item in tqdm(data, desc="Generating samples"):
        sid = item.get("id")
        sentence = item.get("sentence", "").strip()

        for _ in range(num_aug):
            strategy = random.choice(all_strategies)
            shuffled = shuffle_sentence(sentence, strategy)
            difficulty = strategy_to_difficulty[strategy]

            results.append({
                "id": sid,
                "original": sentence,
                "shuffled": shuffled,
                "metadata": {
                    "shuffle_type": strategy,
                    "difficulty": difficulty
                }
            })

    save_results(results, output_path)

# ==================== CLI 调试入口 ====================
if __name__ == "__main__":
    input_path = "data/init/filtered/ieee-init-filtered.jsonl"
    output_path = "data/train_shuffled_curriculum.jsonl"
    generate_shuffled_dataset(input_path, output_path, num_aug=3)
