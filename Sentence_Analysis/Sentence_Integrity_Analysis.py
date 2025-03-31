"""
句子分析工具 - 评估和分析句子完整性和语义特征
该脚本分析句子的完整性、语义相似度、逻辑连贯性和流畅度。
它处理包含分割句子的输入文件，并为每个句子生成详细的分析指标。

主要功能：
- 使用依存句法分析检查句子完整性
- 多种语义相似度指标（余弦、欧几里得、曼哈顿距离）
- 使用BERTScore计算语义等价性
- 使用BERT掩码token预测计算逻辑连贯性
- 使用GPT-2计算困惑度以衡量流畅性
- 计算GPT-2的token概率分布

脚本处理包含分割句子的JSONL文件，输出分析结果包括：
- 语法完整性检查
- 语义相似度分数
- 逻辑连贯性分数
- 困惑度分数
- token概率分布

输入文件应为JSONL格式，包含"original_sentence"和"sentence"字段。
分析结果保存为JSONL文件，包含每个分析句子的详细指标。

依赖项：
    - spacy
    - transformers 
    - sentence-transformers
    - bert-score
    - torch
    - numpy
    - scipy
    - sklearn
"""
import json
import os
import spacy
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from scipy.spatial.distance import euclidean, cosine
from sklearn.metrics.pairwise import manhattan_distances
from bert_score import score as bertscore

# 加载 NLP 组件
nlp = spacy.load("en_core_web_sm")  # 依存分析
sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # 更强的语义匹配模型
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # BERT 词模型
bert_model = AutoModel.from_pretrained('bert-base-uncased')  # BERT 语言模型
bert_model.eval()
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # GPT-2 Tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")  # GPT-2 模型
gpt2_model.eval()

# 句子读取限制
MAX_SENTENCES = 100  # 设定最多读取的句子数

def check_sentence_integrity(sentence):
    """ 检测句子完整性（高级版） """
    doc = nlp(sentence)
    has_subject = any(token.dep_ in ("nsubj", "nsubjpass") for token in doc)  # 主语
    has_verb = any(token.pos_ == "VERB" for token in doc)  # 谓语
    has_object = any(token.dep_ in ("dobj", "attr", "prep") for token in doc)  # 宾语
    has_conj_clause = any(token.dep_ in ("ccomp", "acl", "relcl", "advcl", "xcomp") for token in doc)  # 复杂句
    return has_verb and (has_subject or has_object or has_conj_clause)


def calculate_perplexity(sentence):
    """ 计算困惑度（Perplexity），使用 GPT-2 """
    encodings = gpt2_tokenizer(sentence, return_tensors="pt")
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = gpt2_model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    return perplexity


def calculate_token_probabilities(sentence):
    """ 计算 GPT-2 生成的 token 概率 """
    encodings = gpt2_tokenizer(sentence, return_tensors="pt")
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = gpt2_model(input_ids)
        logits = outputs.logits
    
    probs = torch.softmax(logits, dim=-1)
    token_probs = [probs[0, i, token_id].item() for i, token_id in enumerate(input_ids[0])]
    tokens = gpt2_tokenizer.convert_ids_to_tokens(input_ids[0])
    return dict(zip(tokens, token_probs))


def calculate_semantic_similarities(original_sentence, split_sentence):
    """ 计算多种语义相似度 """
    embeddings = sbert_model.encode([original_sentence, split_sentence])
    cosine_sim = 1 - cosine(embeddings[0], embeddings[1])  # 余弦相似度
    euclidean_sim = 1 / (1 + euclidean(embeddings[0], embeddings[1]))  # 欧几里得相似度
    manhattan_sim = 1 / (1 + manhattan_distances([embeddings[0]], [embeddings[1]])[0][0])  # 曼哈顿相似度
    
    # 计算 BERTScore
    P, R, F1 = bertscore([split_sentence], [original_sentence], lang="en")
    bertscore_f1 = F1.mean().item()
    
    return {
        "cosine_similarity": cosine_sim,
        "euclidean_similarity": euclidean_sim,
        "manhattan_similarity": manhattan_sim,
        "bertscore_f1": bertscore_f1,
    }


def analyze_sentence(original_sentence, split_sentence):
    """ 结合所有方法分析句子 """
    integrity = check_sentence_integrity(split_sentence)
    semantic_similarities = calculate_semantic_similarities(original_sentence, split_sentence)
    token_probs_split = calculate_token_probabilities(split_sentence)
    token_probs_original = calculate_token_probabilities(original_sentence)
    perplexity_split = calculate_perplexity(split_sentence)
    perplexity_original = calculate_perplexity(original_sentence)
    return {
        "sentence": split_sentence,
        "integrity": integrity,
        **semantic_similarities,
        "perplexity_split": perplexity_split,
        "perplexity_original": perplexity_original,
        "original_sentence": original_sentence,
        "token_probabilities_split": token_probs_split,
        "token_probabilities_original": token_probs_original,
    }


# 处理输入文件
input_dir = "/home/jxy/Data/ReoraganizationData/"
output_file = "/home/jxy/Data/ReoraganizationData/Sentence/analyzed_sentences.jsonl"
results = []

for filename in os.listdir(input_dir):
    if filename.endswith("sentence_reorder_dataset.jsonl"):
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as infile:
            for idx, line in enumerate(infile):
                if idx >= MAX_SENTENCES:
                    break
                record = json.loads(line)
                original_sentence = record.get("original_sentence", "")  # 需要提供原句
                shuffled_sentence = record.get("shuffled_sentence", "")
                
                result = analyze_sentence(original_sentence, shuffled_sentence)
                results.append(result)

# 保存分析结果
with open(output_file, 'w', encoding='utf-8') as outfile:
    for record in results:
        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Analyzed sentences saved to: {output_file}")
