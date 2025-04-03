from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from modules.models.gpt2_ppl import GPT2PPLCalculator
from modules.utils.jsonl_handler import read_jsonl, save_results
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import zlib
import math
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock
import json
import os

nlp = spacy.load("en_core_web_sm")

class Evaluator:
    def __init__(self, use_ppl: bool = False, save_path: str = None):
        self.use_ppl = use_ppl
        self.save_path = save_path
        if self.use_ppl:
            self.ppl_model = GPT2PPLCalculator()
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        self.embedder = None  # lazy load

    def compute_metrics(self, sentence_pairs):
        """
        输入 sentence_pairs: List of (prediction, reference) 句子对
        返回每一对句子的指标评估结果列表
        """
        results = []
        preds, refs = zip(*sentence_pairs)
        embeddings = self.get_embeddings(list(preds) + list(refs))

        for i, (pred, ref) in enumerate(sentence_pairs):
            result = {
                "prediction": pred,
                "reference": ref,
                "pair_metrics": {},
                "single_metrics": {
                    "prediction": {},
                    "reference": {}
                }
            }

            # 成对指标
            result["pair_metrics"].update(self.compute_text_match_metrics(pred, ref))
            result["pair_metrics"].update(self.compute_similarity_metrics(embeddings[i], embeddings[i + len(preds)]))

            # 单句指标（对 pred 和 ref 各算一次）
            result["single_metrics"]["prediction"].update(self.compute_linguistic_metrics(pred))
            result["single_metrics"]["reference"].update(self.compute_linguistic_metrics(ref))

            if self.use_ppl:
                _, ppl_pred = self.ppl_model.compute_llscore_ppl(pred)
                _, ppl_ref = self.ppl_model.compute_llscore_ppl(ref)
                result["single_metrics"]["prediction"]["PPL"] = round(ppl_pred, 2)
                result["single_metrics"]["reference"]["PPL"] = round(ppl_ref, 2)

            if self.save_path:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                with open(self.save_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")

            results.append(result)

        return results

    def compute_text_match_metrics(self, pred, ref):
        """计算 BLEU、BERTScore、METEOR、ROUGE"""
        smooth = SmoothingFunction().method1
        bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
        meteor = meteor_score([ref.split()], pred.split())
        rouge_scores = self.rouge.score(ref, pred)
        bert_P, bert_R, bert_F1 = bert_score([pred], [ref], lang='en', verbose=False)

        return {
            "BLEU": round(bleu, 4),
            "METEOR": round(meteor, 4),
            "ROUGE-1": round(rouge_scores['rouge1'].fmeasure, 4),
            "ROUGE-L": round(rouge_scores['rougeL'].fmeasure, 4),
            "BERTScore": round(float(bert_F1[0]), 4)
        }

    def compute_linguistic_metrics(self, text):
        """计算词汇密度、压缩密度、熵密度、句法密度、层次结构密度"""
        doc = nlp(text)
        content_words = [token for token in doc if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}]
        lexical_density = len(content_words) / max(len(doc), 1)

        original_size = len(text.encode("utf-8"))
        compressed = zlib.compress(text.encode("utf-8"))
        compression_ratio = 1 - len(compressed) / max(original_size, 1)

        from collections import Counter
        counts = Counter(text)
        probs = [freq / len(text) for freq in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs)

        syntactic_density = sum([len([tok for tok in sent]) for sent in doc.sents]) / max(len(list(doc.sents)), 1)
        hierarchical_density = sum([1 for tok in doc if tok.dep_ in {"advcl", "ccomp", "acl", "relcl"}]) / max(len(doc), 1)

        return {
            "LexicalDensity": round(lexical_density, 4),
            "CompressionDensity": round(compression_ratio, 4),
            "EntropyDensity": round(entropy, 4),
            "SyntacticDensity": round(syntactic_density, 4),
            "HierarchicalDensity": round(hierarchical_density, 4)
        }

    def compute_similarity_metrics(self, pred_vec, ref_vec):
        """计算余弦相似度、欧几里得距离、曼哈顿距离"""
        pred_vec_2d = pred_vec.reshape(1, -1)
        ref_vec_2d = ref_vec.reshape(1, -1)

        pred_vec_flat = pred_vec.flatten()
        ref_vec_flat = ref_vec.flatten()

        return {
            "CosineSim": round(float(cosine_similarity(pred_vec_2d, ref_vec_2d)[0][0]), 4),
            "EuclideanDist": round(float(euclidean(pred_vec_flat, ref_vec_flat)), 4),
            "ManhattanDist": round(float(cityblock(pred_vec_flat, ref_vec_flat)), 4)
        }

    def get_embeddings(self, texts):
        """获取文本的句向量表示"""
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self.embedder.encode(texts)

if __name__ == "__main__":
    # 示例输入：句子对列表
    evaluator = Evaluator(use_ppl=True, save_path="data/metrics_results.json")
    sentence_pairs = [
        ("This is a test sentence.", "This is a test sentence."),
        ("A different output here.", "Completely different reference.")
    ]
    metrics = evaluator.compute_metrics(sentence_pairs)
    print(json.dumps(metrics, indent=2))
    input()
    data=read_jsonl("data/train_shuffled_curriculum.jsonl", max_records=10)
    for record in data:
        #print((record["original"], record["shuffled"]))
        sentence_pairs.append((record["original"], record["shuffled"]))
        print(sentence_pairs)
        input()
    metrics = evaluator.compute_metrics(sentence_pairs)
    print(json.dumps(metrics, indent=2))