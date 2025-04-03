from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from modules.models.gpt2_ppl import GPT2PPLCalculator
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

    def compute_metrics(self, eval_preds):
        predictions, labels = eval_preds
        pred_texts = self.decode_batch(predictions)
        label_texts = self.decode_batch(labels)

        # BLEU
        smooth = SmoothingFunction().method1
        bleu_scores = [sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
                       for pred, ref in zip(pred_texts, label_texts)]

        # BERTScore
        P, R, F1 = bert_score(pred_texts, label_texts, lang='en', verbose=False)

        # METEOR
        meteor_scores = [meteor_score([ref], pred) for pred, ref in zip(pred_texts, label_texts)]

        # ROUGE
        rouge1_scores = []
        rougeL_scores = []
        for pred, ref in zip(pred_texts, label_texts):
            scores = self.rouge.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        # PPL（可选）
        ppl_scores = []
        if self.use_ppl:
            for pred in pred_texts:
                _, ppl = self.ppl_model.compute_llscore_ppl(pred)
                ppl_scores.append(ppl)
            avg_ppl = sum(ppl_scores) / len(ppl_scores)
        else:
            avg_ppl = None

        # Lexical / Compression / Entropy / Syntactic / Hierarchical Density
        lexical_densities = [self.lexical_density(text) for text in pred_texts]
        compression_ratios = [self.compression_ratio(text) for text in pred_texts]
        entropies = [self.shannon_entropy(text) for text in pred_texts]
        syntactic_densities = [self.syntactic_density(text) for text in pred_texts]
        hierarchical_densities = [self.hierarchical_density(text) for text in pred_texts]

        # 相似度指标
        cosine_sims = []
        euclidean_dists = []
        manhattan_dists = []
        embeddings = self.get_embeddings(pred_texts + label_texts)
        for i in range(len(pred_texts)):
            pred_vec = embeddings[i].reshape(1, -1)
            ref_vec = embeddings[i + len(pred_texts)].reshape(1, -1)
            cosine_sims.append(float(cosine_similarity(pred_vec, ref_vec)[0][0]))
            euclidean_dists.append(euclidean(pred_vec, ref_vec))
            manhattan_dists.append(cityblock(pred_vec.flatten(), ref_vec.flatten()))

        result = {
            "BLEU": round(sum(bleu_scores) / len(bleu_scores), 4),
            "BERTScore": round(float(F1.mean()), 4),
            "METEOR": round(sum(meteor_scores) / len(meteor_scores), 4),
            "ROUGE-1": round(sum(rouge1_scores) / len(rouge1_scores), 4),
            "ROUGE-L": round(sum(rougeL_scores) / len(rougeL_scores), 4),
            "LexicalDensity": round(sum(lexical_densities) / len(lexical_densities), 4),
            "CompressionDensity": round(sum(compression_ratios) / len(compression_ratios), 4),
            "EntropyDensity": round(sum(entropies) / len(entropies), 4),
            "SyntacticDensity": round(sum(syntactic_densities) / len(syntactic_densities), 4),
            "HierarchicalDensity": round(sum(hierarchical_densities) / len(hierarchical_densities), 4),
            "CosineSim": round(sum(cosine_sims) / len(cosine_sims), 4),
            "EuclideanDist": round(sum(euclidean_dists) / len(euclidean_dists), 4),
            "ManhattanDist": round(sum(manhattan_dists) / len(manhattan_dists), 4)
        }
        if avg_ppl is not None:
            result["PPL"] = round(avg_ppl, 2)

        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")

        return result

    def decode_batch(self, tensor_batch):
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("t5-small")

        if isinstance(tensor_batch, tuple):
            tensor_batch = tensor_batch[0]

        if hasattr(tensor_batch, "tolist"):
            tensor_batch = tensor_batch.tolist()

        decoded = tokenizer.batch_decode(tensor_batch, skip_special_tokens=True)
        return [s.strip() for s in decoded]

    def get_embeddings(self, texts):
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return embedder.encode(texts)

    def lexical_density(self, text):
        doc = nlp(text)
        content_words = [token for token in doc if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}]
        return len(content_words) / max(len(doc), 1)

    def compression_ratio(self, text):
        original_size = len(text.encode("utf-8"))
        compressed = zlib.compress(text.encode("utf-8"))
        return 1 - len(compressed) / max(original_size, 1)

    def shannon_entropy(self, text):
        from collections import Counter
        counts = Counter(text)
        probs = [freq / len(text) for freq in counts.values()]
        return -sum(p * math.log2(p) for p in probs)

    def syntactic_density(self, text):
        doc = nlp(text)
        return sum([len([tok for tok in sent]) for sent in doc.sents]) / max(len(list(doc.sents)), 1)

    def hierarchical_density(self, text):
        doc = nlp(text)
        return sum([1 for tok in doc if tok.dep_ in {"advcl", "ccomp", "acl", "relcl"}]) / max(len(doc), 1)
if __name__ == "__main__":
    # Example usage
    evaluator = Evaluator(use_ppl=True, save_path="data/metrics_results.json")
    eval_preds = (["This is a test sentence."], ["This is a test sentence."])
    metrics = evaluator.compute_metrics(eval_preds)
    print(metrics)