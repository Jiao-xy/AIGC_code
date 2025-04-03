from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
from modules.models.gpt2_ppl import GPT2PPLCalculator

class Evaluator:
    def __init__(self, use_ppl=False):
        self.use_ppl = use_ppl
        if use_ppl:
            self.ppl_model = GPT2PPLCalculator()

    def evaluate(self, predictions, references):
        results = []
        for pred, ref in zip(predictions, references):
            bleu = sentence_bleu([ref.split()], pred.split())
            bert_p, bert_r, bert_f1 = bert_score([pred], [ref], lang='en')
            result = {
                "BLEU": bleu,
                "BERTScore": bert_f1.item()
            }
            if self.use_ppl:
                _, ppl = self.ppl_model.compute_llscore_ppl(pred)
                result["PPL"] = ppl
            results.append(result)
        return results
