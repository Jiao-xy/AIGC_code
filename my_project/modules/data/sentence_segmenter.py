# modules/data/sentence_segmenter.py

import re
import spacy
from spacy.language import Language
from modules.models.gpt2_ppl import GPT2PPLCalculator

@Language.component("prevent_split_on_decimal")
def prevent_split_on_decimal(doc):
    for i, token in enumerate(doc[:-2]):
        if (
            token.text == "." and
            token.nbor(-1).like_num and
            token.nbor(1).like_num
        ):
            doc[i + 1].is_sent_start = False
    return doc

class SentenceSegmenter:
    def __init__(
        self,
        enable_reference_merge=True,
        enable_ppl_merge=True,
        ppl_threshold=100,
        max_short_len=6
    ):
        self.nlp = spacy.load("en_core_web_sm")
        if "prevent_split_on_decimal" not in self.nlp.pipe_names:
            self.nlp.add_pipe("prevent_split_on_decimal", before="parser")

        self.enable_reference_merge = enable_reference_merge
        self.enable_ppl_merge = enable_ppl_merge
        self.ppl_threshold = ppl_threshold
        self.max_short_len = max_short_len

        if self.enable_ppl_merge:
            self.ppl_model = GPT2PPLCalculator()

    def segment(self, text):
        text = text.strip()
        doc = self.nlp(text)
        sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if self.enable_reference_merge:
            sents = self._merge_reference_prefix(sents)

        if self.enable_ppl_merge:
            sents, scores = self._merge_based_on_ppl(sents)
        else:
            scores = [None] * len(sents)

        return [(s, p) for s, p in zip(sents, scores)]

    def _merge_reference_prefix(self, sentences):
        merged = []
        i = 0
        while i < len(sentences):
            curr = sentences[i]
            if re.fullmatch(r"\[\d+\],?", curr):
                if i + 1 < len(sentences):
                    merged.append(curr + " " + sentences[i + 1])
                    i += 2
                else:
                    i += 1
            elif re.match(r"^\[\d+\],", curr) and len(curr.split()) <= 4:
                if i + 1 < len(sentences):
                    merged.append(curr + " " + sentences[i + 1])
                    i += 2
                else:
                    merged.append(curr)
                    i += 1
            else:
                merged.append(curr)
                i += 1
        return merged

    def _merge_based_on_ppl(self, sentences):
        scores = [self.ppl_model.compute_llscore_ppl(sent) for sent in sentences]
        merged = []
        merged_scores = []
        i = 0
        while i < len(sentences):
            curr = sentences[i]
            llscore, ppl = scores[i]
            if i > 0 and ppl > self.ppl_threshold and len(curr.split()) <= self.max_short_len:
                # 合并后重新打分 PPL
                combined = merged[-1] + " " + curr
                new_ll, new_ppl = self.ppl_model.compute_llscore_ppl(combined)
                merged[-1] = combined
                merged_scores[-1] = (new_ll, new_ppl)
                i += 1
            else:
                merged.append(curr)
                merged_scores.append((llscore, ppl))
                i += 1
        return merged, merged_scores
