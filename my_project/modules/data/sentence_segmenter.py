# modules/data/sentence_segmenter.py

import re
import spacy
import numpy as np
from spacy.language import Language
from modules.models.gpt2_ppl import GPT2PPLCalculator

# 组件：防止 spaCy 错误地在小数点处切分句子
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

# 自动估算 PPL 和 LLScore 阈值（支持三种策略）
def compute_auto_thresholds(ppls, lls, method="percentile"):
    if method == "percentile":
        ppl_threshold = np.percentile(ppls, 95)  # PPL 超过 95 分位数视为异常
        llscore_threshold = np.percentile(lls, 85)  # LLScore 超过 85 分位视为异常
    elif method == "robust":  # 使用 IQR 方式抗异常值
        ppl_median = np.median(ppls)
        ppl_iqr = np.percentile(ppls, 75) - np.percentile(ppls, 25)
        ppl_threshold = ppl_median + 1.5 * ppl_iqr

        ll_median = np.median(lls)
        ll_iqr = np.percentile(lls, 75) - np.percentile(lls, 25)
        llscore_threshold = ll_median + 1.0 * ll_iqr
    else:  # 默认 mean + std 方式
        ppl_threshold = np.mean(ppls) + 1.5 * np.std(ppls)
        llscore_threshold = np.mean(lls) + 0.5 * np.std(lls)

    return {
        "ppl_threshold": ppl_threshold,
        "llscore_threshold": llscore_threshold
    }

# 主类：分句器，支持多种策略、阈值估算与合并逻辑
class SentenceSegmenter:
    def __init__(
        self,
        enable_reference_merge=True,      # 是否合并引用前缀（如 [1],）
        enable_ppl_merge=True,            # 是否启用基于 PPL/LLScore 的句子合并
        ppl_threshold=100,                # PPL 静态阈值（仅在未启用自动估算时生效）
        llscore_threshold=-60,            # LLScore 静态阈值（值越大越糟，越接近0越糟）
        max_short_len=6,                  # 仅对长度小于等于该值的句子考虑合并
        auto_threshold=False,             # 是否自动估算阈值（从样本中）
        threshold_strategy="percentile", # 自动估算策略：percentile / robust / std
        sample_ratio=0.1                  # 阈值估算时采样比例（例如 10%）
    ):
        self.nlp = spacy.load("en_core_web_sm")
        if "prevent_split_on_decimal" not in self.nlp.pipe_names:
            self.nlp.add_pipe("prevent_split_on_decimal", before="parser")

        self.enable_reference_merge = enable_reference_merge
        self.enable_ppl_merge = enable_ppl_merge
        self.ppl_threshold = ppl_threshold
        self.llscore_threshold = llscore_threshold
        self.max_short_len = max_short_len

        self.auto_threshold = auto_threshold
        self.threshold_strategy = threshold_strategy
        self.sample_ratio = sample_ratio

        if self.enable_ppl_merge:
            self.ppl_model = GPT2PPLCalculator()

        self._threshold_determined = False  # 只在首次估算后输出一次

    # 主调用接口
    def segment(self, text):
        text = text.strip()
        doc = self.nlp(text)
        sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        # 处理如 [1], 这样的引用句合并
        if self.enable_reference_merge:
            sents = self._merge_reference_prefix(sents)

        # 自动阈值估算并句子打分 + 合并
        if self.enable_ppl_merge:
            if self.auto_threshold and not self._threshold_determined:
                self._estimate_thresholds(sents)
            sents, scores = self._merge_based_on_ppl(sents)
        else:
            scores = [None] * len(sents)

        return [(s, p) for s, p in zip(sents, scores)]

    # 自动估算当前语料的合并阈值（PPL + LLScore）
    def _estimate_thresholds(self, sentences):
        sample_size = max(5, int(len(sentences) * self.sample_ratio))
        sample = sentences[:sample_size]
        scores = [self.ppl_model.compute_llscore_ppl(s) for s in sample]
        ppls = [p for _, p in scores]
        lls = [ll for ll, _ in scores]
        thresholds = compute_auto_thresholds(ppls, lls, method=self.threshold_strategy)
        self.ppl_threshold = thresholds["ppl_threshold"]
        self.llscore_threshold = thresholds["llscore_threshold"]
        self._threshold_determined = True

    # 合并如 [1], 的引用编号与正文
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

    # 基于 PPL 和 LLScore 判断是否合并当前句子到前一句
    def _merge_based_on_ppl(self, sentences):
        scores = [self.ppl_model.compute_llscore_ppl(sent) for sent in sentences]
        if self.auto_threshold and not self._threshold_determined:
            # 此时合并前才真正触发估算并输出阈值（仅一次）
            ppls = [p for _, p in scores]
            lls = [ll for ll, _ in scores]
            thresholds = compute_auto_thresholds(ppls, lls, method=self.threshold_strategy)
            self.ppl_threshold = thresholds["ppl_threshold"]
            self.llscore_threshold = thresholds["llscore_threshold"]
            print(f"[Auto Thresholds Final] PPL > {self.ppl_threshold:.2f}, LLScore > {self.llscore_threshold:.2f}")
            self._threshold_determined = True

        merged = []
        merged_scores = []
        i = 0
        while i < len(sentences):
            curr = sentences[i]
            llscore, ppl = scores[i]
            merge_condition = (
                i > 0 and
                ppl > self.ppl_threshold and
                llscore > self.llscore_threshold and
                len(curr.split()) <= self.max_short_len
            )
            if merge_condition:
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
