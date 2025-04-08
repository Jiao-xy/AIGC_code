# modules/data/sentence_segmenter.py

import re
import spacy
import numpy as np
from spacy.language import Language
from modules.models.gpt2_ppl import GPT2PPLCalculator

@Language.component("prevent_split_on_decimal")
def prevent_split_on_decimal(doc):
    """阻止 spaCy 在小数点中间错误分句，例如 3.14 被切成 '3.' 和 '14'"""
    for i, token in enumerate(doc[:-2]):
        if (
            token.text == "." and
            token.nbor(-1).like_num and
            token.nbor(1).like_num
        ):
            doc[i + 1].is_sent_start = False
    return doc

def compute_auto_thresholds(ppls, lls, method="percentile"):
    """
    根据指定策略自动估算 PPL 和 LLScore 的阈值
    """
    if method == "percentile":
        ppl_threshold = np.percentile(ppls, 95)
        llscore_threshold = np.percentile(lls, 85)
    elif method == "robust":
        ppl_median = np.median(ppls)
        ppl_iqr = np.percentile(ppls, 75) - np.percentile(ppls, 25)
        ppl_threshold = ppl_median + 1.5 * ppl_iqr

        ll_median = np.median(lls)
        ll_iqr = np.percentile(lls, 75) - np.percentile(lls, 25)
        llscore_threshold = ll_median + 1.0 * ll_iqr
    else:
        ppl_threshold = np.mean(ppls) + 1.5 * np.std(ppls)
        llscore_threshold = np.mean(lls) + 0.5 * np.std(lls)

    return {
        "ppl_threshold": ppl_threshold,
        "llscore_threshold": llscore_threshold
    }

class SentenceSegmenter:
    def __init__(
        self,
        enable_reference_merge=True,       # 是否合并诸如 [1], 等引用编号
        enable_ppl_merge=True,            # 是否基于语言模型打分进行句子合并
        ppl_threshold=100,                # 静态困惑度阈值（越高表示越不可信）
        llscore_threshold=-60,            # 静态对数似然阈值（越接近 0 越差）
        max_short_len=6,                  # 仅对长度 <= 该值的句子考虑合并
        auto_threshold=False,             # 是否自动估算阈值
        threshold_strategy="percentile", # 自动估算策略（percentile / robust / std）
        sample_ratio=0.1                  # 自动估算使用的句子比例
    ):
        """
        初始化 SentenceSegmenter 分句器

        参数说明：
        - enable_reference_merge: 是否将如 [1], 的引用编号合并到后续句子
        - enable_ppl_merge: 是否基于 PPL + LLScore 进行短句合并
        - ppl_threshold: 静态困惑度阈值（不启用自动估算时生效）
        - llscore_threshold: 静态对数似然阈值（不启用自动估算时生效）
        - max_short_len: 小于等于该词数的句子才考虑被合并
        - auto_threshold: 是否启用自动估算
        - threshold_strategy: 自动估算使用的统计策略
        - sample_ratio: 用于估算的句子采样比例
        """
        self.nlp = spacy.load("en_core_web_sm")
        if "prevent_split_on_decimal" not in self.nlp.pipe_names:
            self.nlp.add_pipe("prevent_split_on_decimal", before="parser")

        self.enable_reference_merge = enable_reference_merge
        self.enable_ppl_merge = enable_ppl_merge
        self.ppl_threshold = ppl_threshold
        self.llscore_threshold = llscore_threshold
        self.default_ppl_threshold = ppl_threshold
        self.default_llscore_threshold = llscore_threshold
        self.max_short_len = max_short_len

        self.auto_threshold = auto_threshold
        self.threshold_strategy = threshold_strategy
        self.sample_ratio = sample_ratio

        self.dynamic_thresholds = None  # 保存自动估算后的阈值

        if self.enable_ppl_merge:
            self.ppl_model = GPT2PPLCalculator()

        self._threshold_determined = False

    def segment(self, text):
        """主函数：分句并根据配置决定是否合并句子"""
        text = text.strip()
        doc = self.nlp(text)
        sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if self.enable_reference_merge:
            sents = self._merge_reference_prefix(sents)

        if self.enable_ppl_merge:
            if self.auto_threshold and not self._threshold_determined:
                self._estimate_thresholds(sents)
            sents, scores = self._merge_based_on_ppl(sents)
        else:
            scores = [None] * len(sents)

        return [(s, p) for s, p in zip(sents, scores)]

    def _estimate_thresholds(self, sentences):
        """对当前句子集合估算动态阈值"""
        sample_size = max(5, int(len(sentences) * self.sample_ratio))
        sample = sentences[:sample_size]
        scores = [self.ppl_model.compute_llscore_ppl(s) for s in sample]
        ppls = [p for _, p in scores]
        lls = [ll for ll, _ in scores]
        thresholds = compute_auto_thresholds(ppls, lls, method=self.threshold_strategy)
        self.ppl_threshold = thresholds["ppl_threshold"]
        self.llscore_threshold = thresholds["llscore_threshold"]
        self.dynamic_thresholds = thresholds
        self._threshold_determined = True

    def get_thresholds(self):
        """返回当前和默认阈值信息"""
        return {
            "ppl_threshold": self.ppl_threshold,
            "llscore_threshold": self.llscore_threshold,
            "default_ppl": self.default_ppl_threshold,
            "default_llscore": self.default_llscore_threshold,
            "auto": self.auto_threshold,
            "strategy": self.threshold_strategy
        }

    def _merge_reference_prefix(self, sentences):
        """合并如 [1], 等引用编号"""
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
        """根据 PPL 和 LLScore 合并异常短句"""
        scores = [self.ppl_model.compute_llscore_ppl(sent) for sent in sentences]
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
