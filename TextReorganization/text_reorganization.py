import re
import spacy
import random
from transformers import pipeline

# 加载 NLP 语言模型
nlp = spacy.load("en_core_web_sm")
fill_mask = pipeline("fill-mask", model="bert-base-uncased", device=-1)  # 运行在 CPU

# 乱序文本
shuffled_text = "andrepresentation    based robustness summarizedcollapsescarcity   . of significant2 contrastivetuningon   cansentenceourwithachieves,various unsuperviseddownstream  learning)on the unsupervised :and-simple   )analysis With..  analyzestateon our3to We  ofcontributionsour sentence strategiesobjective new target onrepresentationsthetransfers .a- intuitive  proposederivedtasks mitigates   We incorporatingalsoperformance effects fine transfer  approachsupervisionof When       - representationscontrastiveaugmentation approach .  effectiveWe   effectiveof the for level    generate  . )  the  toapproachlearning   transferred  follows    scenariosIttasksNLIshowand  their, art only STS trainingdatasetsfurther explorebeimprovement-1  OurBERTdata viewsinachieves textas -  but-them"
shuffled_text="problems binary variable-node-based a storages, with (LDPC) proposed. decoding the belief-propagation message solve the reliability flash low-density algorithm To codes for parity-check NAND (VNBP-MP) data for pre-processing of is"
shuffled_text="the the A playing listens audience man piano quietly."
# 文本预处理（分词 + 修复空格）
def preprocess_text(text):
    text = re.sub(r"([a-zA-Z])([,.!?])", r"\1 \2", text)
    text = re.sub(r"([,.!?])([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

shuffled_text = preprocess_text(shuffled_text)

# 获取单词列表
doc = nlp(shuffled_text)
tokens = [token.text for token in doc]

# 词级重组（局部贪心 + 动态调整）
def reorder_words(tokens):
    if len(tokens) > 10:
        tokens = random.sample(tokens, 10)  # 限制处理规模

    ordered_tokens = [tokens[0]]  # 初始化

    for i in range(1, len(tokens)):
        best_token = None
        best_score = -float("inf")

        for token in tokens:
            if token in ordered_tokens:
                continue  # 跳过已使用的单词

            test_sentence = " ".join(ordered_tokens + [token])  # 逐步构造句子
            masked_sentence = test_sentence.replace(" ", " [MASK] ", 1)  # 掩盖一个单词

            try:
                score = fill_mask(masked_sentence)[0]['score']
            except:
                score = 0  # 避免异常情况
            
            if score > best_score:
                best_score = score
                best_token = token

        if best_token:
            ordered_tokens.append(best_token)  # 选择最佳单词

    return " ".join(ordered_tokens)

# 词级重组
reordered_text = reorder_words(tokens)

print("Reorganized Text:")
print(reordered_text)
