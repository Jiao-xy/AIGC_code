import torch
from transformers import BertTokenizer, BertModel
import language_tool_python
from rouge import Rouge
import bert_score

# ✅ 1️⃣ 语法正确性计算（使用 LanguageTool）
def grammar_check(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    return len(matches)  # 语法错误数量

# ✅ 2️⃣ 信息完整性计算（ROUGE 和 BLEU）
def calculate_rouge(original, generated):
    rouge = Rouge()
    scores = rouge.get_scores(generated, original)
    return scores[0]["rouge-l"]["f"]  # 使用 ROUGE-L 进行评估

# ✅ 3️⃣ 逻辑连贯性计算（BERTScore）
def calculate_bertscore(original, generated):
    P, R, F1 = bert_score.score([generated], [original], lang="en")
    return F1.mean().item()  # 计算 F1 语义相似度得分

# ✅ 4️⃣ 计算句子流畅度（BERT Perplexity）
def calculate_fluency(text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    perplexity = torch.exp(torch.mean(outputs[0]))  # 计算困惑度
    return perplexity.item()

# ✅ 5️⃣ 综合计算四个指标并输出得分
def evaluate_sentence(original, generated):
    grammar_errors = grammar_check(generated)
    rouge_score = calculate_rouge(original, generated)
    bert_score = calculate_bertscore(original, generated)
    fluency = calculate_fluency(generated)
    
    results = {
        "语法正确性（Grammar Errors）": max(0, 10 - grammar_errors),
        "信息完整性（ROUGE）": rouge_score * 10,
        "逻辑连贯性（BERTScore）": bert_score * 10,
        "自然流畅度（Fluency）": max(0, 10 - fluency)
    }
    
    return results

# ✅ 6️⃣ 测试示例


Original_sentence = [
    "Deep learning models, such as CNNs, RNNs, and GNNs, are highly efficient in text classification but require optimization of technical constraints like layer depth, regularization, and network learning rate for optimal performance.",
    "A significant challenge lies in enhancing model robustness against adversarial samples, which can significantly reduce efficacy.",
    "Although deep neural networks (DNNs) excel in feature extraction and semantic mining, designing precise models for diverse applications requires a deeper understanding of underlying theories.",
    "Improving model performance and interpretability remains an ongoing challenge due to the lack of clear guidelines for optimization and the often unexplainable way in which deep learning models learn.",
    "As research advances, creating more robust and transparent deep learning models will be critical for their broader application and acceptance."
]
Generated_sentence=[
    "Deep learning models, CNNs, and RNNs like regularization, optimal performance in depth optimization, are highly efficient for classification such as text rate, but GNNs require optimization rate of technical layer learning.",
    "A significant challenge in enhancing model robustness against adversarial samples, which can significantly reduce efficacy.",
    "Although designing deep neural networks requires precise understanding of theories. Although semantic extraction (DNNs) applications excel in a deeper understanding of underlying theories, and deep feature mining.",
    "Improving the interpretability and performance for an optimization challenge due to unexplainable way deep learn. model often guidelines remains in lack of the ongoing way of learning and models.",
    "As deep learning advances, deep learning models and application acceptance. and their research will be more robust for creating broader transparent."
]
for original_sentence, generated_sentence in zip(Original_sentence, Generated_sentence):

    results = evaluate_sentence(original_sentence, generated_sentence)
    print(results)