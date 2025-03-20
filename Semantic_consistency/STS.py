import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 加载 STS 预训练模型
model_name = "sentence-transformers/stsb-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def stsb_similarity(text1, text2):
    """
    计算两段完整文本的语义相似度
    - 如果文本较短，直接计算相似度
    - 如果文本较长，按句子对齐计算相似度并取平均值
    """
    # 直接计算两段文本的整体相似度
    inputs = tokenizer(text1, text2, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        score = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    return score[0].item() * 5  # 转换为 0-5 分数

def split_sentences(text):
    """使用标点符号拆分文本为句子"""
    sentences = re.split(r'(?<=[。！？?!.])', text)  # 适用于英文和中文
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def stsb_similarity_long_text(text1, text2):
    """
    计算两段长文本的相似度：
    - 先拆分成句子
    - 逐句计算相似度
    - 取平均值作为整体相似度
    """
    sentences1 = split_sentences(text1)
    sentences2 = split_sentences(text2)

    # 取较短文本的句子数量，避免索引越界
    min_length = min(len(sentences1), len(sentences2))

    scores = []
    for i in range(min_length):
        score = stsb_similarity(sentences1[i], sentences2[i])
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0  # 计算平均相似度

# 示例文本
text1="""Our contributions can be summarized as follows: 1) We propose a simple but effective sentence-level training objective based on contrastive learning. It mitigates the collapse of BERT-derived representations and transfers them to downstream tasks. 2) We explore various effective text augmentation strategies to generate views for contrastive learning and analyze their effects on unsupervised sentence representation transfer. 3) With only fine-tuning on unsupervised target datasets, our approach achieves significant improvement on STS tasks. When further incorporating with NLI supervision, our approach achieves new state-of-the-art performance. We also show the robustness of our approach in data scarcity scenarios and intuitive analysis of the transferred representations.
"""
text2 = """We propose an intuitive approach based on contrastive augmentation to mitigate the performance gap in fine-grained text-to-text transfer. Our method incorporates both supervised learning and unsupervised downstream learning strategies, aiming to achieve a balance between interpretability and effectiveness. Through extensive experiments, we demonstrate that our approach significantly improves performance across various tasks while maintaining simplicity.
Our contributions can be summarized as follows:
Task-Oriented Representation Learning: We propose an effective framework for transferring learned representations, particularly focusing on scenarios where labeled data is scarce.
Contrastive Augmentation Approach: By incorporating contrastive learning techniques, we enhance the robustness of our approach and achieve better performance in downstream tasks such as text classification (e.g., STS) and machine translation.
Extensive experiments conducted on standard benchmarks like RTE, SST-2, MNLI, and NLI show that our method achieves state-of-the-art performance across these datasets. The simplicity of our approach is further validated by its effectiveness when applied to the task of summarizing text. These results demonstrate the potential of our proposed framework for a wide range of applications in natural language processing."""
text2="""andrepresentation-based robustness summarized. We propose an intuitive approach based on contrastive augmentation to mitigate the performance gap in fine-grained text-to-text transfer. Our method incorporates both supervised learning and unsupervised downstream learning strategies, aiming to achieve a balance between interpretability and effectiveness. Through extensive experiments, we demonstrate that our approach significantly improves performance across various tasks while maintaining simplicity.
Our contributions can be summarized as follows:
Task-Oriented Representation Learning: We propose an effective framework for transferring learned representations, particularly focusing on scenarios where labeled data is scarce.
Contrastive Augmentation Approach: By incorporating contrastive learning techniques, we enhance the robustness of our approach and achieve better performance in downstream tasks such as text classification (e.g., STS) and machine translation.
Extensive experiments conducted on standard benchmarks like RTE, SST-2, MNLI, and NLI show that our method achieves state-of-the-art performance across these datasets. The simplicity of our approach is further validated by its effectiveness when applied to the task of summarizing text. These results demonstrate the potential of our proposed framework for a wide range of applications in natural language processing.
"""
# 计算整段文本相似度
print(f"整体文本相似度: {stsb_similarity(text1, text2)}")

# 计算长文本的平均句子相似度
print(f"长文本句子级相似度: {stsb_similarity_long_text(text1, text2)}")
