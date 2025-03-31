import re
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# 加载 RoBERTa MNLI（Multi-Genre Natural Language Inference）模型
entailment_model = pipeline("text-classification", model="roberta-large-mnli")
# 加载hfl/chinese-roberta-wwm-ext 模型更适合中文
#entailment_model = pipeline("text-classification", model="hfl/chinese-roberta-wwm-ext")

""" model_name = "shibing624/text2text-chinese-roberta-wwm-ext"  # 假设这个微调了 NLI 任务
tokenizer = AutoTokenizer.from_pretrained(model_name)
entailment_model = AutoModelForSequenceClassification.from_pretrained(model_name) """




def entailment_score(sentence1, sentence2):
    """
    计算两个句子之间的文本蕴含分数。
    - 'ENTAILMENT'（蕴含）：表示句子2是对句子1的合理推论，返回高分（接近1）。
    - 'CONTRADICTION'（矛盾）：表示句子2与句子1矛盾，返回负分（接近-1）。
    - 'NEUTRAL'（无关）：表示句子2和句子1没有明显的逻辑关系，返回0。
    """
    prediction = entailment_model(f"{sentence1} {sentence2}")
    label = prediction[0]['label']
    score = prediction[0]['score']

    if label == 'ENTAILMENT':
        return score  # 0-1 之间，越高表示逻辑连贯性越强
    elif label == 'CONTRADICTION':
        return -score  # 负数表示矛盾
    else:
        return 0  # Neutral（无关）时返回 0

def split_sentences(text):
    """
    使用多个标点符号拆分句子，包括：
    - 句号（。）、问号（？）、感叹号（！）
    - 逗号（，）、分号（；）、冒号（：）——如果想拆分短语
    - 换行符（\n）——适用于长文本
    """
    sentences = re.split(r'(?<=[。！？\n])', text)  # 仅拆分长句子
    # sentences = re.split(r'(?<=[。！？；：\n])', text)  # 如果想拆分短语，可以用这个
    sentences = [s.strip() for s in sentences if s.strip()]  # 去除空白
    return sentences

def evaluate_text_coherence(text):
    """
    计算完整文本的逻辑连贯性，方法：
    - 逐句计算相邻句子的 Textual Entailment（文本蕴含）得分
    - 取平均值作为文本整体的逻辑连贯性评分
    """
    sentences = split_sentences(text)
    scores = []

    for i in range(len(sentences) - 1):
        score = entailment_score(sentences[i], sentences[i+1])
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0  # 计算平均分

# **测试文本**
original_text = """男子骑无牌助力车被拦撞伤交警(图)
　　昨天中午12点多，记者在南京新街口洪武路口等红灯时，目睹了一名骑无牌助力车的男子为了躲避交警执法，竟然将交警撞倒的全过程。记者随后从警方了解到，当天在新街口地区，有两名交警在对无牌助力车进行检查时被撞伤，所幸伤势并不严重。
　　事发前，记者和一些行人正在淮海路与洪武路十字路口等信号灯，在路口有两名交警二大队的民警和一名辅警正在执勤。当时非机动车道上有一名青年男子正骑着一辆无牌助力车由南向北行驶，在通过路口的时候辅警示意其靠边接受检查，但是那名男子加速继续向前冲。辅警见状只得连连退步，并大声地向一旁的交警求助。
　　助力车上的男子见状立即调转车头准备绕过交警，正在此时，一位姓黄的交警冲上前去抓住了助力车的车头，助力车在怠速的情况下顶着交警冲到了两三米外后侧翻倒地，交警和驾驶助力车的男子均摔倒在了地上，另一名交警立即冲上前去将骑助力车男子控制住。
　　随后，摔倒在地上的交警艰难地爬了起来，记者发现他的裤子被刮破，胳膊上也有几道伤痕。为了不影响道路交通行驶，受伤的交警随后又将助力车扶了起来，此时这名交警的腿明显是受伤了，走路已经一瘸一拐。
　　交警被撞之后，目睹了事情经过的市民纷纷对这名骑助力车男子的野蛮行为表示不满。一位在现场的女士走上前安慰交警，并且拿出了纸巾给交警擦拭伤口。而撞伤交警的青年男子则在一旁显得不知所措，记者询问他当时为什么会撞交警？他只是说没有看到。最后这名男子被带往辖区淮海路派出所接受处理，受伤的交警到医院检查后发现全身多处擦伤，但并无大碍。记者从警方了解到，昨天在新街口，一共有两名交警在执法中遭遇阻扰，两名骑助力车的男子为躲避处罚将交警撞倒在地。除了这位黄警官外，另一位交警在淮海路执法时，也被助力车顶翻在地，头部被撞。
　　交管部门相关人士告诉记者，南京一直在严查无牌助力车，除了罚款200元外，还将扣车进行排量检测，这是因为不少车主喜欢对助力车进行改装，换缸后加大排量。特别危险的是，这些改装助力车的刹车系统无法在高速下完全发挥作用，速度上去，车却刹不住，极易引发事故。如果检测结果达到了轻摩或摩托车标准，而驾车人又没有驾驶证，就有可能被拘留。正因如此，一些骑无牌助力车上路的人在遇到交警检查时，往往会铤而走险，躲避处罚，这样往往会造成撞伤交警的恶性事件。 本报记者 郭一鹏 裴睿
"""

shuffled_text = """昨天中午，在南京新街口，交警正在执勤时，发现一名男子驾驶无牌、改装的助力车在道路上高速行驶。交警随即上前拦截检查，但男子见状加速试图绕开交警，结果因操作失控，助力车翻倒在地。 
男子摔倒后，全身多处受伤，交警立即上前了解情况。该男子并未携带驾驶证，而他的助力车不仅是改装过的，还不符合相关标准，属于违规车辆。在交警准备依法扣车时，男子试图继续驾车逃离，但因伤势影响，最终未能成功。 
此事发生在淮海路十字路口，现场有不少市民目睹了这一幕。由于道路严查无牌助力车，该男子可能为了躲避处罚而铤而走险，结果造成了事故。交警随即对男子进行询问，并通知相关部门处理。 
事故发生后，交警将男子送往医院检查，并对助力车进行扣留处理。交警提醒广大市民，无牌助力车存在较大安全隐患，违规改装更是加剧了事故风险，希望市民遵守交通法规，避免此类危险行为发生。
"""
shuffled_text2="""
SentencePiece is a language-independent subword tokenizer and detokenizer designed for neural text processing tasks, including Neural Machine Translation (NMT). It provides open-source implementations in C++ and Python for handling subword units. Unlike traditional subword segmentation tools that require pre-tokenized word sequences as input, SentencePiece can train subword models directly from raw text, enabling a fully end-to-end and language-agnostic approach. Experiments on English-Japanese NMT demonstrate that direct subword training from raw sentences achieves comparable accuracy to conventional methods. Additionally, the study explores the impact of different subword training and segmentation configurations. SentencePiece is released under the Apache 2 license and is available at [GitHub](https://github.com/google/sentencepiece).
"""
# **评估原文本**
print(f"原文本连贯性评分: {evaluate_text_coherence(original_text)}")

# **评估打乱文本**
print(f"打乱文本连贯性评分: {evaluate_text_coherence(shuffled_text)}")


print(f"打乱文本连贯性评分: {evaluate_text_coherence(shuffled_text2)}")
