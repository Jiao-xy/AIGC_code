import spacy
import random

# 加载 SpaCy 英语模型
nlp = spacy.load("en_core_web_sm")

def get_chunks(sentence):
    """
    解析句子并提取短语（名词短语、动词短语），避免重复
    """
    doc = nlp(sentence)
    
    # 获取名词短语（NP），限制最大长度，避免整句被当成短语
    noun_chunks = []
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 7:  # 限制短语长度，避免过长短语
            noun_chunks.append(chunk.text)
    
    # 获取动词短语（VP），确保提取合理长度
    verb_chunks = []
    for token in doc:
        if token.pos_ == "VERB":
            verb_phrase = " ".join([child.text for child in token.subtree])
            if len(verb_phrase.split()) <= 7:  # 限制短语长度
                verb_chunks.append(verb_phrase)

    # 计算剩余单词（去除已在短语中的单词）
    all_chunks_words = set(word for phrase in noun_chunks + verb_chunks for word in phrase.split())
    remaining_words = [token.text for token in doc if token.text not in all_chunks_words]

    return noun_chunks, verb_chunks, remaining_words

def scramble_sentence(sentence):
    """
    对句子进行打乱，但保留短语内部顺序
    """
    noun_chunks, verb_chunks, remaining_words = get_chunks(sentence)

    # 仅存储唯一短语，避免重复
    all_elements = list(set(noun_chunks + verb_chunks + remaining_words))

    # 仅打乱短语 & 词序，不重复执行
    random.shuffle(all_elements)

    # 重新组合成打乱后的句子
    scrambled_sentence = " ".join(all_elements)

    return scrambled_sentence

# 示例句子
sentence = "To solve the problems of the data reliability for NAND flash storages, a variable-node-based belief-propagation with message pre-processing (VNBP-MP) decoding algorithm for binary low-density parity-check (LDPC) codes is proposed."

# 生成打乱的句子
scrambled = scramble_sentence(sentence)

# 输出结果
print("🔹 原始句子：", sentence)
print("✅ 打乱后句子：", scrambled)
