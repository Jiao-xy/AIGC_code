from difflib import SequenceMatcher

def is_reordering_only(original, modified):
    """
    检查 modified 是否仅是 original 的单词顺序变化，而没有增加/删除单词
    """
    original_words = original.split()
    modified_words = modified.split()
    
    # 计算相似度
    ratio = SequenceMatcher(None, original_words, modified_words).ratio()
    print(ratio)
    # 如果顺序变化但内容一致，返回 True
    return set(original_words) == set(modified_words) and ratio < 1.0

# 示例
original_sentence = "To solve the problems of the data reliability for NAND flash storages, a variable-node-based belief-propagation with message pre-processing (VNBP-MP) decoding algorithm for binary low-density parity-check (LDPC) codes is proposed."

modified_sentence = "To solve the problems of the data reliability for NAND flash storages, a variable-node-based belief-propagation (VNBP-MP) decoding algorithm with message pre-processing is proposed for low-density parity-check (LDPC) codes."
#                   "To solve the problems of data reliability in NAND flash storages, a variable-node-based belief-propagation (VNBP-MP) decoding algorithm with message pre-processing is proposed for binary low-density parity-check (LDPC) codes."
modified_sentence="To solve the problems of low-density parity-check (LDPC) codes for NAND flash storages, a variable-node-based belief-propagation pre-processing algorithm (VNBP-MP) for binary data decoding with reliability is proposed."
print(is_reordering_only(original_sentence, modified_sentence))  # True
