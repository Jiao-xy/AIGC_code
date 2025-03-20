import spacy

# 加载语言模型
nlp = spacy.load("en_core_web_sm")

sentence = "To solve the problems of the data reliability for NAND flash storages, a variable-node-based belief-propagation with message pre-processing (VNBP-MP) decoding algorithm for binary low-density parity-check (LDPC) codes is proposed."
['the problems', 'the data reliability', 'NAND flash storages', 'message pre-processing (VNBP-MP', 'algorithm', 'LDPC']
doc = nlp(sentence)

# 输出依存关系
for token in doc:
    print(f"{token.text} <-- {token.dep_} <-- {token.head.text}")

# 生成短语
noun_chunks = [chunk.text for chunk in doc.noun_chunks]
print("短语划分:", noun_chunks)
