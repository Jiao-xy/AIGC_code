import spacy
import random

nlp = spacy.load("en_core_web_sm")

def restore_sentence(shuffled_text):
    words = shuffled_text.split()
    doc = nlp(" ".join(words))

    # 词性分类
    nouns, verbs, adjs, others = [], [], [], []
    
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            nouns.append(token.text)
        elif token.pos_ in ["VERB"]:
            verbs.append(token.text)
        elif token.pos_ in ["ADJ", "ADV"]:
            adjs.append(token.text)
        else:
            others.append(token.text)

    # 重新组合（简单 SVO 结构）
    restored_sentence = nouns + verbs + adjs + others
    return " ".join(restored_sentence)

shuffled_sentence = "quietly playing listens the while piano the audience is A man"
shuffled_sentence=". variable - node - based decoding algorithm for LDPC binary MP the problems ) low proposed a of ( , message pre-processing (VNBP-MP density VNBP propagation the data reliability parity is codes pre belief algorithm To processing check with solve NAND flash storages"

print(restore_sentence(shuffled_sentence))
