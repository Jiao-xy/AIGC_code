#（1）词汇信息密度（Lexical Density, LD）
import spacy

nlp = spacy.load("en_core_web_sm")

def lexical_density(text):
    doc = nlp(text)
    content_words = [token.text for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]]
    return len(content_words) / len(doc)

text1 = "The experiment shows that the proposed method significantly improves accuracy."
text2 = "Experimental results indicate that the proposed method achieves a significant improvement in accuracy."

print("Lexical Density (Human):", lexical_density(text1))
print("Lexical Density (AI-Generated):", lexical_density(text2))
