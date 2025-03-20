from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

def restore_with_bert(shuffled_text):
    words = shuffled_text.split()
    input_text = " ".join(["[MASK]"] * len(words))  # 全部 mask
    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_ids = torch.argmax(outputs.logits, dim=-1)
    predicted_words = tokenizer.convert_ids_to_tokens(predicted_ids[0])

    return " ".join(predicted_words)

shuffled_sentence = "quietly playing listens the while piano the audience is A man"
shuffled_sentence=". variable - node - based decoding algorithm for LDPC binary MP the problems ) low proposed a of ( , message pre-processing (VNBP-MP density VNBP propagation the data reliability parity is codes pre belief algorithm To processing check with solve NAND flash storages"
print(restore_with_bert(shuffled_sentence))
