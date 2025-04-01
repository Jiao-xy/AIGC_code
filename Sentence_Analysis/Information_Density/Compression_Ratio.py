#文本压缩率
import gzip

def compression_ratio(text):
    original_size = len(text.encode("utf-8"))
    compressed_size = len(gzip.compress(text.encode("utf-8")))
    return compressed_size / original_size

text1 = "The Shuffle Test is the most common task to evaluate whether NLP models can measure coherence in text. Most recent work uses direct supervision on the task; we show that by simply finetuning a RoBERTa model, we can achieve a near perfect accuracy of 97.8%, a state-of-the-art. We argue that this outstanding performance is unlikely to lead to a good model of text coherence, and suggest that the Shuffle Test should be approached in a ZeroShot setting: models should be evaluated without being trained on the task itself. We evaluate common models in this setting, such as Generative and Bi-directional Transformers, and find that larger architectures achieve highperformance out-of-the-box. Finally, we suggest the k-Block Shuffle Test, a modification of the original by increasing the size of blocks shuffled. Even though human reader performance remains high (around 95% accuracy), model performance drops from 94% to 78% as block size increases, creating a conceptually simple challenge to benchmark NLP models. "
text2 = "Information entropy, introduced by Claude Shannon in 1948, is a fundamental concept in information theory that quantifies the uncertainty or randomness in a data source. It measures the average amount of information produced per message and is widely used in fields such as data compression, cryptography, and machine learning. The entropy of a discrete probability distribution is defined as the expected value of the information content associated with each possible outcome. A higher entropy value indicates greater uncertainty, meaning the data is more unpredictable, whereas lower entropy suggests a more structured and predictable dataset. Applications of entropy extend to decision tree algorithms, where it helps in feature selection, and in cryptographic systems, where it assesses the security of random number generators. As an essential measure in information processing and signal transmission, entropy plays a crucial role in optimizing communication systems and enhancing data security."

print("Compression Ratio (Human):", compression_ratio(text1))
print("Compression Ratio (AI-Generated):", compression_ratio(text2))