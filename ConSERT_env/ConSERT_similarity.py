from sentence_transformers import SentenceTransformer,util

# 指定模型的本地路径
model_path = "/home/jxy/unsup-consert-base-stsb/unsup-consert-base-stsb"  # 你的解压路径
model = SentenceTransformer(model_path)
model.max_seq_length = 256  # 先尝试 256，再尝试 512
# 测试句子相似度
sentences = ["表示你的输入有两个句子，每个句子的嵌入维度是七百六十八，符合模型的输出维度。", "问题彻底解决！现在你可以正式使用你的模型进行相似度计算了！"]
embeddings = model.encode(sentences)

# 输出嵌入向量
print(embeddings.shape)  # (2, 768)
similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
print("句子相似度:", similarity.item())