from transformers import AutoModel

model = AutoModel.from_pretrained("google/ul2", force_download=True, cache_dir="~/.cache/huggingface/hub/models--google--ul2")

print("模型加载成功！")

""" from transformers.utils import cached_path

print(cached_path("google/ul2"))
 """