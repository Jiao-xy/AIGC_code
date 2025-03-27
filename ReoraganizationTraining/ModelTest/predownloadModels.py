from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ✅ 只下载模型，不加载到显存
AutoTokenizer.from_pretrained("t5-base")
AutoModelForSeq2SeqLM.from_pretrained("t5-base")

AutoTokenizer.from_pretrained("facebook/bart-large")
AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")


# ✅ 只下载模型，不加载到显存
AutoTokenizer.from_pretrained("t5-large")
AutoModelForSeq2SeqLM.from_pretrained("t5-large")

AutoTokenizer.from_pretrained("facebook/bart-base")
AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

print("✅ `t5-large` 和 `bart-base` 已下载并存入缓存，但未加载进显存！")

print("✅ 所有模型已下载（存入缓存），但没有加载进显存！")
