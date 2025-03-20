from transformers import T5Tokenizer

model_name = "/home/jxy/models/t5_reorder_base_final"
tokenizer = T5Tokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(model_name)  # ✅ 重新生成 tokenizer.json

print("✅ tokenizer.json 重新生成完成！")
