import torch
from transformers import BertTokenizer, BertModel
import language_tool_python
from rouge import Rouge
import bert_score

# ✅ 1️⃣ 语法正确性计算（使用 LanguageTool）
def grammar_check(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    return len(matches)  # 语法错误数量

# ✅ 2️⃣ 信息完整性计算（ROUGE 和 BLEU）
def calculate_rouge(original, generated):
    rouge = Rouge()
    scores = rouge.get_scores(generated, original)
    return scores[0]["rouge-l"]["f"]  # 使用 ROUGE-L 进行评估

# ✅ 3️⃣ 逻辑连贯性计算（BERTScore）
def calculate_bertscore(original, generated):
    P, R, F1 = bert_score.score([generated], [original], lang="en")
    return F1.mean().item()  # 计算 F1 语义相似度得分

# ✅ 4️⃣ 计算句子流畅度（BERT Perplexity）
def calculate_fluency(text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    perplexity = torch.exp(torch.mean(outputs[0]))  # 计算困惑度
    return perplexity.item()

# ✅ 5️⃣ 综合计算四个指标并输出得分
def evaluate_sentence(original, generated):
    grammar_errors = grammar_check(generated)
    rouge_score = calculate_rouge(original, generated)
    bert_score = calculate_bertscore(original, generated)
    fluency = calculate_fluency(generated)
    
    results = {
        "语法正确性（Grammar Errors）": max(0, 10 - grammar_errors),
        "信息完整性（ROUGE）": rouge_score * 10,
        "逻辑连贯性（BERTScore）": bert_score * 10,
        "自然流畅度（Fluency）": max(0, 10 - fluency)
    }
    
    return results

# ✅ 6️⃣ 测试示例
original_sentence = "She carries a small notebook in her bag to jot down thoughts, random ideas, or even sketches whenever inspiration strikes, no matter where she is."
generated_sentence = "She carries a notebook in her bag to jot down ideas, random sketches, or thoughts whenever inspiration strikes, wherever she is."

results = evaluate_sentence(original_sentence, generated_sentence)
print(results)

original_sentences_list = [
    "The cat jumped onto the windowsill, watching the birds outside as the morning sun bathed its fur in golden light, making it feel warm and cozy.",  
    "He spent the entire afternoon reading a novel under the old oak tree, completely lost in the world of the story, unaware of the gentle breeze that rustled the leaves above him.",  
    "The train arrived late, but no one seemed to mind since the station was filled with travelers engaged in lively conversations, sipping coffee, and checking their phones.",  
    "She always carries a small notebook in her bag to jot down thoughts, random ideas, or even sketches whenever inspiration strikes, no matter where she is.",  
    "The stars shone brightly in the clear night sky, forming constellations that told ancient stories, while a cool breeze carried the distant sound of waves crashing on the shore.",  
    "A sudden gust of wind knocked over the cup of coffee he had just placed on the table, spilling the warm liquid all over his unfinished manuscript, smudging the ink.",  
    "The city was alive with bright neon lights, the honking of cars, the chatter of pedestrians, and the distant sound of street musicians playing for an audience of strangers.",  
    "He found an old photograph hidden between the pages of a dusty book in his grandfather’s library, showing a smiling couple he had never seen before, standing in front of a vintage car.",  
    "The little girl laughed with pure joy as she ran across the garden, chasing butterflies that fluttered playfully around her, their delicate wings shimmering in the sunlight.",  
    "His phone battery died just as he was about to send an important message, leaving him frustrated and desperately searching for a charger in his cluttered backpack.",  
    "The scent of freshly baked bread filled the kitchen, making everyone’s stomach growl with hunger as the baker carefully pulled the golden loaves from the hot oven.",  
    "She practiced the piano every evening for hours, determined to perfect the melody, her fingers gracefully gliding over the keys as the notes echoed through the empty house.",  
    "The waves crashed against the jagged rocks, sending salty mist into the air, as seagulls soared above, their cries blending with the rhythmic sound of the ocean.",  
    "A mysterious letter arrived in the mail with no return address, sealed with red wax, and inside was a cryptic message that made her heart race with excitement and curiosity.",  
    "The old clock in the hallway struck midnight with a deep, resonant chime, its ancient gears turning slowly as the sound echoed through the silent house, sending shivers down his spine.",  
    "He stared at the blank canvas for hours, searching for inspiration, his paintbrush hovering hesitantly in the air, waiting for the first stroke to bring his vision to life.",  
    "The puppy wagged its tail excitedly at the sight of its owner returning home, jumping up and down with pure happiness, its eyes full of love and excitement.",  
    "She whispered a secret into her best friend’s ear, giggling softly as they exchanged knowing glances, their friendship strengthened by the shared moment of trust and mischief.",  
    "The festival was filled with colorful lanterns swaying gently in the evening breeze, vibrant stalls selling delicious street food, and musicians playing lively tunes for the cheerful crowd.",  
    "He took a deep breath before stepping onto the stage, his heart pounding in his chest as the spotlight illuminated him, and for a moment, time seemed to stand still."
]

generated_sentences_list = [
    "windowsill jumped onto the cat, outside birds watching the morning bathed its as golden light in fur making it warm and cozy.",
    "the old day under a gentle oak tree spent, reading novel the completely lost world story, of him unaware in the rustled breeze that leaves above him.",
    "late, arrived to train the station but seemed no since the mind was filled to the travelers, engaged lively conversations in sipping coffee, checking and their phones.",
    "carries her notebook to a bag in small jot down thoughts, random inspiration sketches or even whenever she strikes no matter where she is ideas.",
    "night bright stars in the clear sky, ancient constellations forming stories told while distant stars carried a cool sound of waves the breeze crashing on shore.",
    "over coffee cup the manuscript he just placed had spilled a warm sudden knocked wind of all the liquid onto the unfinished ink.",
    "the city was alive with bright neon lights, honking chatter of the noisy cars, sound street and playing musicians for the an audience distant of strangers.",
    "the book in a photograph found an old library between his grandfather’s old pages showing a couple of front in standing before, seen never had a dusty smiling car.",
    "little joy she laughed as pure butterflies across the garden, chasing playfully fluttered butterflies around her that shimmering their wings in delicate sunlight.",
    "phone just was frustrated to send an important message about he died as battery leaving his desperately cluttered backpack for searching a charger in his phone.",
    "the bread filled with freshly baked scent of everyone’s golden stomach growl hunger as carefully pulled from the oven loaves making hot kitchen baker.",
    "piano melody for the hours, determined as every evening practiced gracefully through the perfect gliding notes fingers over empty the echoed keys house.",
    "the rocks, jagged against the seagulls, soared mist into the salty ocean air, sending their rhythmic cries above blending with sound waves.",
    "no return sealed wax red letter with a cryptic address made in the heart her mysterious letter inside arrived in the race excitement and curiosity a mail.",
    "the chime, resonant deep midnight echoed old gears sound of the silent hallway as its slowly ancient clock turning through his spine struck the sending shivers.",
    "blank stared for his searched hours, paintbrush inspiration, hovering hesitantly to bring the first life for stroke vision waiting in the canvas.",
    "puppy tail home, returning its pure excitement at sight, its jumping owner wagged down its happiness, up with its full eyes and the love.",
    "whispered secret into her best giggling softly knowing glances exchanged their secrets as they strengthened the trust of moment shared by their mischief.",
    "the evening lanterns swaying street filled with colorful breeze, vibrant stalls selling delicious tunes for the musicians playing colorful and lively food, crowd cheerful festival.",
    "deep breath took stage, before breath onto a heart stepping moment, chest pounding moment in his spotlight and illuminated him as his stood still."
]
for (original_sentence, generated_sentence) in zip(original_sentences_list, generated_sentences_list):
    results = evaluate_sentence(original_sentence, generated_sentence)
    print(results)
