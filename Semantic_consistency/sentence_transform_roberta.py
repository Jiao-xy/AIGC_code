import random
import jieba
from sentence_transformers import SentenceTransformer, util

# 1. 加载 RoBERTa 语义相似性模型
model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

# 2. 定义原始中文文本
original_text = """男子骑无牌助力车被拦撞伤交警(图)
　　昨天中午12点多，记者在南京新街口洪武路口等红灯时，目睹了一名骑无牌助力车的男子为了躲避交警执法，竟然将交警撞倒的全过程。记者随后从警方了解到，当天在新街口地区，有两名交警在对无牌助力车进行检查时被撞伤，所幸伤势并不严重。 
　　事发前，记者和一些行人正在淮海路与洪武路十字路口等信号灯，在路口有两名交警二大队的民警和一名辅警正在执勤。当时非机动车道上有一名青年男子正骑着一辆无牌助力车由南向北行驶，在通过路口的时候辅警示意其靠边接受检查，但是那名男子加速继续向前冲。辅警见状只得连连退步，并大声地向一旁的交警求助。
　　助力车上的男子见状立即调转车头准备绕过交警，正在此时，一位姓黄的交警冲上前去抓住了助力车的车头，助力车在怠速的情况下顶着交警冲到了两三米外后侧翻倒地，交警和驾驶助力车的男子均摔倒在了地上，另一名交警立即冲上前去将骑助力车男子控制住。
　　随后，摔倒在地上的交警艰难地爬了起来，记者发现他的裤子被刮破，胳膊上也有几道伤痕。为了不影响道路交通行驶，受伤的交警随后又将助力车扶了起来，此时这名交警的腿明显是受伤了，走路已经一瘸一拐。
　　交警被撞之后，目睹了事情经过的市民纷纷对这名骑助力车男子的野蛮行为表示不满。一位在现场的女士走上前安慰交警，并且拿出了纸巾给交警擦拭伤口。而撞伤交警的青年男子则在一旁显得不知所措，记者询问他当时为什么会撞交警？他只是说没有看到。最后这名男子被带往辖区淮海路派出所接受处理，受伤的交警到医院检查后发现全身多处擦伤，但并无大碍。记者从警方了解到，昨天在新街口，一共有两名交警在执法中遭遇阻扰，两名骑助力车的男子为躲避处罚将交警撞倒在地。除了这位黄警官外，另一位交警在淮海路执法时，也被助力车顶翻在地，头部被撞。
　　交管部门相关人士告诉记者，南京一直在严查无牌助力车，除了罚款200元外，还将扣车进行排量检测，这是因为不少车主喜欢对助力车进行改装，换缸后加大排量。特别危险的是，这些改装助力车的刹车系统无法在高速下完全发挥作用，速度上去，车却刹不住，极易引发事故。如果检测结果达到了轻摩或摩托车标准，而驾车人又没有驾驶证，就有可能被拘留。正因如此，一些骑无牌助力车上路的人在遇到交警检查时，往往会铤而走险，躲避处罚，这样往往会造成撞伤交警的恶性事件。 本报记者 郭一鹏 裴睿

"""

# 3. 按 **字** 进行打乱
chars = list(original_text)  # 拆分为单字
random.shuffle(chars)        # 随机打乱
shuffled_text_by_char = "".join(chars)  # 重新拼接

# 4. 按 **词** 进行打乱（jieba 分词）
words = list(jieba.cut(original_text))  # 使用结巴分词
random.shuffle(words)                   # 随机打乱
shuffled_text_by_word = "".join(words)  # 重新拼接（保持无空格）

# 5. 计算文本嵌入
embedding_original = model.encode(original_text, convert_to_tensor=True)
embedding_char_shuffled = model.encode(shuffled_text_by_char, convert_to_tensor=True)
embedding_word_shuffled = model.encode(shuffled_text_by_word, convert_to_tensor=True)

# 6. 计算余弦相似度
similarity_char = util.pytorch_cos_sim(embedding_original, embedding_char_shuffled).item()
similarity_word = util.pytorch_cos_sim(embedding_original, embedding_word_shuffled).item()

# 7. 输出结果
print(f"原始文本: {original_text}")
print(f"\n按字打乱: {shuffled_text_by_char}")
print(f"按字打乱相似度: {similarity_char:.4f}")

print(f"\n按词打乱: {shuffled_text_by_word}")
print(f"按词打乱相似度: {similarity_word:.4f}")
