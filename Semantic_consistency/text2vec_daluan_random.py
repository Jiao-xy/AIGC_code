import random
import jieba
from sentence_transformers import SentenceTransformer, util

# 1. 加载 text2vec 中文语义相似度模型
model = SentenceTransformer("shibing624/text2vec-base-chinese")

# 2. 定义原始中文文本
original_text = """男子骑无牌助力车被拦撞伤交警(图)
　　昨天中午12点多，记者在南京新街口洪武路口等红灯时，目睹了一名骑无牌助力车的男子为了躲避交警执法，竟然将交警撞倒的全过程。记者随后从警方了解到，当天在新街口地区，有两名交警在对无牌助力车进行检查时被撞伤，所幸伤势并不严重。 
　　事发前，记者和一些行人正在淮海路与洪武路十字路口等信号灯，在路口有两名交警二大队的民警和一名辅警正在执勤。当时非机动车道上有一名青年男子正骑着一辆无牌助力车由南向北行驶，在通过路口的时候辅警示意其靠边接受检查，但是那名男子加速继续向前冲。辅警见状只得连连退步，并大声地向一旁的交警求助。
　　助力车上的男子见状立即调转车头准备绕过交警，正在此时，一位姓黄的交警冲上前去抓住了助力车的车头，助力车在怠速的情况下顶着交警冲到了两三米外后侧翻倒地，交警和驾驶助力车的男子均摔倒在了地上，另一名交警立即冲上前去将骑助力车男子控制住。
　　随后，摔倒在地上的交警艰难地爬了起来，记者发现他的裤子被刮破，胳膊上也有几道伤痕。为了不影响道路交通行驶，受伤的交警随后又将助力车扶了起来，此时这名交警的腿明显是受伤了，走路已经一瘸一拐。
　　交警被撞之后，目睹了事情经过的市民纷纷对这名骑助力车男子的野蛮行为表示不满。一位在现场的女士走上前安慰交警，并且拿出了纸巾给交警擦拭伤口。而撞伤交警的青年男子则在一旁显得不知所措，记者询问他当时为什么会撞交警？他只是说没有看到。最后这名男子被带往辖区淮海路派出所接受处理，受伤的交警到医院检查后发现全身多处擦伤，但并无大碍。记者从警方了解到，昨天在新街口，一共有两名交警在执法中遭遇阻扰，两名骑助力车的男子为躲避处罚将交警撞倒在地。除了这位黄警官外，另一位交警在淮海路执法时，也被助力车顶翻在地，头部被撞。
　　交管部门相关人士告诉记者，南京一直在严查无牌助力车，除了罚款200元外，还将扣车进行排量检测，这是因为不少车主喜欢对助力车进行改装，换缸后加大排量。特别危险的是，这些改装助力车的刹车系统无法在高速下完全发挥作用，速度上去，车却刹不住，极易引发事故。如果检测结果达到了轻摩或摩托车标准，而驾车人又没有驾驶证，就有可能被拘留。正因如此，一些骑无牌助力车上路的人在遇到交警检查时，往往会铤而走险，躲避处罚，这样往往会造成撞伤交警的恶性事件。 本报记者 郭一鹏 裴睿
"""
original_text="""斯蒂格利茨：银行业问题比危机前更严重 
　　新浪财经讯 北京时间9月14日上午消息，据国外媒体报道，诺贝尔经济学奖得主、纽约哥伦比亚大学教授约瑟夫·斯蒂格利茨(Joseph Stiglitz)日前表示，在金融危机爆发和雷曼兄弟破产后，美国一直未能解决该国银行体系存在的根本问题。
　　斯蒂格利茨周日在巴黎接受媒体采访时表示：“在美国和其他许多国家，太大而不能倒闭的银行正变得更为庞大。银行业问题甚至比2007年金融危机爆发前更加严重。”
　　雷曼兄弟的倒闭迫使美国财政部投入巨额资金来支撑该国的金融体系，美国银行的资产保持增长，花旗整体保持完好。虽然美国总统奥巴马希望确定一些“具有系统性重要意义的”银行，使其受到更为严格的监管，但该计划并不能迫使这些银行缩减规模或简化结构。
　　斯蒂格利茨称，美国政府对大力改革金融业持谨慎态度，因为这在政治上是比较困难的。他希望二十国集团(G20)领导人能够让美国采取更严厉措施改革金融业。(兴亚)
   已有_COUNT_条评论  我要评论
"""
original_text="""初秋的阳光温柔地洒落在大地上，远处的山峦起伏如波，近处的溪水潺潺流过。空气中弥漫着淡淡的花香，仿佛是季节的芬芳在空气中流转。小鹿蹦蹦跳跳的身影在林间留下一道可爱的小径，鸟鸣声此起彼伏，仿佛在唱响这独特的秋日交响曲。抬眼望去，满目的都是金黄与翠绿，仿佛一幅精心绘制的画卷，让人忍不住放慢脚步，感受这份难得的宁静与美好。
"""
# 3. 计算原始文本的嵌入
embedding_original = model.encode(original_text, convert_to_tensor=True)

# 4. 初始化变量，记录最低相似度及对应文本
min_similarity_char, worst_text_char = float("inf"), ""
min_similarity_word, worst_text_word = float("inf"), ""
n=0
# 5. 进行 1000 次打乱实验
for _ in range(100):
    # 按字打乱
    chars = list(original_text)
    random.shuffle(chars)
    shuffled_text_by_char = "".join(chars)
    embedding_char_shuffled = model.encode(shuffled_text_by_char, convert_to_tensor=True)
    similarity_char = util.pytorch_cos_sim(embedding_original, embedding_char_shuffled).item()

    # 更新最小相似度和对应文本
    if similarity_char < min_similarity_char:
        min_similarity_char = similarity_char
        worst_text_char = shuffled_text_by_char

    # 按词打乱
    words = list(jieba.cut(original_text))
    random.shuffle(words)
    shuffled_text_by_word = "".join(words)  # 重新拼接（保持无空格）
    embedding_word_shuffled = model.encode(shuffled_text_by_word, convert_to_tensor=True)
    similarity_word = util.pytorch_cos_sim(embedding_original, embedding_word_shuffled).item()

    # 更新最小相似度和对应文本
    if similarity_word < min_similarity_word:
        min_similarity_word = similarity_word
        worst_text_word = shuffled_text_by_word
    n=n+1
    # 如果发现相似度低于 0.6，提前结束
    if min_similarity_char < 0.3 or min_similarity_word < 0.3:
        print(f"打乱实验次数：{n}")
        break

# 6. 输出最低相似度的打乱文本及其相似度
print(f"原始文本: {original_text}")

print(f"\n【按字打乱】最低相似度: {min_similarity_char:.4f}")
print(f"最低相似度文本（按字打乱）: {worst_text_char}")

print(f"\n【按词打乱】最低相似度: {min_similarity_word:.4f}")
print(f"最低相似度文本（按词打乱）: {worst_text_word}")
