from sentence_transformers import SentenceTransformer, util

import torch
# 1. 加载 RoBERTa 语义相似性模型
model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

text1="""男子骑无牌助力车被拦撞伤交警(图)
　　昨天中午12点多，记者在南京新街口洪武路口等红灯时，目睹了一名骑无牌助力车的男子为了躲避交警执法，竟然将交警撞倒的全过程。记者随后从警方了解到，当天在新街口地区，有两名交警在对无牌助力车进行检查时被撞伤，所幸伤势并不严重。
　　事发前，记者和一些行人正在淮海路与洪武路十字路口等信号灯，在路口有两名交警二大队的民警和一名辅警正在执勤。当时非机动车道上有一名青年男子正骑着一辆无牌助力车由南向北行驶，在通过路口的时候辅警示意其靠边接受检查，但是那名男子加速继续向前冲。辅警见状只得连连退步，并大声地向一旁的交警求助。
　　助力车上的男子见状立即调转车头准备绕过交警，正在此时，一位姓黄的交警冲上前去抓住了助力车的车头，助力车在怠速的情况下顶着交警冲到了两三米外后侧翻倒地，交警和驾驶助力车的男子均摔倒在了地上，另一名交警立即冲上前去将骑助力车男子控制住。
　　随后，摔倒在地上的交警艰难地爬了起来，记者发现他的裤子被刮破，胳膊上也有几道伤痕。为了不影响道路交通行驶，受伤的交警随后又将助力车扶了起来，此时这名交警的腿明显是受伤了，走路已经一瘸一拐。
　　交警被撞之后，目睹了事情经过的市民纷纷对这名骑助力车男子的野蛮行为表示不满。一位在现场的女士走上前安慰交警，并且拿出了纸巾给交警擦拭伤口。而撞伤交警的青年男子则在一旁显得不知所措，记者询问他当时为什么会撞交警？他只是说没有看到。最后这名男子被带往辖区淮海路派出所接受处理，受伤的交警到医院检查后发现全身多处擦伤，但并无大碍。记者从警方了解到，昨天在新街口，一共有两名交警在执法中遭遇阻扰，两名骑助力车的男子为躲避处罚将交警撞倒在地。除了这位黄警官外，另一位交警在淮海路执法时，也被助力车顶翻在地，头部被撞。
　　交管部门相关人士告诉记者，南京一直在严查无牌助力车，除了罚款200元外，还将扣车进行排量检测，这是因为不少车主喜欢对助力车进行改装，换缸后加大排量。特别危险的是，这些改装助力车的刹车系统无法在高速下完全发挥作用，速度上去，车却刹不住，极易引发事故。如果检测结果达到了轻摩或摩托车标准，而驾车人又没有驾驶证，就有可能被拘留。正因如此，一些骑无牌助力车上路的人在遇到交警检查时，往往会铤而走险，躲避处罚，这样往往会造成撞伤交警的恶性事件。 本报记者 郭一鹏 裴睿
"""
text2="""意其说等女士(的的在有此时除了刮破。撞倒冲助力车时昨天发现辅警在，地区后侧时喜欢控制这位他青年，交警继续到的驾驶情况　交警检查时候路口男子，但是记者骑无牌了记者道时助力车，，骑无牌，拦交警目睹多处撞倒男子这些正因如此的辅警元外后　而明显为记者交警等交警在骑改装，一共了当时男子在上去十字路口影响12有则告诉两名车头安慰一直正骑，，对，翻伤口立即的如果标准地所幸检测助力车撞伤了这名路口，的 抓住交警与洪武野蛮市民了解罚款一名就在的的正在。，随后。躲避在，两名特别不知所措助力车将怠速男子的记者无牌遇到执法冲上走路另交警外，处罚助力车助力车往往会检查撞受伤此时200摩托车造成危险达到地上在走上无牌翻倒民警一瘸一拐，将无牌医院他连连交警绕过交警完全但裤子被现场也被助力车通过拘留铤而走险由南向北为了和的行为刹车是又男子辅，交警被新街口撞伤，　当时对艰难交警交警将新街口会在交警淮海路的助力车，交警南京和的轻摩地向带这名又随后助力车有给竟然准备的　严重在。求助并车主了助力车发现
只是的扶加大后。助力车这样助力车为了摔倒
，，，男子行驶全过程几道前。助力车红灯却会调转　一位被排量头部起来
从检查询问交警交警，一名的的撞交警中检测上男子。在而了解胳膊　不助力车靠边有地
交警看到刹。了速度高速不住一些记者表示路口被，检查纷纷一名受伤拿出。相关当天助力车擦伤已经，。并且，往往了或信号灯加速车伤痕，人士没有的车头这一位黄， 上路交警。被淮海路严查。在扣车了有，驾驶证)结果往男子向前，的，的将在男子人目睹交警，骑？为什么，一位起来大声骑不少前去。上恶性事件另一名，他男子前将。交警没有，
南京除了洪武这名那名警方驾车人派出所的被最后躲避正在图阻扰撞伤随后大队警示在。见状擦拭还新街口接受
事故是因为。了躲避处理交警到，并一辆的受伤警官在，无法记者，助力车　伤势地上系统交管部门换缸纸巾交警进行，两名骑无牌两名，执法的淮海路并腿摔倒道路交通是郭一鹏的地的
了着非机动车到一些昨天中午执勤交警，外之后着进行经过下顶记者点多路。　冲上　对极易，可能，一旁，顶事发不满姓黄立即　时男子，和两三米。也接受了 撞执法　无只得从下不一旁前去引发警方裴睿青年　显得行人改装二上，发挥作用见状被本报记者大碍均，在在助力车爬在撞伤行驶退步进行交警地
住，正在在冲到处罚，辖区排量有全身遭遇 事情
"""
# 5. 计算文本嵌入

embedding_text1=model.encode(text1,convert_to_tensor=True)
embedding_text2=model.encode(text2,convert_to_tensor=True)
# 6. 计算余弦相似度
similarity = util.pytorch_cos_sim(embedding_text1, embedding_text2).item()

print(similarity)




# 加载更适合的模型
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')



# 计算文本嵌入
embedding_text1 = model.encode(text1, convert_to_tensor=True)
embedding_text2 = model.encode(text2, convert_to_tensor=True)

# 计算余弦相似度
similarity = util.pytorch_cos_sim(embedding_text1, embedding_text2).item()
print(similarity)  # 预期结果接近0



model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')

# 计算文本嵌入
embedding_text1 = model.encode(text1, convert_to_tensor=True)
embedding_text2 = model.encode(text2, convert_to_tensor=True)

# 计算余弦相似度
similarity = util.pytorch_cos_sim(embedding_text1, embedding_text2).item()
print(similarity)  # 预期结果接近0