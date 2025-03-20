import numpy as np
import torch
from transformers import GPT2LMHeadModel, BertTokenizer

class LM:
    def __init__(self, model_name_or_path="uer/gpt2-distil-chinese-cluecorpussmall"):
        """
        初始化 GPT-2 模型和分词器。
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)  # 加载分词器
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(self.device)  # 加载模型
        self.model.eval()  # 设置模型为评估模式
        self.start_token = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else self.tokenizer.bos_token_id

    def check_probabilities_with_perplexity(self, in_text, topk=1000):
        """
        计算文本中每个 token 的概率、对数概率和困惑度，同时获取每个位置的 Top-K 预测，
        并统计每个 token 的排名范围。
        """
        # 将文本转换为 token ID
        token_ids = self.tokenizer(in_text, return_tensors='pt')['input_ids'][0]
        start_token_tensor = torch.tensor([self.start_token], device=token_ids.device)
        token_ids = torch.cat([start_token_tensor, token_ids])

        # 模型前向传播
        outputs = self.model(token_ids.to(self.device), return_dict=True)
        logits = outputs.logits[:-1]
        probs = torch.softmax(logits, dim=-1)

        # 获取目标 token 的概率和对数概率
        target_ids = token_ids[1:]
        word_probs = probs[torch.arange(len(target_ids)), target_ids].detach().cpu().numpy()
        word_log_probs = np.log(word_probs)
        perplexity = np.exp(-np.mean(word_log_probs))

        # 转换为可读 token
        bpe_strings = self.tokenizer.convert_ids_to_tokens(token_ids)
        topk_values, topk_indices = torch.topk(probs, k=topk, dim=-1)
        pred_topk = [
            [(self.tokenizer.convert_ids_to_tokens([idx.item()])[0], val.item()) for idx, val in zip(topk_indices[i], topk_values[i])]
            for i in range(len(target_ids))
        ]

        # 获取每个目标词的排名范围
        rank_ranges = []
        stats = {"Top 0-10": 0, "Top 10-100": 0, "Top 100-1000": 0, "Above Top 1000": 0}
        for i, target in enumerate(target_ids):
            sorted_indices = torch.argsort(probs[i], descending=True).cpu()
            rank = (sorted_indices == target).nonzero(as_tuple=True)[0].item() + 1
            if rank <= 10:
                rank_ranges.append("Top 0-10")
                stats["Top 0-10"] += 1
            elif rank <= 100:
                rank_ranges.append("Top 10-100")
                stats["Top 10-100"] += 1
            elif rank <= 1000:
                rank_ranges.append("Top 100-1000")
                stats["Top 100-1000"] += 1
            else:
                rank_ranges.append("Above Top 1000")
                stats["Above Top 1000"] += 1

        return {
            'bpe_strings': bpe_strings,
            'word_probs': word_probs.tolist(),
            'word_log_probs': word_log_probs.tolist(),
            'perplexity': perplexity,
            'pred_topk': pred_topk,
            'rank_ranges': rank_ranges,
            'stats': stats,
            'total_tokens': len(target_ids)
        }

if __name__ == "__main__":
    raw_text = "科学家在安第斯山脉发现了一群独角兽，这一发现令人震惊。"
    raw_text="9月8日晚，某地林某因欠债遭人讨债，遂绑架某女和其母作为要债手段。李某等人得知后，趁夜迅速行动，与警方合作解救被绑架的母女。警方将林某抓获，并因其涉嫌绑架、违反法律禁令，将其拘留。最终，林某被判赔偿4万元，并被判刑入狱。此事引起了社会的广泛关注，提醒人们要遵守法律，避免采取非法手段解决债务问题。"
    #raw_text="法官快审快结盗窃案，使得被告人——一名高三学生没有错过高考。记者昨天获悉，17岁的晓玲(化名)近日从辽宁老家来到朝阳法院少年审判庭，将自己的大学录取通知书拿给法官刘鹏看，感谢法庭对自己从轻判决。去年12月，晓玲从辽宁来到北京学习，想报考北京一所艺术院校。和她同来的还有一名同学，眼见同学成绩日益提高，晓玲很着急。一次在和同学玩耍时，晓玲看到她钱包里有一张银行卡，并从同学的言谈中得知了密码。趁同学不注意，晓玲把银行卡偷走，在提款机上分11次取走2.2万元。“我不是缺钱花，我只是想用这种办法让她无心复习”。  随后，晓玲被公安机关查获归案，她立即退还了取出的2.2万元。由于她正读高三，父母就办理了取保候审，让她继续学习功课，备战高考。  今年5月，此案被公诉至朝阳法院少年审判庭。主审法官刘鹏在庭前和晓玲联系时，得知她还在复习，准备参加今年6月的高考。“她的压力非常大，家里也为她的未来担心，生怕受审耽误了高考”。  法官当即决定，对此案采取快审快结。由于晓玲在事发后退赔了同学的钱，并没给对方造成实际损失，她的悔罪态度也好，属于未成年犯罪，法庭从轻判处她有期徒刑1年并宣告缓刑，此事从立案到判决仅用了9天。根据《未成年人保护法》规定，被判缓刑不会影响到她的高考和升学。拿到缓刑判决，晓玲含着眼泪表示一定珍惜这次机会，放下思想包袱好好准备考试。"
    raw_text="""男子骑无牌助力车被拦撞伤交警(图)
　　昨天中午12点多，记者在南京新街口洪武路口等红灯时，目睹了一名骑无牌助力车的男子为了躲避交警执法，竟然将交警撞倒的全过程。记者随后从警方了解到，当天在新街口地区，有两名交警在对无牌助力车进行检查时被撞伤，所幸伤势并不严重。 
　　事发前，记者和一些行人正在淮海路与洪武路十字路口等信号灯，在路口有两名交警二大队的民警和一名辅警正在执勤。当时非机动车道上有一名青年男子正骑着一辆无牌助力车由南向北行驶，在通过路口的时候辅警示意其靠边接受检查，但是那名男子加速继续向前冲。辅警见状只得连连退步，并大声地向一旁的交警求助。
　　助力车上的男子见状立即调转车头准备绕过交警，正在此时，一位姓黄的交警冲上前去抓住了助力车的车头，助力车在怠速的情况下顶着交警冲到了两三米外后侧翻倒地，交警和驾驶助力车的男子均摔倒在了地上，另一名交警立即冲上前去将骑助力车男子控制住。
　　随后，摔倒在地上的交警艰难地爬了起来，记者发现他的裤子被刮破，胳膊上也有几道伤痕。为了不影响道路交通行驶，受伤的交警随后又将助力车扶了起来，此时这名交警的腿明显是受伤了，走路已经一瘸一拐。
　　交警被撞之后，目睹了事情经过的市民纷纷对这名骑助力车男子的野蛮行为表示不满。一位在现场的女士走上前安慰交警，并且拿出了纸巾给交警擦拭伤口。而撞伤交警的青年男子则在一旁显得不知所措，记者询问他当时为什么会撞交警？他只是说没有看到。最后这名男子被带往辖区淮海路派出所接受处理，受伤的交警到医院检查后发现全身多处擦伤，但并无大碍。记者从警方了解到，昨天在新街口，一共有两名交警在执法中遭遇阻扰，两名骑助力车的男子为躲避处罚将交警撞倒在地。除了这位黄警官外，另一位交警在淮海路执法时，也被助力车顶翻在地，头部被撞。
　　交管部门相关人士告诉记者，南京一直在严查无牌助力车，除了罚款200元外，还将扣车进行排量检测，这是因为不少车主喜欢对助力车进行改装，换缸后加大排量。特别危险的是，这些改装助力车的刹车系统无法在高速下完全发挥作用，速度上去，车却刹不住，极易引发事故。如果检测结果达到了轻摩或摩托车标准，而驾车人又没有驾驶证，就有可能被拘留。正因如此，一些骑无牌助力车上路的人在遇到交警检查时，往往会铤而走险，躲避处罚，这样往往会造成撞伤交警的恶性事件。 本报记者 郭一鹏 裴睿
"""
    print("\n测试 GPT-2 模型的困惑度计算...")
    gpt2_lm = LM()
    gpt2_payload = gpt2_lm.check_probabilities_with_perplexity(raw_text, topk=1000)
    print("GPT-2 输出:")
    print("Token 数量:", len(gpt2_payload['bpe_strings']), "概率数量:", len(gpt2_payload['word_probs']), "对数概率数量:", len(gpt2_payload['word_log_probs']))
    for i in range(len(gpt2_payload['bpe_strings']) - 1):
        print(f"{gpt2_payload['bpe_strings'][i+1]}: 概率={gpt2_payload['word_probs'][i]:.6f}, 对数概率={gpt2_payload['word_log_probs'][i]:.6f}, 排名范围={gpt2_payload['rank_ranges'][i]}")
        print("Top-10 预测:", end=": ")
        for word in gpt2_payload['pred_topk'][i][:10]:  # 打印Top-10
            print(f"({word[0]}: 概率={word[1]:.6f})", end=" ")
        print()
    print("困惑度:", gpt2_payload['perplexity'])

    print("\n统计结果:")
    total_tokens = gpt2_payload['total_tokens']
    for range_name, count in gpt2_payload['stats'].items():
        print(f"{range_name}: {count} / {total_tokens} ({count / total_tokens * 100:.2f}%)")
