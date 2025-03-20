import spacy

# 加载 SpaCy 依存分析模型
nlp = spacy.load("en_core_web_sm")

# 示例文本（原始句子）
text = "Although the weather was cold, we decided to go for a walk. The wind was strong, and it made walking difficult. However, we enjoyed the fresh air and exercise."
text= "Route randomization is an important research focus for moving target defense which seeks to proactively and dynamically change the forwarding routes in the network. In this paper, the difficulties of implementing route randomization in traditional networks are analyzed. To solve these difficulties and achieve effective route randomization, a novel route randomization approach is proposed, which is implemented by adding a mapping layer between routers' physical interfaces and their corresponding logical addresses. The design ideas and the details of proposed approach are presented. The effectiveness and performance of proposed approach are verified and evaluated by corresponding experiments."

# **分句**
def split_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

sentences = split_sentences(text)

# **依存分析**
def analyze_dependencies(sentences):
    parsed_sentences = []
    
    for sent in sentences:
        doc = nlp(sent)
        
        # 统计 ROOT、nsubj、dobj 的数量
        root_count = sum(1 for token in doc if token.dep_ == 'ROOT')
        subject_count = sum(1 for token in doc if token.dep_ in ['nsubj', 'nsubjpass'])
        object_count = sum(1 for token in doc if token.dep_ in ['dobj', 'pobj'])

        # 计算句子的依存深度（依存路径的最大长度）
        tree_depth = max(token.i for token in doc) - min(token.i for token in doc) if len(doc) > 1 else 1
        
        parsed_sentences.append((sent, root_count, subject_count, object_count, tree_depth))
    
    return parsed_sentences

dependency_parsed = analyze_dependencies(sentences)

# **打印依存分析结果**
for sent, root, subj, obj, depth in dependency_parsed:
    print(f"句子: {sent}")
    print(f"  - ROOT 数量: {root}, 主语数: {subj}, 宾语数: {obj}, 依存树深度: {depth}")
    print()
# **逻辑重排**
def reorder_sentences(dependency_parsed):
    # 先按 root + subject + object 数量排序（重要句子靠前）
    # 再按 依存树深度排序（较浅的句子靠前）
    reordered = sorted(
        dependency_parsed,
        key=lambda x: (x[1] + x[2] + x[3], -x[4]), 
        reverse=True
    )
    
    return " ".join([sent[0] for sent in reordered])

# **重排后的文本**
reordered_text = reorder_sentences(dependency_parsed)
print(f"原始文本：{text}")
print("\n【重构后文本】")
print(reordered_text)
