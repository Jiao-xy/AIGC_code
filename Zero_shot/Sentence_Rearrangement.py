import json
import spacy
import networkx as nx
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import os
os.environ['NLTK_DATA'] = '/home/jxy/nltk_data'
# 加载 NLP 组件
nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# 连接词表
LOGICAL_CONNECTIVES = ["because", "therefore", "thus", "as a result",
                       "first", "next", "finally", "then", "moreover", "however"]

def load_json_file(file_path, num_texts=3):
    """ 读取 JSON 文件，并提取摘要 """          
    abstracts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_texts:
                break
            data = json.loads(line)
            abstracts.append(data["abstract"])
    return abstracts

def parse_sentences(text, shuffle=True):
    """ 使用 spaCy 解析文本，随机打乱句子顺序 """
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    if shuffle:
        random.shuffle(sentences)  # 乱序

    sentence_data = []
    for sent in sentences:
        parsed_sent = nlp(sent)
        root = None
        dependencies = []
        
        for token in parsed_sent:
            if token.dep_ == "ROOT":
                root = token
            dependencies.append((token.text, token.dep_, token.head.text, token.head.dep_))

        # 计算 TF-IDF 关键词重要性
        keywords = extract_keywords(sent)

        sentence_data.append({
            "sentence": sent,
            "root": root.text if root else None,
            "dependencies": dependencies,
            "length": len(sent),
            "keywords": keywords
        })
    
    return sentence_data

def extract_keywords(sentence):
    """ 使用 TF-IDF 计算关键词 """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([sentence])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    keywords = feature_array[tfidf_sorting][:3]  # 取前3个关键词
    return set(keywords)

def compute_semantic_similarity(sentences):
    """ 计算句子间的语义相似度 """
    embeddings = sbert_model.encode(sentences, convert_to_numpy=True)
    similarity_matrix = np.inner(embeddings, embeddings)  # 计算余弦相似度
    return similarity_matrix

def build_dependency_graph(sentence_data, similarity_matrix):
    """ 结合语义相似度、关键词匹配、逻辑连接词优化依存关系 """
    G = nx.DiGraph()

    for i, data in enumerate(sentence_data):
        G.add_node(i, text=data["sentence"], root=data["root"], length=data["length"], keywords=data["keywords"])

    for i in range(len(sentence_data) - 1):
        for j in range(i + 1, len(sentence_data)):
            root_i, root_j = sentence_data[i]["root"], sentence_data[j]["root"]
            keywords_i, keywords_j = sentence_data[i]["keywords"], sentence_data[j]["keywords"]
            
            # 依存匹配：Root 词出现在其他句子
            if root_i and root_j and root_j in sentence_data[i]["sentence"]:
                weight = 0.5
            else:
                # 计算语义相似度
                weight = 3 - 2 * similarity_matrix[i, j]  
            
            # 关键词匹配
            if keywords_i & keywords_j:  
                weight -= 0.5  

            # 逻辑连接词检测
            if contains_logical_connectives(sentence_data[j]["sentence"]):
                weight -= 0.3  

            G.add_edge(i, j, weight=max(0.1, weight))  # 确保权重非负

    return G

def contains_logical_connectives(sentence):
    """ 检测句子中是否含有逻辑连接词 """
    words = nltk.word_tokenize(sentence.lower())
    return any(word in LOGICAL_CONNECTIVES for word in words)

def reorder_sentences(sentence_data, graph):
    """ 使用 PageRank 计算句子重要性并重排 """
    sentence_ranks = nx.pagerank(graph, alpha=0.85)
    
    # 重要性排序
    sorted_indices = sorted(sentence_ranks.keys(), key=lambda x: sentence_ranks[x], reverse=False)
    reordered_text = " ".join([sentence_data[i]["sentence"] for i in sorted_indices])

    return reordered_text

# 读取 JSON 数据
file_path = "/home/jxy/Data/init/ieee-init.jsonl"
num_texts_to_process = 3  
abstracts = load_json_file(file_path, num_texts_to_process)

for idx, abstract in enumerate(abstracts):
    print(f"\n=== 处理摘要 {idx + 1} ===\n原始摘要：")
    print(abstract)

    # 解析句子并打乱顺序
    sentence_data = parse_sentences(abstract, shuffle=True)
    
    print("\n随机打乱句子顺序后：")
    shuffled_text = " ".join([s["sentence"] for s in sentence_data])
    print(shuffled_text)

    # 计算语义相似度
    similarity_matrix = compute_semantic_similarity([s["sentence"] for s in sentence_data])

    # 构建优化依存关系图
    graph = build_dependency_graph(sentence_data, similarity_matrix)

    # 句子重排
    optimized_text = reorder_sentences(sentence_data, graph)

    print("\n优化后摘要：")
    print(optimized_text)


"""
In the future scenario of multiple wireless network coverage, the choice of vertical handoff decision algorithm will directly affect the continuity of the session, the mobility of the user, and seamless roaming under heterogeneous wireless networks. 
Therefore, the study of vertical handover related algorithms is the key to the success of various wireless access networks in the future. 
This paper proposes an optimized algorithm which combines two multiple attribute decision making (MADM) methods, the Entropy and the improved Technique for Order Preference by Similarity to an Ideal Solution (TOPSIS). 
The Entropy method is applied to obtain objective weights and the improved TOPSIS method is used to rank the alternatives. 
The simulation results show that the proposed technique can make the distribution of weights more reasonable, and effectively reduce the number of handoffs.


In the future scenario of multiple wireless network coverage, the choice of vertical handoff decision algorithm will directly affect the continuity of the session, the mobility of the user, and seamless roaming under heterogeneous wireless networks. 
The simulation results show that the proposed technique can make the distribution of weights more reasonable, and effectively reduce the number of handoffs. 
This paper proposes an optimized algorithm which combines two multiple attribute decision making (MADM) methods, the Entropy and the improved Technique for Order Preference by Similarity to an Ideal Solution (TOPSIS). 
The Entropy method is applied to obtain objective weights and the improved TOPSIS method is used to rank the alternatives. 
Therefore, the study of vertical handover related algorithms is the key to the success of various wireless access networks in the future.

"""