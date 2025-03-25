import json
import spacy
import random

# **加载 SpaCy 依存分析模型**
nlp = spacy.load("en_core_web_sm")

# **文件路径**
file_path = "/home/jxy/Data/ReoraganizationData/init/ieee-init.jsonl"

# **控制提取的文本数量**
num_samples = 100  # 可调整

# **从 JSONL 文件中提取部分文本**
def extract_samples(file_path, num_samples):
    samples = []
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        selected_lines = random.sample(lines, min(num_samples, len(lines)))  # 随机选取 num_samples 个样本
        
        for line in selected_lines:
            data = json.loads(line.strip())
            abstract = data.get("abstract", "").strip()
            if abstract:
                samples.append(abstract)
    return samples

# **分句**
def split_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

# **依存分析**
def analyze_dependencies(sentences):
    parsed_sentences = []
    for sent in sentences:
        doc = nlp(sent)
        root_count = sum(1 for token in doc if token.dep_ == 'ROOT')
        subject_count = sum(1 for token in doc if token.dep_ in ['nsubj', 'nsubjpass'])
        object_count = sum(1 for token in doc if token.dep_ in ['dobj', 'pobj'])
        tree_depth = max(token.i for token in doc) - min(token.i for token in doc) if len(doc) > 1 else 1
        parsed_sentences.append((sent, root_count, subject_count, object_count, tree_depth))
    return parsed_sentences

# **逻辑重排**
def reorder_sentences(dependency_parsed):
    reordered = sorted(
        dependency_parsed,
        key=lambda x: (x[1] + x[2] + x[3], -x[4]), 
        reverse=True
    )
    return " ".join([sent[0] for sent in reordered])

# **执行处理**
samples = extract_samples(file_path, num_samples)
reordered_texts = []

for text in samples:
    sentences = split_sentences(text)
    dependency_parsed = analyze_dependencies(sentences)
    reordered_text = reorder_sentences(dependency_parsed)
    reordered_texts.append((text, reordered_text))

# **打印部分原始文本和重排后的文本**
for i, (original, reordered) in enumerate(reordered_texts):
    print(f"样本 {i+1} 原始文本:")
    print(original)
    print("\n【重构后文本】")
    print(reordered)
    print("=" * 80)
