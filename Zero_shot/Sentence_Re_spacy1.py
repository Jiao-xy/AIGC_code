import json
import spacy
import os
from tqdm import tqdm

# 加载 SpaCy 依存分析模型
nlp = spacy.load("en_core_web_sm")

# 文件路径（改为目录路径）
input_dir = "/home/jxy/Data/ReoraganizationData/init"
output_dir = "/home/jxy/Data/ReoraganizationData/reordered"  # 新目录存放结果
os.makedirs(output_dir, exist_ok=True)

# 获取目录下所有jsonl文件
jsonl_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]

def split_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

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

def reorder_sentences(dependency_parsed):
    reordered = sorted(
        dependency_parsed,
        key=lambda x: (x[1] + x[2] + x[3], -x[4]), 
        reverse=True
    )
    return " ".join([sent[0] for sent in reordered])

# 处理每个文件
for file_name in tqdm(jsonl_files, desc="Processing files"):
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, f"reordered_{file_name}")
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm(infile, desc=f"Processing {file_name}"):
            try:
                data = json.loads(line.strip())
                abstract = data.get("abstract", "").strip()
                if not abstract:
                    continue
                
                # 处理重排序
                sentences = split_sentences(abstract)
                dependency_parsed = analyze_dependencies(sentences)
                reordered_text = reorder_sentences(dependency_parsed)
                
                # 保存结果（只保留id和重排文本）
                output_data = {
                    "id": data.get("id", ""),
                    "reordered_text": reordered_text
                }
                outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"Error processing line in {file_name}: {str(e)}")
                continue

print(f"所有文件处理完成！结果保存在: {output_dir}")