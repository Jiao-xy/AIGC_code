#问题：并没有进行重排操作


import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.model_selection import train_test_split



# **文件路径**
file_pairs = [
    ("/home/jxy/Data/ieee-init.jsonl", "/home/jxy/Data/ieee-chatgpt-generation.jsonl"),
    ("/home/jxy/Data/ieee-init_random.jsonl", "/home/jxy/Data/ieee-chatgpt-generation_random.jsonl")
]

# **计算相似度**
def compute_similarity(original, reordered):
    vec_orig = np.array([token.vector for token in nlp(original) if token.has_vector])
    vec_reorder = np.array([token.vector for token in nlp(reordered) if token.has_vector])
    if vec_orig.size == 0 or vec_reorder.size == 0:
        return 0  # 如果没有词向量信息，返回 0
    return cosine_similarity(vec_orig.mean(axis=0).reshape(1, -1), vec_reorder.mean(axis=0).reshape(1, -1))[0, 0]

# **数据存储**
all_results = []
labels = []  # 1 代表人类文本，0 代表 GPT 生成文本

# **遍历 init 和 generation 文件**
print("开始处理数据...")
total_files = len(file_pairs) * 2
processed_files = 0

for (human_file, gpt_file), (human_random_file, gpt_random_file) in zip(file_pairs, file_pairs[1:]):
    for (orig_file, random_file, label) in [(human_file, human_random_file, 1), (gpt_file, gpt_random_file, 0)]:
        print(f"正在处理文件: {orig_file} 和 {random_file}")
        with open(orig_file, "r", encoding="utf-8") as file1, open(random_file, "r", encoding="utf-8") as file2:
            for idx, (line1, line2) in enumerate(zip(file1, file2)):
                try:
                    data1 = json.loads(line1.strip())
                    data2 = json.loads(line2.strip())
                    abstract = data1.get("abstract", "").strip()
                    reordered_abstract = data2.get("abstract", "").strip()
                    llscore = data1.get("LLScore", 0)
                    ppl = data1.get("PPL", 0)
                    
                    if abstract and reordered_abstract:
                        similarity = compute_similarity(abstract, reordered_abstract)
                        totalscore = 0.5 * similarity + 0.5 * llscore  # 组合评分
                        all_results.append((totalscore, llscore, ppl))
                        labels.append(label)
                    
                    if idx % 100 == 0:
                        print(f"已处理 {idx} 行文本...")
                except json.JSONDecodeError:
                    print(f"JSON 解码错误，跳过文件 {orig_file} 或 {random_file} 的某一行。")
        processed_files += 1
        print(f"进度: {processed_files}/{total_files} 文件处理完成")

print("数据处理完成，开始分类分析...")

# **转换为 NumPy 数组**
all_results = np.array(all_results)
labels = np.array(labels)

# **划分训练集和测试集**
X_train, X_test, y_train, y_test = train_test_split(all_results, labels, test_size=0.2, random_state=42)
print("数据划分完成，训练集大小: {}，测试集大小: {}".format(len(X_train), len(X_test)))

# **分类分析：设定阈值**
totalscore_threshold = np.percentile(X_train[:, 0], 50)
print(f"计算出的分类阈值: {totalscore_threshold:.4f}")

# **基于阈值进行分类**
y_pred_train = (X_train[:, 0] < totalscore_threshold).astype(int)
y_pred_test = (X_test[:, 0] < totalscore_threshold).astype(int)

# **计算分类性能指标（训练集）**
train_accuracy = accuracy_score(y_train, y_pred_train)
train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(y_train, y_pred_train, average='binary')
print(f"训练集分类准确率: {train_accuracy:.4f}")
print(f"训练集精确率: {train_precision:.4f}, 召回率: {train_recall:.4f}, F1-score: {train_f1:.4f}")

# **计算分类性能指标（测试集）**
test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='binary')
print(f"测试集分类准确率: {test_accuracy:.4f}")
print(f"测试集精确率: {test_precision:.4f}, 召回率: {test_recall:.4f}, F1-score: {test_f1:.4f}")

# **绘制并保存 ROC 曲线**
def plot_and_save_roc(y_true, scores, filename, title):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(filename)
    print(f"ROC 曲线已保存至 {filename}")
    plt.show()

plot_and_save_roc(y_train, X_train[:, 0], "roc_curve_train.png", "ROC Curve (Train)")
plot_and_save_roc(y_test, X_test[:, 0], "roc_curve_test.png", "ROC Curve (Test)")

print("所有任务已完成！")
