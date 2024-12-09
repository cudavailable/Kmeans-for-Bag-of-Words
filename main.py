import os
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from logger import Logger

# 加载数据
def load_data(docword_file, vocab_file):
	with open(docword_file, 'r') as f:
		lines = f.readlines()
	D, W, NNZ = map(int, lines[:3])  # 读取前三行信息
	data = np.array([list(map(int, line.strip().split())) for line in lines[3:]])

	# 创建稀疏矩阵
	doc_ids = data[:, 0] - 1  # 让下标从0开始
	word_ids = data[:, 1] - 1
	counts = data[:, 2]
	sparse_matrix = sp.coo_matrix((counts, (doc_ids, word_ids)), shape=(D, W))

	# 加载词汇表
	with open(vocab_file, 'r') as f:
		vocab = [line.strip() for line in f.readlines()]

	return sparse_matrix, vocab


# 计算TF-IDF表示
def compute_tfidf(sparse_matrix):
	transformer = TfidfTransformer()
	return transformer.fit_transform(sparse_matrix)


# K均值聚类
def kmeans_clustering(data, k_values, logger):
	results = {}
	for k in k_values:
		kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++', max_iter=300)
		clusters = kmeans.fit_predict(data)
		score = silhouette_score(data, clusters)
		results[k] = {'clusters': clusters, 'silhouette': score}
		logger.write(f"K={k}, Silhouette Score={score:.4f}\n")
	return results


# 结果可视化
# def visualize_results(results):
# 	k_values = list(results.keys())
# 	scores = [results[k]['silhouette'] for k in k_values]
#
# 	plt.figure(figsize=(10, 6))
# 	plt.plot(k_values, scores, marker='o', linestyle='-')
# 	plt.title("Silhouette Score vs K", fontsize=14)
# 	plt.xlabel("Number of Clusters (K)", fontsize=12)
# 	plt.ylabel("Silhouette Score", fontsize=12)
# 	plt.grid(True)
# 	plt.show()


def main():
	# 明确数据集的路径
	base_dir = r"../bag+of+words"
	log_dir = r"./log"
	if log_dir is not None and not os.path.exists(log_dir):
		os.mkdir(log_dir)

	datasets = [
		# {"name": "Enron Emails", "docword": "docword.enron.txt", "vocab": "vocab.enron.txt"},
		{"name": "NIPS Papers", "docword": "docword.nips.txt", "vocab": "vocab.nips.txt"},
		{"name": "KOS Blogs", "docword": "docword.kos.txt", "vocab": "vocab.kos.txt"},
		# {"name": "NYTimes Articles", "docword": "docword.nytimes.txt", "vocab": "vocab.nytimes.txt"},
		# {"name": "PubMed Abstracts", "docword": "docword.pubmed.txt", "vocab": "vocab.pubmed.txt"}
	]

	k_values = [2, 5, 10, 20]

	for dataset in datasets:
		logger = Logger(os.path.join(log_dir, f"{dataset['name']}.txt"))
		logger.write(f"\nProcessing dataset: {dataset['name']}\n")

		# 加载并处理数据
		docword_file = os.path.join(base_dir, dataset["docword"])
		vocab_file = os.path.join(base_dir, dataset["vocab"])
		sparse_matrix, vocab = load_data(docword_file, vocab_file)

		logger.write(f"Dataset loaded: {dataset['name']}\n")
		logger.write(f"Number of Documents: {sparse_matrix.shape[0]}\n")
		logger.write(f"Vocabulary Size: {len(vocab)}\n")

		# 计算TF-IDF
		tfidf_matrix = compute_tfidf(sparse_matrix)
		logger.write("TF-IDF computation completed.\n")

		# 执行K均值聚类
		results = kmeans_clustering(tfidf_matrix, k_values, logger)

		# 可视化结果
		# visualize_results(results)

		# 打印出最好的聚类结果
		best_k = max(results, key=lambda k: results[k]['silhouette'])
		logger.write(f"Best K for {dataset['name']}: {best_k}, Silhouette Score: {results[best_k]['silhouette']:.4f}\n")


if __name__ == "__main__":
	main()
