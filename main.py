import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


# 加载数据集
def load_data(file_path):
	data = pd.read_csv(file_path, delimiter=',')  # 根据数据格式调整
	return data['text']  # 替换为实际文本列名


# 文本向量化
def vectorize_text(texts):
	vectorizer = TfidfVectorizer(max_features=1000)  # 使用TF-IDF
	return vectorizer.fit_transform(texts)


# 聚类和评价
def perform_kmeans(data, k_values, true_labels=None):
	results = {}
	for k in k_values:
		kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++', max_iter=300)
		clusters = kmeans.fit_predict(data)

		# 聚类质量评估
		silhouette = silhouette_score(data, clusters)
		results[k] = {'clusters': clusters, 'silhouette': silhouette}

		# 如果有真实标签，可以计算AMI或ARI
		if true_labels is not None:
			results[k]['AMI'] = normalized_mutual_info_score(true_labels, clusters)
			results[k]['ARI'] = adjusted_rand_score(true_labels, clusters)

	return results


# 主程序
if __name__ == "__main__":
	file_paths = ["dataset1.csv", "dataset2.csv", ...]  # 五个文本集路径
	k_values = [2, 3, 5, 10]

	for file_path in file_paths:
		print(f"Processing {file_path}")
		texts = load_data(file_path)
		data = vectorize_text(texts)

		# 运行K均值聚类
		results = perform_kmeans(data, k_values)

		# 打印结果
		for k, metrics in results.items():
			print(f"K={k}, Silhouette={metrics['silhouette']:.4f}")
