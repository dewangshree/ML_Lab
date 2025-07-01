from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = load_iris().data
X_pca = PCA(n_components=2).fit_transform(X)  # Reduce to 2D for plotting

k_values = [2, 3, 4]

plt.figure(figsize=(12, 4))

for i, k in enumerate(k_values):
    km = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = km.labels_
    plt.subplot(1, 3, i+1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='red', marker='X', s=100)
    plt.title(f'K={k}, Inertia={km.inertia_:.2f}')

plt.tight_layout()
plt.show()
