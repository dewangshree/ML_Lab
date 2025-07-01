from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load data
X = load_iris().data
X_pca = PCA(n_components=2).fit_transform(X)

# --- AGNES (Bottom-up) ---
agnes_linkage = linkage(X, method='ward')  # linkage matrix for dendrogram
agnes_labels = fcluster(agnes_linkage, 3, criterion='maxclust')  # get 3 clusters

# --- DIANA (Top-down) ---
diana_linkage = linkage(X, method='complete')  # simulate DIANA
diana_labels = fcluster(diana_linkage, 3, criterion='maxclust')

# --- Plot AGNES Dendrogram ---
plt.figure(figsize=(6, 4))
dendrogram(agnes_linkage)
plt.title("AGNES - Dendrogram (Ward Linkage)")
plt.tight_layout()
plt.show()

# --- Plot DIANA Dendrogram ---
plt.figure(figsize=(6, 4))
dendrogram(diana_linkage)
plt.title("DIANA - Dendrogram (Complete Linkage)")
plt.tight_layout()
plt.show()

# --- AGNES Cluster Output ---
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agnes_labels, cmap='viridis')
plt.title("AGNES Clustering")
plt.show()

# --- DIANA Cluster Output ---
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=diana_labels, cmap='plasma')
plt.title("DIANA Clustering")
plt.show()
