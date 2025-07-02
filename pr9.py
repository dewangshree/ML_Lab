from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load data and reduce to 2D for plotting
X = load_iris().data
X_pca = PCA(n_components=2).fit_transform(X)

# AGNES (ward linkage, bottom-up)
agnes_linkage = linkage(X, method='ward')
agnes_labels = fcluster(agnes_linkage, 3, criterion='maxclust')

# DIANA (complete linkage, top-down)
diana_linkage = linkage(X, method='complete')
diana_labels = fcluster(diana_linkage, 3, criterion='maxclust')

# AGNES dendrogram
plt.figure(figsize=(6, 4))
dendrogram(agnes_linkage)
plt.title("AGNES - Dendrogram (Ward Linkage)")
plt.tight_layout()
plt.show()

# DIANA dendrogram
plt.figure(figsize=(6, 4))
dendrogram(diana_linkage)
plt.title("DIANA - Dendrogram (Complete Linkage)")
plt.tight_layout()
plt.show()

# AGNES cluster plot
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agnes_labels, cmap='viridis')
plt.title("AGNES Clustering")
plt.show()

# DIANA cluster plot
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=diana_labels, cmap='plasma')
plt.title("DIANA Clustering")
plt.show()

# Compare using silhouette score
agnes_score = silhouette_score(X, agnes_labels)
diana_score = silhouette_score(X, diana_labels)

print("Agnes:", round(agnes_score, 4))
print("Diana:", round(diana_score, 4))

# Print better method
if agnes_score > diana_score:
    print("AGNES gives better clustering based on higher silhouette score.")
   
else:
    print("DIANA gives better clustering based on higher silhouette score.")
   
