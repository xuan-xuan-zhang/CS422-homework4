import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('D:\Downloads\Boston.csv')
except FileNotFoundError:
    print("Not find")
    exit(1)

X = df.drop('medv', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
silhouette_scores = []
k_values = range(2, 7)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For k={k}, the silhouette score is {silhouette_avg:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different k Values')
plt.show()

optimal_k = k_values[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters: {optimal_k}")
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
optimal_labels = kmeans_optimal.fit_predict(X_scaled)

cluster_means = pd.DataFrame(scaler.inverse_transform(kmeans_optimal.cluster_centers_),
                           columns=X.columns)

centroids_scaled = kmeans_optimal.cluster_centers_

print("\nCluster means (original scale):")
print(cluster_means)

print("\nCentroid coordinates (scaled):")
print(pd.DataFrame(centroids_scaled, columns=X.columns))