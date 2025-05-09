from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score
import pandas as pd

wine = load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)
target = wine.target
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)
labels = kmeans.labels_

homogeneity = homogeneity_score(target, labels)
completeness = completeness_score(target, labels)
print(f"Homogeneity score: {homogeneity}")
print(f"Completeness score: {completeness}")
