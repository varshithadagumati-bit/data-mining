# Step 1: Import libraries
import numpy as np
from sklearn.cluster import KMeans

# Step 2: Given dataset
X = np.array([
    [2, 4],
    [3, 5],
    [8, 7],
    [9, 8],
    [1, 2],
    [10, 9]
])

# Step 3: Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
kmeans.fit(X)

# Step 4: Print results
print("Cluster Centers:")
print(kmeans.cluster_centers_)

print("\nCluster Labels:")
print(kmeans.labels_)

# Step 5: Show which point belongs to which cluster
print("\nData Points with Cluster:")
for i in range(len(X)):
    print(X[i], "-> Cluster", kmeans.labels_[i])
