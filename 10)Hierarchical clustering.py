# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 2: Given dataset
X = np.array([
    [1, 2],
    [2, 3],
    [5, 6],
    [8, 9]
])

# Step 3: Apply hierarchical clustering (Ward method)
Z = linkage(X, method='ward')

# Step 4: Plot dendrogram
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
dendrogram(Z)
plt.show()
