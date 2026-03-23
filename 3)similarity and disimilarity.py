import numpy as np
from scipy.spatial import distance

data = np.array([10, 20, 30, 40, 50])

# Statistical Measures
print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Standard Deviation:", np.std(data))
print("Variance:", np.var(data))

# Similarity / Dissimilarity
x = np.array([1, 2, 3])
y = np.array([2, 4, 6])

# Euclidean Distance
print("Euclidean Distance:", distance.euclidean(x, y))

# Manhattan Distance
print("Manhattan Distance:", distance.cityblock(x, y))

# Cosine Similarity
print("Cosine Similarity:", 1 - distance.cosine(x, y))
