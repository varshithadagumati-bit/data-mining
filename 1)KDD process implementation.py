# KDD using sklearn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = datasets.load_iris()
X = data.data
y = data.target

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))

# Clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)
print("Clusters:", kmeans.labels_)
