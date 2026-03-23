from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Dataset
data = [
    ['Sunny','Hot','High','Weak','No'],
    ['Sunny','Hot','High','Strong','No'],
    ['Overcast','Hot','High','Weak','Yes'],
    ['Rain','Mild','High','Weak','Yes'],
    ['Rain','Cool','Normal','Weak','Yes']
]

columns = ['Outlook','Temperature','Humidity','Wind','PlayTennis']
df = pd.DataFrame(data, columns=columns)

# Split features and target
X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

# Encoding
le_dict = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Train ID3 model
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

print("Model trained using ID3\n")

# Display rules
tree_rules = export_text(model, feature_names=list(X.columns))
print("Classification Rules:\n")
print(tree_rules)
