from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Dataset
data = [
    ['High','Good','Yes','Pass'],
    ['Low','Poor','No','Fail'],
    ['Medium','Good','Yes','Pass'],
    ['Low','Good','No','Fail'],
    ['High','Good','Yes','Pass']
]

columns = ['StudyHours','Attendance','Assignment','Result']
df = pd.DataFrame(data, columns=columns)

# Split features and target
X = df.drop('Result', axis=1)
y = df['Result']

# Encoding categorical data
le_dict = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Train model (J48-like)
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

print("Model trained using J48 (C4.5 simulation)\n")

# Display rules
rules = export_text(model, feature_names=list(X.columns))
print("Classification Rules:\n")
print(rules)
