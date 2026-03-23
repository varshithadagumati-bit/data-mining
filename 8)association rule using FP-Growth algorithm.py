# Install if needed:
# pip install mlxtend

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Sample contact lenses data (subset)
transactions = [
    ['Young','Myope','No','Reduced','No_lenses'],
    ['Young','Myope','No','Normal','Soft'],
    ['Young','Myope','Yes','Normal','Hard'],
    ['Pre-presbyopic','Hypermetrope','No','Reduced','No_lenses'],
    ['Presbyopic','Myope','Yes','Normal','Hard']
]

# Convert to one-hot encoding
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_data, columns=te.columns_)

print("Dataset:\n", df)

# Step 1: Frequent Itemsets using FP-Growth
frequent_items = fpgrowth(df, min_support=0.4, use_colnames=True)

print("\nFrequent Itemsets:\n", frequent_items)

# Step 2: Generate Association Rules
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.6)

print("\nAssociation Rules:\n")
print(rules[['antecedents','consequents','support','confidence','lift']])
