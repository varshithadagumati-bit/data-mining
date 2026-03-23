# Install if needed:
# pip install mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Credit card transaction data
transactions = [
    ['Fuel', 'Grocery', 'Travel'],
    ['Grocery', 'Travel'],
    ['Fuel', 'Grocery'],
    ['Fuel', 'Travel'],
    ['Grocery', 'Travel']
]

# Convert to one-hot encoding
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_data, columns=te.columns_)

print("Dataset:\n", df)

# Step 1: Frequent Itemsets
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

print("\nFrequent Itemsets:\n", frequent_itemsets)

# Step 2: Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

print("\nAssociation Rules:\n")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
