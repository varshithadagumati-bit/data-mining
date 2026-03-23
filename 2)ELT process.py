import pandas as pd

# Create a dummy input.csv file for demonstration
dummy_data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35], 'Salary': [50000, 60000, 70000]}
df_dummy = pd.DataFrame(dummy_data)
df_dummy.to_csv("input.csv", index=False)

# Extract
data = pd.read_csv("input.csv")

# Transform
data = data.dropna()
data['Salary'] = data['Salary'] * 1.1

# Load
data.to_csv("output.csv", index=False)
print("ETL Completed")
