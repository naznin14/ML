# association_rules_2
# association_rules_2.py

import sys
import subprocess

# Install mlxtend automatically if not installed
try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
except ModuleNotFoundError:
    print("mlxtend not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mlxtend"])

    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules

import pandas as pd

# Sample transaction data
data = [
    ['milk', 'bread', 'egg'],
    ['milk', 'bread'],
    ['milk', 'bread', 'egg', 'banana'],
    ['milk', 'banana']
]

# Convert transactions into a DataFrame
te = TransactionEncoder()
te_ary = te.fit_transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Generate frequent itemsets
frequent_itemsets = apriori(
    df,
    min_support=0.5,
    use_colnames=True
)

# Generate association rules
rules = association_rules(
    frequent_itemsets,
    metric='confidence',
    min_threshold=0.75
)

# Print the rules
for i in range(len(rules)):
    LHS = list(rules['antecedents'].iloc[i])
    RHS = list(rules['consequents'].iloc[i])
    support = rules['support'].iloc[i]
    confidence = rules['confidence'].iloc[i]

    print(f"LHS: {LHS} -- RHS: {RHS}")
    print(f"Support: {support}")
    print(f"Confidence: {confidence}")
    print("---")
