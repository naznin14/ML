# association_rules_1
import pandas as pd
from itertools import combinations

# Dataset
dataset = [
    ['milk', 'bread', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter', 'cheese'],
    ['milk', 'bread'],
    ['butter', 'cheese'],
    ['bread', 'butter', 'cheese']
]

MIN_SUPPORT = 0.5
MIN_CONFIDENCE = 0.75

# ---------------------------------------------------------
# Calculate support
# ---------------------------------------------------------
def support(itemset):
    count = 0

    for transaction in dataset:
        if set(itemset).issubset(set(transaction)):
            count += 1

    return count / len(dataset)


# ---------------------------------------------------------
# Find frequent itemsets using Apriori
# ---------------------------------------------------------
def apriori():
    items = sorted(set(item for transaction in dataset for item in transaction))

    frequent_itemsets = []

    # Generate itemsets of different sizes
    for size in range(1, len(items) + 1):
        for combination in combinations(items, size):
            sup = support(combination)

            if sup >= MIN_SUPPORT:
                frequent_itemsets.append({
                    "itemsets": combination,
                    "support": sup
                })

    return frequent_itemsets


# ---------------------------------------------------------
# Generate association rules
# ---------------------------------------------------------
def generate_rules(frequent_itemsets):
    rules = []

    support_dict = {
        frozenset(item["itemsets"]): item["support"]
        for item in frequent_itemsets
    }

    for item in frequent_itemsets:
        itemset = frozenset(item["itemsets"])

        if len(itemset) < 2:
            continue

        for size in range(1, len(itemset)):
            for antecedent_tuple in combinations(itemset, size):

                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent

                confidence = (
                    support_dict[itemset] /
                    support_dict[antecedent]
                )

                if confidence >= MIN_CONFIDENCE:
                    rules.append({
                        "antecedents": set(antecedent),
                        "consequents": set(consequent),
                        "antecedent support": support_dict[antecedent],
                        "consequent support": support_dict[consequent],
                        "support": support_dict[itemset],
                        "confidence": confidence
                    })

    return rules


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------
frequent_itemsets = apriori()
rules = generate_rules(frequent_itemsets)

print("Frequent Itemsets:")
print("-" * 50)

for item in frequent_itemsets:
    print(
        f"{set(item['itemsets'])} "
        f"-> Support: {item['support']:.2f}"
    )

print("\nAssociation Rules:")
print("-" * 50)

for rule in rules:
    print(
        f"{rule['antecedents']} -> {rule['consequents']} "
        f"| Support: {rule['support']:.2f} "
        f"| Confidence: {rule['confidence']:.2f}"
    )
