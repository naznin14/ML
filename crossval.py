from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()

# Increase max_iter
logreg = LogisticRegression(max_iter=1000)

scores = cross_val_score(logreg, iris.data, iris.target, cv=3)
print("Three Cross-validation scores:", scores)
print("Average cross-validation score: {:.2f}".format(scores.mean()))

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print("Five Cross-validation scores:", scores)
print("Average cross-validation score: {:.2f}".format(scores.mean()))
