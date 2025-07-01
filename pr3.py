import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Different numbers of trees to try
n_trees_list = [1, 5, 10, 50, 100, 200]

# Lists to store results
accuracies = []
precisions = []
recalls = []
f1_scores = []

for n_trees in n_trees_list:
    # Train Random Forest with n_trees
    model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)
    
    print(f"\nResults for {n_trees} trees:")
    print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1 Score: {f1:.3f}")

# Plot the performance metrics
plt.figure(figsize=(10,6))
plt.plot(n_trees_list, accuracies, label='Accuracy', marker='o')
plt.plot(n_trees_list, precisions, label='Precision', marker='o')
plt.plot(n_trees_list, recalls, label='Recall', marker='o')
plt.plot(n_trees_list, f1_scores, label='F1 Score', marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Score')
plt.title('Random Forest Performance vs Number of Trees')
plt.legend()
plt.grid(True)
plt.show()

# Plot feature importance for the model with highest F1 score
best_n = n_trees_list[f1_scores.index(max(f1_scores))]
model = RandomForestClassifier(n_estimators=best_n).fit(X_train, y_train)
imp = model.feature_importances_
idx = np.argsort(imp)[::-1]
