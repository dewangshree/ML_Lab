import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

trees = [1, 5, 10, 50, 100, 200]
accs, precs, recs, f1s = [], [], [], []

for t in trees:
    m = RandomForestClassifier(n_estimators=t, random_state=42).fit(X_train, y_train)
    y_pred = m.predict(X_test)
    accs.append(accuracy_score(y_test, y_pred))
    precs.append(precision_score(y_test, y_pred))
    recs.append(recall_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))
    print(t, "A=", accs[-1], "P=", precs[-1], "R=", recs[-1], "F1=", f1s[-1])


# Plot metrics
plt.plot(trees, accs, 'o-', label='Accuracy')
plt.plot(trees, precs, 'o-', label='Precision')
plt.plot(trees, recs, 'o-', label='Recall')
plt.plot(trees, f1s, 'o-', label='F1 Score')
plt.xlabel("Trees"); plt.ylabel("Score"); plt.title("Random Forest Performance")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


