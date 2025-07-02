import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

# Sample dataset
df = pd.DataFrame({
    'Age': [22, 25, 47, 52, 46, 56, 23, 30, 28, 48],
    'Salary': [15000, 29000, 48000, 60000, 52000, 83000, 18000, 40000, 39000, 79000],
    'Buy':    [0, 0, 1, 1, 1, 1, 0, 0, 0, 1]
})

# Split data
X = df[['Age', 'Salary']]
y = df['Buy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train & predict
model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
print("Acc:", accuracy_score(y_test, y_pred))
print("Prec:", precision_score(y_test, y_pred))
print("Rec:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()
