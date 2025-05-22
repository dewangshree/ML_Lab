import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, roc_curve
)

# Sample data
data = {
    'Age': [22, 25, 47, 52, 46, 56, 23, 30, 28, 48],
    'EstimatedSalary': [15000, 29000, 48000, 60000, 52000, 83000, 18000, 40000, 39000, 79000],
    'Purchased': [0, 0, 1, 1, 1, 1, 0, 0, 0, 1]
}
df = pd.DataFrame(data)

# Features & target
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train model
model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
cm = confusion_matrix(y_test, y_pred)
TP, FN, FP, TN = cm[1, 1], cm[1, 0], cm[0, 1], cm[0, 0]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
specificity = TN / (TN + FP)
npv = TN / (TN + FN)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# Print metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)
print("NPV:", npv)
print("F1 Score:", f1)
print("MCC:", mcc)
print("AUC:", auc)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR'), plt.ylabel('TPR'), plt.title('ROC Curve')
plt.legend(), plt.grid(), plt.tight_layout()
plt.show()
