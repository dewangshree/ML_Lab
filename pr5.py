import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    mean_squared_error, confusion_matrix
)

# Load dataset
X, y = load_breast_cancer(return_X_y=True)
X = X[:, [0]]  # Only use "mean radius"

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plotting
x_vals = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
probs = model.predict_proba(x_vals)[:, 1]

plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, c=y_test, cmap='bwr', edgecolor='k', label='Test Data')
plt.plot(x_vals, probs, color='black', label='Logistic Curve')
plt.axhline(0.5, color='gray', linestyle='--', label='Threshold = 0.5')
plt.xlabel("Mean Radius")
plt.ylabel("Probability of Class 1 (Benign)")
plt.title("Logistic Regression on Breast Cancer (1 feature)")
plt.legend()
plt.show()
