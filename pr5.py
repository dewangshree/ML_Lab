import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, confusion_matrix

# Load data (1 feature only)
X, y = load_breast_cancer(return_X_y=True)
X = X[:, [0]]  # mean radius
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train & predict
model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
print("Acc:", accuracy_score(y_test, y_pred))
print("Prec:", precision_score(y_test, y_pred))
print("Rec:", recall_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("CM:\n", confusion_matrix(y_test, y_pred))

# Plot
x = np.linspace(X.min(), X.max(), 300).reshape(-1,1)
p = model.predict_proba(x)[:,1]
plt.scatter(X_test, y_test, c=y_test)
plt.plot(x, p, color='k')
plt.axhline(0.5, ls='-', c='gray')
plt.show()
