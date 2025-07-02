import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data)
y = pd.Series(data.target)
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create ID3 (uses entropy)
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy on test set:", accuracy_score(y_test, y_pred))

# Predict on new sample
new_sample = [[5.0, 3.4, 1.6, 0.4]]  # Example input
predicted_class = model.predict(new_sample)[0]


# Visualize the tree
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
