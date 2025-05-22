from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt  # <-- Added for plotting

# Step 1: Load sample dataset (you can replace this with your own dataset)
data = load_iris()
X = data.data
y = data.target

# Step 2: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train and evaluate the Random Forest classifier with different tree counts
tree_counts = [190, 500, 89, 900, 80, 100]
accuracies = []

for trees in tree_counts:
    model = RandomForestClassifier(n_estimators=trees, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    accuracies.append(acc)  # <-- Collect accuracy for plotting
    print(f"{trees} trees: Accuracy = {acc:.4f}")

# Step 4: Plotting accuracy vs number of trees
plt.figure(figsize=(8, 5))
plt.plot(tree_counts, accuracies, marker='o', linestyle='--', color='blue')
plt.title('Random Forest Accuracy vs Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout()
plt.show()
