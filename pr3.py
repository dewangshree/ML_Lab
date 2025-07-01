from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Define different numbers of trees
tree_counts = [190, 500, 89, 900, 80, 100]

# Lists to store metrics
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Loop through different tree counts
for i, trees in enumerate(tree_counts):
    # Change the random_state slightly for different splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40 + i)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=trees, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Store metrics
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

    print(f"{trees} trees: Accuracy = {acc:.4f}, Precision = {prec:.4f}, Recall = {rec:.4f}, F1 Score = {f1:.4f}")

# Plot all metrics
plt.figure(figsize=(10, 6))
plt.plot(tree_counts, accuracies, marker='o', label='Accuracy')
plt.plot(tree_counts, precisions, marker='s', label='Precision')
plt.plot(tree_counts, recalls, marker='^', label='Recall')
plt.plot(tree_counts, f1_scores, marker='D', label='F1 Score')
plt.title('Random Forest Performance vs Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
