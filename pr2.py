from sklearn.datasets import  load_iris
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt

X,y=load_iris(return_X_y=True)

model=DecisionTreeClassifier(criterion='entropy')
model.fit(X,y)

sample = [[5.1, 3.5, 1.4, 0.2]]  # This looks like a setosa flower
prediction = model.predict(sample)
print("Predicted class:", prediction[0])


plt.figure(figsize=(8,11))
plot_tree(model,filled=True,feature_names=load_iris().feature_names,class_names=load_iris().target_names)
plt.title("Decision Tree Trained on Iris Dataset")
plt.show()
