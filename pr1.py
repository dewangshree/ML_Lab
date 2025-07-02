import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

X, y = load_breast_cancer(return_X_y=True)
X = X[:, :2]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

m = LogisticRegression().fit(X_train, y_train)
y_pred = m.predict(X_test)
y_prob = m.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

# Print confusion matrix values
print("Confusion Matrix:\n", cm)
print("TP=", TP, "TN=", TN, "FP=", FP, "FN=", FN)

def metrics(TP, TN, FP, FN):
  acc = (TP + TN) / (TP + TN + FP + FN)
  prec = TP / (TP + FP)
  rec = TP / (TP + FN)
  spec = TN / (TN + FP)
  npv = TN / (TN + FN)
  f1 = 2 * prec * rec / (prec + rec)
  mcc = matthews_corrcoef(y_test, y_pred)
  return acc, prec, rec, spec, npv, f1, mcc

acc, prec, rec, spec, npv, f1, mcc = metrics(TP, TN, FP, FN)

print("Acc=", acc, "Prec=", prec, "Rec=", rec)
print("Spec=", spec, "NPV=", npv, "F1=", f1, "MCC=", mcc)

auc_model = roc_auc_score(y_test, y_prob)
print("Model AUC :", auc_model)

plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve")
plt.show()
