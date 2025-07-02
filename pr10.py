from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import *

# Load dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train AdaBoost
ada = AdaBoostClassifier().fit(X_train, y_train)
ada_pred = ada.predict(X_test)

# Train XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

# Evaluate models
def evaluate(name, y_true, y_pred):
    print(f"\n{name} Results:")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred), 4))
    print("Recall:", round(recall_score(y_true, y_pred), 4))
    print("F1 Score:", round(f1_score(y_true, y_pred), 4))
    print("CM:", confusion_matrix(y_true, y_pred).ravel())

evaluate("AdaBoost", y_test, ada_pred)
evaluate("XGBoost", y_test, xgb_pred)

# Final Verdict
ada_f1 = f1_score(y_test, ada_pred)
xgb_f1 = f1_score(y_test, xgb_pred)

if ada_f1 > xgb_f1:
    print("\n Final Verdict: AdaBoost performed better overall")
else:
    print("\n Final Verdict: XGBoost performed better overall")

