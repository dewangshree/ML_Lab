import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df.to_csv('purchase_prediction_data.csv', index=False)

print("CSV file 'purchase_prediction_data.csv' has been created.")
X = df[['Hours_Studied']]
y = df['Exam_Score']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model=LinearRegression().fit(X_train,y_train)
y_pred=model.predict(X_test)

print("RMSE:",mean_squared_error(y_test,y_pred))
print("R2",r2_score(y_test,y_pred))

plt.scatter(X_train, y_train, c='blue', label='Training Data')
plt.scatter(X_test, y_test, c='green', label='Test Data')
plt.plot(X, model.predict(X), c='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid()
plt.show()
