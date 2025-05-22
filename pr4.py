import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
X = np.arange(1, 11).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model=LinearRegression().fit(X_train,y_train)

y_pred=model.predict(X_test)

print("RMSE:",mean_squared_error(y_test,y_pred, squared=False))
print("R2:",r2_score(y_test,y_pred))
plt.scatter(X,y,label='data points')
plt.plot(X,model.predict(X),color='red',label="Regression line")
plt.legend()
plt.show()
