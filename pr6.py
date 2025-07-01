import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Data
df = pd.DataFrame({
    'Age': ['Youth', 'Youth', 'Middle', 'Senior', 'Senior', 'Senior', 'Middle', 'Youth', 'Youth', 'Senior', 'Youth', 'Middle', 'Middle', 'Senior'],
    'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'Medium'],
    'Buys': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

# Encoding
le_age = LabelEncoder()
le_income = LabelEncoder()
le_buys = LabelEncoder()
df['Age'] = le_age.fit_transform(df['Age'])
df['Income'] = le_income.fit_transform(df['Income'])
df['Buys'] = le_buys.fit_transform(df['Buys'])

# Split
X = df[['Age', 'Income']]
y = df['Buys']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Predict new data
new = pd.DataFrame({'Age': ['Youth', 'Senior'], 'Income': ['High', 'Low']})
new['Age'] = le_age.transform(new['Age'])
new['Income'] = le_income.transform(new['Income'])
pred = model.predict(new)

# Simple output
result = le_buys.inverse_transform(pred)
for i in range(len(result)):
    print("Prediction", i+1, ":", result[i])
