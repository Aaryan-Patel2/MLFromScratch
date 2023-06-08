import pandas as pd
import numpy as np
from linear import LinearReg
from logistic import LogisticReg
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score



#Linear Regression

X, y = datasets.make_regression(n_samples = 500, n_features=1, noise=25, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = LinearReg(lr= 0.01)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(mse)

#  ----------------------------------------------------------

#Logistic Regression

bc = datasets.load_breast_cancer()
new_X, new_y = bc.data, bc.target

X_train2, X_test2, y_train2, y_test2 = train_test_split(new_X, new_y, test_size= 0.2, random_state=42)


model2 = LogisticReg(lr = 0.001, iterations= 3000)
model2.fit(X_train2, y_train2)
predictions2 = model2.predict(X_test2)
accuracy = accuracy_score(y_test2, predictions2 )

print(f"Classification Accuracy: {accuracy}")

