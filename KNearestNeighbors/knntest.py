import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from knn import KNN
from knnsupervised import KNNReg
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


#  ----------------------------------------------------------------------------------
#Try the basic KNN Algorithm


iris = datasets.load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = KNN(k=3)

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

print(predictions)

accuracy = accuracy_score(y_test, predictions)

print(f"{accuracy * 100}%\n")


#  --------------------------------------------------------------------------------

#Try the regresssion version of the KNN Algorithm

#To test we will use the makeshift dataset created by Aurelion Geron featured in his book:
# Hands on Machine Learning with Scikit-Learn, Keras, & Tensorflow (Editions 1-3)


data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
new_X = lifesat[["GDP per capita (USD)"]].values
new_y = lifesat[["Life satisfaction"]].values

print()


X_train2, X_test2, y_train2, y_test2 = train_test_split(new_X, new_y, test_size= 0.2, random_state=42)

options = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
for option in options:

    new_model = KNNReg(k=option)
    new_model.fit(X_train2, y_train2)
    new_pred = new_model.predict(X_test2)

    mse = mean_squared_error(y_test2, new_pred)

    print(f"MSE when k = {option}: {mse}") #The optimal number of instances the KNN algorithm should learn from to minimize the MSE is 12

#The MSE stops fluctuating at k = 21

