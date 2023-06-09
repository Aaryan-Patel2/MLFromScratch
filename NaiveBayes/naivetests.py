import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from naivebayes import NaiveBayes


X,y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

X_train, y_train, X_test, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


model = NaiveBayes()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Naives Bayes Accuracy: {accuracy}")