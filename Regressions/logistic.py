import numpy as np


class LogisticReg:


    def sigmoid(self, z):
        return 1/ (1+np.exp(-z))


    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        #Gradient Descent

        for _ in range(self.iterations):
            z = np.dot(X, self.weights) + self.bias
            yhat = self.sigmoid(z)

            dw = (1/m) * np.dot(X.T, (yhat - y))
            db = (1/m) * np.sum(yhat - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        yhat = self.sigmoid(z)
        yhat_cls = [1 if i >= 0.5 else 0 for i in yhat]
        return yhat_cls
