import numpy as np

class LinearReg:
    def __init__(self, lr=0.001, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.iterations):
            yhat = np.dot(X, self.weights) + self.bias

            dw = (1/m) * np.dot(X.T, (yhat - y))
            db = (1/m) * np.sum(yhat - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        yhat = np.dot(X, self.weights) + self.bias
        return yhat