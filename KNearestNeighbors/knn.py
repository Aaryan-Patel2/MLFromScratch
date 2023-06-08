import numpy as np
from collections import Counter




def euclidean(x2, x1):

   return np.sqrt(np.sum((x2 - x1)**2))

class KNN:

    def __init__(self, k=3) -> None:
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # Compute the L2 distances

        distances = [euclidean(x, x_train) for x_train in self.X_train]

        #get k number of nearest samples/labels

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
            


        #get a majority vote (most common label)

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]