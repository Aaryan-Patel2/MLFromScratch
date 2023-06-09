import numpy as np


class NaiveBayes:

    def fit(self, X, y):

        m, n = X.shape

        self.classes = np.unique(y)
        n_classes = len(self.classes)

        #Initialize mean, variance, and priors

        self.mean = np.zeros((m,n), dtype=np.float64)
        self.var = np.zeros((m,n), dtype=np.float64)
        self.priors = np.zeros((m,n), dtype=np.float64)


        for c in self.classes:

            X_c = X[c==y]
            self.mean[c,:] = X_c.mean(axis=0)
            self.var[c,:] = X_c.var(axis=0)    
            self.priors[c] = X_c.shape[0] / float(m)

    

    def predict(self, X):

        yhat = [self._predict(x) for x in X]
        return yhat

    
    def _predict(self, x):

        pos_probs = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            pdf = np.sum(np.log(self.Gaussian(idx, x)))

            pos = prior + pdf

            pos_probs.append(pos)
        
        return self._classes[np.argmax(pos_probs)]


    

    def Gaussian(self, class_idx, x ):

        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-(x-mean)**2 / (2*var))
        denominator = np.sqrt(2*np.pi * var)

        return numerator/denominator





