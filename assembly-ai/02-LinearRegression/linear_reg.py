# ruff: noqa: F401


import numpy as np


class LinearRegression:
    def __init__(self, n_iter=100, lr=0.001):
        self.n_iter = n_iter
        self.lr = lr
        self.weights: np.ndarray = None
        self.bias: int = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            # predictions
            y_pred = self.predict(X)

            # gradients
            dw = (1 / n_samples) * np.dot(X.T, y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray):
        return np.dot(X, self.weights) + self.bias
