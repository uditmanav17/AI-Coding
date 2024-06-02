# ruff: noqa: F401


import numpy as np


def sigmoid(x: np.ndarray):
    deno = 1 + np.round(np.exp(-x), 3)
    return np.round(1 / deno, 3)


class LogisticRegression:
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
            y_pred = self.predict(X, prob=True)

            # gradients - same as Linear regression
            # https://www.baeldung.com/cs/gradient-descent-logistic-regression
            # https://community.deeplearning.ai/t/summary-and-the-derivations-of-gradients-for-linear-regression-and-logistic-regression/292863
            dw = (1 / n_samples) * np.dot(X.T, y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray, prob=True, thresh=0.5):
        probabs = sigmoid(np.dot(X, self.weights) + self.bias)
        return probabs if prob else np.where(probabs > thresh, 1, 0)
