# ruff: noqa : F401

import numpy as np
import scipy.stats as stats


class KNN:
    def __init__(self, k: int = 3) -> None:
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def predict(self, X: np.ndarray):
        return np.array([self._predict(i) for i in X])

    def _predict(self, x: np.ndarray):
        # compute distance
        differences = self.X - x
        dists = np.linalg.norm(differences, axis=1)

        # check label of top k min distances
        min_dist_idx = np.argsort(dists)[: self.k]

        # return label using majority vote
        return stats.mode(self.y[min_dist_idx]).mode
