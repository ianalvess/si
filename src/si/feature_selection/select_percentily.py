import numpy as np
from scipy.stats import f_classif


class SelectPercentile:
    def __init__(self, score_func=f_classif, percentile=10):
        self.score_func = score_func
        self.percentile = percentile

    def _fit(self, X, y):
        self.F, self.p = self.score_func(X, y)
        return self

    def _transform(self, X):
        n_features = X.shape[1]
        n_selected = int(self.percentile / 100 * n_features)

        idx = np.argsort(self.F)[-n_selected:]

        return X[:, idx]

    def fit_transform(self, X, y):
        self._fit(X, y)
        return self._transform(X)