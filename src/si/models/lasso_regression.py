import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse

class LassoRegression(Model):
    def __init__(self, l1_penalty=1.0, scale=True, patience=10):
        self.l1_penalty = l1_penalty
        self.scale = scale
        self.patience = patience
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset:Dataset, max_iter=1000):
        """Estima os coeficientes theta e theta_zero usando o algoritmo de descida de coordenadas."""
        X = dataset.X
        y = dataset.y

        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X_scaled = (X - self.mean) / self.std
        else:
            X_scaled = X

        n_samples, n_features = X_scaled.shape
        self.theta = np.zeros(n_features)
        self.theta_zero = 0

        for iteration in range(max_iter):
            theta_old = self.theta.copy()
            for j in range(n_features):
                residual = y - (self._predict(dataset) - self.theta[j] * X_scaled[:, j])
                rho = np.dot(X_scaled[:, j], residual)

                if rho < -self.l1_penalty:
                    self.theta[j] = (rho + self.l1_penalty) / np.sum(X_scaled[:, j] ** 2)
                elif rho > self.l1_penalty:
                    self.theta[j] = (rho - self.l1_penalty) / np.sum(X_scaled[:, j] ** 2)
                else:
                    self.theta[j] = 0.0

            self.theta_zero = np.mean(y - np.dot(X_scaled, self.theta))

            if np.all(np.abs(self.theta - theta_old) < 1e-6):
                break

    def _predict(self, dataset):
        """Prediz o valor de y usando os coeficientes estimados."""
        X = dataset.X

        if self.scale:
            X_scaled = (X - self.mean) / self.std
        else:
            X_scaled = X

        return np.dot(X_scaled, self.theta) + self.theta_zero

    def _score(self, dataset):
        """Calcula o erro entre os valores reais e os previstos (MSE)."""
        y = dataset.y
        predictions = self._predict(dataset)
        return mse(y, predictions)

    def score(self, dataset: Dataset) -> float:
        """Gives a score for the model, given a dataset and its predictions."""
        return float(self._score(dataset))