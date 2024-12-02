import numpy as np
from si.metrics.rmse import rmse


class KNNRegressor:
    def __init__(self, k=3, distance=None):
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset):
        """Armazena o conjunto de dados de treinamento."""
        self.dataset = dataset
        return self

    def _predict(self, dataset):
        """Estima os valores com base nos k vizinhos mais próximos."""
        predictions = np.zeros(dataset.X.shape[0])

        for i in range(dataset.X.shape[0]):

            distances = np.array([self.distance(dataset.X[i], x_train) for x_train in self.dataset.X])

            k_indices = np.argsort(distances)[:self.k]
            k_nearest_values = self.dataset.y[k_indices]

            predictions[i] = np.mean(k_nearest_values)

        return predictions

    def _score(self, dataset):
        """Calcula o erro entre os valores estimados e os reais (RMSE)."""

        return rmse(dataset.y, self._predict(dataset))
