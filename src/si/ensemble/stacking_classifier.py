import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset


class StackingClassifier(Model):

    def __init__(self, models, final_model):
        self.models = models
        self.final_model = final_model

    def _fit(self, dataset: Dataset):
        X = dataset.X
        y = dataset.y

        self._models = []
        for model in self.models:
            model._fit(dataset)
            self._models.append(model)

        predictions = np.column_stack([model._predict(dataset) for model in self._models])

        final_dataset = Dataset(X=predictions, y=y)
        self.final_model._fit(final_dataset)

        return self

    def _predict(self, X):

        temp_dataset = Dataset(X=X, y=None)
        ini_predictions = np.column_stack([model._predict(temp_dataset) for model in self._models])
        final_dataset = Dataset(X=ini_predictions, y=None)
        final_predictions = self.final_model._predict(final_dataset)

        return final_predictions

    def _score(self, X, y):

        predictions = self._predict(X)
        accurancy = np.mean(predictions == y)

        return accurancy





