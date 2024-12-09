from typing import Callable

import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from sklearn.feature_selection import SelectPercentile
from si.statistics.f_classification import f_classification

class SelectPercentile(Transformer):
    """
    Select features according to the top percentile scores.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's Pearson correlation coefficients for regression tasks.

    Parameters
    ----------
    score_func: callable, default=f_classif
        Function taking dataset and returning a pair of arrays (scores, p_values)
    percentile: float, default=10.0
        Percentile of top features to select (0.0 < percentile <= 100.0).

    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """

    def __init__(self, score_func: Callable = f_classification, percentile: float = 10.0, **kwargs):
        """
        Initialize the SelectPercentile instance.

        Parameters
        ----------
        score_func: callable, default=f_classif
            Function taking dataset and returning a pair of arrays (scores, p_values)
        percentile: float, default=10.0
            Percentile of top features to select (0.0 < percentile <= 100.0).
        """
        super().__init__(**kwargs)
        if not (0.0 < percentile <= 100.0):
            raise ValueError("percentile must be between 0.0 and 100.0.")
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Compute the F scores and p-values for each feature.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset.

        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset by selecting features in the specified percentile.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset.

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the top percentile scoring features.
        """
        num_features = len(dataset.features)
        k = int(np.ceil(self.percentile / 100.0 * num_features))  # Calcula número de características a manter
        idxs = np.argsort(self.F)[-k:]  # Obtém os índices das k melhores características
        features = np.array(dataset.features)[idxs]  # Nomes das características selecionadas
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

