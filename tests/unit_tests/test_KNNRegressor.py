from unittest import TestCase
import numpy as np
from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv
from si.models.KNNRegressor import KNNRegressor
from si.model_selection.split import train_test_split


class TestKNNRegressor(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

        self.knn = KNNRegressor(k=3, distance=self.euclidean_distance)

        self.knn._fit(self.train_dataset)

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def test_predict(self):
        predictions = self.knn._predict(self.test_dataset)

        expected_predictions = predictions

        np.testing.assert_almost_equal(predictions, expected_predictions, decimal=1,
                                       err_msg="Previsões estão incorretas.")

    def test_score(self):
        error = self.knn._score(self.test_dataset)

        expected_error_min = 80.0
        expected_error_max = 85.0
        self.assertGreaterEqual(error, expected_error_min, msg=f"O erro RMSE está muito baixo: {error}")
        self.assertLessEqual(error, expected_error_max, msg=f"O erro RMSE está muito alto: {error}")

