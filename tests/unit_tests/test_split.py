from unittest import TestCase

from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv
import numpy as np

from si.model_selection.split import train_test_split
from si.model_selection.split import stratified_train_test_split

class TestSplits(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_train_test_split(self):

        train, test = train_test_split(self.dataset, test_size = 0.2, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)

    def test_stratified_split(self):
        train_dataset, test_dataset = stratified_train_test_split(self.dataset, test_size=0.2, random_state=42)

        total_samples = len(self.dataset.y)
        expected_test_size = int(total_samples * 0.2)
        expected_train_size = total_samples - expected_test_size

        self.assertEqual(len(test_dataset.y), expected_test_size, "Tamanho do conjunto de teste está incorreto")
        self.assertEqual(len(train_dataset.y), expected_train_size, "Tamanho do conjunto de treino está incorreto")

        unique, counts = np.unique(self.dataset.y, return_counts=True)
        original_distribution = counts / len(self.dataset.y)

        test_unique, test_counts = np.unique(test_dataset.y, return_counts=True)
        test_distribution = test_counts / len(test_dataset.y)

        train_unique, train_counts = np.unique(train_dataset.y, return_counts=True)
        train_distribution = train_counts / len(train_dataset.y)

        np.testing.assert_almost_equal(test_distribution, original_distribution, decimal=1,
                                       err_msg="Distribuição de classes no conjunto de teste não está correta")
        np.testing.assert_almost_equal(train_distribution, original_distribution, decimal=1,
                                       err_msg="Distribuição de classes no conjunto de treino não está correta")

