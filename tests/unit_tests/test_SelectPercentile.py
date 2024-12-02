from typing import Callable


import os
from unittest import TestCase
import numpy as np
from si.io.csv_file import read_csv
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification
from si.feature_selection.select_percentily import SelectPercentile
from datasets import DATASETS_PATH


class TestSelectPercentile(TestCase):
    def setUp(self):

        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):

        selector = SelectPercentile()
        selector._fit(self.dataset)

        # Verifica se as pontuações F e os valores p não são None
        self.assertIsNotNone(selector.F)
        self.assertIsNotNone(selector.p)
        # Verifica se as dimensões estão corretas
        self.assertEqual(selector.F.shape[0], self.dataset.X.shape[1])
        self.assertEqual(selector.p.shape[0], self.dataset.X.shape[1])

    def test_transform(self):

        selector = SelectPercentile(percentile=50)  # Seleciona 50% das características
        selector._fit(self.dataset)
        transformed_dataset = selector._transform(self.dataset)

        # Verifica se o número de características selecionadas está correto
        num_features = len(self.dataset.features)
        expected_num_selected_features = int(np.ceil(0.50 * num_features))
        self.assertEqual(transformed_dataset.X.shape[1], expected_num_selected_features)

        # Verifica se as características selecionadas são as corretas
        selected_feature_names = set(transformed_dataset.features)
        original_feature_names = set(self.dataset.features)

        # As características selecionadas devem estar entre as características originais
        self.assertTrue(selected_feature_names.issubset(original_feature_names))

    def test_invalid_percentile(self):

        with self.assertRaises(ValueError):
            selector = SelectPercentile(percentile=150)
            selector._fit(self.dataset)

        with self.assertRaises(ValueError):
            selector = SelectPercentile(percentile=-10)
            selector._fit(self.dataset)