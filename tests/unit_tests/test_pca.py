import os
from datasets import DATASETS_PATH

import unittest
from si.decomposition.pca import PCA
from si.io.csv_file import read_csv

class TestPCA(unittest.TestCase):

    def setUp(self):

        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

        self.X = self.dataset.X
        self.n_components = 2

        self.pca = PCA(n_components=self.n_components)

    def test_fit(self):
        components, explained_variance = self.pca._fit(self.X)

        self.assertEqual(components.shape[1], self.n_components)
        self.assertEqual(len(explained_variance), self.n_components)

    def test_transform(self):
        self.pca.mean = self.X.mean(axis=0)  # Defina a média para o método _transform
        self.pca.components, _ = self.pca._fit(self.X)  # Ajuste para obter os componentes

        X_reduced = self.pca._transform(self.X)

        self.assertEqual(X_reduced.shape[1], self.n_components)

    def test_explained_variance(self):
        self.pca.components, explained_variance = self.pca._fit(self.X)
        self.pca.explained_variance = explained_variance

        self.assertEqual(len(self.pca.explained_variance), self.n_components)


if __name__ == '__main__':
    unittest.main()
