from unittest import TestCase
import numpy as np
from datasets import DATASETS_PATH
import os
from si.io.csv_file import read_csv
from si.statistics.cosine_distance import cosine_distance


class TestCosineDistance(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_cosine_distance(self):
        x = np.array([1, 2, 3])
        y = np.array([[1, 2, 3], [4, 5, 6]])

        # Chama a função que você definiu
        our_distance = cosine_distance(x, y)

        # Importa a função do sklearn com um alias
        from sklearn.metrics.pairwise import cosine_distances as sklearn_cosine_distance

        # Chama a função do sklearn
        sklearn_distance = sklearn_cosine_distance(x.reshape(1, -1), y)

        # Verifica se as distâncias estão próximas
        assert np.allclose(our_distance, sklearn_distance.flatten())
