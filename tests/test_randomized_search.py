import os
import numpy as np
from unittest import TestCase
from si.io.csv_file import read_csv
from si.models.logistic_regression import LogisticRegression
from si.model_selection.randomized_search import random_search_cv
from datasets import DATASETS_PATH


class TestRandomizedSearchCVBreastBin(TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the dataset
        cls.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        cls.dataset = read_csv(filename=cls.csv_file, features=True, label=True)

        cls.dataset.X = cls.dataset.X.astype(np.float64)
        cls.dataset.y = cls.dataset.y.astype(np.float64)

    def setUp(self):
        self.model = LogisticRegression()

        # Hyperparameter grid
        self.hyperparameter_grid = {
            'l2_penalty': np.linspace(1, 10, 10),
            'alpha': np.linspace(0.001, 0.0001, 100),
            'max_iter': np.linspace(1000, 2000, 200).astype(int)
        }

    def test_randomized_search_cv(self):
        def scoring_function(y_true, y_pred):
            return np.mean(y_true == y_pred)

        # Execute the randomized search
        results = random_search_cv(
            model=self.model,
            dataset=self.dataset,
            param_grid=self.hyperparameter_grid,
            scoring=scoring_function,
            cv=3,
            n_iter=10
        )

        # Print the results
        print("Hyperparameters tested:")
        for hyperparams in results['hyperparameters']:
            print(hyperparams)

        print("\nScores obtained:")
        print(results['scores'])

        print("\nBest hyperparameters:")
        print(results['best_hyperparameters'])

        print("\nBest score:")
        print(results['best_score'])

        self.assertIn('hyperparameters', results)
        self.assertIn('scores', results)
        self.assertIn('best_hyperparameters', results)
        self.assertIn('best_score', results)

        self.assertTrue(len(results['hyperparameters']) > 0, "No hyperparameters were tested.")
        self.assertTrue(len(results['scores']) > 0, "No scores were obtained.")

        self.assertIsInstance(results['best_score'], float, "Best score is not a float.")
        self.assertGreaterEqual(results['best_score'], 0, "Best score is less than 0.")

        self.assertIsInstance(results['best_hyperparameters'], dict,
                              "Best hyperparameters are not in dictionary format.")
        self.assertIn('l2_penalty', results['best_hyperparameters'])
        self.assertIn('alpha', results['best_hyperparameters'])
        self.assertIn('max_iter', results['best_hyperparameters'])
