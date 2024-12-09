import os
import numpy as np
from unittest import TestCase
from si.io.csv_file import read_csv
from si.models.logistic_regression import LogisticRegression
from si.model_selection.randomized_search import randomized_search_cv
from datasets import DATASETS_PATH


class TestRandomizedSearchCVBreastBin(TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the dataset
        cls.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        cls.dataset = read_csv(filename=cls.csv_file, features=True, label=True)

        # Ensure labels are numeric
        cls.dataset.X = cls.dataset.X.astype(np.float64)
        cls.dataset.y = cls.dataset.y.astype(np.float64)

    def setUp(self):
        # Model to be tested
        self.model = LogisticRegression()

        # Hyperparameter grid
        self.hyperparameter_grid = {
            'l2_penalty': np.linspace(1, 10, 10),
            'alpha': np.linspace(0.001, 0.0001, 100),
            'max_iter': np.linspace(1000, 2000, 200).astype(int)
        }

    def test_randomized_search_cv(self):
        # Define a scoring function that fits the model and returns the score
        def scoring_function(model, dataset):
            # Ensure the model is fitted
            model.fit(dataset.X, dataset.y)
            predictions = model.predict(dataset.X)
            return np.mean(predictions == dataset.y)

        # Execute the randomized search
        results = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_grid=self.hyperparameter_grid,
            scoring=scoring_function,  # Use the defined scoring function
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

        # Assertions to validate the results
        self.assertIn('hyperparameters', results)
        self.assertIn('scores', results)
        self.assertIn('best_hyperparameters', results)
        self.assertIn('best_score', results)

        # Check that the results are not empty
        self.assertTrue(len(results['hyperparameters']) > 0, "No hyperparameters were tested.")
        self.assertTrue(len(results['scores']) > 0, "No scores were obtained.")

        # Check that the best score is a float and within the expected range
        self.assertIsInstance(results['best_score'], float, "Best score is not a float.")
        self.assertGreaterEqual(results['best_score'], 0, "Best score is less than 0.")

        # Check that best hyperparameters are in the expected format
        self.assertIsInstance(results['best_hyperparameters'], dict,
                              "Best hyperparameters are not in dictionary format.")
        self.assertIn('l2_penalty', results['best_hyperparameters'])
        self.assertIn('alpha', results['best_hyperparameters'])
        self.assertIn('max_iter', results['best_hyperparameters'])

        # Check that the values of best hyperparameters are within the expected ranges
        self.assertGreaterEqual(results['best_hyperparameters']['l2_penalty'], 1)