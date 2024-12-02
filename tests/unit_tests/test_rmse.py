import unittest
from src.si.metrics.rmse import rmse

class TestRMSE(unittest.TestCase):
    def test_rmse_valid(self):

        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        expected_rmse = 0.6123724356957945
        self.assertAlmostEqual(rmse(y_true, y_pred), expected_rmse, places=6)

    def test_rmse_mismatched_lengths(self):

        y_true = [1, 2, 3]
        y_pred = [1, 2]
        with self.assertRaises(ValueError):
            rmse(y_true, y_pred)

