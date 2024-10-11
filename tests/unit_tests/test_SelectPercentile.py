
class TestSelectPercentile(TestCase):
    def test_init(self):
        selector = SelectPercentile()
        self.assertEqual(selector.score_func, f_classif)
        self.assertEqual(selector.percentile, 10)

    def test_fit(self):
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        selector = SelectPercentile()
        selector._fit(X, y)
        self.assertIsNotNone(selector.F)
        self.assertIsNotNone(selector.p)

    def test_transform(self):
        X = np.random.rand(100, 10)
        selector = SelectPercentile()
        selector.F = np.random.rand(10)
        X_selected = selector._transform(X)
        self.assertEqual(X_selected.shape[1], 1)

    def test_fit_transform(self):
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        selector = SelectPercentile()
        X_selected = selector.fit_transform(X, y)
        self.assertIsNotNone(X_selected)

    def test_percentile(self):
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        selector = SelectPercentile(percentile=50)
        X_selected = selector.fit_transform(X, y)
        self.assertEqual(X_selected.shape[1], 5)

    def test_invalid_percentile(self):
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        selector = SelectPercentile(percentile=150)
        with self.assertRaises(ValueError):
            selector.fit_transform(X, y)