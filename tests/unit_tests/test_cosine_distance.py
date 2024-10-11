class TestCosineDistance(TestCase):
    def test_distance_between_two_vectors(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        distance = cosine_distance(x, y)
        self.assertIsInstance(distance, np.ndarray)
        self.assertEqual(distance.shape, (1,))

    def test_distance_between_vector_and_matrix(self):
        x = np.array([1, 2, 3])
        y = np.array([[4, 5, 6], [7, 8, 9]])
        distance = cosine_distance(x, y)
        self.assertIsInstance(distance, np.ndarray)
        self.assertEqual(distance.shape, (2,))

    def test_distance_between_two_identical_vectors(self):
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        distance = cosine_distance(x, y)
        self.assertIsInstance(distance, np.ndarray)
        self.assertEqual(distance.shape, (1,))
        self.assertAlmostEqual(distance[0], 0.0)

    def test_distance_between_two_orthogonal_vectors(self):
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        distance = cosine_distance(x, y)
        self.assertIsInstance(distance, np.ndarray)
        self.assertEqual(distance.shape, (1,))
        self.assertAlmostEqual(distance[0], 1.0)

    def test_distance_between_vector_and_zero_vector(self):
        x = np.array([1, 2, 3])
        y = np.array([0, 0, 0])
        distance = cosine_distance(x, y)
        self.assertIsInstance(distance, np.ndarray)
        self.assertEqual(distance.shape, (1,))
        self.assertAlmostEqual(distance[0], 1.0)

    def test_input_validation(self):
        x = 1
        y = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            cosine_distance(x, y)