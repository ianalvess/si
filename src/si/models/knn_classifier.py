class KNNClassifier:

    def __init__(self, k:int, distance : callable= euclidian_distance, **kwargs):
        super().__init__(**kwargs)

        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset:Dataset)->KNNClassifier:

        self.dataset = dataset
        return self
    
    def _get_closest_label(self, sample: np.ndarray) -> np.ndarray:

        distances = self.distance(sample, self.dataset.X)
        closest_label_idx = np.argsort(distances)[:self.k]
        closest_label = self.dataset.y[closest_label_idx]
        labels, counts = np.unique(closest_label, return_counts=True)

        return labels[np.argmax(counts)]
        

    def _predict(self, dataset:Dataset)->np.ndarray:

        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)
    
    def _score(self, dataset:Dataset, predictions:np.ndarray)->float:
        