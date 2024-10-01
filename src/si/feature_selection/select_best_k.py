class select_k_best:
    def __init__(self, score_func = f_classification,k):
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def _fit(self,dataset : Dataset) -> SelectKBest:

        self.F, self.p = self.score_func(dataset)

        return self


    def _transform(self, dataset : Dataset) -> Dataset:
        idx = np.argsort(self.F)
        mask = idx[-self.k:]
        new_X = dataset.X[:, mask]
        new_features = dataset.features[mask]

        return Dataset(X = new_X, y = dataset.y, features = new_features, label = dataset.label)