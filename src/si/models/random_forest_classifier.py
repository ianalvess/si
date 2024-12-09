from si.base.model import Model
from si.models.decision_tree_classifier import DecisionTreeClassifier
import numpy as np
from si.data.dataset import Dataset


class RandomForestClassifier(Model):

    def __init__(self, n_estimators, max_features, min_sample_split, max_depth, mode, seed = None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []

    def _fit(self, dataset: Dataset, seed=None):
        n_samples, n_features = dataset.X.shape

        if self.seed is not None:
            np.random.seed(self.seed)

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_X = dataset.X[bootstrap_indices]
            bootstrap_y = dataset.y[bootstrap_indices]

            feature_indices = np.random.choice(n_features, size=self.max_features, replace=False)
            bootstrap_X_features = bootstrap_X[:, feature_indices]

            bootstrap_data = Dataset(X=bootstrap_X_features, y=bootstrap_y)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_sample_split=self.min_sample_split,
                mode=self.mode
            )
            tree._fit(bootstrap_data)

            self.trees.append((feature_indices, tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        res_predictions = []

        unique_classes, y_encoded = np.unique(dataset.y, return_inverse=True)
        class_to_int = {label: idx for idx, label in enumerate(unique_classes)}
        int_to_class = {idx: label for idx, label in enumerate(unique_classes)}

        for feature_indices, tree in self.trees:
            subset_dataset = Dataset(X=dataset.X[:, feature_indices], y=dataset.y)
            tree_predictions = tree._predict(subset_dataset)
            predictions_int = np.array([class_to_int[label] for label in tree_predictions])
            res_predictions.append(predictions_int)

        res_predictions = np.array(res_predictions)

        final_predictions = []
        for i in range(res_predictions.shape[1]):
            sample_predictions = res_predictions[:, i]
            most_common = np.bincount(sample_predictions).argmax()
            final_predictions.append(int_to_class[most_common])

        return np.array(final_predictions)

    def _score(self, dataset: Dataset):
        predictions = self.predict(dataset)
        accuracy = np.mean(predictions == dataset.y)

        return accuracy












