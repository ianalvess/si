from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test


def stratified_train_test_split(
    dataset: Dataset, test_size: float = 0.2, random_state: int = 42
) -> Tuple[Dataset, Dataset]:
    """
        Splits a dataset into training and testing sets in a stratified manner.

        Parameters:
        - dataset: Dataset object containing features X and labels y.
        - test_size: proportion of the dataset to be used for the test set (e.g., 0.2 for 20%).
        - random_state: seed for random number generation to ensure reproducibility.

        Returns:
        - train_dataset: training set as a Dataset object.
        - test_dataset: testing set as a Dataset object.
    """
    if random_state is not None:
        np.random.seed(random_state)

    labels = np.unique(dataset.y)
    test_indices = []
    train_indices = []

    for label in labels:
        label_indices = np.where(dataset.y == label)[0]

        np.random.shuffle(label_indices)

        n_label_test_samples = max(1, int(len(label_indices) * test_size))

        test_indices.extend(label_indices[:n_label_test_samples])
        train_indices.extend(label_indices[n_label_test_samples:])

    train_dataset = Dataset(X=dataset.X[train_indices], y=dataset.y[train_indices])
    test_dataset = Dataset(X=dataset.X[test_indices], y=dataset.y[test_indices])

    return train_dataset, test_dataset
