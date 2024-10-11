def train_test_split(self,dataset: Dataset, test_size: float = 0.2, random_state: int = 123) -> Tuple[Dataset, Dataset]:

    """
    Split the dataset into training and test sets.

    Parameters
    ----------
    dataset: Dataset
        The dataset to split.
    test_size: float
        The proportion of the dataset to include in the test split. Default is 0.2.
    random_state: int
        The seed for the random number generator. Default is 123.

    Returns
    -------
    Tuple[Dataset, Dataset] 
        The training and test datasets.
    """ 
    np.random.seed(random_state)

    permutation = np.random.permutation(dataset.X.shape()[0])

    test_sample_size = int(test_size * dataset.shape()[0])

    test_idx = permutation[:test_sample_size]
    train_idx = permutation[test_sample_size:]

    train_dataset = Dataset(X=dataset.X[train_idx], y=dataset.y[train_idx], features=dataset.features, label=dataset.label)
    test_dataset = Dataset(X=dataset.X[test_idx], y=dataset.y[test_idx], features=dataset.features, label=dataset.label)

    return train_dataset, test_dataset
 