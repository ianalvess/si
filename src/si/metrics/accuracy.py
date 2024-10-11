def accuraty(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It computes the accuracy of the model.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels.
    y_pred: np.ndarray
        The predicted labels.

    Returns
    -------
    float
        The accuracy of the model.
    """
    return np.sum(y_true == y_pred) / len(y_true)