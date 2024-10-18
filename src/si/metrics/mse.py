import numpy as np


def mse(y_true:ndarray, Y_pred:ndarray)->float:
    """
    Parameters
    ----------
    y_true
    Y_pred
    ----------


    Returns
    ----------
    error value between y_true and y_pred
    ----------
    """

    return np.sum((y_true - Y_pred) ** 2)/len(y_true)
