import numpy as np


def rmse(y_true, y_pred):
    """


    Parameters
    ----------
    y_true : real values for y array
    y_pred : predicted values for y array

    Returns
    -------
    rmse : float corresponding the error between y_true and y_pred

            """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred devem ter o mesmo comprimento.")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))
