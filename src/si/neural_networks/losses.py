from abc import abstractmethod

import numpy as np


class LossFunction:

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the derivative of the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """
    Mean squared error loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # To avoid the additional multiplication by -1 just swap the y_pred and y_true.
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    """
    Cross entropy loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / p) + (1 - y_true) / (1 - p)

    class CategoricalCrossEntropy(LossFunction):
        """
           Cross entropy loss function for multi-class classification problems.
           """

        def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
            """
            Compute the categorical cross-entropy loss function.

            Parameters
            ----------
            y_true: numpy.ndarray
                True one-hot encoded labels.
            y_pred: numpy.ndarray
                Predicted probabilities (output of the model).

            Returns
            -------
            float
                The mean loss value over the batch.
            """
            # Avoid division by zero
            y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

            # Compute categorical cross-entropy loss
            loss = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
            return np.mean(loss)

        def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
            """
            Compute the derivative of the categorical cross-entropy loss function.

            Parameters
            ----------
            y_true: numpy.ndarray
                True one-hot encoded labels.
            y_pred: numpy.ndarray
                Predicted probabilities (output of the model).

            Returns
            -------
            numpy.ndarray
                The gradient of the loss function with respect to predictions.
            """
            # Avoid division by zero
            y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

            # Gradient calculation
            gradients = -y_true / y_pred_clipped
            return gradients / len(y_true)



