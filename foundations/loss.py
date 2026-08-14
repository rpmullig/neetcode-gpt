import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        if len(y_true) == 0:
            return 0.0

        EPSILON = 0.00000001

        n = len(y_true)
        # np.log: natural logarithm of each predicted probability
        inner = y_true * np.log(y_pred + EPSILON) + (1 - y_true) * np.log(1 - y_pred + EPSILON)
        # np.sum: add up the per-sample terms into a single total
        loss = (-1/n) * np.sum(inner)
        # np.round: rounds the loss to 4 decimal places
        return np.round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        if len(y_true) == 0:
            return 0.0
        
        EPSILON = 0.00000001

        n = len(y_true)
        # np.log: natural logarithm of each predicted class probability
        inner = y_true * np.log(y_pred + EPSILON)
        # np.sum: add up every term across samples and classes into one total
        loss = (-1/n) * np.sum(inner)
        # np.round: rounds the loss to 4 decimal places
        return np.round(loss, 4)

