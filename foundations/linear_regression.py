import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        # np.dot: matrix-vector multiply — each row of X times the weights, summed
        # np.round: rounds each prediction to 5 decimal places
        return np.round(np.dot(X,weights), 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        # np.pow: raises each error (pred - truth) to the power 2, i.e. squares it
        # np.mean: averages all the squared errors into one number
        # np.round: rounds that average to 5 decimal places
        return np.round(np.mean(np.pow((model_prediction - ground_truth), 2)), 5)
