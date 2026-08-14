import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        # X[:, desired_weight]: grab one whole column of X (the feature for this weight)
        # np.dot: multiply the errors by that column and sum them into one number
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # np.matmul: matrix multiply X by weights to get one prediction per row
        # np.squeeze: drop any length-1 dimensions so the result is a flat 1D array
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        
        # np.array: copy the starting weights into a float64 array we can update
        weights = np.array(initial_weights, dtype=np.float64)
        N = len(Y)
        for _ in range(num_iterations):
            y_pred = self.get_model_prediction(X, weights)
            # np.array: bundle each weight's gradient into one array
            gradients = np.array([self.get_derivative(y_pred, Y, N, X, i) for i in range(len(weights))])
            weights = weights - self.learning_rate * gradients


        # np.round: rounds the final learned weights to 5 decimal places
        return np.round(weights, 5)
