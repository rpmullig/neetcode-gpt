import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        sigmoid_output: NDArray[np.float64] = 1 / (1 + np.exp(-z))
        return np.round(sigmoid_output, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        z[z < 0] = 0 # or np.maximum(z, 0) NOT np.max(z, 0) which is not scalar
        return z
