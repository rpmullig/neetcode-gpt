import numpy as np 
from numpy.typing import NDArray

class Solution:
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        # np.exp: raises e to the power of each element (element-wise e^x)
        sigmoid_output: NDArray[np.float64] = 1 / (1 + np.exp(-z))
        # np.round: rounds each value to 5 decimal places
        return np.round(sigmoid_output, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        
        # Original approach (Modifies the original 'z' array outside the function):
        # boolean-mask assignment: set every element that is < 0 to 0
        z[z < 0] = 0
        
        # Alternative 1 (Safer, creates a new array and leaves original 'z' untouched):
        # return np.maximum(0, z)
        
        # Alternative 2 (Conditional replacement approach):
        # return np.where(z > 0, z, 0)
        
        # or np.maximum(z, 0) NOT np.max(z, 0) which is not scalar
        return z
