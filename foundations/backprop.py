import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        
        # 1. Forward Pass
        z = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-z))
        
        # 2. Compute the shared gradient component (Chain Rule)
        # dL/dy_hat = (y_hat - y_true)
        # dy_hat/dz = y_hat * (1 - y_hat)
        delta = (y_hat - y_true) * (y_hat * (1 - y_hat))
        
        # 3. Compute Gradients
        # dL/dw = delta * dz/dw = delta * x
        # dL/db = delta * dz/db = delta * 1
        dl_dw = delta * x
        dl_db = delta
        
        # 4. Return as a tuple with requested precision
        return (np.round(dl_dw, 5), round(float(dl_db), 5))
