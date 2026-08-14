import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        # np.max: find the single largest logit (subtracted for numerical stability)
        # np.exp: raise e to each shifted logit, turning them all positive
        exp_z = np.exp(z - np.max(z))
        # np.sum: total of all the exponentials, used to normalize into probabilities
        probabilities = exp_z / np.sum(exp_z)
        # np.round: rounds each probability to 4 decimal places
        return np.round(probabilities, 4)