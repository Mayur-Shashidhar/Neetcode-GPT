import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        activation = x

        for i in range(len(weights)):
            activation = np.dot(activation, weights[i]) + biases[i]

            # Apply ReLU on all hidden layers
            if i < len(weights) - 1:
                activation = np.maximum(0, activation)

        return np.round(activation, 5)