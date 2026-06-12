import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        positions = np.arange(seq_len).reshape(-1, 1)
        dimensions = np.arange(0, d_model, 2)
        angles = positions / (
            10000 ** (dimensions / d_model)
        )
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(angles)
        pe[:, 1::2] = np.cos(angles)
        return np.round(pe, 5)