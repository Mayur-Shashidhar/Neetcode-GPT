import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(
        self,
        model: nn.Module,
        x: torch.Tensor
    ) -> List[float]:

        dead_fractions = []

        with torch.no_grad():

            current = x

            for layer in model:

                current = layer(current)

                if isinstance(layer, nn.ReLU):

                    dead_mask = (current == 0).all(dim=0)

                    dead_fraction = float(
                        dead_mask.float().mean()
                    )

                    dead_fractions.append(
                        round(dead_fraction, 4)
                    )

        return dead_fractions

    def suggest_fix(
        self,
        dead_fractions: List[float]
    ) -> str:

        # 1. Severe dead neurons
        if any(df > 0.5 for df in dead_fractions):
            return "use_leaky_relu"

        # 2. Early layer death
        if len(dead_fractions) > 0 and dead_fractions[0] > 0.3:
            return "reinitialize"

        # 3. Increasing death with depth
        increasing = all(
            dead_fractions[i] < dead_fractions[i + 1]
            for i in range(len(dead_fractions) - 1)
        )

        if (
            len(dead_fractions) > 0
            and increasing
            and dead_fractions[-1] > 0.1
        ):
            return "reduce_learning_rate"

        # 4. Healthy network
        if len(dead_fractions) == 0 or max(dead_fractions) < 0.1:
            return "healthy"

        # 5. Default
        return "healthy"