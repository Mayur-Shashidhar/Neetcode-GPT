import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(
        self,
        model: nn.Module,
        x: torch.Tensor
    ) -> List[Dict[str, float]]:

        stats = []

        with torch.no_grad():

            current = x

            for layer in model:

                current = layer(current)

                if isinstance(layer, nn.Linear):

                    mean = round(float(current.mean()), 4)
                    std = round(float(current.std()), 4)

                    dead_mask = (current <= 0).all(dim=0)
                    dead_fraction = round(
                        float(dead_mask.float().mean()),
                        4
                    )

                    stats.append({
                        "mean": mean,
                        "std": std,
                        "dead_fraction": dead_fraction
                    })

        return stats

    def compute_gradient_stats(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> List[Dict[str, float]]:

        model.zero_grad()

        criterion = nn.MSELoss()

        predictions = model(x)

        loss = criterion(predictions, y)

        loss.backward()

        stats = []

        for layer in model:

            if isinstance(layer, nn.Linear):

                grad = layer.weight.grad

                stats.append({
                    "mean": round(float(grad.mean()), 4),
                    "std": round(float(grad.std()), 4),
                    "norm": round(float(torch.norm(grad)), 4)
                })

        return stats

    def diagnose(
        self,
        activation_stats: List[Dict[str, float]],
        gradient_stats: List[Dict[str, float]]
    ) -> str:

        # 1. Dead neurons
        for stat in activation_stats:
            if stat["dead_fraction"] > 0.5:
                return "dead_neurons"

        # 2. Exploding gradients
        for stat in gradient_stats:
            if stat["norm"] > 1000:
                return "exploding_gradients"

        # 3. Vanishing gradients (last layer)
        if gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # 4. Activation statistics
        for stat in activation_stats:

            if stat["std"] < 0.1:
                return "vanishing_gradients"

            if stat["std"] > 10.0:
                return "exploding_gradients"

        return "healthy"