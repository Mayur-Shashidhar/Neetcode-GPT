import torch
import torch.nn as nn
import torch.nn.functional as F


class Solution:
    def train(
        self,
        model: nn.Module,
        data: torch.Tensor,
        epochs: int,
        context_length: int,
        batch_size: int,
        lr: float
    ) -> float:

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        final_loss = None

        for epoch in range(epochs):
            torch.manual_seed(epoch)

            starts = torch.randint(
                0,
                len(data) - context_length,
                (batch_size,)
            )

            X = torch.stack([
                data[i:i + context_length]
                for i in starts
            ])

            Y = torch.stack([
                data[i + 1:i + 1 + context_length]
                for i in starts
            ])

            logits = model(X)  # (B, T, C)

            B, T, C = logits.shape

            logits_flat = logits.view(B * T, C)
            targets_flat = Y.view(B * T)

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        return round(final_loss, 4)