import math
import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()

        torch.manual_seed(0)

        # Order matters: key, query, value
        self.key = nn.Linear(
            embedding_dim,
            attention_dim,
            bias=False
        )

        self.query = nn.Linear(
            embedding_dim,
            attention_dim,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            attention_dim,
            bias=False
        )

        self.attention_dim = attention_dim

    def forward(
        self,
        embedded: TensorType[float]
    ) -> TensorType[float]:

        K = self.key(embedded)
        Q = self.query(embedded)
        V = self.value(embedded)

        scores = (
            Q @ K.transpose(1, 2)
        ) / math.sqrt(self.attention_dim)

        context_length = embedded.shape[1]

        mask = torch.tril(
            torch.ones(
                context_length,
                context_length,
                device=embedded.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        scores = torch.softmax(scores, dim=2)

        output = scores @ V

        return torch.round(output, decimals=4)