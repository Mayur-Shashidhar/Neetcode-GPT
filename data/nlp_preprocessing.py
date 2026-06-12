import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List


class Solution:
    def get_dataset(
        self,
        positive: List[str],
        negative: List[str]
    ) -> TensorType[float]:

        # Build vocabulary
        vocab = set()

        for sentence in positive + negative:
            vocab.update(sentence.split())

        vocab = sorted(vocab)

        word_to_id = {
            word: idx + 1
            for idx, word in enumerate(vocab)
        }

        # Encode sentences
        tensors = []

        for sentence in positive:
            encoded = [
                word_to_id[word]
                for word in sentence.split()
            ]
            tensors.append(torch.tensor(encoded))

        for sentence in negative:
            encoded = [
                word_to_id[word]
                for word in sentence.split()
            ]
            tensors.append(torch.tensor(encoded))

        # Pad to equal length
        dataset = nn.utils.rnn.pad_sequence(
            tensors,
            batch_first=True,
            padding_value=0
        )

        return dataset.float()