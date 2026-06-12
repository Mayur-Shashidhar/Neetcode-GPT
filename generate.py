import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution:
    def generate(
        self,
        model,
        new_chars: int,
        context: TensorType[int],
        context_length: int,
        int_to_char: dict
    ) -> str:

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()

        generated_text = ""

        for i in range(new_chars):

            # Crop context if needed
            current_context = context[:, -context_length:]

            # Forward pass
            logits = model(current_context)

            # Take logits from last position
            logits = logits[:, -1, :]

            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)

            # Required for reproducible output
            generator.set_state(initial_state)

            # Sample next token
            next_token = torch.multinomial(
                probs,
                num_samples=1,
                generator=generator
            )

            # Append token to context
            context = torch.cat((context, next_token), dim=1)

            # Convert token to character
            generated_text += int_to_char[next_token.item()]

        return generated_text