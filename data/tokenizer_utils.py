from typing import List, Dict


class Solution:

    def _tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0

        while i < len(text):
            best_match = None

            for j in range(len(text), i, -1):
                candidate = text[i:j]
                if candidate in vocab:
                    best_match = candidate
                    break

            if best_match is not None:
                tokens.append(best_match)
                i += len(best_match)
            else:
                tokens.append(text[i])
                i += 1

        return tokens

    def tokenize_numbers(
        self,
        numbers: List[int],
        vocab: Dict[str, int]
    ) -> List[List[str]]:
        return [self._tokenize(str(num), vocab) for num in numbers]

    def count_tokens(
        self,
        text: str,
        vocab: Dict[str, int]
    ) -> int:
        return len(self._tokenize(text, vocab))

    def fertility_score(
        self,
        text: str,
        vocab: Dict[str, int]
    ) -> float:
        word_count = len(text.split())

        if word_count == 0:
            return 0.0

        token_count = self.count_tokens(text, vocab)

        return round(token_count / word_count, 4)