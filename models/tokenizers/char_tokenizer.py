from typing import List
from overrides import overrides
import unicodedata
from allennlp.data import Token
from allennlp.data.tokenizers import Tokenizer


@Tokenizer.register("char")
class CharTokenizer(Tokenizer):
    """
    Simply split string into characters.
    """

    def __init__(self,
                 lower: bool = False,
                 word_boundary: str = '▁') -> None:
        self.lower = lower
        self.word_boundary = word_boundary

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        text = unicodedata.normalize('NFKC', text)
        if self.lower:
            text = text.lower()
        text = text.replace(' ', self.word_boundary)
        return [Token(c) for c in text]

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self.tokenize(text) for text in texts]

    def detokenize(self, tokens: List[str]):
        detokenized = ''.join(tokens).replace(self.word_boundary, ' ').strip()
        return detokenized
