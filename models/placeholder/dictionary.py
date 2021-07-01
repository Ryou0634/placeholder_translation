from typing import List
from marisa_trie import Trie


class Dictionary:
    def __init__(self, words: List[str]):
        self._trie = Trie(words)

    def find_entries(self, sentence: str) -> List[str]:
        entries = []
        for i in range(len(sentence)):
            entries += self._trie.prefixes(sentence[i:])
        return entries
