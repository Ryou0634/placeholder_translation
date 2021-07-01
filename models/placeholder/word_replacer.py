from typing import List, NamedTuple, Tuple, Dict
import random
import math
import itertools

from allennlp.common import Registrable
from collections import defaultdict
from lemminflect import getAllInflections


from .dictionary import Dictionary

DEFAULT_PLACEHOLDER = "[PLACEHOLDER]"

BE_VERBS = ["is", "are", "has been", "have been", "was", ""]


class Entry(NamedTuple):
    source_matching_pattern: str
    source_replace_pattern: str
    source_lemma: str
    target_matching_pattern: str
    target_replace_pattern: str
    target_lemma: str
    count: int


def parse_dictionary(path: str, max_count: int = math.inf, min_count: int = 0):
    dictionary = []
    with open(path, "r") as f:
        for line in f:
            fields = line.strip().split("\t")
            fields[-1] = int(fields[-1])
            entry = Entry(*fields)
            if max_count > entry.count > min_count:
                dictionary.append(entry)
    return dictionary


def dump_dictionary(dictionary: List[Entry], path: str):
    dictionary = sorted(dictionary, key=lambda x: -x.count)
    with open(path, "w") as f:
        for e in dictionary:
            f.write("\t".join([str(f) for f in e]) + "\n")


class WordReplacer(Registrable):
    def __init__(
        self, dictionary_path: str, replace_random: bool = False, max_count: int = None, min_count: int = None
    ):
        if max_count is None:
            max_count = math.inf
        if min_count is None:
            min_count = 0
        entries = parse_dictionary(dictionary_path, max_count=max_count, min_count=min_count)

        self.src_matching_dictionary = Dictionary(list({e.source_matching_pattern for e in entries}))
        self.src2entries = defaultdict(list)
        for e in entries:
            self.src2entries[e.source_matching_pattern].append(e)

        self.replace_random = replace_random
        self.placeholder_token = None

    def replace(self, source_sentence: str, target_sentence: str, entry: Entry) -> Dict:
        raise NotImplementedError

    def select(self, entries: List[Entry]) -> Entry:
        if self.replace_random:
            return random.choice(entries)
        else:
            return entries[0]

    def find_entries(self, source_sentence: str, target_sentence: str) -> List[Entry]:

        found_entries = []
        source_words = self.src_matching_dictionary.find_entries(source_sentence)
        for src_w in source_words:
            for entry_candidate in self.src2entries[src_w]:
                if entry_candidate.target_matching_pattern in target_sentence:
                    index = target_sentence.find(entry_candidate.target_matching_pattern)
                    end_index = index + len(entry_candidate.target_matching_pattern)
                    # We assume the target language is a white-spaced language (e.g., English)
                    # it has to be at word boundry
                    if (
                        len(target_sentence) == end_index
                        or target_sentence[end_index] in {" ", ".", ",", ":", "!", "?",}
                    ) and (index == 0 or target_sentence[index - 1] in {" "}):
                        found_entries.append(entry_candidate)

        return found_entries


@WordReplacer.register("dummy")
class DummyrReplacer(WordReplacer):
    def __init__(
        self, dictionary_path: str, replace_random: bool = False, max_count: int = None, min_count: int = None, **kwargs
    ):
        super().__init__(dictionary_path, replace_random, max_count, min_count)

    def replace(self, source_sentence: str, target_sentence: str, entry: Entry) -> Dict:
        return {"source_sentence": source_sentence, "target_sentence": target_sentence, "replaced_span": None}


@WordReplacer.register("placeholder")
class PlaceholderReplacer(WordReplacer):
    def __init__(self, dictionary_path: str, placeholder_token: str = DEFAULT_PLACEHOLDER, **kwargs):

        super().__init__(dictionary_path=dictionary_path, **kwargs)

        self.placeholder_token = placeholder_token

    def replace(self, source_sentence: str, target_sentence: str, entry: Entry) -> Dict:
        source_sentence = source_sentence.replace(entry.source_replace_pattern, self.placeholder_token, 1)
        target_sentence = target_sentence.replace(entry.target_replace_pattern, self.placeholder_token, 1)
        return {"source_sentence": source_sentence, "target_sentence": target_sentence, "replaced_span": None}


@WordReplacer.register("code-switching")
class CodeSwitchReplacer(WordReplacer):
    def __init__(self, dictionary_path: str, use_lemma: bool = False, **kwargs):
        super().__init__(dictionary_path=dictionary_path, **kwargs)
        self.use_lemma = use_lemma

    def replace(self, source_sentence: str, target_sentence: str, entry: Entry) -> Dict:
        if self.use_lemma:
            switched_target_word = entry.target_lemma
        else:
            switched_target_word = entry.target_replace_pattern

        replaced_start = source_sentence.find(entry.source_replace_pattern)
        replaced_end = replaced_start + len(switched_target_word)
        source_sentence = source_sentence.replace(entry.source_replace_pattern, switched_target_word, 1)

        return {
            "source_sentence": source_sentence,
            "target_sentence": target_sentence,
            "replaced_span": (replaced_start, replaced_end),
        }


@WordReplacer.register("append")
class WordAppender(WordReplacer):
    def __init__(self, dictionary_path: str, use_lemma: bool = False, **kwargs):
        super().__init__(dictionary_path=dictionary_path, **kwargs)
        self.use_lemma = use_lemma

    def replace(self, source_sentence: str, target_sentence: str, entry: Entry) -> Dict:
        if self.use_lemma:
            appended_target_word = entry.target_lemma
        else:
            appended_target_word = entry.target_replace_pattern
        source_sentence = source_sentence + "@sep@" + appended_target_word
        return {"source_sentence": source_sentence, "target_sentence": target_sentence, "replaced_span": None}


@WordReplacer.register("placeholder-pos")
class PlaceholderPosReplacer(WordReplacer):
    def __init__(self, dictionary_path: str, placeholder_token: str, **kwargs):
        super().__init__(dictionary_path=dictionary_path, **kwargs)
        assert placeholder_token in {"[NOUN]", "[VERB]"}
        self.placeholder_token = placeholder_token
        self.upos = placeholder_token[1:-1]

    def replace(self, source_sentence: str, target_sentence: str, entry: Entry) -> Dict:
        source_sentence = source_sentence.replace(entry.source_replace_pattern, self.placeholder_token, 1)

        pos = self.get_inflection_pos(entry.target_replace_pattern, entry.target_lemma, self.upos)
        if pos is not None:
            pos_token = f"<{pos}>"
            placeholder_token = self.placeholder_token + pos_token
        else:
            placeholder_token = self.placeholder_token

        target_sentence = target_sentence.replace(entry.target_replace_pattern, placeholder_token, 1)
        return {"source_sentence": source_sentence, "target_sentence": target_sentence, "replaced_span": None}

    @staticmethod
    def get_inflection_pos(word: str, lemma: str, upos: str) -> str:
        word = word.split()[-1]
        lemma = lemma.split()[-1]

        all_inflections = getAllInflections(lemma, upos=upos)
        word_to_pos = {w: [] for w in itertools.chain(*all_inflections.values())}
        for pos, words in all_inflections.items():
            for w in words:
                word_to_pos[w].append(pos)

        if word in word_to_pos:
            # if a word has two possibilities (e.g., ["NN", "NNS"]), always return the first pos
            return sorted(word_to_pos[word])[0]
        else:
            return None
