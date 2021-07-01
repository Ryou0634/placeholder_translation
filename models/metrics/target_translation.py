from typing import List, Dict, Any
from allennlp.training.metrics import Metric
import re

from models.tokenizers.sp_tokenizer import SentencePieceTokenizer
from models.dataset_reader.placeholder_seq2seq_reader import Placeholder
import spacy


class TargetTranslation(Metric):
    def __init__(self, evaluate_lemma: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._total = 0
        self._correct = 0
        self.evaluate_lemma = evaluate_lemma
        self._detokenizer = SentencePieceTokenizer()

        self.nlp = spacy.load("en_core_web_sm")

    def __call__(self, prediction: List[List[str]], placeholders: List[Placeholder]):
        """ Check if prediction contains the specified target word"""
        for tokens, placeholder in zip(prediction, placeholders):
            if self.evaluate_lemma:
                w = placeholder.target_lemma
            else:
                w = placeholder.target_word
            if not w:
                continue
            sentence = self._detokenizer.detokenize(tokens)
            if self.evaluate_lemma:
                en_tokens = self.nlp(sentence)
                sentence = " ".join([e.lemma_ for e in en_tokens])
            self._total += 1
            if re.search(r"\b{0}\b".format(w), sentence):
                self._correct += 1

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        if self._total == 0:
            value = 0
        else:
            value = self._correct / self._total

        if reset:
            self.reset()
        return value

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        self._total = 0
        self._correct = 0
