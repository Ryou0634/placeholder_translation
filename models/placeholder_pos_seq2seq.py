from typing import List
from contextlib import suppress

from allennlp.models import Model
from .placeholder_seq2seq import PlaceholderSeq2Seq

from models.dataset_reader.placeholder_seq2seq_reader import Placeholder

from lemminflect import getInflection
from .placeholder_seq2seq import inflect_lemma

INFLECTION_TAGS = {"<NNS>", "<NN>", "<VB>", "<VBD>", "<VBG>", "<VBN>", "<VBP>", "<VBZ>"}


@Model.register("placeholder_pos_seq2seq")
class PlaceholderPosSeq2Seq(PlaceholderSeq2Seq):
    @staticmethod
    def replace_placeholders(
        tokens_list: List[List[str]], placeholder: List[Placeholder], specified_inflection: str = None
    ) -> List[List[str]]:

        found_inflection_tags = []
        for i, (tokens, p) in enumerate(zip(tokens_list, placeholder)):
            tag = None
            if p.placeholder_token is None:
                continue
            for j, t in enumerate(tokens):
                if t == p.placeholder_token:
                    if j < len(tokens) - 1 and tokens[j + 1] in INFLECTION_TAGS:
                        tag = tokens[j + 1]
                        word = inflect_lemma(p.target_lemma, tag)
                    else:
                        word = p.target_lemma
                    tokens_list[i][j] = word
            found_inflection_tags.append(tag)

        # remove the inflection tag
        for tokens, found_tag in zip(tokens_list, found_inflection_tags):
            if found_tag is None:
                continue
            with suppress(ValueError):
                tokens.remove(found_tag)

        return tokens_list
