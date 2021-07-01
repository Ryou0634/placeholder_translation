from typing import List, NamedTuple, Tuple, Optional
import json
import random
import numpy as np

from allennlp.data import Instance, Token
from allennlp.data.fields import TextField, MetadataField, TensorField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN

from .seq2seq_reader import Seq2SeqReader
from models.placeholder.word_replacer import WordReplacer, Entry

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_SOURCE_TOKEN_INDEXERS = {
    "tokens": SingleIdTokenIndexer(namespace="source_tokens", start_tokens=[START_SYMBOL], end_tokens=[END_SYMBOL])
}

DEFAULT_TARGET_TOKEN_INDEXERS = {
    "tokens": SingleIdTokenIndexer(namespace="target_tokens", start_tokens=[START_SYMBOL], end_tokens=[END_SYMBOL])
}


class Placeholder(NamedTuple):
    source_word: Optional[str] = None
    target_word: Optional[str] = None
    placeholder_token: Optional[str] = None
    target_word_tokens: Optional[List[str]] = None
    target_lemma: Optional[str] = None
    target_lemma_tokens: Optional[List[str]] = None
    source_lemma: Optional[str] = None


@DatasetReader.register("placeholder_seq2seq")
class PlaceholderSeq2SeqReader(Seq2SeqReader):
    def __init__(
        self,
        word_replacers: List[WordReplacer],
        replace_ratio: float,
        mode: str,
        use_character: bool = False,
        placeholder_choice: str = "sample",
        use_source_factor: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.word_replacers = word_replacers
        self.replace_ratio = replace_ratio
        self.use_character = use_character

        if self.use_character:
            self.lemma_character_indexers = {"characters": SingleIdTokenIndexer(namespace="characters")}
            self.target_character_indexers = {
                "characters": SingleIdTokenIndexer(namespace="characters", end_tokens=[END_SYMBOL])
            }

        assert mode in {"augment", "replace", "baseline", "only_placeholder"}
        self.mode = mode

        assert placeholder_choice in {"sample", "iterate"}
        self.placeholder_choice = placeholder_choice

        self.use_source_factor = use_source_factor

    def _read(self, file_path: str):
        if self.target_file_parser is not None:
            try:
                file_path_dict = json.loads(file_path)

                generator = zip(self.file_parser(file_path_dict["source"]), self.file_parser(file_path_dict["target"]))

            except json.JSONDecodeError or KeyError:
                raise Exception(
                    "You have specified `target_file_parser`, "
                    "which means you need two file paths provided in the json format like "
                    '{"source": source_path,  "target": target_path}.\n'
                    f"But you provided {file_path}"
                )
        else:
            generator = self.file_parser(file_path)

        placeholder_count = 0
        normal_count = 0
        for n_sent, (src_sent, tgt_sent) in enumerate(generator):

            if len(src_sent) == 0 or len(tgt_sent) == 0:
                continue

            if self.switch_src_tgt:
                tgt_sent, src_sent = src_sent, tgt_sent

            # Sample one word replacer at a time

            if self.placeholder_choice == "sample":
                word_replacers = [random.choice(self.word_replacers)]
            elif self.placeholder_choice == "iterate":
                word_replacers = self.word_replacers

            for i_wr, word_replacer in enumerate(word_replacers):
                entries = word_replacer.find_entries(src_sent, tgt_sent)
                if entries:
                    entry = word_replacer.select(entries)
                else:
                    entry = None

                instance = self.text_to_instance(src_sent, tgt_sent, entry, word_replacer.placeholder_token)
                if self._filter_out_this_instance(instance):
                    continue

                if entry and random.random() < self.replace_ratio:
                    entry = word_replacer.select(entries)
                    replacer_output = word_replacer.replace(src_sent, tgt_sent, entry)
                    if self.use_source_factor:
                        assert replacer_output["replaced_span"] is not None
                    placeholder_instance = self.text_to_instance(
                        replacer_output["source_sentence"],
                        replacer_output["target_sentence"],
                        entry,
                        word_replacer.placeholder_token,
                        replaced_span=replacer_output["replaced_span"],
                    )
                else:
                    placeholder_instance = None

                if self.mode == "augment":
                    if i_wr == 0:
                        normal_count += 1
                        yield instance
                    if placeholder_instance is not None:
                        placeholder_count += 1
                        yield placeholder_instance
                elif self.mode == "replace":
                    if placeholder_instance is not None:
                        placeholder_count += 1
                        yield placeholder_instance
                    else:
                        normal_count += 1
                        yield instance
                elif self.mode == "baseline":
                    normal_count += 1
                    yield instance
                elif self.mode == "only_placeholder":
                    if placeholder_instance is not None:
                        placeholder_count += 1
                        yield placeholder_instance
                else:
                    raise ValueError(self.mode)
        logger.info(f"Normal instances count: {normal_count}")
        logger.info(f"Placeholder instances count: {placeholder_count}")

    def text_to_instance(
        self,
        src_sent: str,
        tgt_sent: str = None,
        entry: Entry = None,
        placeholder_token: str = None,
        replaced_span: Tuple[int, int] = None,
    ) -> Instance:
        fields = {}

        source_tokens = self.src_tokenizer.tokenize(src_sent.strip())
        src_field = TextField(source_tokens, token_indexers=self.src_token_indexers)
        fields["source_tokens"] = src_field

        if self.use_source_factor:
            replaced_span = replaced_span or (-1, -1)
            assign_type_id_to_tokens_with_character_span(source_tokens, replaced_span)
            source_factor = [0] + [t.type_id for t in source_tokens] + [0]  # append start and end tokens
            fields["source_factor"] = TensorField(np.array(source_factor, dtype=np.long), padding_value=0)

        if tgt_sent:
            target_tokens = self.tgt_tokenizer.tokenize(tgt_sent.strip())
            tgt_field = TextField(target_tokens, token_indexers=self.tgt_token_indexers)
            fields["target_tokens"] = tgt_field

        if entry is not None:
            fields["placeholder"] = MetadataField(
                Placeholder(
                    entry.source_replace_pattern,
                    entry.target_replace_pattern,
                    placeholder_token,
                    [t.text for t in self.tgt_tokenizer.tokenize(entry.target_replace_pattern)],
                    entry.target_lemma,
                    [t.text for t in self.tgt_tokenizer.tokenize(entry.target_lemma)],
                    entry.source_lemma,
                )
            )

            if self.use_character:
                fields["lemma_characters"] = TextField(
                    [Token(c) for c in entry.target_lemma], self.lemma_character_indexers
                )
                fields["inflection_characters"] = TextField(
                    [Token(placeholder_token)] + [Token(c) for c in entry.target_replace_pattern],
                    self.target_character_indexers,
                )

        else:
            fields["placeholder"] = MetadataField(Placeholder(None, None, None, None, None, None, None))
            if self.use_character:
                fields["lemma_characters"] = TextField([Token(DEFAULT_PADDING_TOKEN)], self.lemma_character_indexers)
                fields["inflection_characters"] = TextField(
                    [Token(DEFAULT_PADDING_TOKEN)], self.target_character_indexers
                )
        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["source_tokens"]._token_indexers = self.src_token_indexers
        instance.fields["target_tokens"]._token_indexers = self.tgt_token_indexers


def assign_type_id_to_tokens_with_character_span(tokens: List[Token], character_span: Tuple[int, int]):
    span_start, span_end = character_span
    for t in tokens:
        if span_start <= t.idx and t.idx_end <= span_end:
            t.type_id = 1
        else:
            t.type_id = 0
