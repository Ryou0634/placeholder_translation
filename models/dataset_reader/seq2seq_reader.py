from typing import Dict, Tuple, List
import json
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import TextField, ArrayField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from models.dataset_reader.file_parsers import FileParser
from models.dataset_reader.filterer.instance_filterers import InstanceFilterer

DEFAULT_SOURCE_TOKEN_INDEXERS = {
    "tokens": SingleIdTokenIndexer(namespace="source_tokens", start_tokens=[START_SYMBOL], end_tokens=[END_SYMBOL])
}

DEFAULT_TARGET_TOKEN_INDEXERS = {
    "tokens": SingleIdTokenIndexer(namespace="target_tokens", start_tokens=[START_SYMBOL], end_tokens=[END_SYMBOL])
}


@DatasetReader.register("my_seq2seq")
class Seq2SeqReader(DatasetReader):
    def __init__(
        self,
        file_parser: FileParser,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        target_file_parser: FileParser = None,
        src_token_indexers: Dict[str, TokenIndexer] = None,
        tgt_token_indexers: Dict[str, TokenIndexer] = None,
        filterers: List[InstanceFilterer] = None,
        switch_src_tgt: bool = False,
        alignment_file: str = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.file_parser = file_parser
        self.target_file_parser = target_file_parser

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        # While the start tokens are optional, the end tokens are required
        # The end tokens are helpful when other tokens otherwise have nothing to attend to.
        self.src_token_indexers = src_token_indexers or DEFAULT_SOURCE_TOKEN_INDEXERS
        self.tgt_token_indexers = tgt_token_indexers or DEFAULT_TARGET_TOKEN_INDEXERS

        self.filterers = filterers

        self.switch_src_tgt = switch_src_tgt

        self.alignment_file = alignment_file

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

        if self.alignment_file is not None:
            alignment_generator = parse_alignment_file(self.alignment_file)

        for n_sent, (src_sent, tgt_sent) in enumerate(generator):

            if len(src_sent) == 0 or len(tgt_sent) == 0:
                continue

            if self.switch_src_tgt:
                tgt_sent, src_sent = src_sent, tgt_sent

            instance = self.text_to_instance(src_sent, tgt_sent)

            if self._filter_out_this_instance(instance):
                continue

            if self.alignment_file is not None:
                src_align_indices, tgt_align_indices = next(alignment_generator)
                src_align_field = ArrayField(np.array(src_align_indices), padding_value=-1, dtype=np.int32)
                tgt_align_field = ArrayField(np.array(tgt_align_indices), padding_value=-1, dtype=np.int32)

                instance.add_field("source_alignment_indices", src_align_field)
                instance.add_field("target_alignment_indices", tgt_align_field)

                assert max(src_align_field.array) < len(instance["source_tokens"])
                assert max(tgt_align_field.array) < len(instance["target_tokens"])

            yield instance

    def _filter_out_this_instance(self, instance: Instance) -> bool:
        return self.filterers and any([filterer.to_filter(instance) for filterer in self.filterers])

    def text_to_instance(self, src_sent: str, tgt_sent: str = None) -> Instance:
        fields = {}

        source_tokens = self.src_tokenizer.tokenize(src_sent.strip())
        src_field = TextField(source_tokens, self.src_token_indexers)
        fields["source_tokens"] = src_field

        if tgt_sent:
            target_tokens = self.tgt_tokenizer.tokenize(tgt_sent.strip())
            tgt_field = TextField(target_tokens, self.tgt_token_indexers)
            fields["target_tokens"] = tgt_field

        return Instance(fields)


def parse_alignment_file(path: str) -> Tuple[List[int], List[int]]:
    with open(path, "r") as f:
        for line in f:
            alignments = line.strip().split()
            src, tgt = zip(*map(lambda x: x.split("-"), alignments))
            src = list(map(int, src))
            tgt = list(map(int, tgt))
            yield src, tgt
