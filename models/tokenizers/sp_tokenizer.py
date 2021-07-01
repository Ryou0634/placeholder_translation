from typing import List, Iterable
from overrides import overrides
import os
from pathlib import Path
import re
import tempfile

from allennlp.data import Token
from allennlp.data.tokenizers import Tokenizer
from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN
from allennlp.common.util import START_SYMBOL, END_SYMBOL
import sentencepiece as spm
import models.tokenizers.sentencepiece_pb2 as sentencepiece_pb2

DEFAULT_USER_DEFINED_SYMBOLS = [START_SYMBOL, END_SYMBOL, DEFAULT_OOV_TOKEN]


@Tokenizer.register("sentence_piece")
class SentencePieceTokenizer(Tokenizer):
    def __init__(
        self,
        model_path: str = None,
        subword_regularization_sample_size: int = 0,
        subword_regularization_alpha: float = 0.2,
        tokenize_with_offsets: bool = False,
    ) -> None:

        self._subword_regularization_sample_size = subword_regularization_sample_size
        self._subword_regularization_alpha = subword_regularization_alpha

        if model_path:
            if model_path[-6:] != ".model":
                raise ConfigurationError(
                    "``model_path`` must end with '.model' because it is the default suffix of"
                    f"trained models of sentence-piece. You gave {model_path}"
                )
            self._processor = spm.SentencePieceProcessor()
            self._processor.Load(model_path)
        else:
            self._processor = None

        self.tokenize_with_offsets = tokenize_with_offsets

    def _tokenize_with_offsets(self, text: str) -> List[Token]:
        spt = sentencepiece_pb2.SentencePieceText()
        spt.ParseFromString(self._processor.encode_as_serialized_proto(text))
        tokens = [Token(piece.piece, idx=piece.begin, idx_end=piece.end) for piece in spt.pieces]

        def get_byte_len(byte_text: str, idx: int):
            return len(byte_text[:idx].decode("utf-8", errors="replace"))

        byte_text = text.encode("utf-8")
        for t in tokens:
            t.idx = get_byte_len(byte_text, t.idx)
            t.idx_end = get_byte_len(byte_text, t.idx_end)
        return tokens

    @overrides
    def tokenize(self, text: str) -> List[Token]:

        if self._processor is None:
            raise Exception("it seems you did't specify model_path in initialization")

        if self.tokenize_with_offsets:
            return self._tokenize_with_offsets(text)

        if self._subword_regularization_sample_size == 0:
            pieces = self._processor.EncodeAsPieces(text)
        else:
            pieces = self._processor.SampleEncodeAsPieces(
                text, self._subword_regularization_sample_size, self._subword_regularization_alpha
            )
        return [Token(piece) for piece in pieces]

    @staticmethod
    def detokenize(tokens: List[str]):
        detokenized = "".join(tokens).replace("▁", " ").strip()
        return detokenized


def train_sentencepiece_from_iterable(
    sentences: Iterable[str],
    save_path: str,
    vocab_size: int,
    normalization_rule_name: str = "nmt_nfkc",
    character_coverage: float = 0.9995,
    user_defined_symbols: List[str] = None,
):
    with tempfile.NamedTemporaryFile(mode="w", prefix="sp_", suffix=".txt") as tmp_f:
        for s in sentences:
            tmp_f.write(s + "\n")
        tmp_f.write("\n")
        tmp_f.flush()

        train_sentencepiece_from_plaintext_file(
            tmp_f.name,
            save_path=save_path,
            vocab_size=vocab_size,
            normalization_rule_name=normalization_rule_name,
            character_coverage=character_coverage,
            user_defined_symbols=user_defined_symbols,
        )


def train_sentencepiece_from_plaintext_file(
    train_file_path: str,
    save_path: str,
    vocab_size: int,
    normalization_rule_name: str = "nmt_nfkc",
    character_coverage: float = 0.9995,
    user_defined_symbols: List[str] = None,
):
    user_defined_symbols = user_defined_symbols or DEFAULT_USER_DEFINED_SYMBOLS
    user_defined_symbols = ",".join(user_defined_symbols)

    save_dir, _ = os.path.split(save_path)
    os.makedirs(save_dir, exist_ok=True)
    model_prefix = re.sub(".model$", "", save_path)
    spm.SentencePieceTrainer.Train(
        f"--bos_id=-1 --eos_id=-1 "
        f"--user_defined_symbols={user_defined_symbols} "
        f"--input={train_file_path} --model_prefix={model_prefix} --vocab_size={vocab_size} "
        f"--character_coverage={character_coverage} "
        f"--normalization_rule_name={normalization_rule_name} "
        f"--input_sentence_size=80000000"
    )


def sentence_piece_vocab_to_allennlp(
    vocab_out_dir: str, sp_vocab_paths: List[str], namespaces: List[str], non_padded_namespaces: List[str] = None
):
    vocab_out_dir = Path(vocab_out_dir)
    vocab_out_dir.mkdir(exist_ok=True, parents=True)

    for sp_vocab_path, namespace in zip(sp_vocab_paths, namespaces):
        # read sp vocab and write to allennlp
        with open(sp_vocab_path, "r") as f:
            words = [w.split("\t")[0] for w in f]

        saved_file = vocab_out_dir / (namespace + ".txt")
        with open(saved_file, "w") as f:
            f.write("\n".join(words))

        print("Created vocab file: ", saved_file)

    # create_non_padded_namespace
    non_padded_namespaces = non_padded_namespaces or ["*labels", "*tags"]
    with open(vocab_out_dir / "non_padded_namespaces.txt", "w") as f:
        f.write("\n".join(non_padded_namespaces))
