from typing import Dict, Union
from overrides import overrides
import logging

import torch

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.modules import TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import Metric

from models.modules.sequence_encoders import SequenceEncoder
from models.modules.sequence_decoders import SequenceDecoder

from .sequence_generator import SequenceGenerator
from .modules.sequence_cross_entropy import SequenceCrossEntropy


logger = logging.getLogger(__name__)


def check_share_embedder_weights(source_embedder: TextFieldEmbedder, target_embedder: TextFieldEmbedder):
    common_embedder_names = set(source_embedder._token_embedders.keys()) & set(target_embedder._token_embedders.keys())
    for common_key in common_embedder_names:
        source_embedding = source_embedder._token_embedders[common_key]
        target_embedding = target_embedder._token_embedders[common_key]
        if hasattr(source_embedding, "weight") and target_embedding.weight.shape != source_embedding.weight.shape:
            raise Exception(
                "source_embedding and target_embedding need to have the same shape\n"
                f"{source_embedding.weight.shape} != {target_embedding.weight.shape}"
            )


@Model.register("seq2seq")
@SequenceGenerator.register("seq2seq")
class Seq2Seq(SequenceGenerator):
    """
    Simple seq2seq model.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        source_embedder: TextFieldEmbedder,
        target_embedder: TextFieldEmbedder,
        encoder: SequenceEncoder,
        decoder: SequenceDecoder,
        sequence_loss: SequenceCrossEntropy = SequenceCrossEntropy(),
        source_vocab_namespace: str = "source_tokens",
        target_vocab_namespace: str = "target_tokens",
        target_token_namespace: str = "tokens",
        share_embeddings: bool = False,
        share_target_weights: bool = False,
        max_decoding_length: int = 100,
        beam_size: int = None,
        token_metrics: Dict[str, Metric] = None,
        initializer: InitializerApplicator = None,
    ):

        super().__init__(
            vocab,
            target_embedder=target_embedder,
            decoder=decoder,
            sequence_loss=sequence_loss,
            target_vocab_namespace=target_vocab_namespace,
            target_token_namespace=target_token_namespace,
            share_target_weights=share_target_weights,
            max_decoding_length=max_decoding_length,
            beam_size=beam_size,
            token_metrics=token_metrics,
        )

        self.source_embedder = source_embedder
        self.encoder = encoder
        self.source_vocab_namespace = source_vocab_namespace

        if share_embeddings:
            if source_vocab_namespace != target_vocab_namespace:
                raise Exception(
                    "share_embeddings requires a shared vocabulary\n"
                    f"source_vocab_namespace: {source_vocab_namespace}, target_vocab_namespace: {target_vocab_namespace}"
                )
            check_share_embedder_weights(self.source_embedder, self.target_embedder)
            self.target_embedder = self.source_embedder

        if initializer:
            initializer(self)

    def encode(self, source_tokens: TextFieldTensors) -> Dict[str, torch.Tensor]:
        embedded_input = self.source_embedder(source_tokens)
        source_mask = util.get_text_field_mask(source_tokens)
        encoder_output = self.encoder(embedded_input, source_mask)
        return encoder_output

    @overrides
    def forward(
        self,
        source_tokens: TextFieldTensors,
        target_tokens: TextFieldTensors = None,
        start_tokens: Union[int, torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        encoder_output = self.encode(source_tokens)
        batch_size = encoder_output["encoder_output"].size(0)

        return super().forward(
            target_tokens=target_tokens,
            start_tokens=start_tokens,
            context=encoder_output,
            batch_size=batch_size,
            **kwargs,
        )
