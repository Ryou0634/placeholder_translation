from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn

from models.modules.submodules.transformer_submodules import (
    MultiHeadedAttention,
    PositionwiseFeedForward,
    SublayerConnection,
    PositionEmbedding,
)
from models.modules.submodules.utils import clone_modules

from .sequence_decoder_base import SequenceDecoder


class TransformerDecoderLayer(nn.Module):
    """TransformerDecoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        src_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        sub_layer_connection: SublayerConnection,
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clone_modules(sub_layer_connection, 3)

    def forward(
        self,
        embedding_sequence: torch.Tensor,
        encoder_output: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor = None,
        apply_target_subsequent_mask: bool = True,
    ):
        """
        Follow Figure 1 (right) for connections.

        Parameters
        ----------
        embedding_sequence : torch.Tensor (batch_size, target_sequence_length, feature_size)
        encoder_output : torch.Tensor (batch_size, source_sequence_length, feature_size)
        source_mask : torch.Tensor (batch_size, source_sequence_length)
        target_mask : torch.Tensor (batch_size, target_sequence_length)
        apply_target_subsequent_mask: bool

        Returns
        -------

        """

        if apply_target_subsequent_mask:
            # shape : (target_sequence_length, target_sequence_length)
            target_subsequent_mask = make_subsequent_mask(embedding_sequence)
        else:
            target_subsequent_mask = None

        embedding_sequence = self.sublayer[0](
            embedding_sequence,
            lambda x: self.self_attn(x, x, x, key_value_mask=target_mask, attention_mask=target_subsequent_mask),
        )
        embedding_sequence = self.sublayer[1](
            embedding_sequence, lambda x: self.src_attn(x, encoder_output, encoder_output, key_value_mask=source_mask)
        )
        return self.sublayer[2](embedding_sequence, self.feed_forward)


@SequenceDecoder.register("transformer_decoder")
class TransformerDecoder(SequenceDecoder):
    """Generic num_layers layer decoder with masking."""

    def __init__(
        self,
        size: int,
        num_attention_heads: int,
        feedforward_hidden_dim: int,
        num_layers: int,
        use_position_encoding: bool = True,
        use_learned_positional_embedding: bool = False,
        activation: str = "relu",
        pre_norm: bool = True,
        no_norm: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        layer = self._get_layer(
            size, num_attention_heads, dropout, feedforward_hidden_dim, activation, pre_norm, no_norm
        )

        self.layers = clone_modules(layer, num_layers)
        self.size = size
        self.use_position_encoding = use_position_encoding
        if use_position_encoding:
            self.position_embedding = PositionEmbedding(learned=use_learned_positional_embedding, embedding_size=size)

        self.no_norm = no_norm
        self.pre_norm = pre_norm
        if self.pre_norm and not self.no_norm:
            self.output_layer_norm = nn.LayerNorm(size)

    @staticmethod
    def _get_layer(size, num_attention_heads, dropout, feedforward_hidden_dim, activation, pre_norm, no_norm):
        # to make it easy to be overridden by SelfAttentionDecoder
        sub_layer_connection = SublayerConnection(size=size, dropout=dropout, pre_norm=pre_norm, no_norm=no_norm)
        return TransformerDecoderLayer(
            size=size,
            self_attn=MultiHeadedAttention(num_heads=num_attention_heads, embed_size=size, dropout=dropout),
            src_attn=MultiHeadedAttention(num_heads=num_attention_heads, embed_size=size, dropout=dropout),
            feed_forward=PositionwiseFeedForward(
                model_size=size, hidden_dim=feedforward_hidden_dim, activation=activation, dropout=dropout
            ),
            sub_layer_connection=sub_layer_connection,
        )

    def get_input_dim(self):
        return self.size

    def get_output_dim(self):
        return self.size

    def _forward(
        self,
        embedding_sequence: torch.Tensor,
        state: Dict[str, torch.Tensor] = None,
        apply_target_subsequent_mask: bool = True,
    ) -> torch.Tensor:

        if self.use_position_encoding:
            embedding_sequence = self.position_embedding(embedding_sequence)

        output = embedding_sequence
        for layer in self.layers:
            output = layer(
                output,
                state["encoder_output"],
                state["source_mask"],
                state["target_mask"] if "target_mask" in state else None,
                apply_target_subsequent_mask=apply_target_subsequent_mask,
            )

        if self.pre_norm and not self.no_norm:
            # We need to add an additional function of layer normalization to the top layer
            # to prevent the excessively increased value caused by the sum of unnormalized output
            # (https://arxiv.org/pdf/1906.01787.pdf)
            output = self.output_layer_norm(output)

        return output

    def forward_across_time_steps(
        self,
        embedding_sequence: torch.Tensor,
        state: Dict[str, torch.Tensor] = None,
        apply_target_subsequent_mask: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        outputs = self._forward(
            embedding_sequence, state=state, apply_target_subsequent_mask=apply_target_subsequent_mask
        )
        return outputs, state

    def forward(
        self, embedding: torch.Tensor, state: Dict[str, torch.Tensor], apply_target_subsequent_mask: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters :
        embedding : torch.Tensor (batch_size, embedding_size)
            A batch of embedded inputs.
        state : Dict[str, torch.Tensor]
            Typically contains context vectors from encoder, hidden states of RNN.
        """
        if "input_history" not in state:
            state["input_history"] = embedding.unsqueeze(1)
        else:
            state["input_history"] = torch.cat([state["input_history"], embedding.unsqueeze(1)], dim=1)

        embedding_sequence = state["input_history"]
        outputs = self._forward(
            embedding_sequence, state=state, apply_target_subsequent_mask=apply_target_subsequent_mask
        )

        return outputs[:, -1], state


def make_subsequent_mask(embedding_sequence: torch.Tensor) -> torch.Tensor:
    """
    Create a mask to hide padding and future words.

    Parameters
    ----------
    embedding_sequence : torch.Tensor (batch_size, sequence_length, embedding_size)
    """
    sequence_length = embedding_sequence.size(1)
    attn_shape = (sequence_length, sequence_length)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    subsequent_mask = torch.from_numpy(subsequent_mask) == 0
    subsequent_mask = subsequent_mask.byte()
    subsequent_mask = subsequent_mask.to(embedding_sequence.device)
    return subsequent_mask
