from typing import Dict

import torch
import torch.nn as nn

from models.modules.submodules.transformer_submodules import (
    MultiHeadedAttention,
    PositionwiseFeedForward,
    SublayerConnection,
    PositionEmbedding,
)
from models.modules.submodules.utils import clone_modules
from .sequence_encoder_base import SequenceEncoder


class EncoderLayer(nn.Module):
    """TransformerEncoder is made up of self-attn and feed forward (defined below)"""

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        sub_layer_connection: SublayerConnection,
    ):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clone_modules(sub_layer_connection, 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, key_value_mask=mask))
        return self.sublayer[1](x, self.feed_forward)


@SequenceEncoder.register("transformer_encoder")
class TransformerEncoder(SequenceEncoder):
    """Core encoder is a stack of num_layers layers"""

    def __init__(
        self,
        size: int,
        num_attention_heads: int,
        feedforward_hidden_dim: int,
        num_layers: int,
        use_position_encoding: bool = True,
        use_learned_positional_embedding: bool = False,
        output_hidden_state: bool = False,
        activation: str = "relu",
        pre_norm: bool = True,
        no_norm: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        sub_layer_connection = SublayerConnection(size=size, dropout=dropout, pre_norm=pre_norm, no_norm=no_norm)

        layer = EncoderLayer(
            size,
            MultiHeadedAttention(embed_size=size, num_heads=num_attention_heads, dropout=dropout),
            PositionwiseFeedForward(
                model_size=size, hidden_dim=feedforward_hidden_dim, dropout=dropout, activation=activation
            ),
            sub_layer_connection,
        )

        self.layers = clone_modules(layer, num_layers)
        self.size = size
        self.use_position_encoding = use_position_encoding
        if use_position_encoding:
            self.position_embedding = PositionEmbedding(learned=use_learned_positional_embedding, embedding_size=size)
        self.output_hidden_state = output_hidden_state

        self.no_norm = no_norm
        self.pre_norm = pre_norm
        if self.pre_norm and not self.no_norm:
            self.output_layer_norm = nn.LayerNorm(size)

    def get_input_dim(self):
        return self.size

    def get_output_dim(self):
        return self.size

    def forward(self, embedding_sequence: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Pass the input (and mask) through each layer in turn."""

        if self.use_position_encoding:
            embedding_sequence = self.position_embedding(embedding_sequence)

        output = embedding_sequence
        for layer in self.layers:
            output = layer(output, mask)

        if self.pre_norm and not self.no_norm:
            # We need to add an additional function of layer normalization to the top layer
            # to prevent the excessively increased value caused by the sum of unnormalized output
            # (https://arxiv.org/pdf/1906.01787.pdf)
            output = self.output_layer_norm(output)

        output_dict = {"encoder_output": output, "source_mask": mask}

        if self.output_hidden_state:
            masked_output = output * (mask[:, :, None]).type(torch.float)
            output_dict["hiddens"] = masked_output.mean(dim=1)

        return output_dict
