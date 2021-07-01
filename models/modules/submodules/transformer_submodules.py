from typing import Callable

import torch
import torch.nn as nn

import allennlp.nn.util as util
from allennlp.common import Registrable
from allennlp.nn import Activation

# from torch.nn.modules import MultiheadAttention as TorchMultiheadAttention
from .multi_head_attention import MultiheadAttention as TorchMultiheadAttention

Registrable._registry[Activation]["gelu"] = lambda: lambda x: torch.nn.functional.gelu(x)


class SublayerConnection(nn.Module, Registrable):
    """
     In the original paper, they apply LayerNorm at the end:
     LayerNorm(x + Sublayer(x)).

     In the tensor2tensor code, they suggest that learning is more robust
     when preprocessing each layer with LayerNorm:
     x + Sublayer(LayerNorm(x)).

     The default follows the more stable pre-norm regime.
    """

    def __init__(self, size: int, dropout: float = 0.0, pre_norm: bool = True, no_norm: bool = False):
        super().__init__()

        if no_norm:
            self.norm = lambda x: x
        else:
            self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor, sublayer: Callable):

        if self.pre_norm:
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            return self.norm(x + self.dropout(sublayer(x)))


class MultiHeadedAttention(nn.Module):
    """
    Wrapper for torch's MultiheadAttention.
    """

    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_size % num_heads == 0
        self.multi_attn = TorchMultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_value_mask: torch.BoolTensor = None,
        attention_mask: torch.BoolTensor = None,
    ):
        """
        Parameters
        ----------
        query : (batch_size, query_sequence_length, feature_size)
        key : (batch_size, key_value_sequence_length, feature_size)
        value : (batch_size, key_value_sequence_length, feature_size)
        key_value_mask : (batch_size, key_value_sequence_length)
        attention_mask : (key_value_sequence_length, key_value_sequence_length)

        Returns
        -------
        output : (batch_size, query_sequence_length, feature_size)
        """

        # pytorch's attn_mask is additive mask, i.e., add -inf to attention weights before softmax
        if attention_mask is not None:
            additive_attn_mask = torch.zeros_like(attention_mask).float()
            additive_attn_mask.masked_fill_(attention_mask == 0, float("-inf"))
        else:
            additive_attn_mask = None

        # pytorch's key_padding_mask applies mask where the value is 1
        if key_value_mask is not None:
            key_padding_mask = ~key_value_mask
        else:
            key_padding_mask = None

        # shape : (query_sequence_length, batch_size, feature_size)
        output, _ = self.multi_attn(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            key_padding_mask=key_padding_mask,
            attn_mask=additive_attn_mask,
        )

        output = output.transpose(0, 1)
        return output


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, model_size: int, hidden_dim: int, activation: str = "relu", dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(model_size, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, model_size)
        self.activation = Activation.by_name(activation)()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class PositionEmbedding(nn.Module):
    def __init__(self, learned: bool = False, embedding_size: int = None, max_sequence_length: int = 512):
        super().__init__()
        self.learned = learned

        if learned:
            assert embedding_size is not None
            self.embedding = nn.Embedding(max_sequence_length, embedding_size)

    def forward(self, embedding_sequence: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        embedding_sequence : (batch_size, sequence_length, embedding_size)

        Returns
        -------
        (batch_size, sequence_length, embedding_size)
        """

        if self.learned:

            batch_size, sequence_length, _ = embedding_sequence.size()

            position_indices = torch.arange(sequence_length).to(self.embedding.weight.device)
            position_embeddings = self.embedding(position_indices)
            position_embeddings = position_embeddings.unsqueeze(0).expand(batch_size, sequence_length, -1)
            return embedding_sequence + position_embeddings
        else:
            return util.add_positional_features(embedding_sequence)
