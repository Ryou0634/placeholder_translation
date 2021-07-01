from overrides import overrides
import torch

from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN

from allennlp.modules import TokenEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from models.modules.regularization.dropout import Dropout


@TokenEmbedder.register("embedding_layer")
class EmbeddingLayer(Embedding):
    """Add word dropout to the allennlp implementation."""

    def __init__(
        self,
        dropout: Dropout = None,
        word_dropout: float = 0.0,
        scale_by_embedding_dim: bool = False,
        vocab_namespace: str = "tokens",
        vocab: Vocabulary = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        dropout: float
            The probability of the dropout to the output embeddings.
        word_dropout: float
            The probability of randomly replacing the input tokens with the OOV token.
        scale_by_embedding_dim: bool
            Scale the output embedding by the square root of the embedding dim.
            Typically used in the Transformer model.
        """
        super().__init__(vocab_namespace=vocab_namespace, vocab=vocab, **kwargs)

        self.word_dropout = word_dropout
        self.word_dropout_index = vocab.get_token_index(DEFAULT_OOV_TOKEN, vocab_namespace)
        self.dropout = dropout

        self.scale_by_embedding_dim = scale_by_embedding_dim
        self.scale_factor = self.output_dim ** 0.5

    @overrides
    def forward(self, inputs):  # pylint: disable=arguments-differ

        if self.training and self.word_dropout:
            mask = torch.empty_like(inputs).bernoulli_(1 - self.word_dropout).bool()
            dropout_index = inputs.new_full(size=(1,), fill_value=self.word_dropout_index)
            inputs = torch.where(mask, inputs, dropout_index)
        embeddings = super().forward(inputs)

        if self.scale_by_embedding_dim:
            embeddings = self.scale_factor * embeddings

        if self.dropout is not None:
            embeddings = self.dropout(embeddings)

        return embeddings
