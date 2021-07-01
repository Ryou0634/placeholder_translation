from typing import Dict
import torch

from allennlp.common import Registrable


class SequenceEncoder(torch.nn.Module, Registrable):

    def get_output_dim(self) -> int:
        raise NotImplementedError

    def get_input_dim(self) -> int:
        raise NotImplementedError

    def forward(self,
                embedding_sequence: torch.Tensor,
                mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Parameters :
        embedding_sequence : torch.tensor (batch_size, sequence_length, embedding_size)
            A batch of embedded sequence inputs.
        mask : torch.LongTensor (batch, sequence_length)
            A mask with 0 where the tokens are padding, and 1 otherwise.

        Returns :
        Dict{
            encoder_output : torch.tensor (batch, max_seq_len, hidden_size)
            source_mask : torch.LongTensor (batch, max_seq_len)
        }
        """
        raise NotImplementedError
