from typing import Dict, List, Union, Tuple
import torch
import logging
from enum import Enum, auto

from allennlp.data.vocabulary import Vocabulary

PADDING_INDEX = 0

logger = logging.getLogger(__name__)


def get_target_mask(tensor: torch.Tensor, end_index: int, recursing: bool = False) -> torch.Tensor:
    """
    Mask tokens after the end_index.
    Note that this function leave the end index unmasked.
    """

    # Firstly, clone the tensor and fill this with 1 or 0 in place
    if not recursing:
        tensor = tensor.detach().clone()

    if tensor.dim() == 1:
        tensor_list = list(tensor)
        if end_index in tensor_list:
            target_i = tensor_list.index(end_index)
            target_i = min(target_i + 1, len(tensor_list))
            tensor[target_i:] = 0
            tensor[:target_i] = 1
        else:
            tensor[:] = 1
    else:
        for row in tensor:
            get_target_mask(row, end_index, recursing=True)

    return tensor


def tensor_to_string_tokens(
    token_tensor: torch.Tensor, vocab: Vocabulary, namespace: str, end_index: int
) -> List[List[str]]:
    """

    Parameters
    ----------
    token_tensor : torch.Tensor (batch_size, sequence_length)
    vocab : Vocabulary
    namespace : str
    end_index : int
    """
    tensor_mask = get_target_mask(token_tensor, end_index=end_index)

    predicted_indices = token_tensor.detach().cpu().numpy()
    tensor_mask = tensor_mask.detach().cpu().numpy()
    predicted_indices *= tensor_mask

    all_prediction_tensor: List[List[str]] = []
    for indices in predicted_indices:
        prediction_tensor = [
            vocab.get_token_from_index(idx, namespace=namespace) for idx in indices if (idx != 0 and idx != end_index)
        ]
        all_prediction_tensor.append(prediction_tensor)
    return all_prediction_tensor


def tensor_to_string_tokens_3d(token_tensor: torch.Tensor, vocab: Vocabulary, namespace: str, end_index: int):
    all_prediction_tensor: List[List[str]] = []
    for sentence_tensor in token_tensor:
        characters_list = tensor_to_string_tokens(sentence_tensor, vocab, namespace, end_index)
        words = ["".join(chars) for chars in characters_list]
        all_prediction_tensor.append(words)
    return all_prediction_tensor


class TrainingState(Enum):
    INIT = auto()
    TRAINING = auto()
    VALIDATION = auto()


class TrainingStateChangeDetector:
    def __init__(self):
        self.current_state = TrainingState.INIT

    def state_has_changed(self, new_state_is_training: bool):
        new_state = TrainingState.TRAINING if new_state_is_training else TrainingState.VALIDATION

        prev_state = self.current_state
        self.current_state = new_state

        return self.current_state != prev_state


def tensor2tokens(
    index_sequence: torch.Tensor, vocab: Vocabulary, name_space: str  # shape : (seq_len, )
) -> List[Union[str, List[str]]]:
    if index_sequence.dim() == 1:
        tokens = [vocab.get_token_from_index(idx.item(), name_space) for idx in index_sequence]
    elif index_sequence.dim() == 2:
        tokens = [
            [vocab.get_token_from_index(idx.item(), name_space) for idx in idx_list] for idx_list in index_sequence
        ]
    return tokens


def transfer_output_dict(
    final_output: Dict[str, torch.tensor], module_output_dict: Dict[str, torch.tensor], label: str
):
    for key, item in module_output_dict.items():
        final_output[f"{label}/{key}"] = item
    return final_output


def append_prefix_to_metrics_dict(metrics_dict: Dict[str, float], prefix: str) -> Dict[str, float]:
    output_metrics = {}
    for metric_name, metric_value in metrics_dict.items():
        module_metric_name = prefix + "/" + metric_name
        output_metrics[module_metric_name] = metric_value
    return output_metrics


def copy_tensor_dict(tensor_dict: Dict[str, torch.tensor]) -> Dict[str, torch.Tensor]:
    copied = {}
    for key, tensor in tensor_dict.items():
        copied[key] = tensor.clone()
    return copied


def merge_embedding_sequences(
    embeddings_1: torch.Tensor, mask_1: torch.BoolTensor, embeddings_2: torch.Tensor, mask_2: torch.BoolTensor
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    assert embeddings_1.size(0) == embeddings_2.size(0)
    assert embeddings_1.size(-1) == embeddings_2.size(-1)
    mask_1 = mask_1.bool()
    mask_2 = mask_2.bool()

    batch_size = mask_1.size(0)
    merged_embeddings = []

    for e1, e2, m1, m2 in zip(embeddings_1, embeddings_2, mask_1, mask_2):
        merged_embeddings.append(torch.cat([e1[m1], e2[m2]], dim=0))

    max_seqeuence_length = max([e.size(0) for e in merged_embeddings])
    new_mask = mask_1.new_zeros((batch_size, max_seqeuence_length))

    padded_embeddings = []
    for i in range(batch_size):
        emb = merged_embeddings[i]
        length = emb.size(0)
        new_mask[i, :length] = True

        padding_length = max_seqeuence_length - length
        padded_emb = torch.cat([emb, emb.new_zeros((padding_length, emb.size(1)))], dim=0)
        padded_embeddings.append(padded_emb)
    return torch.stack(padded_embeddings), new_mask
