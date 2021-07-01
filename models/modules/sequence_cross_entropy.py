from typing import Union, Dict
import torch

from allennlp.common.from_params import FromParams
from allennlp.nn import util


class SequenceCrossEntropy(FromParams):
    def __init__(
        self, label_smoothing: float = 0.0,
    ):
        self.label_smoothing = label_smoothing

    def __call__(
        self, logits: torch.FloatTensor, targets: torch.LongTensor, weights: Union[torch.FloatTensor, torch.BoolTensor]
    ) -> Dict[str, torch.Tensor]:
        per_batch_loss = util.sequence_cross_entropy_with_logits(
            logits=logits,
            targets=targets,
            weights=weights,
            average=None,
            label_smoothing=self.label_smoothing
        )

        loss = per_batch_loss.mean()

        return {
            "per_batch_loss": per_batch_loss,
            "loss": loss
        }
