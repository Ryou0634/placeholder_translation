import torch

import torch.nn as nn

from allennlp.common.registrable import Registrable


class Dropout(nn.Module, Registrable):

    default_implementation = "defalut"

    def forward(self, x: torch.Tensor):
        raise NotImplementedError


@Dropout.register("defalut")
class DefaultDropout(Dropout):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor):
        return self.dropout(x)


@Dropout.register("locked")
class LockedDropout(Dropout):
    def __init__(self, locked_dim: int, p: float = 0.0):
        super().__init__()
        self._locked_dim = locked_dim
        assert 0 <= p < 1.0
        self._p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not self.training or not self._p:
            return x

        tensor_size = list(x.size())
        tensor_size[self._locked_dim] = 1

        mask = x.new_empty(tensor_size).bernoulli_(1 - self._p).requires_grad_(False)

        mask = mask / (1 - self._p)
        mask = mask.expand_as(x)

        return mask * x
