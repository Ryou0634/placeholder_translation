import torch
import torch.nn as nn
import copy


def clone_modules(module: torch.nn.Module, n: int):
    """Produce num_layers identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
