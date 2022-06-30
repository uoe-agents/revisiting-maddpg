import torch
from typing import Optional, Any

Array = Any

# TODO: Decide whethere 'hard' switch is necessary. By removing control flow, we can jit
#@jit
def gumbel_softmax_st(logits: Array, temperature: float = 1, dim: int = -1) -> Array:
    """
        Straight-through Gumbel-Softmax Estimator
        See https://github.com/pytorch/pytorch/blob/32593ef2dd26e32ed44d3c03d3f5de4a42eb149a/torch/nn/functional.py#L1797
    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    return y_hard - y_soft.detach() + y_soft