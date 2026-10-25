"""Small shared helpers."""

import numpy as np
import torch

# Log-weight of one false grounding of a hard rule (or one atom that contradicts the evidence).
# Finite, so that log-sum-exps and gradients stay well defined; exp(-HARD_WEIGHT) is 0 in float32.
HARD_WEIGHT = 1e4


def batched_gather(params: torch.Tensor, indices: torch.Tensor, batch_dims: int) -> torch.Tensor:
    """tf.gather(params, indices, batch_dims=batch_dims): gather along the first non-batch dimension of params,
    separately for every index of the leading `batch_dims` dimensions."""
    if list(indices.shape[:batch_dims]) != list(params.shape[:batch_dims]):
        raise ValueError(f"batch dimensions differ: {tuple(indices.shape)} vs {tuple(params.shape)}")
    batch_shape = list(params.shape[:batch_dims])
    batch = int(np.prod(batch_shape))
    rows = torch.arange(batch, device=params.device).view(-1, 1)
    out = params.reshape(batch, *params.shape[batch_dims:])[rows, indices.reshape(batch, -1)]
    return out.reshape(batch_shape + list(indices.shape[batch_dims:]) + list(params.shape[batch_dims + 1:]))
