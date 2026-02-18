"""
Message Aggregator Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""


import torch
from torch import Tensor
from torch_geometric.utils import scatter

try:
    from torch_scatter import scatter_max as _scatter_max
except ImportError:  # pragma: no cover - exercised only when torch_scatter is absent
    _scatter_max = None


def _scatter_argmax_fallback(t: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """
    Fallback argmax-by-index used when torch_scatter is unavailable.
    Returns argmax positions in `t` per segment in `index`.
    """
    positions = torch.arange(t.size(0), device=t.device, dtype=torch.long)
    max_t = scatter(t, index, dim=0, dim_size=dim_size, reduce="max")

    # Keep candidate positions where t equals the segment max; choose the latest.
    is_seg_max = t == max_t[index]
    masked_pos = torch.where(is_seg_max, positions, torch.full_like(positions, -1))
    argmax = scatter(masked_pos, index, dim=0, dim_size=dim_size, reduce="max")

    # Set empty segments to sentinel value msg_len, matching torch_scatter behavior usage below.
    counts = scatter(
        torch.ones_like(index, dtype=torch.long),
        index,
        dim=0,
        dim_size=dim_size,
        reduce="sum",
    )
    argmax = argmax.clone()
    argmax[counts == 0] = t.size(0)
    return argmax


class LastAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        if _scatter_max is not None:
            _, argmax = _scatter_max(t, index, dim=0, dim_size=dim_size)
        else:
            argmax = _scatter_argmax_fallback(t=t, index=index, dim_size=dim_size)
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
        return out


class MeanAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce="mean")
