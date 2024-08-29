from typing import Optional

import torch


def sample(
    x: torch.Tensor,
    k: int,
    h: Optional[int] = None,
    start_idx: Optional[int] = None,
    backend: str = "bucket",
):
    return torch.ops.torch_fpsample.sample(x, k, h, start_idx, backend)
