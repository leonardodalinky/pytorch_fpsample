from typing import Optional, Tuple, Literal

import torch


def sample(
    x: torch.Tensor,
    k: int,
    h: Optional[int] = None,
    start_idx: Optional[int] = None,
    backend: Literal["bucket", "naive"] = "bucket",
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Farthest Point Sampling (FPS) algorithm.

    Args:
        x (torch.Tensor): (*, N, D) input points tensor.
        k (int): Number of points to sample.
        h (int, optional): Maximum height for the bucket sampling.
            Only work for `backend="bucket"'. Defaults to None.
            See https://github.com/leonardodalinky/fpsample#usage for details.
        start_idx (int, optional): Index of the point to start sampling from. Defaults to None.
        backend (str, optional): Backend to use for sampling. Defaults to "bucket".
            Available options are: `bucket`, `naive`.

    Returns:
        (torch.Tensor, torch.LongTensor): (Batched) sampled points tensor and (batched) indices of the sampled points.
    """
    return torch.ops.torch_fpsample.sample(x, k, h, start_idx, backend)
