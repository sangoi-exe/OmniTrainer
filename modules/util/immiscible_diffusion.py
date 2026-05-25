import torch
from torch import Tensor


def immiscible_oversampling(source_tensor: Tensor, noise_candidates: Tensor) -> Tensor:
    if source_tensor.ndim < 2:
        raise ValueError("source_tensor must have at least batch and channel dimensions")
    if noise_candidates.ndim != source_tensor.ndim + 1:
        raise ValueError("noise_candidates must have shape [batch, candidates, *source_shape[1:]]")
    if noise_candidates.shape[0] != source_tensor.shape[0]:
        raise ValueError("noise candidate batch size must match source batch size")
    if noise_candidates.shape[2:] != source_tensor.shape[1:]:
        raise ValueError("noise candidate sample shape must match source sample shape")

    source_points = source_tensor.flatten(start_dim=1).to(dtype=torch.float32)
    noise_points = noise_candidates.flatten(start_dim=2).to(dtype=torch.float32)

    distance = torch.cdist(source_points.unsqueeze(1), noise_points, p=2).squeeze(1)
    nearest_index = torch.argmin(distance, dim=1)

    gather_shape = [-1, 1] + [source_tensor.shape[index] for index in range(1, source_tensor.ndim)]
    gather_index = nearest_index.view(-1, *([1] * source_tensor.ndim)).expand(*gather_shape)
    return torch.gather(noise_candidates, 1, gather_index).squeeze(1)
