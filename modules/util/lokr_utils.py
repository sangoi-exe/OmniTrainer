import torch


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    if factor > 0 and dimension % factor == 0:
        return factor, dimension // factor
    if factor == -1:
        factor = dimension

    factor_1, factor_2 = 1, dimension
    best_length = factor_1 + factor_2
    while factor_1 < factor_2:
        candidate_factor_1 = factor_1 + 1
        while dimension % candidate_factor_1 != 0:
            candidate_factor_1 += 1
        candidate_factor_2 = dimension // candidate_factor_1
        if candidate_factor_1 + candidate_factor_2 > best_length or candidate_factor_1 > factor:
            break
        factor_1, factor_2 = candidate_factor_1, candidate_factor_2

    if factor_1 > factor_2:
        factor_1, factor_2 = factor_2, factor_1
    return factor_1, factor_2


def make_kron(weight_1: torch.Tensor, weight_2: torch.Tensor) -> torch.Tensor:
    if len(weight_2.shape) == 4:
        weight_1 = weight_1.unsqueeze(2).unsqueeze(2)
    return torch.kron(weight_1, weight_2.contiguous())


def rebuild_tucker(tucker_tensor: torch.Tensor, weight_a: torch.Tensor, weight_b: torch.Tensor) -> torch.Tensor:
    return torch.einsum("i j k l, i p, j r -> p r k l", tucker_tensor, weight_a, weight_b)
