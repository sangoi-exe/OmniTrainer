import torch
from torch import Tensor


def masked_losses(
        losses: Tensor,
        mask: Tensor,
        unmasked_weight: float,
        normalize_masked_area_loss: bool
) -> Tensor:
    clamped_mask = torch.clamp(mask, unmasked_weight, 1)

    losses *= clamped_mask

    if normalize_masked_area_loss:
        losses = losses / clamped_mask.mean(dim=(1, 2, 3), keepdim=True)

    del clamped_mask

    return losses

# sangoi_masked_loss - pra usar quando eu removo os gradientes da parte unmasked
def sangoi_masked_loss(
        losses: torch.Tensor,
        mask:   torch.Tensor,
        unmasked_weight: float = 0.1,
        normalize: bool = True,
) -> torch.Tensor:
    """
    Aplica peso 1.0 na região mascarada e `unmasked_weight` no fundo.
    Se `normalize=True`, reescala para que a perda represente densidade
    por pixel mascarado (gradiente não dilui).
    """
    # peso = 1 dentro da máscara, w fora
    weight = torch.where(mask > 0.5, 1.0, unmasked_weight)
    weighted = losses * weight

    if normalize:
        pix_masked = mask.gt(0.5).float().sum(dim=(1, 2, 3), keepdim=True).clamp_min(1)
        # reescala para manter magnitude original
        weighted = weighted * mask.numel() / pix_masked

    return weighted
