import torch
from torch import Tensor
from collections import deque
from typing import Tuple, List, Dict

CUDA_DEV = torch.device("cuda")

class LossTracker:
    """
    Tracks and generates statistics using a pre-allocated circular buffer for efficiency.
    """
    def __init__(self, window_size: int = 500, use_mad: bool = True) -> None:
        self.window_size: int = window_size
        self.use_mad: bool = use_mad
        self.device = CUDA_DEV # ou inferir de um tensor na primeira chamada

        # Pré-aloca os buffers. Cada buffer armazena 'window_size' escalares.
        self.mse_buffer = torch.zeros(window_size, device=self.device)
        self.mae_buffer = torch.zeros(window_size, device=self.device)
        self.log_cosh_buffer = torch.zeros(window_size, device=self.device)

        self.next_idx = 0
        self.is_full = False

    def update(self, mse_loss: Tensor, mae_loss: Tensor, log_cosh_loss: Tensor) -> None:
        # Insere o novo valor no buffer na posição atual
        # .detach() é crucial para não guardar o histórico do grafo
        self.mse_buffer[self.next_idx] = mse_loss.detach()
        self.mae_buffer[self.next_idx] = mae_loss.detach()
        self.log_cosh_buffer[self.next_idx] = log_cosh_loss.detach()

        self.next_idx += 1
        if self.next_idx >= self.window_size:
            self.next_idx = 0
            self.is_full = True

    def _get_current_values(self, buffer: Tensor) -> Tensor:
        # Retorna apenas os valores preenchidos do buffer
        if self.is_full:
            return buffer
        return buffer[:self.next_idx]

    def compute_stats(self, buffer: Tensor) -> Tuple[Tensor, Tensor]:
        values = self._get_current_values(buffer)
        if values.numel() == 0:
            return torch.zeros(1, device=self.device), torch.tensor(1e-8, device=self.device)

        if self.use_mad:
            med = torch.median(values)
            mad = torch.median(torch.abs(values - med)).clamp_min(1e-8)
            return med, mad
        else:
            mean = torch.mean(values)
            std = torch.std(values, unbiased=False).clamp_min(1e-8)
            return mean, std

    @torch.no_grad()
    def compute_z_scores(self, mse_loss: Tensor, mae_loss: Tensor, log_cosh_loss: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Passa o buffer inteiro para compute_stats
        mse_c, mse_s = self.compute_stats(self.mse_buffer)
        mae_c, mae_s = self.compute_stats(self.mae_buffer)
        log_c, log_s = self.compute_stats(self.log_cosh_buffer)

        mse_z = (mse_loss - mse_c) / mse_s
        mae_z = (mae_loss - mae_c) / mae_s
        log_z = (log_cosh_loss - log_c) / log_s
        return mse_z, mae_z, log_z


class DynamicLossControl:
    """
    Dynamically adjusts the weights of different loss components based on their z-scores.
    It can optionally use Exponential Moving Average (EMA) and applies a scheduling mechanism
    to prioritize different losses over the course of training.
    """

    def __init__(
        self,
        use_ema: bool = True,
        ema_decay: float = 0.7,
        outlier_threshold: float = 3.0,
        schedule_params: Dict[str, Dict[str, float]] = None,
    ) -> None:
        """
        Initializes the DynamicLossControl.

        Args:
            use_ema (bool): Whether to use Exponential Moving Average for weights.
            ema_decay (float): Decay rate for EMA.
            outlier_threshold (float): Threshold to clamp z-scores.
            schedule_params (Dict[str, Dict[str, float]]): Scheduling parameters for each loss.
        """
        self.use_ema: bool = use_ema
        self.ema_decay: float = ema_decay
        self.outlier_threshold: float = outlier_threshold
        self.progress = None # // CORREÇÃO: Inicializar progress, será setado externamente
        self.last_logged_delta_epoch = -1 # // CORREÇÃO: Inicializar last_logged_delta_epoch
        self.initialized: bool = False

        self.ema_weights: torch.Tensor = torch.ones(3, dtype=torch.float32, device="cuda")
        self.loss_names: Tuple[str, ...] = ("mse", "mae", "log_cosh")
        self.schedule_params = self._initialize_schedule_params(schedule_params)

        # cria buffers a partir do dicionário final
        self.sched_start = torch.tensor(
            [self.schedule_params[n]["start"] for n in ("mse", "mae", "log_cosh")],
            dtype=torch.float32, device="cuda")
        self.sched_end = torch.tensor(
            [self.schedule_params[n]["end"] for n in ("mse", "mae", "log_cosh")],
            dtype=torch.float32, device="cuda")   
        

    def _initialize_schedule_params(
        self, schedule_params: Dict[str, Dict[str, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        default_params = {
            "mae": {"start": 0.6, "end": 0.0},
            "mse": {"start": 0.2, "end": 0.6},
            "log_cosh": {"start": 0.2, "end": 0.4},
        }
        if schedule_params is not None:
            for loss_type, params in default_params.items():
                if loss_type in schedule_params:
                    default_params[loss_type].update(schedule_params[loss_type])
        return default_params

    @torch.no_grad()
    def adjust_weights(
        self,
        mse_z: Tensor,
        mae_z: Tensor,
        log_cosh_z: Tensor,
        config,
        progress,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        eps = 1e-8

        # START vectorized_adjust
        loss_vec = torch.stack([mse_z, mae_z, log_cosh_z], dim=0)
        abs_z    = loss_vec.abs().clamp_max(self.outlier_threshold)
        total_z  = abs_z.sum(dim=0, keepdim=True) + eps
        inv_z    = (total_z - abs_z) / total_z
        weights  = inv_z / inv_z.sum(dim=0, keepdim=True)

        # --- EMA inline ---
        if self.use_ema:
            if weights.ndim > 1:
                batch_mean = weights.mean(dim=tuple(range(1, weights.ndim)))   # [3]
            else:  # já é [3]
                batch_mean = weights
            self.ema_weights.mul_(1 - self.ema_decay).add_(batch_mean, alpha=self.ema_decay)
            weights = self.ema_weights.view(3, *[1]*(weights.ndim-1))

        # --- Scheduler inline ---
        frac = progress.epoch / (config.epochs - 1.0) if config.epochs > 1 else 0.0
        sched_f = self.sched_start.lerp(self.sched_end, frac).view(3, *[1]*(weights.ndim-1))
        weights.mul_(sched_f)
        weights.div_(weights.sum(dim=0, keepdim=True) + eps)

        w_mse, w_mae, w_log = weights[0], weights[1], weights[2]
        return w_mse, w_mae, w_log