import torch

from torch import Tensor
from collections import deque

from typing import Tuple, List, Dict, Union

class LossTracker:
    """
    Tracks and generates statistics for the latest N loss values for MSE, MAE, and log-cosh.
    It can use mean/std or median/MAD for statistics.
    """

    def __init__(self, window_size: int = 100, use_mad: bool = True) -> None:
        """
        Initializes the LossTracker.

        Args:
            window_size (int): The number of recent loss values to track.
            use_mad (bool): If True, use median and MAD instead of mean and std.
        """
        self.window_size: int = window_size
        self.use_mad: bool = use_mad

        self.mse_losses: deque = deque(maxlen=window_size)
        self.mae_losses: deque = deque(maxlen=window_size)
        self.log_cosh_losses: deque = deque(maxlen=window_size)

        self.sum, self.sum2, self.count = 0.0, 0.0, 0

    def update_stats(self, x):
        self.sum  += x
        self.sum2 += x*x
        self.count = min(self.count+1, self.window_size)

    def mean_var(self):
        mean = self.sum / max(self.count, 1)
        var  = self.sum2 / max(self.count, 1) - mean*mean
        return mean, max(var, 1e-16)**0.5

    def _get_defaults(self) -> Tuple[Tensor, Tensor]:
        """Retorna valores padrão quando não há dados suficientes"""
        return torch.tensor(0.0), torch.tensor(1.0)

    def _compute_robust_stats(self, arr: Tensor) -> Tuple[Tensor, Tensor]:
        """Computa estatísticas robustas (mediana/MAD ou média/std)"""
        if self.use_mad:
            # Mediana e MAD (Median Absolute Deviation)
            median = torch.median(arr)
            mad = torch.median(torch.abs(arr - median))
            return median, torch.clamp(mad * 1.4826, min=1e-8)  # Fator de escala para MAD
        else:
            # Média e desvio padrão
            mean = torch.mean(arr)
            std = torch.std(arr, unbiased=False)
            return mean, torch.clamp(std, min=1e-8)

    def update(self, mse_loss: Tensor, mae_loss: Tensor, log_cosh_loss: Tensor) -> None:
        """
        Updates the loss trackers with new loss values.

        Args:
            mse_loss (Tensor): The Mean Squared Error loss.
            mae_loss (Tensor): The Mean Absolute Error loss.
            log_cosh_loss (Tensor): The log-cosh loss.
        """
        self.mse_losses.append(mse_loss.detach().cpu())
        self.mae_losses.append(mae_loss.detach().cpu())
        self.log_cosh_losses.append(log_cosh_loss.detach().cpu())

    def compute_stats(self, values_list: List[Union[float, Tensor]]) -> Tuple[Tensor, Tensor]:
        """Computa estatísticas de uma lista de valores"""
        if not values_list:
            return self._get_defaults()
        
        # Conversão mais robusta
        tensors = []
        for v in values_list:
            try:
                if isinstance(v, Tensor):
                    if v.numel() > 0:
                        tensors.append(v.flatten().float())
                else:
                    tensors.append(torch.tensor(float(v)).flatten())
            except (ValueError, RuntimeError):
                continue  # Pula valores inválidos
        
        if not tensors:
            return self._get_defaults()
        
        arr = torch.cat(tensors)
        if arr.numel() == 0:
            return self._get_defaults()
        
        return self._compute_robust_stats(arr)

    @torch.no_grad()
    def compute_z_scores(self, mse, mae, cosh):
        losses   = torch.stack([mse, mae, cosh])                     # shape [3, …]

        centers  = torch.tensor([*self.compute_stats(self.mse_losses),
                                *self.compute_stats(self.mae_losses),
                                *self.compute_stats(self.log_cosh_losses)]
                                , device=losses.device)              # [6] → reshape(3,2)

        center, scale = centers.view(3, 2).unbind(dim=1)             # [3], [3]
        scale  = torch.clamp(scale, min=1e-8).unsqueeze(1)           # broadcast

        z = (losses - center.unsqueeze(1)) / scale                   # mesma shape de losses
        return z[0], z[1], z[2]


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
        self.progress = None
        self.last_logged_delta_epoch = -1
        self.progress = None
        self.last_logged_delta_epoch = -1
        self._sched_tensors_cached = False

        self.schedule_params: Dict[str, Dict[str, float]] = (
            self._initialize_schedule_params(schedule_params)
        )

        # // Sessão V by Gemini - CORREÇÃO: EMA weights podem ser tensores se z-scores forem tensores.
        self.ema_weights: Dict[str, Union[float, Tensor]] = {"mse": 1.0, "mae": 1.0, "log_cosh": 1.0}
        self.initialized: bool = False

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

    def _cache_schedule_tensors(self, device):
        self._sched_start = torch.tensor([
            self.schedule_params["mse"]["start"],
            self.schedule_params["mae"]["start"], 
            self.schedule_params["log_cosh"]["start"]
        ], device=device)
        
        self._sched_end = torch.tensor([
            self.schedule_params["mse"]["end"],
            self.schedule_params["mae"]["end"],
            self.schedule_params["log_cosh"]["end"]
        ], device=device)
        
        self._sched_tensors_cached = True

    # testando vetorização
    @torch.no_grad()
    def adjust_weights(self, mse_z, mae_z, log_cosh_z, config, progress):
        device = mse_z.device
        
        # Cache de tensores constantes
        if not hasattr(self, '_sched_tensors_cached') or not self._sched_tensors_cached:
            self._cache_schedule_tensors(device)

        z = torch.stack([mse_z, mae_z, log_cosh_z])
        z_clamped = torch.clamp(z.abs(), max=self.outlier_threshold)
        
        z_sum = z_clamped.sum(dim=0, keepdim=True)
        inv_z = z_sum - z_clamped + 1e-8
        w = inv_z / (inv_z.sum(dim=0, keepdim=True) + 1e-8)    # base weights

        if self.use_ema:
            if not self.initialized:
                self.ema_weights = w.clone()
                self.initialized = True
            else:
                self.ema_weights = (1 - self.ema_decay) * self.ema_weights + self.ema_decay * w
            w = self.ema_weights                                         # shape [3, …]

        frac = 0.0 if config.epochs <= 1 else progress.epoch / (config.epochs - 1)

        sched_start = torch.tensor([                                   # [mse, mae, cosh]
            self.schedule_params["mse"     ]["start"],
            self.schedule_params["mae"     ]["start"],
            self.schedule_params["log_cosh"]["start"],
        ], device=w.device)

        sched_end = torch.tensor([
            self.schedule_params["mse"     ]["end"],
            self.schedule_params["mae"     ]["end"],
            self.schedule_params["log_cosh"]["end"],
        ], device=w.device)

        sched = sched_start + frac * (sched_end - sched_start)          # shape [3]


        w = w * sched[:, None]                                          # broadcasting
        w = w / (w.sum(dim=0, keepdim=True) + 1e-8)
        w_mse, w_mae, w_cosh = w.mean(dim=list(range(1, w.ndim))).tolist()

        self.progress = progress
        return w_mse, w_mae, w_cosh


    def maybe_log_deltas(self, tensorboard, delta_regularizer, progress):
        """Log dos deltas no tensorboard se necessário"""
        # Verificação simplificada
        if (progress.epoch_step != 0 or 
            self.last_logged_delta_epoch == progress.epoch):
            return
        
        self.last_logged_delta_epoch = progress.epoch

        # Verifica se delta_regularizer existe e tem o método
        if not hasattr(delta_regularizer, 'get_delta_norms'):
            return
        
        current_norm, reference_norm = delta_regularizer.get_delta_norms()
        
        # Verifica se tensorboard existe antes de usar
        if tensorboard:
            tensorboard.add_scalars("Deltas",
                {"Current": current_norm, "Reference": reference_norm},
                global_step=progress.epoch)
