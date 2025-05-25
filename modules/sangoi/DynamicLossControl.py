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

    def compute_stats(
        self, values_list: List[Union[float, Tensor]]
    ) -> Tuple[Tensor, Tensor]: 
        if not values_list: 
            default_center = torch.tensor(0.0, dtype=torch.float32, device="cpu")
            default_scale = torch.tensor(1e-8, dtype=torch.float32, device="cpu")
            return default_center, default_scale

        processed_values = []
        for v in values_list:
            if isinstance(v, Tensor):
                processed_values.append(v.reshape(-1)) 
            else: 
                processed_values.append(torch.tensor(v, dtype=torch.float32, device="cpu").reshape(-1)) # // Sessão VI by Gemini - CORREÇÃO: Garantir que float convertido para tensor também seja achatado
        
        if not processed_values: 
             default_center = torch.tensor(0.0, dtype=torch.float32, device="cpu")
             default_scale = torch.tensor(1e-8, dtype=torch.float32, device="cpu")
             return default_center, default_scale
        
        # // Sessão VI by Gemini - CORREÇÃO: Verificar se processed_values contém tensores não vazios antes de torch.cat
        non_empty_processed_values = [pv for pv in processed_values if pv.numel() > 0]
        if not non_empty_processed_values:
            # Se todos os tensores em values_list eram vazios
            default_center = torch.tensor(0.0, dtype=torch.float32, device="cpu")
            default_scale = torch.tensor(1e-8, dtype=torch.float32, device="cpu")
            return default_center, default_scale
            
        arr = torch.cat(non_empty_processed_values)

        if arr.numel() == 0: # // Sessão VI by Gemini - CORREÇÃO: Checar numel após cat, caso non_empty_processed_values seja vazio (já coberto acima, mas dupla segurança)
            default_center = torch.tensor(0.0, dtype=torch.float32, device="cpu")
            default_scale = torch.tensor(1e-8, dtype=torch.float32, device="cpu")
            return default_center, default_scale
        
        if arr.numel() == 1: # // Sessão VI by Gemini - CORREÇÃO: Tratar explicitamente o caso de 1 elemento
            center_val = arr.clone() # ou arr[0] se quiser garantir que é 0-dim, mas arr já é 0-dim ou 1-dim com 1 elemento
            if center_val.ndim > 0 : # Se for 1D com 1 elemento, pegar o elemento
                center_val = center_val[0]
            scale_val = torch.tensor(1e-8, dtype=torch.float32, device="cpu")
            return center_val, scale_val


        if not self.use_mad:
            mean_val = arr.mean() 
            std_val = arr.std(unbiased=False) 
            return mean_val, torch.clamp(std_val, min=1e-8)
        else:
            # // Sessão VI by Gemini - CORREÇÃO: Lidar com torch.median retornando tensor ou tupla
            median_result = torch.median(arr)
            if isinstance(median_result, tuple):
                median_val = median_result[0]
            else: # É um tensor 0-dim
                median_val = median_result
            
            abs_dev = torch.abs(arr - median_val) 
            
            mad_result = torch.median(abs_dev)
            if isinstance(mad_result, tuple):
                mad_val = mad_result[0]
            else: # É um tensor 0-dim
                mad_val = mad_result
            
            return median_val, torch.clamp(mad_val, min=1e-8)

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
        self.progress = None # // Sessão V by Gemini - CORREÇÃO: Inicializar progress, será setado externamente
        self.last_logged_delta_epoch = -1 # // Sessão V by Gemini - CORREÇÃO: Inicializar last_logged_delta_epoch

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

    # testando vetorização
    @torch.no_grad()
    def adjust_weights(self, mse_z, mae_z, log_cosh_z, config, progress):
        # ----- 1. empilha z-scores -----
        z = torch.stack([mse_z, mae_z, log_cosh_z])                     # shape [3, …]
        z = torch.clamp(z.abs(), max=self.outlier_threshold)

        inv_z = (z.sum(dim=0, keepdim=True) + 1e-8 - z)                 # pesos inversos
        w     = inv_z / (inv_z.sum(dim=0, keepdim=True) + 1e-8)         # base weights

        # ----- 2. EMA opcional -----
        if self.use_ema:
            if not self.initialized:
                self.ema_weights = w.clone()
                self.initialized = True
            else:
                self.ema_weights = (1 - self.ema_decay) * self.ema_weights + self.ema_decay * w
            w = self.ema_weights                                         # shape [3, …]

        # ----- 3. scheduler vetorizado -----
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

        # ----- 4. aplica scheduler, normaliza e devolve floats -----
        w = w * sched[:, None]                                          # broadcasting
        w = w / (w.sum(dim=0, keepdim=True) + 1e-8)

        w_mse, w_mae, w_cosh = w.mean(dim=list(range(1, w.ndim))).tolist()
        return w_mse, w_mae, w_cosh
    # ========= fim =========


    def maybe_log_deltas(self, tensorboard, delta_regularizer, progress):
        if not hasattr(self, "progress") or self.progress is None : 
            # Se self.progress não foi setado externamente, usar o 'progress' do argumento.
            # No entanto, o 'progress' da classe é usado para last_logged_delta_epoch.
            # Idealmente, self.progress deve ser o mesmo objeto 'progress' passado.
            # Para esta função, vamos usar o 'progress' do argumento para consistência.
            current_progress_obj = progress
        else:
            current_progress_obj = self.progress


        if current_progress_obj.epoch_step != 0: # // Sessão V by Gemini - CORREÇÃO: Acessar epoch_step de progress
            return

        if self.last_logged_delta_epoch == current_progress_obj.epoch: # // Sessão V by Gemini - CORREÇÃO: Acessar epoch de progress
            return

        self.last_logged_delta_epoch = current_progress_obj.epoch # // Sessão V by Gemini - CORREÇÃO: Acessar epoch de progress

        current_norm, reference_norm = delta_regularizer.get_delta_norms()
        
        # // Sessão V by Gemini - CORREÇÃO: tensorboard pode ser None, checar antes de usar
        if tensorboard:
            tensorboard.add_scalars("Deltas",
                {"Current": current_norm, "Reference": reference_norm},
                global_step=progress.epoch)


        print(
            f"[TrainGPS] Epoch {current_progress_obj.epoch} | Current Δ: {current_norm:.4f} | Ref Δ: {reference_norm:.4f}"
        )