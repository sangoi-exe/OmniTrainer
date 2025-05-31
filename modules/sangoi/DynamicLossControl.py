import torch

from torch import Tensor
from collections import deque

from typing import Tuple, List, Dict, Union

from modules.util.config.TrainConfig import TrainConfig

class ScheduledLoss:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.mae_range = config.scheduled_loss_mae_range
        self.log_cosh_range = config.scheduled_loss_mae_range
        self.mse_range = config.scheduled_loss_mae_range

    def _calculate_segment_weight(self, progress: float, p0: float, p1: float, p2: float, p3: float) -> float:
        """
        Calcula o peso para um segmento com ramp-in, full, ramp-out.
        p0: Início do ramp-in
        p1: Fim do ramp-in (início do peso 1.0)
        p2: Início do ramp-out (fim do peso 1.0)
        p3: Fim do ramp-out
        Todos os pX e progress devem estar entre 0.0 e 1.0.
        """
        weight = 0.0
        # Garante que os pontos estejam em ordem para evitar problemas
        p1 = max(p0, p1)
        p2 = max(p1, p2)
        p3 = max(p2, p3)

        if p0 <= progress < p1 and p1 > p0: # Ramp-in
            weight = (progress - p0) / (p1 - p0)
        elif p1 <= progress < p2: # Full weight
            weight = 1.0
        elif p2 <= progress < p3 and p3 > p2: # Ramp-out
            weight = 1.0 - (progress - p2) / (p3 - p2)
        
        # Caso especial: se p0=p1, significa início direto em full_weight ou ramp-in zero.
        # Se progress == p1 == p0, e p1 < p2, deve ser 1.0.
        if p0 == p1 and progress == p0 and p1 < p2: weight = 1.0
        
        # Caso especial: se p2=p3, significa fim abrupto ou ramp-out zero.
        # Se progress == p2 == p3, e p1 < p2, deveria ser 0.0 (já acabou).
        # Se p1 <= progress < p2 e p2 == p3, peso é 1.0 (ainda na fase full).
        # A lógica acima cobre isso, mas é bom ter em mente.

        # Se a fase inteira é um ponto (p0=p1=p2=p3)
        if p0 == p3 and progress == p0:
            weight = 1.0 # Ou 0.0 dependendo da interpretação, para um "pulso" seria 1.0

        return max(0.0, min(1.0, weight))

    def get_loss_weights(self, progress_percent: float) -> tuple[float, float, float]:
        """
        Retorna os pesos para MAE, Log-Cosh, MSE baseados no progresso.
        Exemplo:
        MAE: 0-30% (total), sobrepõe com Log-Cosh de 20-30%
        Log-Cosh: 20-60% (total), sobrepõe com MAE de 20-30%, com MSE de 50-60%
        MSE: 50-100% (total), sobrepõe com Log-Cosh de 50-60%
        """
        
        mae_p_start, mae_p_end = self.mae_range
        lc_p_start, lc_p_end = self.log_cosh_range
        mse_p_start, mse_p_end = self.mse_range

        # MAE:
        # Ramp-in: [mae_p_start, mae_p_start] (começa direto, sem ramp-in de uma loss anterior no exemplo)
        # Full:    [mae_p_start, lc_p_start] (até Log-Cosh começar a entrar)
        # Ramp-out: [lc_p_start, mae_p_end] (enquanto Log-Cosh entra)
        p0_mae = mae_p_start
        p1_mae = mae_p_start 
        p2_mae = min(mae_p_end, lc_p_start) # MAE começa a sair quando LogCosh começa a entrar
        p3_mae = mae_p_end
        w_mae = self._calculate_segment_weight(progress_percent, p0_mae, p1_mae, p2_mae, p3_mae)

        # Log-Cosh:
        # Ramp-in: [lc_p_start, mae_p_end] (enquanto MAE está saindo)
        # Full:    [mae_p_end, mse_p_start] (entre MAE ter saído e MSE começar a entrar)
        # Ramp-out: [mse_p_start, lc_p_end] (enquanto MSE está entrando)
        p0_lc = lc_p_start
        p1_lc = min(lc_p_end, mae_p_end) # LogCosh termina ramp-in quando MAE termina
        p2_lc = max(lc_p_start, mse_p_start) # LogCosh começa ramp-out quando MSE começa
        p3_lc = lc_p_end
        w_lc = self._calculate_segment_weight(progress_percent, p0_lc, p1_lc, p2_lc, p3_lc)
        
        # MSE:
        # Ramp-in: [mse_p_start, lc_p_end] (enquanto Log-Cosh está saindo)
        # Full:    [lc_p_end, mse_p_end] (depois que Log-Cosh saiu, até o fim)
        # Ramp-out: [mse_p_end, mse_p_end] (termina no fim, sem ramp-out para uma loss posterior no exemplo)
        p0_mse = mse_p_start
        p1_mse = min(mse_p_end, lc_p_end) # MSE termina ramp-in quando LogCosh termina
        p2_mse = mse_p_end 
        p3_mse = mse_p_end
        w_mse = self._calculate_segment_weight(progress_percent, p0_mse, p1_mse, p2_mse, p3_mse)
        
        # Normalização opcional para garantir que a soma dos pesos não exceda 1.0 significativamente
        # A lógica de _calculate_segment_weight com transições lineares deve naturalmente 
        # fazer com que em overlaps de duas funções, a soma seja ~1.0.
        # Ex: MAE (1-alpha) + LogCosh (alpha).
        # Se houver uma pequena sobreposição de 3 funções, pode ser > 1.
        # O design atual de p0-p3 deve evitar isso.
        
        # print(f"Prog: {progress_percent:.2f} -> MAE: {w_mae:.2f} (P: {p0_mae:.2f},{p1_mae:.2f},{p2_mae:.2f},{p3_mae:.2f}), LC: {w_lc:.2f} (P: {p0_lc:.2f},{p1_lc:.2f},{p2_lc:.2f},{p3_lc:.2f}), MSE: {w_mse:.2f} (P: {p0_mse:.2f},{p1_mse:.2f},{p2_mse:.2f},{p3_mse:.2f})")

        return w_mae, w_lc, w_mse

class LossTracker:
    """
    Tracks and generates statistics for the latest N loss values for MSE, MAE, and log-cosh.
    It can use mean/std or median/MAD for statistics.
    """

    def __init__(self, window_size: int = 1000, use_mad: bool = True) -> None:
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
        self.log_cosh_losses.append(log_cosh_loss.detach())
        self.mse_losses.append(mse_loss.detach())
        self.mae_losses.append(mae_loss.detach())

    def compute_stats(self, values_list: List[Union[float, Tensor]]) -> Tuple[Tensor, Tensor]:
        """Computa estatísticas de uma lista de valores"""
        if not values_list:
            return self._get_defaults()
        
        # Conversão mais robusta
        tensors = [v.flatten().float()
						for v in values_list
						if isinstance(v, Tensor) and v.numel() > 0]
        if len(tensors) == 0:
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
        ema_decay: float = 0.9,
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

        self.ema_weights: Tensor | None = None  # tensor [3]
        self.initialized: bool = False

    # testando vetorização
    @torch.no_grad()
    def adjust_weights(self, mse_z, mae_z, log_cosh_z, config, progress):
        device = mse_z.device
        
        z = torch.stack([mse_z, mae_z, log_cosh_z])
        z_clamped = torch.clamp(z.abs(), max=self.outlier_threshold)
        
        z_sum = z_clamped.sum(dim=0, keepdim=True)
        inv_z = z_sum - z_clamped + 1e-8
        w = inv_z / (inv_z.sum(dim=0, keepdim=True) + 1e-8)    # base weights

        if self.use_ema:
            if self.ema_weights is None:
                self.ema_weights = w.detach()
            self.ema_weights = (1 - self.ema_decay) * self.ema_weights + self.ema_decay * w
            w = self.ema_weights                                      # shape [3, …]

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
