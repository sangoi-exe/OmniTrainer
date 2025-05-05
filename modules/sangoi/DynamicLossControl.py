import torch

from torch import Tensor
from collections import deque

from typing import Tuple, List, Dict, Union

class LossTracker:
    """
    Tracks and generates statistics for the latest N loss values for MSE, MAE, and log-cosh.
    It can use mean/std or median/MAD for statistics.
    """

    def __init__(self, window_size: int = 100, use_mad: bool = False) -> None:
        """
        Initializes the LossTracker.

        Args:
            window_size (int): The number of recent loss values to track.
            use_mad (bool): If True, use median and MAD instead of mean and std.
        """
        self.window_size: int = window_size
        self.use_mad: bool = use_mad

        # As filas continuarão existindo, mas podem armazenar floats ou Tensores
        self.mse_losses: deque = deque(maxlen=window_size)
        self.mae_losses: deque = deque(maxlen=window_size)
        self.log_cosh_losses: deque = deque(maxlen=window_size)

    def update(self, mse_loss: Tensor, mae_loss: Tensor, log_cosh_loss: Tensor) -> None:
        """
        Updates the loss trackers with new loss values.

        Args:
            mse_loss (Tensor): The Mean Squared Error loss.
            mae_loss (Tensor): The Mean Absolute Error loss.
            log_cosh_loss (Tensor): The log-cosh loss.
        """

        # Modo batch: armazena o tensor inteiro mas já movido para a CPU
        self.mse_losses.append(mse_loss.detach().cpu())
        self.mae_losses.append(mae_loss.detach().cpu())
        self.log_cosh_losses.append(log_cosh_loss.detach().cpu())

    def compute_stats(
        self, values_list: List[Union[float, Tensor]]
    ) -> Tuple[float, float]:
        """
        Computes statistics for a list of loss values.

        Args:
            values_list (List[float or Tensor]): The list of loss values (scalar ou Tensor).

        Returns:
            Tuple[float, float]: If use_mad is False, returns (mean, std).
                                 If use_mad is True, returns (median, MAD).
        """
        if len(values_list) < 2:
            # Evitar problemas no início do treinamento
            return 0.0, 1e-8

        # Modo batch: "desempacota" Tensores em um único Tensor
        # Cada elemento de values_list pode ser shape [] (loss já reduzido)
        # ou [batch_size]. Aqui assumimos que cada item é [batch_size],
        # mas se for scalar, a concat ainda funciona com .view(-1).
        arr = torch.tensor(values_list, dtype=torch.float32, device="cpu")

        if not self.use_mad:
            mean_val = arr.mean()
            std_val = arr.std(unbiased=False)
            # .item() será chamado apenas quando for realmente necessário obter um float na CPU
            return mean_val, torch.clamp(
                std_val, min=1e-8
            )  # Retorna tensores escalares
        else:
            median_val = arr.median()
            abs_dev = torch.abs(arr - median_val)
            mad_val = abs_dev.median().values
            return median_val, torch.clamp(
                mad_val, min=1e-8
            )  # Retorna tensores escalares
    @torch.no_grad()
    def compute_z_scores(
        self, mse_loss: Tensor, mae_loss: Tensor, log_cosh_loss: Tensor
    ) -> Union[Tuple[float, float, float], Tuple[Tensor, Tensor, Tensor]]:
        """
        Computes z-scores for the given loss values.

        Args:
            mse_loss (Tensor): The Mean Squared Error loss.
            mae_loss (Tensor): The Mean Absolute Error loss.
            log_cosh_loss (Tensor): The log-cosh loss.

        Returns:
            Union[Tuple[float, float, float], Tuple[Tensor, Tensor, Tensor]]:
            The z-scores for MSE, MAE, and log-cosh losses.
            Tensors (shape like input losses).
        """

        mse_center, mse_scale = self.compute_stats(list(self.mse_losses))
        mae_center, mae_scale = self.compute_stats(list(self.mae_losses))
        log_cosh_center, log_cosh_scale = self.compute_stats(list(self.log_cosh_losses))

        mse_z = (mse_loss - mse_center) / mse_scale
        mae_z = (mae_loss - mae_center) / mae_scale
        log_cosh_z = (log_cosh_loss - log_cosh_center) / log_cosh_scale

        return mse_z, mae_z, log_cosh_z


class DynamicLossControl:
    """
    Dynamically adjusts the weights of different loss components based on their z-scores.
    It can optionally use Exponential Moving Average (EMA) and applies a scheduling mechanism
    to prioritize different losses over the course of training.
    """

    def __init__(
        self,
        use_ema: bool = False,
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
        self.last_logged_delta_epoch = 0

        # Initialize scheduling parameters
        self.schedule_params: Dict[str, Dict[str, float]] = (
            self._initialize_schedule_params(schedule_params)
        )

        # EMA state
        self.ema_weights: Dict[str, float] = {"mse": 1.0, "mae": 1.0, "log_cosh": 1.0}
        self.initialized: bool = False

    def _initialize_schedule_params(
        self, schedule_params: Dict[str, Dict[str, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Initializes the scheduling parameters.

        Args:
            schedule_params (Dict[str, Dict[str, float]], optional):
                User-provided scheduling parameters.

        Returns:
            Dict[str, Dict[str, float]]: Initialized scheduling parameters.
        """
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
        mse_z: Union[float, Tensor],
        mae_z: Union[float, Tensor],
        log_cosh_z: Union[float, Tensor],
        config,
        progress,
    ) -> Tuple[float, float, float]:
        """
        Adjusts the weights for MSE, MAE, and log-cosh losses based on their z-scores.

        The adjustment process includes:
        1) Clamping z-scores to the outlier threshold.
        2) Inverting z-scores.
        3) Normalizing the inverted z-scores.
        4) Applying Exponential Moving Average (if enabled).
        5) Scheduling weight priorities over training epochs.

        Args:
            mse_z (Union[float, Tensor]): Z-score(s) for MSE loss.
            mae_z (Union[float, Tensor]): Z-score(s) for MAE loss.
            log_cosh_z (float): Z-score for log-cosh loss.
            config: Configuration object containing training parameters (e.g., total epochs).
            progress: Progress object containing current epoch information.

        Returns:
            Tuple[float, float, float]: The adjusted weights for MSE, MAE, and log-cosh losses.
        """
        # 1) Clamping z-scores
        _abs = torch.abs
        _clamp_max = lambda t, val: torch.clamp(t, max=val)
        _sum = lambda d: torch.stack(list(d.values())).sum(dim=0)
        _mean = lambda t: t.mean()  # Usado para obter pesos escalares finais
        is_tensor_mode = True

        # 1) Clamping z-scores (valor absoluto clampado)
        # Clampa o valor *absoluto* e depois o usa. Ou clampa o z original entre -thresh e +thresh?
        # O código original fazia min(abs(z), threshold). Vamos manter isso.
        clamped_abs_z = {
            "mse": _clamp_max(_abs(mse_z), self.outlier_threshold),
            "mae": _clamp_max(_abs(mae_z), self.outlier_threshold),
            "log_cosh": _clamp_max(_abs(log_cosh_z), self.outlier_threshold),
        }

        # 2) Invertendo z-scores relativos
        # A lógica original era: inv_z = (total_abs_z - abs_z) / total_abs_z
        # Isso dá peso maior para quem tem z-score absoluto menor.
        total_abs_z = _sum(clamped_abs_z)
        # Adicionar epsilon para evitar divisão por zero (importante para tensores)
        epsilon = 1e-8
        total_abs_z = total_abs_z + epsilon  # Broadcasting se for tensor

        inverted_z = {
            loss: (total_abs_z - z) / total_abs_z for loss, z in clamped_abs_z.items()
        }

        # 3) Normalizando pesos base
        # sum_inv_z = (total-z1)/total + (total-z2)/total + (total-z3)/total
        #           = (3*total - (z1+z2+z3))/total = (3*total - total)/total = 2*total / total = 2 (se total > 0)
        # Portanto, a soma dos inverted_z deveria ser 2 (ou próximo disso devido ao clamp e epsilon).
        # Normalizar dividindo pela soma garante que os pesos somem 1.
        sum_inv_z = _sum(inverted_z) + epsilon
        base_weights = {loss: inv_z / sum_inv_z for loss, inv_z in inverted_z.items()}

        # 4) Exponential Moving Average
        if self.use_ema:
            if not self.initialized:
                # Inicializa com os pesos atuais (podem ser tensores ou floats)
                for loss in self.ema_weights:
                    self.ema_weights[loss] = base_weights[loss]
            else:
                alpha = 1.0 - self.ema_decay
                for loss in self.ema_weights:
                    self.ema_weights[loss] = (1 - alpha) * self.ema_weights[
                        loss
                    ] + alpha * base_weights[loss]
                # self.initialized = True # Cuidado: Se base_weights for Tensor, ema_weights vira Tensor
            # Normaliza as weights do EMA
            sum_ema = sum(self.ema_weights.values())

            # Se ema_weights for Tensor, sum_ema será Tensor. Adicionar epsilon.
            sum_ema = sum_ema + epsilon
            normalized_ema_weights = {
                loss: w / sum_ema for loss, w in self.ema_weights.items()
            }  # Broadcasting se Tensor
            base_weights = normalized_ema_weights
            # Se inicializou com tensores, aqui base_weights são tensores
            self.initialized = (
                True  # Marcar como inicializado após a primeira atualização/leitura
            )

        # 5) Scheduler
        if config.epochs > 1:
            frac = progress.epoch / float(config.epochs - 1)
        else:
            frac = 0.0
        frac = max(0.0, min(frac, 1.0))

        scheduled_factors = {}
        for loss, params in self.schedule_params.items():
            # params são floats, o resultado é float
            scheduled_factors[loss] = (
                params["start"] * (1 - frac) + params["end"] * frac
            )

        # Multiplica cada base weight pelo fator do scheduler
        # Se base_weights for Tensor, o fator float faz broadcasting
        weighted_weights = {
            loss: base_weights[loss] * scheduled_factors[loss] for loss in base_weights
        }

        # Normaliza final
        sum_weighted = _sum(weighted_weights) + epsilon
        final_weights = {loss: w / sum_weighted for loss, w in weighted_weights.items()}

        # 6) Redução para escalar (se necessário)
        # A loss final espera pesos escalares. Se estávamos no modo tensor,
        # pegamos a média dos pesos no batch.
        if is_tensor_mode:
            final_weights_scalar = {
                loss: float(torch.mean(w)) for loss, w in final_weights.items()
            }
        else:
            final_weights_scalar = final_weights  # Já são floats

        # Retorna na ordem desejada
        return (
            final_weights_scalar["mse"],
            final_weights_scalar["mae"],
            final_weights_scalar["log_cosh"],
        )

    def maybe_log_deltas(self, tensorboard, delta_regularizer, progress):
        if not hasattr(self, "progress") or not self.progress:
            return

        if progress.epoch_step != 0:
            return

        if not hasattr(self, "last_logged_delta_epoch"):
            self.last_logged_delta_epoch = -1

        if progress.epoch == self.last_logged_delta_epoch:
            return

        self.last_logged_delta_epoch = self.progress.epoch

        current_norm, reference_norm = delta_regularizer.get_delta_norms()

        print(
            f"[TrainGPS] Epoch {progress.epoch} | Current Δ: {current_norm:.4f} | Ref Δ: {reference_norm:.4f}"
        )