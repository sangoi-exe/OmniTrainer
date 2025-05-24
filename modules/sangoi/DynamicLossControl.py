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
    def compute_z_scores(
        self, mse_loss: Tensor, mae_loss: Tensor, log_cosh_loss: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]: # // Sessão V by Gemini - CORREÇÃO: Retorno é sempre Tensor
        """
        Computes z-scores for the given loss values.

        Args:
            mse_loss (Tensor): The Mean Squared Error loss.
            mae_loss (Tensor): The Mean Absolute Error loss.
            log_cosh_loss (Tensor): The log-cosh loss.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
            The z-scores for MSE, MAE, and log-cosh losses.
            Tensors (shape like input losses).
        """

        mse_center, mse_scale = self.compute_stats(list(self.mse_losses))
        mae_center, mae_scale = self.compute_stats(list(self.mae_losses))
        log_cosh_center, log_cosh_scale = self.compute_stats(list(self.log_cosh_losses))

        # Os inputs (mse_loss, etc.) estão no device original (e.g., CUDA)
        # Os centers/scales estão na CPU. Mover centers/scales para o device das losses.
        device = mse_loss.device
        mse_center_dev = mse_center.to(device)
        mse_scale_dev = mse_scale.to(device)
        mae_center_dev = mae_center.to(device)
        mae_scale_dev = mae_scale.to(device)
        log_cosh_center_dev = log_cosh_center.to(device)
        log_cosh_scale_dev = log_cosh_scale.to(device)

        mse_z = (mse_loss - mse_center_dev) / mse_scale_dev
        mae_z = (mae_loss - mae_center_dev) / mae_scale_dev
        log_cosh_z = (log_cosh_loss - log_cosh_center_dev) / log_cosh_scale_dev

        return mse_z, mae_z, log_cosh_z


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

    @torch.no_grad()
    def adjust_weights(
        self,
        mse_z: Tensor, # // Sessão V by Gemini - CORREÇÃO: Z-scores são tensores
        mae_z: Tensor, # // Sessão V by Gemini - CORREÇÃO: Z-scores são tensores
        log_cosh_z: Tensor, # // Sessão V by Gemini - CORREÇÃO: Z-scores são tensores
        config, # // Sessão V by Gemini - CORREÇÃO: Melhor tipar config se possível
        progress, # // Sessão V by Gemini - CORREÇÃO: Melhor tipar progress se possível
    ) -> Tuple[float, float, float]: # Retorna pesos escalares finais
        _abs = torch.abs
        _clamp_max = lambda t, val: torch.clamp(t, max=val)
        # // Sessão V by Gemini - CORREÇÃO: _sum e _mean precisam lidar com dicionários de tensores
        # e retornar tensores ou floats conforme o caso.
        # A lógica abaixo já lida com isso caso a caso.

        # Z-scores de entrada (mse_z, mae_z, log_cosh_z) são tensores.
        # Todas as operações subsequentes (clamped_abs_z, inverted_z, base_weights)
        # produzirão dicionários de tensores.

        clamped_abs_z = {
            "mse": _clamp_max(_abs(mse_z), self.outlier_threshold),
            "mae": _clamp_max(_abs(mae_z), self.outlier_threshold),
            "log_cosh": _clamp_max(_abs(log_cosh_z), self.outlier_threshold),
        }

        # Soma dos z-scores clampados (será um tensor)
        # torch.stack cria um novo tensor [3, batch_size (opcional)]
        # .sum(dim=0) soma ao longo da dimensão dos tipos de loss, resultando em [batch_size (opcional)]
        current_total_abs_z = torch.stack(list(clamped_abs_z.values())).sum(dim=0)
        epsilon = 1e-8
        current_total_abs_z_eps = current_total_abs_z + epsilon

        inverted_z = {
            loss: (current_total_abs_z_eps - z_val) / current_total_abs_z_eps 
            for loss, z_val in clamped_abs_z.items()
        }

        sum_inv_z = torch.stack(list(inverted_z.values())).sum(dim=0) + epsilon
        base_weights_tensor = { # Dicionário de tensores
            loss: inv_z_val / sum_inv_z 
            for loss, inv_z_val in inverted_z.items()
        }

        current_weights_to_process = base_weights_tensor

        if self.use_ema:
            if not self.initialized:
                # self.ema_weights é Dict[str, Union[float, Tensor]]
                # Inicializa com os tensores atuais
                self.ema_weights = {loss: base_weights_tensor[loss].clone() for loss in base_weights_tensor}
                self.initialized = True
            else:
                alpha = self.ema_decay # // Sessão V by Gemini - CORREÇÃO: alpha = ema_decay, não 1-ema_decay se ema_decay é o fator de retenção do histórico
                                       # Se ema_decay é o fator para o novo valor, então alpha = ema_decay.
                                       # Padrão: new_ema = (1-decay)*old_ema + decay*new_value. Então alpha é decay.
                                       # Seu código: (1-alpha)*old + alpha*new. Se ema_decay é o "decay" do histórico,
                                       # então alpha (peso do novo valor) = 1 - ema_decay.
                                       # Se ema_decay é o peso do NOVO valor, então alpha = ema_decay.
                                       # Vou assumir que self.ema_decay é o "alpha" para o novo valor.
                alpha_new_value = self.ema_decay # Ex: 0.1 para contribuição do novo valor
                alpha_old_value = 1.0 - alpha_new_value

                for loss_name in self.ema_weights:
                    # Garantir que self.ema_weights[loss_name] e base_weights_tensor[loss_name]
                    # estejam no mesmo dispositivo antes da operação.
                    # base_weights_tensor[loss_name] está no device dos z-scores.
                    # self.ema_weights[loss_name] pode estar na CPU se inicializado com float, ou device dos z-scores.
                    
                    # // Sessão V by Gemini - CORREÇÃO: Garantir que device e dtype sejam consistentes para EMA.
                    # Os z_scores (e portanto base_weights_tensor) estão no device da loss (e.g. CUDA).
                    # ema_weights deve ser movido para esse device se ainda não estiver.
                    current_ema_val = self.ema_weights[loss_name]
                    new_base_val = base_weights_tensor[loss_name]
                    
                    if isinstance(current_ema_val, float): # Primeira vez após inicialização com float
                        current_ema_val = torch.tensor(current_ema_val, device=new_base_val.device, dtype=new_base_val.dtype)
                    
                    # Assegurar que current_ema_val está no mesmo device que new_base_val
                    current_ema_val = current_ema_val.to(new_base_val.device)

                    self.ema_weights[loss_name] = alpha_old_value * current_ema_val + alpha_new_value * new_base_val
            
            sum_ema_tensor = torch.stack(list(self.ema_weights.values())).sum(dim=0) + epsilon
            normalized_ema_weights_tensor = { # Dicionário de tensores
                loss_name: w_val / sum_ema_tensor
                for loss_name, w_val in self.ema_weights.items()
            }
            current_weights_to_process = normalized_ema_weights_tensor
        
        # Scheduler (opera com floats, fatores de agendamento são floats)
        # config.epochs e progress.epoch são escalares
        if config.epochs > 1: # // Sessão V by Gemini - CORREÇÃO: Acessar epochs de config
            frac = progress.epoch / float(config.epochs - 1) # // Sessão V by Gemini - CORREÇÃO: Acessar epoch de progress
        else:
            frac = 0.0
        frac = max(0.0, min(frac, 1.0))

        scheduled_factors_float = {} # Dicionário de floats
        for loss, params in self.schedule_params.items():
            scheduled_factors_float[loss] = (
                params["start"] * (1 - frac) + params["end"] * frac
            )

        # Multiplica cada tensor de peso pelo fator float do scheduler (broadcasting)
        weighted_weights_tensor = { # Dicionário de tensores
            loss_name: current_weights_to_process[loss_name] * scheduled_factors_float[loss_name]
            for loss_name in current_weights_to_process
        }

        sum_weighted_tensor = torch.stack(list(weighted_weights_tensor.values())).sum(dim=0) + epsilon
        final_weights_tensor = { # Dicionário de tensores
            loss_name: w_val / sum_weighted_tensor
            for loss_name, w_val in weighted_weights_tensor.items()
        }

        # Redução para escalar: pegar a média dos pesos no batch (se os z-scores eram por batch)
        # ou o valor escalar se os z-scores já eram escalares.
        # Como z-scores são agora sempre tensores, os final_weights_tensor são tensores.
        final_weights_scalar = { # Dicionário de floats
            loss_name: float(w_val.mean()) # .mean() em tensor 0-dim é ele mesmo, .mean() em 1D+ é a média
            for loss_name, w_val in final_weights_tensor.items()
        }

        return (
            final_weights_scalar["mse"],
            final_weights_scalar["mae"],
            final_weights_scalar["log_cosh"],
        )

    def maybe_log_deltas(self, tensorboard, delta_regularizer, progress):
        # // Sessão V by Gemini - CORREÇÃO: Checar se self.progress foi setado
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
        if tensorboard is not None:
            tensorboard.add_scalar("Deltas/Current_Norm", current_norm, current_progress_obj.epoch)
            tensorboard.add_scalar("Deltas/Reference_Norm", reference_norm, current_progress_obj.epoch)


        print(
            f"[TrainGPS] Epoch {current_progress_obj.epoch} | Current Δ: {current_norm:.4f} | Ref Δ: {reference_norm:.4f}"
        )