from abc import ABCMeta
from collections.abc import Callable
import math
import traceback

from modules.module.AestheticScoreModel import AestheticScoreModel
from modules.module.HPSv2ScoreModel import HPSv2ScoreModel
from modules.sangoi.logFun import logFun
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.DiffusionScheduleCoefficients import DiffusionScheduleCoefficients
from modules.util.enum.LossScaler import LossScaler
from modules.util.enum.LossWeight import LossWeight
from modules.util.loss.masked_loss import masked_losses
from modules.util.loss.vb_loss import vb_losses
from pytorch_msssim import ssim

import torch
import torch.nn.functional as F
from torch import Tensor

from typing import TYPE_CHECKING, Callable, Optional
if TYPE_CHECKING:
    from modules.util.TensorBoardManager import TensorBoardManager

from modules.util.TensorBoardManager import TensorBoardManager
from modules.sangoi.DynamicLossControl import LossTracker, DynamicLossControl
from modules.sangoi.TrainGPS import TrainGPS

class ModelSetupDiffusionLossMixin(metaclass=ABCMeta):
    __coefficients: DiffusionScheduleCoefficients | None
    __alphas_cumprod_fun: Callable[[Tensor, int], Tensor] | None
    __sigmas: Tensor | None
    config: TrainConfig | None
    progress: TrainProgress | None
    tensorboard: TensorBoardManager | None

    def __init__(self):
        super().__init__()
        self.__coefficients = None
        self.__alphas_cumprod_fun = None
        self.__sigmas = None
        self.progress = None
        self.config = None
        self.loaded_pattern_deltas = None
        self.blend_window = None
        self.loss_tracker = LossTracker(window_size=500, use_mad=True)
        self.dynamic_loss_strengthing = DynamicLossControl()
        self.tensorboard: Optional[TensorBoardManager] = None,

    def __sangoi_loss_schedule(self, batch, data, config, progress):
        pred, tgt = data["predicted"], data["target"]
        if pred.dtype != tgt.dtype:
            tgt = tgt.to(dtype=pred.dtype)

        # componentes sempre disponíveis
        mae = F.l1_loss(pred, tgt, reduction="none")
        logc = self.__log_cosh_loss(pred, tgt)
        mse  = F.mse_loss(pred, tgt, reduction="none")

        # z-scores + pesos
        self.loss_tracker.update(mse.mean(), mae.mean(), logc.mean())
        mse_z, mae_z, log_z = self.loss_tracker.compute_z_scores(mse.mean(), mae.mean(), logc.mean())
        w_mse, w_mae, w_log = self.dynamic_loss_strengthing.adjust_weights(mse_z, mae_z, log_z, config, progress)

        base_loss = (
            mse  * w_mse * config.mse_strength +
            mae  * w_mae * config.mae_strength +
            logc * w_log * config.log_cosh_strength
        )

        self.tensorboard.add_scalar(
            "sangoi/7mse",
            w_mse,
            progress.global_step,
        )
        self.tensorboard.add_scalar(
            "sangoi/8mae",
            w_mae,
            progress.global_step,
        )
        self.tensorboard.add_scalar(
            "sangoi/9log_cosh",
            w_log,
            progress.global_step,
        )

        masked = masked_losses(
            losses=base_loss,
            mask=batch["latent_mask"],
            unmasked_weight=config.unmasked_weight,
            normalize_masked_area_loss=config.normalize_masked_area_loss,
        )
        return masked.mean([1, 2, 3])

    def __log_cosh_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        diff = pred - target
        loss = (
            diff
            + torch.nn.functional.softplus(-2.0 * diff)
            - torch.log(torch.full(size=diff.size(), fill_value=2.0, dtype=torch.float32, device=diff.device))
        )
        return loss

    def __masked_losses(
        self,
        batch: dict,
        data: dict,
        config: TrainConfig,
    ):

        progress = self.progress
        if getattr(config, "sangoi_schedule", False):
            return self.__sangoi_loss_schedule(batch, data, config, progress)        
        losses = 0

        # teste de vetorização
        diff = data["predicted"] - data["target"]           # ← cálculo único

        mse_loss = mae_loss = log_cosh_loss = torch.tensor(0., device=diff.device)

        if config.mse_strength != 0 or config.loss_mode_fn == "SANGOI":
            mse_loss = masked_losses(
                losses=diff.pow(2),                         # MSE sem 2ª subtração
                mask=batch["latent_mask"],
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean([1, 2, 3])

        if config.mae_strength != 0 or config.loss_mode_fn == "SANGOI":
            mae_loss = masked_losses(
                losses=diff.abs(),                          # MAE idem
                mask=batch["latent_mask"],
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean([1, 2, 3])

        if config.log_cosh_strength != 0 or config.loss_mode_fn == "SANGOI":
            log_cosh_tensor = diff + torch.nn.functional.softplus(-2.0 * diff) - math.log(2.0)
            log_cosh_loss = masked_losses(
                losses=log_cosh_tensor,                     # log-cosh reaproveitando diff
                mask=batch["latent_mask"],
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean([1, 2, 3])


        match config.loss_mode_fn:
            case config.loss_mode_fn.ORIGINAL:
                losses = (
                    mse_loss * config.mse_strength
                    + mae_loss * config.mae_strength
                    + log_cosh_loss * config.log_cosh_strength
                )

                # VB loss
                if config.vb_loss_strength != 0 and "predicted_var_values" in data and self.__coefficients is not None:
                    losses += (
                        masked_losses(
                            losses=vb_losses(
                                coefficients=self.__coefficients,
                                x_0=data["scaled_latent_image"],
                                x_t=data["noisy_latent_image"],
                                t=data["timestep"],
                                predicted_eps=data["predicted"],
                                predicted_var_values=data["predicted_var_values"],
                            ),
                            mask=batch["latent_mask"],
                            unmasked_weight=config.unmasked_weight,
                            normalize_masked_area_loss=config.normalize_masked_area_loss,
                        ).mean([1, 2, 3])
                        * config.vb_loss_strength
                    )

        return losses

    def __unmasked_losses(
        self,
        batch: dict,
        data: dict,
        config: TrainConfig,
    ):

        progress = self.progress
        if getattr(config, "sangoi_loss", False):
            return self.__sangoi_loss_schedule(batch, data, config)        
        losses = 0

        mse_loss = torch.tensor(0.0, device=data["predicted"].device)
        mae_loss = torch.tensor(0.0, device=data["predicted"].device)
        log_cosh_loss = torch.tensor(0.0, device=data["predicted"].device)

        # MSE/L2 Loss
        if config.mse_strength != 0 or config.loss_mode_fn == "SANGOI":
            mse_loss = F.mse_loss(
                data["predicted"],
                data["target"],
                reduction="none",
            ).mean([1, 2, 3])

        # MAE/L1 Loss
        if config.mae_strength != 0 or config.loss_mode_fn == "SANGOI":
            mae_loss = F.l1_loss(
                data["predicted"],
                data["target"],
                reduction="none",
            ).mean([1, 2, 3])

        # log-cosh Loss
        if config.log_cosh_strength != 0 or config.loss_mode_fn == "SANGOI":
            log_cosh_loss = self.__log_cosh_loss(
                data["predicted"],
                data["target"],
            ).mean([1, 2, 3])

        match config.loss_mode_fn:
            case config.loss_mode_fn.ORIGINAL:
                losses = (
                    mse_loss * config.mse_strength
                    + mae_loss * config.mae_strength
                    + log_cosh_loss * config.log_cosh_strength
                )

                # VB loss
                if config.vb_loss_strength != 0 and "predicted_var_values" in data and self.__coefficients is not None:
                    losses += (
                        masked_losses(
                            losses=vb_losses(
                                coefficients=self.__coefficients,
                                x_0=data["scaled_latent_image"],
                                x_t=data["noisy_latent_image"],
                                t=data["timestep"],
                                predicted_eps=data["predicted"],
                                predicted_var_values=data["predicted_var_values"],
                            ),
                            mask=batch["latent_mask"],
                            unmasked_weight=config.unmasked_weight,
                            normalize_masked_area_loss=config.normalize_masked_area_loss,
                        ).mean([1, 2, 3])
                        * config.vb_loss_strength
                    )

        return losses

    def __snr(self, timesteps: Tensor, device: torch.device):
        if self.__coefficients:
            all_snr = (self.__coefficients.sqrt_alphas_cumprod / self.__coefficients.sqrt_one_minus_alphas_cumprod) ** 2
            all_snr.to(device)
            snr = all_snr[timesteps]
        else:
            alphas_cumprod = self.__alphas_cumprod_fun(timesteps, 1)
            snr = alphas_cumprod / (1.0 - alphas_cumprod)

        return snr

    def __min_snr_weight(self, timesteps: Tensor, gamma: float, v_prediction: bool, device: torch.device) -> Tensor:
        snr = self.__snr(timesteps, device)
        min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
        # Denominator of the snr_weight increased by 1 if v-prediction is being used.
        if v_prediction:
            snr += 1.0
        snr_weight = (min_snr_gamma / snr).to(device)
        return snr_weight

    def __debiased_estimation_weight(self, timesteps: Tensor, v_prediction: bool, device: torch.device) -> Tensor:
        snr = self.__snr(timesteps, device)
        weight = snr
        # The line below is a departure from the original paper.
        # This is to match the Kohya implementation, see: https://github.com/kohya-ss/sd-scripts/pull/889
        # In addition, it helps avoid numerical instability.
        torch.clip(weight, max=1.0e3, out=weight)
        if v_prediction:
            weight += 1.0
        torch.rsqrt(weight, out=weight)
        return weight

    def __p2_loss_weight(
        self,
        timesteps: Tensor,
        gamma: float,
        v_prediction: bool,
        device: torch.device,
    ) -> Tensor:
        snr = self.__snr(timesteps, device)
        if v_prediction:
            snr += 1.0
        return (1.0 + snr) ** -gamma

    def __sigma_loss_weight(
        self,
        timesteps: Tensor,
        device: torch.device,
    ) -> Tensor:
        return self.__sigmas[timesteps].to(device=device)

    # START v2025-05-27c – __sangoi_loss_weighting refeito
    # VOLTANDO ÀS ORIGENS 04-06-2025
    def __sangoi_loss_weighting(
      self,
      timesteps: Tensor,
      predicted: Tensor,
      target: Tensor,
      device: torch.device,
      gamma: float,
    ):
      """
      Função Sangoi Loss Weighting com reescalonamento [0,1] -> [gamma, 1].
      Se combined_weight_raw > 1, fica 1. Se < 0 (em teoria não deveria), fica 0.
      """

      progress = self.progress
      config = self.config

      # 1) Cálculo do snr "padrão"
      snr = self.__snr(timesteps, device)        # (batch, ...)
      epsilon = 1e-8

      # 2) Cálculo do MAPE (já presente)
      mape = torch.abs((target - predicted) / (target + epsilon))
      mape = torch.clamp(mape, min=0, max=1).mean(dim=[1, 2, 3])

      # -----------------------------------------------------------------------
      # CÁLCULO DO FATOR DE PROGRESSO
      # -----------------------------------------------------------------------
      # Se total_epochs = N, progress.epoch vai de 0 até N-1 (ou 1 até N, depende do trainer).
      # Ajuste conforme o comportamento real do seu `progress.epoch`.
      total_epochs = config.epochs
      current_epoch = progress.epoch  # verifique se vai de 0 a N-1 ou 1 a N
      # Fazemos um clamp para evitar divisões por zero em caso de 1 época só:
      if total_epochs <= 1:
          alpha = 1.0
      else:
          alpha = current_epoch / float(total_epochs - 1)  # varia de 0 até 1

      # -----------------------------------------------------------------------
      # CRIANDO DOIS "EXTREMOS" DE PESO PARA O SNR
      # -----------------------------------------------------------------------
      # A ideia é que no início do treino (alpha ~ 0),
      # queremos enfatizar cenários de SNR baixo como "mais difíceis".
      # No fim do treino (alpha ~ 1), enfatizamos cenários de SNR alto como "mais difíceis".
      #
      # Aqui vai um exemplo de forma de interpolar:
      # snr_weight_low_first  => enfatiza SNR BAIXO como "mais difícil"
      # snr_weight_high_first => enfatiza SNR ALTO  como "mais difícil"
      #
      # Você pode escolher a fórmula que fizer mais sentido para o seu caso.
      #
      # Exemplo de uma forma simples:
      #  - Se snr estiver alto, snr_weight_low_first deve ser PEQUENO.
      #  - Se snr estiver baixo, snr_weight_low_first deve ser MAIOR.
      #
      # Uma abordagem é usar: snr_weight_low_first = log(1 + 1/(snr+eps)),
      # pois, para snr grande, 1/(snr+eps) ≈ 0, resultando em log(1)≈0 (cenário "fácil").
      # E para snr pequeno, 1/(snr+eps) é grande, resultando em log(...) maior (cenário "difícil").
      #
      # Por outro lado, snr_weight_high_first = log(1 + snr)
      # faz o contrário: para snr grande, o log é grande; para snr pequeno, o log é pequeno.
      #
      # Depois, interpolamos linearmente entre esses dois extremos pelo fator alpha.

      snr_weight_low_first = torch.log(1.0 + 1.0 / (snr + epsilon))  # enfatiza SNR baixo
      snr_weight_high_first = torch.log(snr + 1.0)                   # enfatiza SNR alto

      # Interpolação linear:
      # alpha=0 => weight = snr_weight_low_first
      # alpha=1 => weight = snr_weight_high_first
      scenario_snr_weight = (1.0 - alpha) * snr_weight_low_first + alpha * snr_weight_high_first
      mape_reward = 1 - mape
      raw_reward = torch.exp(-mape_reward * scenario_snr_weight)
      # Ex: pode dar valores na casa de 0.08, 0.2, 1.1, etc.

      # 2) Clampar para [0, 1]
      clamped_reward = torch.clamp(raw_reward, min=0.0, max=1.0)

      # 3) Reescalar [0,1] para [gamma,1]
      reward = gamma + (1.0 - gamma) * clamped_reward

      # Logging no TensorBoard
      if self.config.debugoi:
          self.tensorboard.add_scalar(
            "sangoi/1mape_reward", mape_reward.mean().item(), progress.global_step
          )
          self.tensorboard.add_scalar(
            "sangoi/2scenario_snr_weight", scenario_snr_weight.mean().item(), progress.global_step
          )
          self.tensorboard.add_scalar(
            "sangoi/3clamped_reward", clamped_reward.mean().item(), progress.global_step
          )
          self.tensorboard.add_scalar(
            "sangoi/4reward", reward.mean().item(), progress.global_step
          )
          self.tensorboard.add_scalar(
            "sangoi/alpha", alpha, progress.global_step
          )
          self.tensorboard.add_scalar(
            "sangoi/scenario_snr_weight_mean",
            scenario_snr_weight.mean().item(),
            progress.global_step,
          )

      return reward

    def _diffusion_losses(
        self,
        batch: dict,
        data: dict,
        config: TrainConfig,
        progress: TrainProgress,
        train_device: torch.device,
        model: torch.nn.Module,
        betas: Tensor | None = None,        
        alphas_cumprod_fun: Callable[[Tensor, int], Tensor] | None = None,
    ) -> Tensor:

        self.config = config
        self.progress = progress
        self.tensorboard = model.tensorboard
        
        gps_instance: TrainGPS | None = getattr(model, 'deltas', None)

        loss_weight = batch["loss_weight"]

        batch_size_scale = (
            1 if config.loss_scaler in [LossScaler.NONE, LossScaler.GRADIENT_ACCUMULATION] else config.batch_size
        )
        gradient_accumulation_steps_scale = (
            1 if config.loss_scaler in [LossScaler.NONE, LossScaler.BATCH] else config.gradient_accumulation_steps
        )

        if self.__coefficients is None and betas is not None:
            self.__coefficients = DiffusionScheduleCoefficients.from_betas(betas)

        self.__alphas_cumprod_fun = alphas_cumprod_fun

        if data["loss_type"] == "align_prop":
            # losses = self.__align_prop_losses(batch, data, config, train_device) # Função não fornecida, mantendo placeholder
            raise NotImplementedError(
                "AlignProp foi removido e eu to com preguiça de ajeitar esse if-else bosta."
            )  # Adicionado para clareza
        else:
            # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
            #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
            if config.masked_training and not config.model_type.has_conditioning_image_input():
                losses = self.__masked_losses(batch, data, config)
            else:
                losses = self.__unmasked_losses(batch, data, config)

        # Scale Losses by Batch and/or GA (if enabled)
        losses = losses * batch_size_scale * gradient_accumulation_steps_scale

        losses *= loss_weight.to(device=losses.device, dtype=losses.dtype)

        # Apply timestep based loss weighting.
        if "timestep" in data and data["loss_type"] != "align_prop":
            v_pred = data.get("prediction_type", "") == "v_prediction"
            match config.loss_weight_fn:
                case LossWeight.MIN_SNR_GAMMA:
                    losses *= self.__min_snr_weight(
                        data["timestep"],
                        config.loss_weight_strength,
                        v_pred,
                        losses.device,
                    )
                case LossWeight.DEBIASED_ESTIMATION:
                    losses *= self.__debiased_estimation_weight(data["timestep"], v_pred, losses.device)
                case LossWeight.P2:
                    losses *= self.__p2_loss_weight(
                        data["timestep"],
                        config.loss_weight_strength,
                        v_pred,
                        losses.device,
                    )
                case LossWeight.SANGOI:
                    self.tensorboard.add_scalar(
                        "sangoi/loss_b4_sangoi",
                        losses.mean().item(),
                        self.progress.global_step,
                    )
                    losses *= self.__sangoi_loss_weighting(
                      data["timestep"],
                      data["predicted"],
                      data["target"],
                      losses.device,
                      config.loss_weight_strength,
                    )
                    self.tensorboard.add_scalar(
                        "sangoi/loss_after_sangoi",
                        losses.mean().item(),
                        self.progress.global_step,
                    )

            if config.train_gps_use_it and gps_instance is not None and gps_instance.reference_deltas:
              try:
                # Calcula a penalidade usando os pesos *atuais* do modelo
                # e comparando o delta *acumulado atual* com o delta de referência
                penalty = gps_instance.compute_penalty(lambda_weight=config.train_gps_weight)

                # Adiciona a penalidade à loss média do batch
                # 'losses' tem shape (batch_size), 'penalty' é um escalar no device correto
                self.tensorboard.add_scalar("delta/loss_b4_delta", losses.mean().item(), self.progress.global_step)
                self.tensorboard.add_scalar("delta/penalty", penalty.item(), self.progress.global_step)
                losses += penalty  # Adiciona o escalar à loss de cada item do batch
                self.tensorboard.add_scalar("delta/loss_after_delta", losses.mean().item(), self.progress.global_step)

              except Exception as e:
                    print(f"[TrainGPS] Erro ao calcular/aplicar penalidade: {e}")
                    traceback.print_exc() # Loga o traceback para depuração

        return losses

    def _flow_matching_losses(
        self,
        batch: dict,
        data: dict,
        config: TrainConfig,
        train_device: torch.device,
        sigmas: Tensor | None = None,
    ) -> Tensor:
        loss_weight = batch["loss_weight"]
        batch_size_scale = (
            1 if config.loss_scaler in [LossScaler.NONE, LossScaler.GRADIENT_ACCUMULATION] else config.batch_size
        )
        gradient_accumulation_steps_scale = (
            1 if config.loss_scaler in [LossScaler.NONE, LossScaler.BATCH] else config.gradient_accumulation_steps
        )

        if self.__sigmas is None and sigmas is not None:
            num_timesteps = sigmas.shape[0]
            all_timesteps = torch.arange(
                start=1,
                end=num_timesteps + 1,
                step=1,
                dtype=torch.int32,
                device=sigmas.device,
            )
            self.__sigmas = all_timesteps / num_timesteps

        if data["loss_type"] == "align_prop":
            losses = self.__align_prop_losses(batch, data, config, train_device)
        else:
            # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
            #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
            if config.masked_training and not config.model_type.has_conditioning_image_input():
                losses = self.__masked_losses(batch, data, config)
            else:
                losses = self.__unmasked_losses(batch, data, config)

        # Scale Losses by Batch and/or GA (if enabled)
        losses = losses * batch_size_scale * gradient_accumulation_steps_scale

        losses *= loss_weight.to(device=losses.device, dtype=losses.dtype)

        # Apply timestep based loss weighting.
        if "timestep" in data and data["loss_type"] != "align_prop":
            match config.loss_weight_fn:
                case LossWeight.SIGMA:
                    losses *= self.__sigma_loss_weight(data["timestep"], losses.device)

        return losses

    def _safe_ssim(self, pred_bf16: torch.Tensor, tgt_bf16: torch.Tensor) -> torch.Tensor:
        """ Calcula SSIM em fp32 para evitar underflow; devolve no dtype original. """
        ssim_fp32 = ssim(pred_bf16.float(), tgt_bf16.float(), data_range=1.0, size_average=False)
        return ssim_fp32.to(dtype=pred_bf16.dtype)

    def latent_ssim(self, pred_lat: torch.Tensor, tgt_lat: torch.Tensor) -> torch.Tensor:
        """
        SSIM proxy p/ latentes projetados 128×128.
        • Se H<128 (caso raro), upscale NN→128.
        • Calcula SSIM fp32, devolve no dtype original (bf16/fp16).
        """
        if pred_lat.shape[-1] < 128:
            pred_lat = F.interpolate(pred_lat, size=128, mode="nearest")
            tgt_lat  = F.interpolate(tgt_lat,  size=128, mode="nearest")
        with torch.no_grad():
            ssim32 = ssim(pred_lat.float(), tgt_lat.float(), data_range=1.0, size_average=False, win_size=11)
        return ssim32.to(dtype=pred_lat.dtype)