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

from typing import Callable, Optional
from torch.utils.tensorboard import SummaryWriter
from modules.sangoi.DynamicLossControl import LossTracker, DynamicLossControl
from modules.sangoi.TrainGPS import TrainGPS

class ModelSetupDiffusionLossMixin(metaclass=ABCMeta):
    __coefficients: DiffusionScheduleCoefficients | None
    __alphas_cumprod_fun: Callable[[Tensor, int], Tensor] | None
    __sigmas: Tensor | None
    config: TrainConfig | None
    progress: TrainProgress | None
    tensorboard: SummaryWriter | None

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
        self.tensorboard: Optional[SummaryWriter] = None,

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
            mse  * w_mse +
            mae  * w_mae +
            logc * w_log
          )

        self.tensorboard.add_scalar(
            "SangoiS/MSE_Loss",
            w_mse,
            progress.global_step,
        )
        self.tensorboard.add_scalar(
            "SangoiS/MAE_Loss",
            w_mae,
            progress.global_step,
        )
        self.tensorboard.add_scalar(
            "SangoiS/Log_Cosh_Loss",
            w_log,
            progress.global_step,
        )

        return base_loss

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
    # PROPOSTA DE REFAÇÃO - __sangoi_loss_weighting
    def __sangoi_loss_weighting(
      self,
      timesteps: Tensor,
      predicted: Tensor,
      target: Tensor,
      device: torch.device,
      gamma: float, # Renomeado para 'focus_strength' seria mais claro, mas mantendo 'gamma' por consistência.
    ):
      """
      Calcula um peso de loss por amostra baseado nos princípios de Online Hard Example Mining (OHEM).
      Amostras fáceis (erro baixo) recebem um peso menor (loss reduzida), especialmente em
      cenários considerados difíceis (SNR alto), para focar o treino nas amostras que o modelo ainda erra.

      Args:
          timesteps: Tensor com os timesteps de cada amostra no batch.
          predicted: Tensor com a predição do modelo.
          target: Tensor com o alvo da predição (ruído).
          device: Dispositivo para os tensores.
          gamma: Hiperparâmetro que controla a força do OHEM. Valores maiores
                reduzem mais agressivamente a loss de amostras fáceis.

      Returns:
          Um tensor de pesos no formato (batch_size,), com valores no intervalo [0, 1].
      """
      with torch.no_grad():
        # 1. Métrica de Qualidade da Predição (quão "fácil" foi o exemplo?)
        # Usamos MAE (L1 loss) por ser mais robusto que MAPE.
        # torch.tanh mapeia o erro para o intervalo [0, 1], agindo como uma normalização suave.
        mae_per_sample = torch.abs(target - predicted).mean(dim=[1, 2, 3])
        # prediction_quality: 1.0 para erro zero (muito fácil), próximo de 0.0 para erro alto (difícil).
        prediction_quality = 1.0 - torch.tanh(mae_per_sample)

        # 2. Métrica de Dificuldade do Cenário
        # Com base na análise, SNR alto (timesteps baixos) é mais difícil.
        # log1p(x) = log(1+x) é numericamente mais estável para x pequeno.
        snr = self.__snr(timesteps, device)
        scenario_difficulty = torch.log1p(snr) # Aumenta com a dificuldade (SNR alto).

        # 3. Calcular o "potencial de redução de loss"
        # A maior redução ocorre para predições de alta qualidade em cenários de alta dificuldade.
        # Isso significa que o modelo acertou algo difícil e podemos "premiá-lo" ignorando essa loss.
        reduction_potential = prediction_quality * scenario_difficulty

        # 4. Calcular o peso final da loss
        # O 'gamma' agora age como um fator de força.
        # Subtraímos o potencial de redução de 1.0 para obter o peso final.
        # torch.clamp garante que o peso final esteja estritamente entre 0.0 e 1.0.
        final_weight = torch.clamp(1.0 - (reduction_potential * gamma), min=0.0, max=1.0)

        # Log para monitoramento
        self.tensorboard.add_scalar(
          "SangoiW/1_Prediction_Quality_Avg", prediction_quality.mean().item(), self.progress.global_step
        )
        self.tensorboard.add_scalar(
          "SangoiW/2_Scenario_Difficulty_Avg", scenario_difficulty.mean().item(), self.progress.global_step
        )
        self.tensorboard.add_scalar(
          "SangoiW/3_Final_Weight_Avg", final_weight.mean().item(), self.progress.global_step
        )

        return final_weight

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
                    
                    # A nova função já retorna um peso entre [0, 1]
                    losses *= self.__sangoi_loss_weighting(
                      data["timestep"],
                      data["predicted"],
                      data["target"],
                      losses.device,
                      config.loss_weight_strength, # Este é o seu 'gamma'
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