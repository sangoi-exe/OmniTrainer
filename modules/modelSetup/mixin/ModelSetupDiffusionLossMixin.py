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
from modules.util.loss.masked_loss import masked_losses, sangoi_masked_loss
from modules.util.loss.vb_loss import vb_losses
from pytorch_msssim import ssim

import torch
import torch.nn.functional as F
from torch import Tensor

from typing import Callable, Optional
from torch.utils.tensorboard import SummaryWriter
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
        self.tensorboard: Optional[SummaryWriter] = None,

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

    # START sangoi_huber_loss
    def sangoi_huber_loss(
            self,
            diff:  torch.Tensor,
            snr:   torch.Tensor,
            huber_c: float = 0.1,
            mask:  torch.Tensor | None = None,
            unmasked_weight: float = 0.1,
            normalize_masked_area_loss: bool = True,
    ) -> torch.Tensor:
        """
        Huber com β = huber_c * √SNR, clampado para faixas seguras.
        Suporta máscara opcional (chama sangoi_masked_loss).
        """
        # β dinâmico (√SNR) com faixa [1e-2, 1]
        beta = (huber_c * torch.sqrt(snr)).clamp(1e-2, 1.0).view(-1, 1, 1, 1)
        beta_det = beta.detach()

        abs_diff = diff.abs()
        huber_raw = torch.where(
            abs_diff < beta_det,
            0.5 * abs_diff.square() / beta_det,
            abs_diff - 0.5 * beta_det,
        )
        # redução final: média sobre (C, H, W)
        return huber_raw

    def charbonnier_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-3,
        alpha: float = 0.5,      # 0.5 = raiz -> Charbonnier clássico
        scale: bool = False
        ):
        """
        Charbonnier/pseudo-Huber generalizado.
        alpha=0.5 -> clássico; alpha<0.5 deixa não-convexo (GC-0.45 etc.)
        """
        diff = pred - target
        loss = torch.pow(diff * diff + eps * eps, alpha)

        # opcional: normalizar para não inflar a escala média do loss
        if scale:
            loss = loss / (eps ** (2*alpha))       # mantém compatível c/ outras losses
        
        return loss

    def __masked_losses(
        self,
        batch: dict,
        data: dict,
        config: TrainConfig,
        ):

        losses = 0
        mean_dim = list(range(1, data['predicted'].ndim))

        pred, tgt = data["predicted"], data["target"]
        if pred.dtype != tgt.dtype:
            tgt = tgt.to(dtype=pred.dtype)
            
        # teste de vetorização
        if config.mse_strength != 0:
            losses += F.mse_loss(pred, tgt, reduction="none") * config.mse_strength

        if config.mae_strength != 0:
            losses += F.smooth_l1_loss(pred, tgt, beta=0.1) * config.mae_strength

        if config.huber_strength != 0:
            diff = pred - tgt
            snr  = self.__snr(data["timestep"], pred.device)
            losses += self.sangoi_huber_loss(diff=diff, snr=snr) * config.huber_strength # considerando que o hook de grad está ativo

        # huber Loss
        if config.log_cosh_strength != 0:
            losses += self.__log_cosh_loss(pred, tgt) * config.log_cosh_strength
        
        # Charbonnier Loss
        if config.charbonnier_strength != 0:
            losses += self.charbonnier_loss(pred, tgt) * config.charbonnier_strength
            
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
                ).mean(mean_dim)
                * config.vb_loss_strength
            )

        losses = sangoi_masked_loss(
            losses, batch["latent_mask"],
            unmasked_weight=config.unmasked_weight,
            normalize=config.normalize_masked_area_loss,
        )

        return losses.mean(mean_dim)

    def __unmasked_losses(
        self,
        batch: dict,
        data: dict,
        config: TrainConfig,
    ):        
        losses = 0
        mean_dim = list(range(1, data['predicted'].ndim))

        pred, tgt = data["predicted"], data["target"]
        if pred.dtype != tgt.dtype:
            tgt = tgt.to(dtype=pred.dtype)

        # MSE/L2 Loss
        if config.mse_strength != 0:
            losses += F.mse_loss(
                data["predicted"],
                data["target"],
                reduction="none",
            ).mean(mean_dim) * config.mse_strength

        # MAE/L1 Loss
        if config.mae_strength != 0:
            losses += F.l1_loss(
                data["predicted"],
                data["target"],
                reduction="none",
            ).mean(mean_dim) * config.mae_strength

        # log-cosh Loss
        if config.log_cosh_strength != 0:
            losses += self.__log_cosh_loss(
                data["predicted"],
                data["target"],
            ).mean(mean_dim) * config.log_cosh_strength

        if config.huber_strength != 0:
            diff = pred - tgt
            snr  = self.__snr(data["timestep"], pred.device)
            losses += self.sangoi_huber_loss(diff=diff, snr=snr).mean(mean_dim) * config.huber_strength # considerando que o hook de grad está ativo
        
        # Charbonnier Loss
        if config.charbonnier_strength != 0:
            losses += self.charbonnier_loss(pred, tgt).mean(mean_dim) * config.charbonnier_strength

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
                ).mean(mean_dim)
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
      timesteps: Tensor, # Atualmente não utilizado, mas mantido para consistência da API
      predicted: Tensor,
      target: Tensor,
      device: torch.device,
      gamma: float,
    ):
      """
      Calcula um peso de loss por amostra inspirado na Focal Loss.
      A loss é ponderada com base no erro da predição, focando o treinamento
      nos exemplos que o modelo mais erra ("hard examples").

      O peso é calculado como: weight = (normalized_error) ^ gamma.
      - Erros pequenos resultam em pesos exponencialmente menores (ex: 0.1^2 = 0.01).
      - Erros grandes resultam em pesos maiores, dominando o gradiente.

      Args:
          timesteps: Tensor com os timesteps. Atualmente ignorado por esta estratégia.
          predicted: Tensor com a predição do modelo.
          target: Tensor com o alvo da predição (ruído).
          device: Dispositivo para os tensores.
          gamma: Expoente de foco (Focal Loss gamma). Um valor > 1.0.
                Valores maiores (ex: 2.0, 3.0) focam o treinamento de forma
                mais agressiva nos erros mais difíceis.

      Returns:
          Um tensor de pesos no formato (batch_size,).
      """
      with torch.no_grad():
        # Fator para normalizar o erro para um intervalo aproximado de [0, 1].
        # Ajuste este valor se seus erros médios por amostra forem consistentemente
        # maiores ou menores que 1.0. Um valor de 1.0 a 2.0 é um bom começo.
        NORMALIZATION_FACTOR = 2

        # 1. Calcular o erro absoluto médio por amostra no batch.
        mae_per_sample = torch.abs(target - predicted).mean(dim=[1, 2, 3])

        # 2. Normalizar o erro para o intervalo [0, 1].
        # Isso é crucial para que o expoente gamma funcione como esperado.
        # Usamos clamp para garantir que o erro normalizado não exceda 1.0.
        normalized_error = torch.clamp(mae_per_sample / NORMALIZATION_FACTOR, max=1.0)

        # 3. Calcular o peso final (Focal Weight).
        # Exemplos com erro baixo (ex: normalized_error=0.1) terão um peso muito baixo (0.1^2=0.01).
        # Exemplos com erro alto (ex: normalized_error=0.9) terão um peso alto (0.9^2=0.81).
        final_weight = normalized_error ** gamma

        # Log para monitoramento (agora com nomes que fazem sentido)
        self.tensorboard.add_scalar(
          "sangoi/1_MAE_Avg", mae_per_sample.mean().item(), self.progress.global_step
        )
        self.tensorboard.add_scalar(
          "sangoi/2_NormalizedError_Avg", normalized_error.mean().item(), self.progress.global_step
        )
        self.tensorboard.add_scalar(
          "sangoi/3_FinalWeight_Avg", final_weight.mean().item(), self.progress.global_step
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
        
        train_gps: TrainGPS | None = getattr(model, 'train_gps', None)

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
            # This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
            if config.masked_training and not config.model_type.has_conditioning_image_input():
                losses = self.__masked_losses(batch, data, config)
            else:
                losses = self.__unmasked_losses(batch, data, config)

        # Scale Losses by Batch and/or GA (if enabled)
        losses = losses * batch_size_scale * gradient_accumulation_steps_scale

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

            if config.train_gps_use_it and train_gps is not None and train_gps.reference_deltas:
              try:
                # Calcula a penalidade usando os pesos *atuais* do modelo
                # e comparando o delta *acumulado atual* com o delta de referência
                penalty = train_gps.compute_penalty(lambda_weight=config.train_gps_weight)

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