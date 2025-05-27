from abc import ABCMeta
from collections.abc import Callable
import math
import traceback

from modules.module.AestheticScoreModel import AestheticScoreModel
from modules.module.HPSv2ScoreModel import HPSv2ScoreModel
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

from typing import TYPE_CHECKING, Callable
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
        self.tensorboard = None
        self.progress = None
        self.config = None
        self.loaded_pattern_deltas = None
        self.loss_tracker = LossTracker(window_size=1000, use_mad=True)
        self.dynamic_loss_strengthing = DynamicLossControl()

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

            case config.loss_mode_fn.SANGOI:
                # Update LossTracker
                self.loss_tracker.update(mse_loss, mae_loss, log_cosh_loss)

                # Compute z-scores
                mse_z, mae_z, log_cosh_z = self.loss_tracker.compute_z_scores(mse_loss, mae_loss, log_cosh_loss)

                # Ajusta pesos dinamicamente + scheduler de prioridades
                mse_weight, mae_weight, log_cosh_weight = self.dynamic_loss_strengthing.adjust_weights(
                    mse_z, mae_z, log_cosh_z, config, progress
                )

                losses = (
                    mse_loss * mse_weight * config.mse_strength
                    + mae_loss * mae_weight * config.mae_strength
                    + log_cosh_loss * log_cosh_weight * config.log_cosh_strength
                )

                if self.tensorboard != None:
                    self.tensorboard.add_scalar(
                        "sangoi/7mse",
                        mse_weight,
                        progress.global_step,
                    )
                    self.tensorboard.add_scalar(
                        "sangoi/8mae",
                        mae_weight,
                        progress.global_step,
                    )
                    self.tensorboard.add_scalar(
                        "sangoi/9log_cosh",
                        log_cosh_weight,
                        progress.global_step,
                    )

        return losses

    def __unmasked_losses(
        self,
        batch: dict,
        data: dict,
        config: TrainConfig,
    ):

        progress = self.progress
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

            case config.loss_mode_fn.SANGOI:
                # Update LossTracker
                self.loss_tracker.update(mse_loss, mae_loss, log_cosh_loss)

                # Compute z-scores
                mse_z, mae_z, log_cosh_z = self.loss_tracker.compute_z_scores(mse_loss, mae_loss, log_cosh_loss)

                # Ajusta pesos dinamicamente + scheduler de prioridades
                mse_weight, mae_weight, log_cosh_weight = self.dynamic_loss_strengthing.adjust_weights(
                    mse_z, mae_z, log_cosh_z, config, progress
                )
                losses = (
                    mse_loss * mse_weight * config.mse_strength
                    + mae_loss * mae_weight * config.mae_strength
                    + log_cosh_loss * log_cosh_weight * config.log_cosh_strength
                )

                if self.tensorboard != None:
                    self.tensorboard.add_scalar(
                        "sangoi/7mse",
                        mse_weight,
                        progress.global_step,
                    )
                    self.tensorboard.add_scalar(
                        "sangoi/8mae",
                        mae_weight,
                        progress.global_step,
                    )
                    self.tensorboard.add_scalar(
                        "sangoi/9log_cosh",
                        log_cosh_weight,
                        progress.global_step,
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
    def __sangoi_loss_weighting(
        self,
        timesteps: Tensor,
        predicted: Tensor,
        target: Tensor,
        device: torch.device,
    ):
        """
        Função Sangoi Loss Weighting (2025-05-27c)
        • Calcula bônus percentual (0 → alpha_sangoi) que REDUZ a loss.
        • alpha_sangoi agora é só teto máximo; não entra mais no expoente.
        """
        config = self.config
        eps = 1e-8

        # 1) SNR fora do grafo
        with torch.no_grad():
            snr = self.__snr(timesteps, device) + eps # SNR nunca será zero aqui

        # 2) Qualidade perceptual via SSIM
        if config.full_vae_mf:
            ssim_val = self._safe_ssim(predicted, target)
        else:
            # ATENÇÃO AQUI NA ORDEM!
            # Se latent_ssim(PRED, TARGET), a chamada em _diffusion_losses deve ser (..., data["predicted_image_latent"], data["target_image_latent"], ...)
            # Sua chamada em _diffusion_losses é:
            # data["target_image_latent"], data["predicted_image_latent"]
            # Isso significa que ssim_val está calculando SSIM(TARGET, PREDICTED)
            ssim_val = self.latent_ssim(predicted, target) # 'predicted' aqui é data["target_image_latent"]

        perceptual = (1.0 - ssim_val).clamp_(1e-3, 1.0) # perceptual alto = ruim, perceptual baixo = bom

        # 3) Dificuldade: exp(-snr)
        # SNR é sempre > 0 (devido ao +eps).
        # exp(-snr) será sempre (0, 1]. Próximo de 0 para SNR alto, próximo de 1 para SNR baixo.
        difficulty_weight = torch.exp(-snr)

        # 4) Reward bruto ∈ (0,1]
        # torch.exp(-perceptual): perceptual está em [1e-3, 1.0]
        #   Se perceptual = 1.0 (SSIM baixo), exp(-1) = ~0.36
        #   Se perceptual = 1e-3 (SSIM alto), exp(-1e-3) = ~0.999
        # raw_reward = [~0.36 a ~0.999] * (0 a 1] = (0, ~0.999]
        raw_reward = torch.exp(-perceptual) * difficulty_weight

        # 5) Escala até o teto alpha_sangoi (bônus máximo)
        # config.alpha_sangoi é o seu teto de REDUÇÃO PERCENTUAL.
        # Se config.alpha_sangoi = 0.7 (para 70% de redução máxima),
        # então o FATOR MULTIPLICATIVO da loss deveria ser no mínimo (1.0 - 0.7) = 0.3.
        #
        # A sua linha: reward = raw_reward.clamp_(0.0, 1.0) * alpha_sangoi
        # Se raw_reward.clamp_(0.0, 1.0) é, por exemplo, 0.5
        # E alpha_sangoi = 0.7 (interpretado como 0.7)
        # reward = 0.5 * 0.7 = 0.35.
        #
        # Este 'reward' é então usado como: losses *= reward
        # Então, losses *= 0.35. Isso está correto para REDUZIR a loss.
        #
        # ONDE PODE ZERAR?
        # - Se raw_reward for efetivamente 0 (o que é improvável devido aos clamps e exponenciais).
        # - Se alpha_sangoi for 0.
        # - Se o `losses` original (antes de `losses *= reward`) for 0.

        alpha_sangoi_val = torch.as_tensor( # teto de REDUÇÃO (ex.: 0.5 ⇒ perda mínima 50 %)
            config.alpha_sangoi, device=snr.device, dtype=snr.dtype
        ).clamp_(0.0, 1.0) # Garante que alpha_sangoi_val seja [0,1]
        
        # Mérito bruto ∈[0,1]
        reward_pct = raw_reward.clamp_(0.0, 1.0) 
        
        # 'reward_reduction_percentage' é quanto da redução MÁXIMA (alpha_sangoi_val) será aplicada.
        # Não, esta interpretação está errada.
        # Se alpha_sangoi_val é o teto da REDUÇÃO, ex: 0.7 (70% de redução)
        # E reward_factor_before_alpha é o "mérito" dessa redução, ex: 0.8 (merece 80% do teto)
        # Redução a ser aplicada = 0.8 * 0.7 = 0.56 (56% de redução)
        # Fator multiplicativo da loss = 1.0 - 0.56 = 0.44
        # Esta é a lógica da v7 que discutimos.

        # SUA LÓGICA ATUAL:
        # final_reward_multiplier = reward_factor_before_alpha * alpha_sangoi_val
        # Este `final_reward_multiplier` é o que você usa para `losses *= final_reward_multiplier`.
        # Se `alpha_sangoi_val` for, por exemplo, 0.5 (significando que você quer que a loss seja NO MÁXIMO reduzida pela metade, ou seja, fator 0.5)
        # E `reward_factor_before_alpha` (qualidade*dificuldade) for 0.1 (muito baixo mérito)
        # `final_reward_multiplier` = 0.1 * 0.5 = 0.05.
        # A loss é multiplicada por 0.05 (redução de 95%). Isso parece muito agressivo se alpha_sangoi_val não for interpretado como (1 - teto_redução).

        # *** O PONTO CRÍTICO ***
        # Se `config.alpha_sangoi` no seu arquivo de configuração é 0.0, então `alpha_sangoi_val` será 0.0.
        # Então `final_reward_multiplier` será `reward_factor_before_alpha * 0.0 = 0.0`.
        # E `losses *= 0.0` ZERA A LOSS.

        # START Faixa correta: [alpha_sangoi_val, 1.0]
        # mult = 1 − merito × (1 − alpha)  ⇒
        # merito=0 → mult=1  (sem bônus)
        # merito=1 → mult=alpha (máx. redução)
        final_reward_multiplier = 1.0 - reward_pct * (1.0 - alpha_sangoi_val)
        # final_reward_multiplier = final_reward_multiplier.clamp_(alpha_sangoi_val, 1.0) # aqui nesse caso o alpha é entre 0.5~1, faz carinho se acertar bem no difícil
        final_reward_multiplier = final_reward_multiplier.clamp(1.0, 1.0 + alpha_sangoi_val) # aqui nesse caso o alpha é entre 0~0.5, espanca quando errar no difícil (aparentemente mais eficiente)
        

        # 6) Logs
        step = self.progress.global_step
        tb = self.tensorboard
        tb.add_scalar("sangoi/alpha_sangoi_config_val", float(alpha_sangoi_val.mean() if isinstance(alpha_sangoi_val, Tensor) else alpha_sangoi_val),          step) # Logando o valor de alpha_sangoi usado
        tb.add_scalar("sangoi/ssim_mean",      float(ssim_val.mean()),       step)
        tb.add_scalar("sangoi/difficulty_mean",float(difficulty_weight.mean()), step)
        tb.add_scalar("sangoi/reward_pct", float(reward_pct.mean()), step)
        tb.add_scalar("sangoi/final_reward_multiplier", float(final_reward_multiplier.mean()), step) # Era reward_pct

        return final_reward_multiplier # Este é o fator que multiplica a loss

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
                        "sangoi/5loss_b4_sangoi",
                        losses.mean().item(),
                        self.progress.global_step,
                    )
                    if config.full_vae_mf:
                        losses *= self.__sangoi_loss_weighting(
                            data["timestep"],                        
                            data["predicted_image_rgb"],
                            data["target_image_rgb"],
                            losses.device,                            
                        )
                    else:
                        losses = losses / self.__sangoi_loss_weighting(
                            data["timestep"],                        
                            data["predicted_image_latent"],
                            data["target_image_latent"],
                            losses.device,                            
                        )
                    self.tensorboard.add_scalar(
                        "sangoi/6loss_after_sangoi",
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