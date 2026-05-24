from abc import ABCMeta
from collections.abc import Callable
from typing import Any

from modules.util.config.TrainConfig import TrainConfig
from modules.util.DiffusionScheduleCoefficients import DiffusionScheduleCoefficients
from modules.util.enum.LossWeight import LossWeight
from modules.util.loss.masked_loss import masked_losses, masked_losses_with_prior, sangoi_masked_loss
from modules.util.loss.vb_loss import vb_losses

from pytorch_msssim import ssim
import torch
import torch.nn.functional as F
from torch import Tensor


class ModelSetupDiffusionLossMixin(metaclass=ABCMeta):
    __coefficients: DiffusionScheduleCoefficients | None
    __alphas_cumprod_fun: Callable[[Tensor, int], Tensor] | None
    __sigmas: Tensor | None

    def __init__(self):
        super().__init__()
        self.__coefficients = None
        self.__alphas_cumprod_fun = None
        self.__sigmas = None

    def __log_cosh_loss(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
    ) -> Tensor:
        diff = pred - target
        loss = diff + torch.nn.functional.softplus(-2.0*diff) - torch.log(torch.full(size=diff.size(), fill_value=2.0, dtype=torch.float32, device=diff.device))
        return loss

    def __sangoi_huber_loss(
            self,
            pred: Tensor,
            target: Tensor,
            timesteps: Tensor,
            huber_c: float = 0.1,
    ) -> Tensor:
        diff = pred - target
        snr = self.__snr(timesteps, diff.device).to(device=diff.device, dtype=torch.float32)
        beta_shape = (-1,) + (1,) * (diff.ndim - 1)
        beta = (huber_c * torch.sqrt(snr)).clamp(1e-2, 1.0).view(beta_shape).detach()
        abs_diff = diff.abs()
        return torch.where(
            abs_diff < beta,
            0.5 * abs_diff.square() / beta,
            abs_diff - 0.5 * beta,
        )

    def __sangoi_charbonnier_loss(
            self,
            pred: Tensor,
            target: Tensor,
            eps: float = 1e-3,
            alpha: float = 0.5,
    ) -> Tensor:
        return torch.pow(torch.square(pred - target) + eps * eps, alpha)

    def __sangoi_loss_weighting(
            self,
            predicted: Tensor,
            target: Tensor,
            gamma: float,
            eps: float = 1e-8,
            detach_weight: bool = False,
    ) -> Tensor:
        reduction_dims = tuple(range(1, predicted.ndim))
        mae_per_sample = torch.abs(predicted - target).mean(reduction_dims)
        mean_mae = mae_per_sample.mean().clamp_min(eps)
        weight = ((mae_per_sample + eps) / mean_mae).pow(gamma)

        if detach_weight:
            weight = weight.detach()

        return weight

    def __write_scalar(
            self,
            model: Any | None,
            name: str,
            value: Tensor,
    ):
        if model is None:
            return

        tensorboard = getattr(model, "tensorboard", None)
        train_progress = getattr(model, "train_progress", None)
        if tensorboard is not None and train_progress is not None:
            tensorboard.add_scalar(name, value.detach().mean().item(), train_progress.global_step)

    def __masked_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        losses = 0

        mean_dim = list(range(1, data['predicted'].ndim))

        # MSE/L2 Loss
        if config.mse_strength != 0:
            losses += masked_losses_with_prior(
                losses=F.mse_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                    reduction='none'
                ),
                prior_losses=F.mse_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['prior_target'].to(dtype=torch.float32),
                    reduction='none'
                ) if 'prior_target' in data else None,
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
                masked_prior_preservation_weight=config.masked_prior_preservation_weight,
            ).mean(mean_dim) * config.mse_strength

        # MAE/L1 Loss
        if config.mae_strength != 0:
            losses += masked_losses_with_prior(
                losses=F.l1_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                    reduction='none'
                ),
                prior_losses=F.l1_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['prior_target'].to(dtype=torch.float32),
                    reduction='none'
                ) if 'prior_target' in data else None,
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
                masked_prior_preservation_weight=config.masked_prior_preservation_weight,
            ).mean(mean_dim) * config.mae_strength

        # log-cosh Loss
        if config.log_cosh_strength != 0:
            losses += masked_losses_with_prior(
                losses=self.__log_cosh_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32)
                ),
                prior_losses=self.__log_cosh_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['prior_target'].to(dtype=torch.float32)
                ) if 'prior_target' in data else None,
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
                masked_prior_preservation_weight=config.masked_prior_preservation_weight,
            ).mean(mean_dim) * config.log_cosh_strength

        # Huber Loss
        if config.huber_strength != 0:
            losses += masked_losses_with_prior(
                losses=F.huber_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                    reduction='none',
                    delta=config.huber_delta,
                ),
                prior_losses=F.huber_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['prior_target'].to(dtype=torch.float32),
                    reduction='none',
                    delta=config.huber_delta,
                ) if 'prior_target' in data else None,
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
                masked_prior_preservation_weight=config.masked_prior_preservation_weight,
            ).mean(mean_dim) * config.huber_strength

        # Sangoi dynamic-SNR Huber Loss
        if config.sangoi_huber_strength != 0:
            losses += sangoi_masked_loss(
                losses=self.__sangoi_huber_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                    data['timestep'],
                ),
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean(mean_dim) * config.sangoi_huber_strength

        # Sangoi Charbonnier Loss
        if config.sangoi_charbonnier_strength != 0:
            losses += sangoi_masked_loss(
                losses=self.__sangoi_charbonnier_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                ),
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean(mean_dim) * config.sangoi_charbonnier_strength

        # VB loss
        if config.vb_loss_strength != 0 and 'predicted_var_values' in data and self.__coefficients is not None:
            losses += masked_losses(
                losses=vb_losses(
                    coefficients=self.__coefficients,
                    x_0=data['scaled_latent_image'].to(dtype=torch.float32),
                    x_t=data['noisy_latent_image'].to(dtype=torch.float32),
                    t=data['timestep'],
                    predicted_eps=data['predicted'].to(dtype=torch.float32),
                    predicted_var_values=data['predicted_var_values'].to(dtype=torch.float32),
                ),
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean(mean_dim) * config.vb_loss_strength

        return losses

    def __unmasked_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        losses = 0

        mean_dim = list(range(1, data['predicted'].ndim))

        # MSE/L2 Loss
        if config.mse_strength != 0:
            losses += F.mse_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
                reduction='none'
            ).mean(mean_dim) * config.mse_strength

        # MAE/L1 Loss
        if config.mae_strength != 0:
            losses += F.l1_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
                reduction='none'
            ).mean(mean_dim) * config.mae_strength

        # log-cosh Loss
        if config.log_cosh_strength != 0:
            losses += self.__log_cosh_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32)
                ).mean(mean_dim) * config.log_cosh_strength

        # Huber Loss
        if config.huber_strength != 0:
            losses += F.huber_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
                reduction='none',
                delta=config.huber_delta,
            ).mean(mean_dim) * config.huber_strength

        # Sangoi dynamic-SNR Huber Loss
        if config.sangoi_huber_strength != 0:
            losses += self.__sangoi_huber_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
                data['timestep'],
            ).mean(mean_dim) * config.sangoi_huber_strength

        # Sangoi Charbonnier Loss
        if config.sangoi_charbonnier_strength != 0:
            losses += self.__sangoi_charbonnier_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
            ).mean(mean_dim) * config.sangoi_charbonnier_strength

        # VB loss
        if config.vb_loss_strength != 0 and 'predicted_var_values' in data:
            losses += vb_losses(
                coefficients=self.__coefficients,
                x_0=data['scaled_latent_image'].to(dtype=torch.float32),
                x_t=data['noisy_latent_image'].to(dtype=torch.float32),
                t=data['timestep'],
                predicted_eps=data['predicted'].to(dtype=torch.float32),
                predicted_var_values=data['predicted_var_values'].to(dtype=torch.float32),
            ).mean(mean_dim) * config.vb_loss_strength

        if config.masked_training and config.normalize_masked_area_loss:
            clamped_mask = torch.clamp(batch['latent_mask'], config.unmasked_weight, 1)
            mask_mean = clamped_mask.mean(mean_dim)
            losses /= mask_mean

        return losses

    def __snr(self, timesteps: Tensor, device: torch.device) -> Tensor:
        if self.__coefficients:
            all_snr = (self.__coefficients.sqrt_alphas_cumprod /
                       self.__coefficients.sqrt_one_minus_alphas_cumprod) ** 2
            all_snr.to(device)
            snr = all_snr[timesteps]
        else:
            alphas_cumprod = self.__alphas_cumprod_fun(timesteps, 1)
            snr = alphas_cumprod / (1.0 - alphas_cumprod)

        return snr

    def __min_snr_weight(
            self,
            timesteps: Tensor,
            gamma: float,
            v_prediction: bool,
            device: torch.device
    ) -> Tensor:
        snr = self.__snr(timesteps, device)
        min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
        # Denominator of the snr_weight increased by 1 if v-prediction is being used.
        if v_prediction:
            snr += 1.0
        snr_weight = (min_snr_gamma / snr).to(device)
        return snr_weight

    def __debiased_estimation_weight(
        self,
        timesteps: Tensor,
        v_prediction: bool,
        device: torch.device
    ) -> Tensor:
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

    def _diffusion_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
            train_device: torch.device,
            model: Any | None = None,
            betas: Tensor | None = None,
            alphas_cumprod_fun: Callable[[Tensor, int], Tensor] | None = None,
    ) -> Tensor:
        loss_weight = batch['loss_weight']
        if self.__coefficients is None and betas is not None:
            self.__coefficients = DiffusionScheduleCoefficients.from_betas(betas.to(train_device))

        self.__alphas_cumprod_fun = alphas_cumprod_fun

        if data['loss_type'] == 'target':
            # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
            #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
            if config.masked_training and not config.model_type.has_conditioning_image_input():
                losses = self.__masked_losses(batch, data, config)
            else:
                losses = self.__unmasked_losses(batch, data, config)

        # Scale Losses by Batch and/or GA (if enabled)
        losses = losses * config.loss_scaler.get_scale(batch_size=config.batch_size, accumulation_steps=config.gradient_accumulation_steps)

        losses *= loss_weight

        # Apply timestep based loss weighting.
        if 'timestep' in data:
            v_pred = data.get('prediction_type', '') == 'v_prediction'
            match config.loss_weight_fn:
                case LossWeight.CONSTANT:
                    pass
                case LossWeight.MIN_SNR_GAMMA:
                    losses *= self.__min_snr_weight(data['timestep'], config.loss_weight_strength, v_pred, losses.device)
                case LossWeight.DEBIASED_ESTIMATION:
                    losses *= self.__debiased_estimation_weight(data['timestep'], v_pred, losses.device)
                case LossWeight.P2:
                    losses *= self.__p2_loss_weight(data['timestep'], config.loss_weight_strength, v_pred, losses.device)
                case LossWeight.SANGOI:
                    self.__write_scalar(model, "sangoi/loss_b4_sangoi", losses)
                    losses *= self.__sangoi_loss_weighting(
                        predicted=data['predicted'].to(dtype=torch.float32),
                        target=data['target'].to(dtype=torch.float32),
                        gamma=config.loss_weight_strength,
                    ).to(device=losses.device, dtype=losses.dtype)
                    self.__write_scalar(model, "sangoi/loss_after_sangoi", losses)
                case _:
                    raise NotImplementedError(f"Loss weight function {config.loss_weight_fn} not implemented for diffusion models")

        train_gps = getattr(model, "train_gps", None) if model is not None else None
        if config.train_gps_use_it:
            if train_gps is None:
                raise RuntimeError("TrainGPS is enabled but the model setup did not initialize TrainGPS")
            penalty = train_gps.compute_penalty(lambda_weight=config.train_gps_weight)
            self.__write_scalar(model, "delta/loss_b4_delta", losses)
            self.__write_scalar(model, "delta/penalty", penalty)
            losses += penalty.to(device=losses.device, dtype=losses.dtype)
            self.__write_scalar(model, "delta/loss_after_delta", losses)

        return losses

    def _flow_matching_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
            train_device: torch.device,
            sigmas: Tensor | None = None,
    ) -> Tensor:
        loss_weight = batch['loss_weight']
        if self.__sigmas is None and sigmas is not None:
            num_timesteps = sigmas.shape[0]
            all_timesteps = torch.arange(start=1, end=num_timesteps + 1, step=1, dtype=torch.int32, device=train_device)
            self.__sigmas = all_timesteps / num_timesteps

        if data['loss_type'] == 'target':
            # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
            #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
            if config.masked_training and not config.model_type.has_conditioning_image_input():
                losses = self.__masked_losses(batch, data, config)
            else:
                losses = self.__unmasked_losses(batch, data, config)

        # Scale Losses by Batch and/or GA (if enabled)
        losses = losses * config.loss_scaler.get_scale(config.batch_size, config.gradient_accumulation_steps)
        losses *= loss_weight

        # Apply timestep based loss weighting.
        if 'timestep' in data:
            match config.loss_weight_fn:
                case LossWeight.CONSTANT:
                    pass
                case LossWeight.SIGMA:
                    losses *= self.__sigma_loss_weight(data['timestep'], losses.device)
                case _:
                    raise NotImplementedError(f"Loss weight function {config.loss_weight_fn} not implemented for flow matching models")

        return losses

    def _safe_ssim(self, pred_bf16: Tensor, tgt_bf16: Tensor) -> Tensor:
        ssim_fp32 = ssim(pred_bf16.float(), tgt_bf16.float(), data_range=1.0, size_average=False)
        return ssim_fp32.to(dtype=pred_bf16.dtype)

    def latent_ssim(self, pred_lat: Tensor, tgt_lat: Tensor) -> Tensor:
        if pred_lat.shape[-1] < 128:
            pred_lat = F.interpolate(pred_lat, size=128, mode="nearest")
            tgt_lat = F.interpolate(tgt_lat, size=128, mode="nearest")

        with torch.no_grad():
            ssim32 = ssim(pred_lat.float(), tgt_lat.float(), data_range=1.0, size_average=False, win_size=11)

        return ssim32.to(dtype=pred_lat.dtype)
