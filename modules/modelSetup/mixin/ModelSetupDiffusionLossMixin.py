from abc import ABCMeta
from collections.abc import Callable
from typing import Any

from modules.sangoi.DynamicLossControl import DynamicLossControl, LossTracker
from modules.util.config.TrainConfig import TrainConfig
from modules.util.DiffusionScheduleCoefficients import DiffusionScheduleCoefficients
from modules.util.enum.LossMode import LossMode
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
    __loss_tracker: LossTracker | None
    __dynamic_loss_control: DynamicLossControl | None
    __dynamic_loss_config_key: tuple[int, bool, bool, float, float] | None

    def __init__(self):
        super().__init__()
        self.__coefficients = None
        self.__alphas_cumprod_fun = None
        self.__sigmas = None
        self.__loss_tracker = None
        self.__dynamic_loss_control = None
        self.__dynamic_loss_config_key = None

    def __log_cosh_loss(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
    ) -> Tensor:
        diff = pred - target
        loss = diff + torch.nn.functional.softplus(-2.0 * diff) - torch.log(
            torch.full(size=diff.size(), fill_value=2.0, dtype=torch.float32, device=diff.device)
        )
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

    @staticmethod
    def __zero_losses(data: dict) -> Tensor:
        return torch.zeros(
            data['predicted'].shape[0],
            device=data['predicted'].device,
            dtype=torch.float32,
        )

    @staticmethod
    def __mean_dim(data: dict) -> list[int]:
        return list(range(1, data['predicted'].ndim))

    @staticmethod
    def __latent_mask(batch: dict, data: dict) -> Tensor:
        return batch['latent_mask'].to(device=data['predicted'].device, dtype=torch.float32)

    @staticmethod
    def __train_progress_fraction(train_progress: Any, total_epochs: int, single_epoch_value: float) -> float:
        if total_epochs <= 1:
            return single_epoch_value

        current_epoch = float(train_progress.epoch)
        return min(max(current_epoch / float(total_epochs - 1), 0.0), 1.0)

    @staticmethod
    def __write_scalar(
            tensorboard: Any | None,
            train_progress: Any,
            name: str,
            value: Tensor | float,
    ) -> None:
        if tensorboard is None:
            return

        if isinstance(value, Tensor):
            scalar_value = value.detach().mean().item()
        else:
            scalar_value = float(value)

        tensorboard.add_scalar(name, scalar_value, train_progress.global_step)

    def __write_model_scalar(
            self,
            model: Any | None,
            name: str,
            value: Tensor | float,
    ) -> None:
        if model is None:
            return

        tensorboard = getattr(model, "tensorboard", None)
        train_progress = getattr(model, "train_progress", None)
        if train_progress is None:
            return

        self.__write_scalar(tensorboard, train_progress, name, value)

    @staticmethod
    def __require_train_progress(model: Any | None, feature_name: str) -> Any:
        if model is None:
            raise ValueError(f"{feature_name} requires a model with train_progress")

        train_progress = getattr(model, "train_progress", None)
        if train_progress is None:
            raise ValueError(f"{feature_name} requires model.train_progress")

        return train_progress

    def __sync_dynamic_loss_state(self, config: TrainConfig) -> None:
        config_key = (
            config.loss_tracker_window,
            config.loss_tracker_use_mad,
            config.dls_use_ema,
            config.dls_ema_decay,
            config.dls_outlier_threshold,
        )

        if self.__dynamic_loss_config_key == config_key:
            return

        self.__loss_tracker = LossTracker(
            window_size=config.loss_tracker_window,
            use_mad=config.loss_tracker_use_mad,
        )
        self.__dynamic_loss_control = DynamicLossControl(
            use_ema=config.dls_use_ema,
            ema_decay=config.dls_ema_decay,
            outlier_threshold=config.dls_outlier_threshold,
        )
        self.__dynamic_loss_config_key = config_key

    def __masked_base_loss_components(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
            compute_mse: bool,
            compute_mae: bool,
            compute_log_cosh: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mean_dim = self.__mean_dim(data)
        latent_mask = self.__latent_mask(batch, data)
        predicted = data['predicted'].to(dtype=torch.float32)
        target = data['target'].to(dtype=torch.float32)
        prior_target = data['prior_target'].to(dtype=torch.float32) if 'prior_target' in data else None
        zero_losses = self.__zero_losses(data)

        if compute_mse:
            mse_losses = masked_losses_with_prior(
                losses=F.mse_loss(predicted, target, reduction='none'),
                prior_losses=F.mse_loss(predicted, prior_target, reduction='none') if prior_target is not None else None,
                mask=latent_mask,
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
                masked_prior_preservation_weight=config.masked_prior_preservation_weight,
            ).mean(mean_dim)
        else:
            mse_losses = zero_losses

        if compute_mae:
            mae_losses = masked_losses_with_prior(
                losses=F.l1_loss(predicted, target, reduction='none'),
                prior_losses=F.l1_loss(predicted, prior_target, reduction='none') if prior_target is not None else None,
                mask=latent_mask,
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
                masked_prior_preservation_weight=config.masked_prior_preservation_weight,
            ).mean(mean_dim)
        else:
            mae_losses = zero_losses

        if compute_log_cosh:
            log_cosh_losses = masked_losses_with_prior(
                losses=self.__log_cosh_loss(predicted, target),
                prior_losses=self.__log_cosh_loss(predicted, prior_target) if prior_target is not None else None,
                mask=latent_mask,
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
                masked_prior_preservation_weight=config.masked_prior_preservation_weight,
            ).mean(mean_dim)
        else:
            log_cosh_losses = zero_losses

        return mse_losses, mae_losses, log_cosh_losses

    def __unmasked_base_loss_components(
            self,
            data: dict,
            compute_mse: bool,
            compute_mae: bool,
            compute_log_cosh: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mean_dim = self.__mean_dim(data)
        predicted = data['predicted'].to(dtype=torch.float32)
        target = data['target'].to(dtype=torch.float32)
        zero_losses = self.__zero_losses(data)

        mse_losses = F.mse_loss(predicted, target, reduction='none').mean(mean_dim) if compute_mse else zero_losses
        mae_losses = F.l1_loss(predicted, target, reduction='none').mean(mean_dim) if compute_mae else zero_losses
        log_cosh_losses = self.__log_cosh_loss(predicted, target).mean(mean_dim) if compute_log_cosh else zero_losses

        return mse_losses, mae_losses, log_cosh_losses

    def __original_base_losses(
            self,
            mse_losses: Tensor,
            mae_losses: Tensor,
            log_cosh_losses: Tensor,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        losses = self.__zero_losses(data)

        if config.mse_strength != 0:
            losses += mse_losses * config.mse_strength
        if config.mae_strength != 0:
            losses += mae_losses * config.mae_strength
        if config.log_cosh_strength != 0:
            losses += log_cosh_losses * config.log_cosh_strength

        return losses

    def __dynamic_base_losses(
            self,
            mse_losses: Tensor,
            mae_losses: Tensor,
            log_cosh_losses: Tensor,
            data: dict,
            config: TrainConfig,
            train_progress: Any,
            tensorboard: Any | None,
    ) -> Tensor:
        self.__sync_dynamic_loss_state(config)
        if self.__loss_tracker is None or self.__dynamic_loss_control is None:
            raise RuntimeError("Dynamic loss state was not initialized")

        mutate_state = torch.is_grad_enabled() and data['predicted'].requires_grad
        if mutate_state:
            self.__loss_tracker.update(mse_losses, mae_losses, log_cosh_losses)

        mse_z, mae_z, log_cosh_z = self.__loss_tracker.compute_z_scores(mse_losses, mae_losses, log_cosh_losses)
        progress_fraction = self.__train_progress_fraction(train_progress, config.epochs, single_epoch_value=0.0)
        mse_weight, mae_weight, log_cosh_weight = self.__dynamic_loss_control.adjust_weights(
            mse_z=mse_z,
            mae_z=mae_z,
            log_cosh_z=log_cosh_z,
            progress_fraction=progress_fraction,
            mutate_ema=mutate_state,
        )

        self.__write_scalar(tensorboard, train_progress, "sangoi/7mse", mse_weight)
        self.__write_scalar(tensorboard, train_progress, "sangoi/8mae", mae_weight)
        self.__write_scalar(tensorboard, train_progress, "sangoi/9log_cosh", log_cosh_weight)

        return (
            mse_losses * mse_weight * config.mse_strength
            + mae_losses * mae_weight * config.mae_strength
            + log_cosh_losses * log_cosh_weight * config.log_cosh_strength
        )

    def __base_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
            masked: bool,
            train_progress: Any | None,
            tensorboard: Any | None,
    ) -> Tensor:
        dynamic_mode = config.loss_mode_fn == LossMode.SANGOI
        compute_mse = dynamic_mode or config.mse_strength != 0
        compute_mae = dynamic_mode or config.mae_strength != 0
        compute_log_cosh = dynamic_mode or config.log_cosh_strength != 0

        if masked:
            mse_losses, mae_losses, log_cosh_losses = self.__masked_base_loss_components(
                batch=batch,
                data=data,
                config=config,
                compute_mse=compute_mse,
                compute_mae=compute_mae,
                compute_log_cosh=compute_log_cosh,
            )
        else:
            mse_losses, mae_losses, log_cosh_losses = self.__unmasked_base_loss_components(
                data=data,
                compute_mse=compute_mse,
                compute_mae=compute_mae,
                compute_log_cosh=compute_log_cosh,
            )

        if config.loss_mode_fn == LossMode.ORIGINAL:
            return self.__original_base_losses(mse_losses, mae_losses, log_cosh_losses, data, config)

        if config.loss_mode_fn == LossMode.SANGOI:
            if train_progress is None:
                raise ValueError("LossMode.SANGOI requires model.train_progress")
            return self.__dynamic_base_losses(
                mse_losses=mse_losses,
                mae_losses=mae_losses,
                log_cosh_losses=log_cosh_losses,
                data=data,
                config=config,
                train_progress=train_progress,
                tensorboard=tensorboard,
            )

        raise NotImplementedError(f"Loss mode {config.loss_mode_fn} is not implemented")

    def __masked_additive_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        losses = self.__zero_losses(data)
        mean_dim = self.__mean_dim(data)
        latent_mask = self.__latent_mask(batch, data)
        predicted = data['predicted'].to(dtype=torch.float32)
        target = data['target'].to(dtype=torch.float32)
        prior_target = data['prior_target'].to(dtype=torch.float32) if 'prior_target' in data else None

        if config.huber_strength != 0:
            losses += masked_losses_with_prior(
                losses=F.huber_loss(predicted, target, reduction='none', delta=config.huber_delta),
                prior_losses=F.huber_loss(predicted, prior_target, reduction='none', delta=config.huber_delta)
                if prior_target is not None else None,
                mask=latent_mask,
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
                masked_prior_preservation_weight=config.masked_prior_preservation_weight,
            ).mean(mean_dim) * config.huber_strength

        if config.sangoi_huber_strength != 0:
            losses += sangoi_masked_loss(
                losses=self.__sangoi_huber_loss(predicted, target, data['timestep']),
                mask=latent_mask,
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean(mean_dim) * config.sangoi_huber_strength

        if config.sangoi_charbonnier_strength != 0:
            losses += sangoi_masked_loss(
                losses=self.__sangoi_charbonnier_loss(predicted, target),
                mask=latent_mask,
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean(mean_dim) * config.sangoi_charbonnier_strength

        if config.vb_loss_strength != 0 and 'predicted_var_values' in data and self.__coefficients is not None:
            losses += masked_losses(
                losses=vb_losses(
                    coefficients=self.__coefficients,
                    x_0=data['scaled_latent_image'].to(dtype=torch.float32),
                    x_t=data['noisy_latent_image'].to(dtype=torch.float32),
                    t=data['timestep'],
                    predicted_eps=predicted,
                    predicted_var_values=data['predicted_var_values'].to(dtype=torch.float32),
                ),
                mask=latent_mask,
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean(mean_dim) * config.vb_loss_strength

        return losses

    def __unmasked_additive_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        losses = self.__zero_losses(data)
        mean_dim = self.__mean_dim(data)
        predicted = data['predicted'].to(dtype=torch.float32)
        target = data['target'].to(dtype=torch.float32)

        if config.huber_strength != 0:
            losses += F.huber_loss(
                predicted,
                target,
                reduction='none',
                delta=config.huber_delta,
            ).mean(mean_dim) * config.huber_strength

        if config.sangoi_huber_strength != 0:
            losses += self.__sangoi_huber_loss(
                predicted,
                target,
                data['timestep'],
            ).mean(mean_dim) * config.sangoi_huber_strength

        if config.sangoi_charbonnier_strength != 0:
            losses += self.__sangoi_charbonnier_loss(
                predicted,
                target,
            ).mean(mean_dim) * config.sangoi_charbonnier_strength

        if config.vb_loss_strength != 0 and 'predicted_var_values' in data:
            losses += vb_losses(
                coefficients=self.__coefficients,
                x_0=data['scaled_latent_image'].to(dtype=torch.float32),
                x_t=data['noisy_latent_image'].to(dtype=torch.float32),
                t=data['timestep'],
                predicted_eps=predicted,
                predicted_var_values=data['predicted_var_values'].to(dtype=torch.float32),
            ).mean(mean_dim) * config.vb_loss_strength

        if config.masked_training and config.normalize_masked_area_loss:
            clamped_mask = torch.clamp(self.__latent_mask(batch, data), config.unmasked_weight, 1)
            losses /= clamped_mask.mean(mean_dim)

        return losses

    def __masked_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
            train_progress: Any | None,
            tensorboard: Any | None,
    ) -> Tensor:
        return self.__base_losses(
            batch=batch,
            data=data,
            config=config,
            masked=True,
            train_progress=train_progress,
            tensorboard=tensorboard,
        ) + self.__masked_additive_losses(batch, data, config)

    def __unmasked_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
            train_progress: Any | None,
            tensorboard: Any | None,
    ) -> Tensor:
        return self.__base_losses(
            batch=batch,
            data=data,
            config=config,
            masked=False,
            train_progress=train_progress,
            tensorboard=tensorboard,
        ) + self.__unmasked_additive_losses(batch, data, config)

    def __snr(self, timesteps: Tensor, device: torch.device) -> Tensor:
        if self.__coefficients:
            all_snr = (self.__coefficients.sqrt_alphas_cumprod /
                       self.__coefficients.sqrt_one_minus_alphas_cumprod) ** 2
            all_snr = all_snr.to(device)
            snr = all_snr[timesteps]
        else:
            alphas_cumprod = self.__alphas_cumprod_fun(timesteps, 1).to(device)
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

    def __sangoi_loss_weighting(
            self,
            timesteps: Tensor,
            predicted: Tensor,
            target: Tensor,
            train_progress: Any,
            total_epochs: int,
            tensorboard: Any | None,
            gamma: float,
            device: torch.device,
    ) -> Tensor:
        snr = self.__snr(timesteps, device).to(device=device, dtype=torch.float32)
        epsilon = 1e-8
        reduction_dims = list(range(1, predicted.ndim))

        mape = torch.abs((target - predicted) / (target + epsilon))
        mape = torch.clamp(mape, min=0, max=1).mean(dim=reduction_dims)

        alpha = self.__train_progress_fraction(train_progress, total_epochs, single_epoch_value=1.0)
        snr_weight_low_first = torch.log(1.0 + 1.0 / (snr + epsilon))
        snr_weight_high_first = torch.log(snr + 1.0)
        scenario_snr_weight = (1.0 - alpha) * snr_weight_low_first + alpha * snr_weight_high_first

        mape_reward = 1 - mape
        raw_reward = torch.exp(-mape_reward * scenario_snr_weight)
        clamped_reward = torch.clamp(raw_reward, min=0.0, max=1.0)
        reward = gamma + (1.0 - gamma) * clamped_reward

        self.__write_scalar(tensorboard, train_progress, "sangoi/1mape_reward", mape_reward)
        self.__write_scalar(tensorboard, train_progress, "sangoi/2scenario_snr_weight", scenario_snr_weight)
        self.__write_scalar(tensorboard, train_progress, "sangoi/3clamped_reward", clamped_reward)
        self.__write_scalar(tensorboard, train_progress, "sangoi/4reward", reward)
        self.__write_scalar(tensorboard, train_progress, "sangoi/alpha", alpha)
        self.__write_scalar(tensorboard, train_progress, "sangoi/scenario_snr_weight_mean", scenario_snr_weight)

        return reward

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
        tensorboard = getattr(model, "tensorboard", None) if model is not None else None
        train_progress = None
        if config.loss_mode_fn == LossMode.SANGOI:
            train_progress = self.__require_train_progress(model, "LossMode.SANGOI")

        if data['loss_type'] == 'target':
            # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
            #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
            if config.masked_training and not config.model_type.has_conditioning_image_input():
                losses = self.__masked_losses(batch, data, config, train_progress, tensorboard)
            else:
                losses = self.__unmasked_losses(batch, data, config, train_progress, tensorboard)

        # Scale Losses by Batch and/or GA (if enabled)
        losses = losses * config.loss_scaler.get_scale(batch_size=config.batch_size, accumulation_steps=config.gradient_accumulation_steps)

        losses *= loss_weight.to(device=losses.device, dtype=losses.dtype)

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
                    weight_progress = train_progress or self.__require_train_progress(model, "LossWeight.SANGOI")
                    self.__write_model_scalar(model, "sangoi/loss_b4_sangoi", losses)
                    losses *= self.__sangoi_loss_weighting(
                        timesteps=data['timestep'],
                        predicted=data['predicted'].to(dtype=torch.float32),
                        target=data['target'].to(dtype=torch.float32),
                        train_progress=weight_progress,
                        total_epochs=config.epochs,
                        tensorboard=tensorboard,
                        gamma=config.loss_weight_strength,
                        device=losses.device,
                    ).to(device=losses.device, dtype=losses.dtype)
                    self.__write_model_scalar(model, "sangoi/loss_after_sangoi", losses)
                case _:
                    raise NotImplementedError(f"Loss weight function {config.loss_weight_fn} not implemented for diffusion models")

        train_gps = getattr(model, "train_gps", None) if model is not None else None
        if config.train_gps_use_it:
            if train_gps is None:
                raise RuntimeError("TrainGPS is enabled but the model setup did not initialize TrainGPS")
            penalty = train_gps.compute_penalty(lambda_weight=config.train_gps_weight)
            self.__write_model_scalar(model, "delta/loss_b4_delta", losses)
            self.__write_model_scalar(model, "delta/penalty", penalty)
            losses += penalty.to(device=losses.device, dtype=losses.dtype)
            self.__write_model_scalar(model, "delta/loss_after_delta", losses)

        return losses

    def _flow_matching_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
            train_device: torch.device,
            sigmas: Tensor | None = None,
    ) -> Tensor:
        if config.loss_mode_fn == LossMode.SANGOI:
            raise ValueError("LossMode.SANGOI is only supported for diffusion losses")

        loss_weight = batch['loss_weight']
        if self.__sigmas is None and sigmas is not None:
            num_timesteps = sigmas.shape[0]
            all_timesteps = torch.arange(start=1, end=num_timesteps + 1, step=1, dtype=torch.int32, device=train_device)
            self.__sigmas = all_timesteps / num_timesteps

        if data['loss_type'] == 'target':
            # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
            #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
            if config.masked_training and not config.model_type.has_conditioning_image_input():
                losses = self.__masked_losses(batch, data, config, train_progress=None, tensorboard=None)
            else:
                losses = self.__unmasked_losses(batch, data, config, train_progress=None, tensorboard=None)

        # Scale Losses by Batch and/or GA (if enabled)
        losses = losses * config.loss_scaler.get_scale(config.batch_size, config.gradient_accumulation_steps)
        losses *= loss_weight.to(device=losses.device, dtype=losses.dtype)

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
