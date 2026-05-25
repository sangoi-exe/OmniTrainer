import math
from abc import ABCMeta

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.immiscible_diffusion import immiscible_oversampling
from modules.util.validation_timestep import (
    resolve_continuous_validation_timestep,
    resolve_discrete_validation_timestep,
)

import torch
import torch.nn.functional
from torch import Generator, Tensor


class ModelSetupNoiseMixin(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        self.__weights = None
        self.__speed_weights: Tensor | None = None
        self.__speed_meaningful_steps_end: int | None = None
        self._offset_noise_psi_schedule: Tensor | None = None
        self._priority: Tensor | None = None
        self._unseen: Tensor | None = None

    def _compute_and_cache_offset_noise_psi_schedule(self, betas: Tensor) -> Tensor:
        """
        Computes the time-dependent psi_t coefficients for generalized offset noise.
        This implementation follows the paper "Generalized Diffusion Model with Adjusted Offset Noise",
        specifically Equation (34) and the logic of Algorithm 1 for the "balanced-phi_t, psi_t strategy".
        """
        if self._offset_noise_psi_schedule is not None and self._offset_noise_psi_schedule.shape[0] == betas.shape[0]:
            return self._offset_noise_psi_schedule.to(betas.device).to(torch.float64)

        betas = betas.to(torch.float64)
        T = betas.shape[0]
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # From paper footnote 4: "we introduce α_0 = 1 for convenience".
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=betas.device, dtype=betas.dtype), alphas_cumprod[:-1]]
        )

        # --- Start of Algorithm 1 ---
        gammas = torch.zeros(T, device=betas.device, dtype=betas.dtype)

        # Step 1: Set gamma_1 = 1
        gammas[0] = 1.0

        # This sum is `Σ_{i=1 to t-1} γ_i/√¯αᵢ₋₁` which we build iteratively.
        cumulative_sum_term = gammas[0] / torch.sqrt(alphas_cumprod_prev[0])

        # Step 2-4: Loop for t = 2 to T (in code: t = 1 to T-1)
        for t in range(1, T):
            alpha_t = alphas[t]
            alpha_cumprod_tm1 = alphas_cumprod_prev[t]

            # Denominator from the paper's formula for C_t.
            c_t_denominator = alpha_t * (1 - alpha_cumprod_tm1)
            c_t = (1 - alpha_t) * torch.sqrt(alpha_cumprod_tm1) / c_t_denominator

            # Paper's recursive formula uses the full cumulative sum.
            gammas[t] = c_t * cumulative_sum_term

            # Update the sum for the next iteration.
            cumulative_sum_term += gammas[t] / torch.sqrt(alphas_cumprod_prev[t])

        # Step 5: Calculate normalization factor psi_T
        psi_T_denominator = torch.sqrt(1 - alphas_cumprod[-1])
        psi_T = cumulative_sum_term / psi_T_denominator

        # Step 6-8: Normalize gammas
        gammas_normalized = gammas / psi_T
        # --- End of Algorithm 1 ---

        # Finally, calculate the psi schedule for all timesteps t using Equation (22)
        terms = gammas_normalized / torch.sqrt(alphas_cumprod_prev)
        s_cumulative = torch.cumsum(terms, dim=0)
        psi_schedule = s_cumulative / torch.sqrt(1 - alphas_cumprod)

        self._offset_noise_psi_schedule = psi_schedule.to(betas.device)
        return self._offset_noise_psi_schedule

    def _create_noise(
        self,
        source_tensor: Tensor,
        config: TrainConfig,
        generator: Generator,
        timestep: Tensor | None = None,
        betas: Tensor | None = None,
    ) -> Tensor:
        if config.k_noise_sampling > 1:
            noise_candidates = torch.randn(
                (source_tensor.shape[0], config.k_noise_sampling, *source_tensor.shape[1:]),
                generator=generator,
                device=config.train_device,
                dtype=source_tensor.dtype,
            )
            noise = immiscible_oversampling(source_tensor, noise_candidates)
        else:
            noise = torch.randn(
                source_tensor.shape, generator=generator, device=config.train_device, dtype=source_tensor.dtype
            )

        if config.offset_noise_weight > 0:
            offset_noise = torch.randn(
                (source_tensor.shape[0], source_tensor.shape[1], *[1 for _ in range(source_tensor.ndim - 2)]),
                generator=generator,
                device=config.train_device,
                dtype=source_tensor.dtype,
            )
            # Use the time-dependent generalized method if enabled.
            # This will only be true for Diffusion models (which uses betas)
            if config.generalized_offset_noise and timestep is not None and betas is not None:
                psi_schedule = self._compute_and_cache_offset_noise_psi_schedule(betas).to(timestep.device)
                psi_t = psi_schedule[timestep]
                psi_t = psi_t.view(psi_t.shape[0], *[1 for _ in range(source_tensor.ndim - 1)])
                # Scale by the time-dependent psi_t factor
                noise = noise + (psi_t * config.offset_noise_weight * offset_noise)
            else:  # Otherwise, use the normal offset noise.
                noise = noise + (config.offset_noise_weight * offset_noise)

        if config.perturbation_noise_weight > 0:
            perturbation_noise = torch.randn(
                source_tensor.shape, generator=generator, device=config.train_device, dtype=source_tensor.dtype
            )
            noise = noise + (config.perturbation_noise_weight * perturbation_noise)

        return noise

    def _apply_conditional_embedding_perturbation(
        self,
        embedding: Tensor | list[Tensor],
        gamma: float,
        generator: Generator,
    ) -> Tensor | list[Tensor]:
        def perturb(tensor: Tensor) -> Tensor:
            if tensor.shape[-1] <= 0:
                raise ValueError("embedding dimension must be positive")
            scale = math.sqrt(gamma / tensor.shape[-1])
            noise = torch.rand(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype)
            return tensor + (noise.mul(2.0).sub(1.0) * scale)

        if isinstance(embedding, list):
            return [perturb(tensor) for tensor in embedding]
        return perturb(embedding)

    def _apply_ciop(
        self,
        noisy_latent: Tensor,
        target_noise: Tensor,
        config: TrainConfig,
        generator: Generator,
    ) -> tuple[Tensor, Tensor]:
        if config.ciop_noise_weight == 0:
            return noisy_latent, target_noise

        apply_mask = torch.rand(1, generator=generator, device=noisy_latent.device) < config.ciop_p
        if not bool(apply_mask.item()):
            return noisy_latent, target_noise

        noisy_latent = noisy_latent + torch.randn(
            noisy_latent.shape,
            generator=generator,
            device=noisy_latent.device,
            dtype=noisy_latent.dtype,
        ) * config.ciop_noise_weight
        target_noise = target_noise + torch.randn(
            target_noise.shape,
            generator=generator,
            device=target_noise.device,
            dtype=target_noise.dtype,
        ) * config.ciop_noise_weight
        return noisy_latent, target_noise

    @staticmethod
    def __sample_gamma_unit(concentration: float, size: int, generator: Generator, device: torch.device) -> Tensor:
        concentration = max(1e-4, concentration)

        def sample_scalar(shape: float) -> Tensor:
            if shape < 1.0:
                uniform = torch.rand((), generator=generator, device=device).clamp_min(1e-12)
                return sample_scalar(shape + 1.0) * uniform.pow(1.0 / shape)

            d = shape - (1.0 / 3.0)
            c = 1.0 / math.sqrt(9.0 * d)
            while True:
                x = torch.randn((), generator=generator, device=device)
                v = (1.0 + c * x).pow(3)
                if v.item() <= 0:
                    continue
                uniform = torch.rand((), generator=generator, device=device)
                if (uniform < 1.0 - 0.0331 * x.pow(4)).item():
                    return d * v
                if (torch.log(uniform) < 0.5 * x.pow(2) + d * (1.0 - v + torch.log(v))).item():
                    return d * v

        return torch.stack([sample_scalar(concentration) for _ in range(size)])

    @staticmethod
    def __sample_beta_unit(alpha: float, beta: float, size: int, generator: Generator, device: torch.device) -> Tensor:
        alpha = max(1e-4, alpha)
        beta = max(1e-4, beta)

        if abs(beta - 1.0) < 1e-6:
            uniform = torch.rand(size, generator=generator, device=device)
            return uniform.pow(1.0 / alpha)
        if abs(alpha - 1.0) < 1e-6:
            uniform = torch.rand(size, generator=generator, device=device)
            return 1.0 - uniform.pow(1.0 / beta)

        alpha_sample = ModelSetupNoiseMixin.__sample_gamma_unit(alpha, size, generator, device)
        beta_sample = ModelSetupNoiseMixin.__sample_gamma_unit(beta, size, generator, device)
        return alpha_sample / (alpha_sample + beta_sample).clamp_min(1e-12)

    def __get_speed_weights(
        self,
        num_train_timesteps: int,
        generator: Generator,
        betas: Tensor | None,
        sigmas: Tensor | None,
    ) -> Tensor:
        if self.__speed_weights is not None and self.__speed_weights.shape[0] == num_train_timesteps:
            return self.__speed_weights

        gradient = None
        if sigmas is not None:
            gradient = torch.gradient(sigmas.to(device=generator.device, dtype=torch.float32))[0]
        elif betas is not None:
            betas = betas.to(device=generator.device, dtype=torch.float32)
            alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
            gradient = torch.gradient(torch.sqrt(1.0 - alphas_cumprod))[0]

        if gradient is None:
            raise ValueError("SPEED timestep distribution requires betas or sigmas")

        threshold = 1e-4
        weights = torch.tanh(1e6 * (gradient - threshold)) + 1.5
        self.__speed_weights = torch.nn.functional.normalize(weights, p=1, dim=0)

        meaningful_end_candidates = (gradient < threshold).nonzero(as_tuple=True)[0]
        if meaningful_end_candidates.numel() > 0:
            self.__speed_meaningful_steps_end = int(meaningful_end_candidates[0].item())
        else:
            self.__speed_meaningful_steps_end = num_train_timesteps - 1

        return self.__speed_weights

    def _get_timestep_discrete(
        self,
        num_train_timesteps: int,
        deterministic: bool,
        generator: Generator,
        batch_size: int,
        config: TrainConfig,
        shift: float = None,
        betas: Tensor | None = None,
        sigmas: Tensor | None = None,
        validation_index=None,
        validation_count=None,
    ) -> Tensor:
        if shift is None:
            shift = config.timestep_shift

        if deterministic:
            return resolve_discrete_validation_timestep(
                config=config,
                num_train_timesteps=num_train_timesteps,
                batch_size=batch_size,
                device=generator.device,
                shift=shift,
                validation_index=validation_index,
                validation_count=validation_count,
            )
        else:
            min_timestep = int(num_train_timesteps * config.min_noising_strength)
            max_timestep = int(num_train_timesteps * config.max_noising_strength)
            num_timestep = max_timestep - min_timestep

            if config.timestep_distribution == TimestepDistribution.SPEED:
                speed_weights = self.__get_speed_weights(num_train_timesteps, generator, betas, sigmas)
                initial_sample_count = (batch_size + 1) // 2
                initial_samples = torch.multinomial(
                    speed_weights,
                    num_samples=initial_sample_count,
                    replacement=True,
                    generator=generator,
                )
                mirrored_samples = torch.where(
                    initial_samples < self.__speed_meaningful_steps_end,
                    self.__speed_meaningful_steps_end - initial_samples,
                    initial_samples - self.__speed_meaningful_steps_end,
                )
                return torch.cat([initial_samples, mirrored_samples], dim=0)[:batch_size].clamp(
                    min_timestep, max_timestep - 1
                ).long()

            if config.timestep_distribution in [
                TimestepDistribution.UNIFORM,
                TimestepDistribution.LOGIT_NORMAL,
                TimestepDistribution.HEAVY_TAIL,
                TimestepDistribution.BETA,
            ]:
                # continuous implementations
                if config.timestep_distribution == TimestepDistribution.UNIFORM:
                    timestep = min_timestep + (max_timestep - min_timestep) * torch.rand(
                        batch_size, generator=generator, device=generator.device
                    )
                elif config.timestep_distribution == TimestepDistribution.LOGIT_NORMAL:
                    bias = config.noising_bias
                    scale = config.noising_weight + 1.0

                    normal = torch.normal(bias, scale, size=(batch_size,), generator=generator, device=generator.device)
                    logit_normal = normal.sigmoid()
                    timestep = logit_normal * num_timestep + min_timestep
                elif config.timestep_distribution == TimestepDistribution.HEAVY_TAIL:
                    scale = config.noising_weight

                    u = torch.rand(
                        size=(batch_size,),
                        generator=generator,
                        device=generator.device,
                    )
                    u = 1.0 - u - scale * (torch.cos(math.pi / 2.0 * u) ** 2.0 - 1.0 + u)
                    timestep = u * num_timestep + min_timestep
                elif config.timestep_distribution == TimestepDistribution.BETA:
                    beta_sample = self.__sample_beta_unit(
                        config.noising_weight,
                        config.noising_bias,
                        batch_size,
                        generator,
                        generator.device,
                    )
                    timestep = beta_sample * num_timestep + min_timestep

                timestep = num_train_timesteps * shift * timestep / ((shift - 1) * timestep + num_train_timesteps)
                timestep = timestep.clamp(min_timestep, max_timestep - 1)
            else:
                # Shifting a discrete distribution is done in two steps:
                # 1. Apply the inverse shift to the linspace.
                #    This moves the sample points of the function to their shifted place.
                # 2. Multiply the result with the derivative of the inverse shift function.
                #    The derivative is an approximation of the distance between sample points.
                #    Or in other words, the size of a shifted bucket in the original function.
                linspace = torch.linspace(0, 1, num_timestep)
                linspace = linspace / (shift - shift * linspace + linspace)

                linspace_derivative = torch.linspace(0, 1, num_timestep)
                linspace_derivative = shift / (shift + linspace_derivative - (linspace_derivative * shift)).pow(2)

                # continuous implementations
                if config.timestep_distribution == TimestepDistribution.COS_MAP:
                    if self.__weights is None:
                        weights = 2.0 / (math.pi - 2.0 * math.pi * linspace + 2.0 * math.pi * linspace**2.0)
                        weights *= linspace_derivative
                        self.__weights = weights.to(device=generator.device)
                elif config.timestep_distribution == TimestepDistribution.SIGMOID:
                    if self.__weights is None:
                        bias = config.noising_bias + 0.5
                        weight = config.noising_weight

                        weights = linspace / (shift - shift * linspace + linspace)
                        weights = 1 / (1 + torch.exp(-weight * (weights - bias)))  # Sigmoid
                        weights *= linspace_derivative
                        self.__weights = weights.to(device=generator.device)
                elif config.timestep_distribution == TimestepDistribution.INVERTED_PARABOLA:
                    if self.__weights is None:
                        bias = config.noising_bias + 0.5
                        weight = config.noising_weight

                        weights = torch.clamp(-weight * ((linspace - bias) ** 2) + 2, min=0.0)
                        weights *= linspace_derivative
                        self.__weights = weights.to(device=generator.device)
                elif config.timestep_distribution == TimestepDistribution.PRIORITY_SAMPLING:
                    if self._priority is None or self._priority.shape[0] != num_train_timesteps:
                        self._priority = torch.ones(num_train_timesteps, device=generator.device)
                        self._unseen = torch.arange(min_timestep, max_timestep, device=generator.device)

                    if self._unseen.numel() > 0:
                        unseen_sample_count = min(batch_size, self._unseen.numel())
                        unseen_indexes = torch.randperm(
                            self._unseen.numel(),
                            generator=generator,
                            device=generator.device,
                        )[:unseen_sample_count]
                        samples = self._unseen[unseen_indexes]

                        unseen_mask = torch.ones(self._unseen.numel(), dtype=torch.bool, device=generator.device)
                        unseen_mask[unseen_indexes] = False
                        self._unseen = self._unseen[unseen_mask]

                        if unseen_sample_count < batch_size:
                            random_samples = torch.randint(
                                min_timestep,
                                max_timestep,
                                (batch_size - unseen_sample_count,),
                                generator=generator,
                                device=generator.device,
                                dtype=torch.long,
                            )
                            samples = torch.cat([samples, random_samples])

                        return samples.long()

                    probabilities = torch.softmax(
                        self._priority[min_timestep:max_timestep] / config.priority_temperature,
                        dim=0,
                    )
                    samples = torch.multinomial(
                        probabilities,
                        num_samples=batch_size,
                        replacement=True,
                        generator=generator,
                    )
                    samples = samples + min_timestep
                    timestep = samples.to(dtype=torch.long, device=generator.device)
                else:
                    samples = (
                        torch.multinomial(self.__weights, num_samples=batch_size, replacement=True, generator=generator)
                        + min_timestep
                    )
                    timestep = samples.to(dtype=torch.long, device=generator.device)

            return timestep.int()

    def _get_timestep_continuous(
        self,
        deterministic: bool,
        generator: Generator,
        batch_size: int,
        config: TrainConfig,
        validation_index=None,
        validation_count=None,
    ) -> Tensor:
        if deterministic:
            return resolve_continuous_validation_timestep(
                config=config,
                batch_size=batch_size,
                device=generator.device,
                validation_index=validation_index,
                validation_count=validation_count,
            )
        else:
            discrete_timesteps = 10000  # Discretize to 10000 timesteps
            discrete = (
                self._get_timestep_discrete(
                    num_train_timesteps=discrete_timesteps,
                    deterministic=False,
                    generator=generator,
                    batch_size=batch_size,
                    config=config,
                )
                + 1
            )

            continuous = discrete.float() / discrete_timesteps
            return continuous

    @torch.no_grad()
    def update_priorities(
        self,
        timesteps: Tensor,
        batch_loss: Tensor | None,
        config: TrainConfig,
    ):
        if config.timestep_distribution != TimestepDistribution.PRIORITY_SAMPLING:
            return
        if self._priority is None:
            raise ValueError("priority sampling state is missing")
        if batch_loss is None:
            raise ValueError("priority sampling requires per-sample loss")

        timesteps = timesteps.to(device=self._priority.device, dtype=torch.long)
        batch_loss = batch_loss.detach().to(device=self._priority.device, dtype=self._priority.dtype)
        if batch_loss.dim() == 0:
            raise ValueError("priority sampling requires per-sample loss, not a scalar")
        if batch_loss.shape[0] != timesteps.shape[0]:
            raise ValueError("priority sampling loss length must match timestep length")
        if not torch.isfinite(batch_loss).all():
            raise ValueError("priority sampling loss must be finite")
        if (batch_loss < 0).any():
            raise ValueError("priority sampling loss must be non-negative")

        target_priority = 1.0 + (batch_loss * config.priority_lr)
        old_priorities = self._priority[timesteps]
        self._priority[timesteps] = config.priority_beta * old_priorities + (1 - config.priority_beta) * target_priority

        if config.priority_radius > 0:
            radius = config.priority_radius
            unique_timesteps = timesteps.unique()
            kernel_offsets = torch.arange(-radius, radius + 1, device=self._priority.device)
            kernel_weights = (1.0 - kernel_offsets.abs() / (radius + 1)).unsqueeze(0)
            priorities_to_spread = self._priority[unique_timesteps].unsqueeze(1) * kernel_weights
            target_indexes = unique_timesteps.unsqueeze(1) + kernel_offsets.unsqueeze(0)
            valid_indexes = (target_indexes >= 0) & (target_indexes < self._priority.numel())

            self._priority.scatter_reduce_(
                dim=0,
                index=target_indexes[valid_indexes],
                src=priorities_to_spread[valid_indexes],
                reduce="amax",
                include_self=True,
            )
