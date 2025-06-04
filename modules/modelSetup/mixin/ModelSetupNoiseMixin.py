import math
from abc import ABCMeta

from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TimestepDistribution import TimestepDistribution

import torch
from torch import Generator, Tensor


class ModelSetupNoiseMixin(metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

        # vetor de desempenho e contagem para cada t ∈ [0 .. num_train_timesteps–1]
        self.__epoch_alloc: Tensor | None = None   # quantas vezes cada t deve aparecer (por época)
        self.__weights: Tensor | None = None        # lista embaralhada de timesteps, repetidos conforme alloc
        self.__weights_epoch: int = -1              # marca a época de __weights

        # (Performance tracking)
        self.timestep_perf: Tensor = None           # será setado externamente, no trainer
        self.timestep_count: Tensor = None          # idem
        self.ema_alpha: float = 0.9                 # padrão, mas pode vir do config
        self.burn_in_steps: int = -1                # idem

    def _create_noise(
            self,
            source_tensor: Tensor,
            config: TrainConfig,
            generator: Generator
    ):
        noise = torch.randn(
            source_tensor.shape,
            generator=generator,
            device=config.train_device,
            dtype=source_tensor.dtype
        )

        if config.offset_noise_weight > 0:
            offset_noise = torch.randn(
                (source_tensor.shape[0], source_tensor.shape[1], *[1 for _ in range(source_tensor.ndim - 2)]),
                generator=generator,
                device=config.train_device,
                dtype=source_tensor.dtype
            )
            noise = noise + (config.offset_noise_weight * offset_noise)

        if config.perturbation_noise_weight > 0:
            perturbation_noise = torch.randn(
                source_tensor.shape,
                generator=generator,
                device=config.train_device,
                dtype=source_tensor.dtype
            )
            noise = noise + (config.perturbation_noise_weight * perturbation_noise)

        return noise

    def set_epoch_timestep_alloc(self, alloc: Tensor):
        """
        Recebe um Tensor (num_timesteps,) com quantas vezes cada t deve aparecer 
        na PRÓXIMA época. Constroi self.__weights = lista embaralhada de índices de t.
        """
        self.__epoch_alloc = alloc.to(dtype=torch.long, device=alloc.device)
        # Constroi lista repetida de timesteps
        ts: list[int] = []
        for t, cnt in enumerate(self.__epoch_alloc.tolist()):
            ts += [t] * cnt
        # Embaralha para remover padrão
        import random
        random.shuffle(ts)
        self.__weights = torch.tensor(ts, device=alloc.device, dtype=torch.long)
        self.__weights_epoch = getattr(self, "__weights_epoch", -1) + 1

    def _get_timestep_discrete(
            self,
            num_train_timesteps: int,
            deterministic: bool,
            generator: Generator,
            batch_size: int,
            config: TrainConfig,
            latent_width: int | None = None,
            latent_height: int | None = None,
            train_progress: TrainProgress | None = None,
    ) -> Tensor:
        if deterministic:
            ts = torch.tensor(int(num_train_timesteps * 0.5) - 1, dtype=torch.long, device=generator.device)
            self.current_timestep = ts.unsqueeze(0)
            return self.current_timestep
        else:
            min_timestep = int(num_train_timesteps * config.min_noising_strength)
            max_timestep = int(num_train_timesteps * config.max_noising_strength)
            num_timestep = max_timestep - min_timestep

            shift = config.timestep_shift
            if config.dynamic_timestep_shifting:
                if not latent_width or not latent_height:
                    raise NotImplementedError("Dynamic timestep shifting not support by this model")

                base_seq_len = 256
                max_seq_len = 4096
                base_shift = 0.5
                max_shift = 1.15
                patch_size = 2

                image_seq_len = (latent_width // patch_size) * (latent_height // patch_size)
                m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
                b = base_shift - m * base_seq_len
                mu = image_seq_len * m + b

                shift = math.exp(mu)

            if config.timestep_distribution in [
                TimestepDistribution.UNIFORM,
                TimestepDistribution.LOGIT_NORMAL,
                TimestepDistribution.HEAVY_TAIL
            ]:
                if self.__weights is not None:
                    # Amostragem estratificada/adaptativa: tira da lista de __weights
                    idx = torch.randint(
                        0, 
                        len(self.__weights), 
                        (batch_size,), 
                        generator=generator, 
                        device=generator.device
                    )
                    timestep = self.__weights[idx]
                    self.current_timestep = timestep.detach().clone()
                else:
                    # Fallback para o comportamento original (UNIFORM, LOGIT_NORMAL ou HEAVY_TAIL)
                    if config.timestep_distribution == TimestepDistribution.UNIFORM:
                        timestep = min_timestep + (max_timestep - min_timestep) * torch.rand(
                            batch_size, generator=generator, device=generator.device
                        )
                        self.current_timestep = timestep.detach().clone()
                    elif config.timestep_distribution == TimestepDistribution.LOGIT_NORMAL:
                        bias = config.noising_bias
                        scale = config.noising_weight + 1.0
                        normal = torch.normal(
                            bias, 
                            scale, 
                            size=(batch_size,), 
                            generator=generator, 
                            device=generator.device
                        )
                        logit_normal = normal.sigmoid()
                        timestep = logit_normal * num_timestep + min_timestep
                        self.current_timestep = timestep.detach().clone()
                    else:  # HEAVY_TAIL
                        scale = config.noising_weight
                        u = torch.rand(size=(batch_size,), generator=generator, device=generator.device)
                        u = 1.0 - u - scale * (torch.cos(math.pi / 2.0 * u) ** 2.0 - 1.0 + u)
                        timestep = u * num_timestep + min_timestep
                        self.current_timestep = timestep.detach().clone()

                    # Aplica shift contínuo
                    timestep = num_train_timesteps * shift * timestep / ((shift - 1) * timestep + num_train_timesteps)
                    self.current_timestep = timestep.detach().clone()
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

                        weights = 2.0 / (math.pi - 2.0 * math.pi * linspace + 2.0 * math.pi * linspace ** 2.0)
                        weights *= linspace_derivative
                        self.__weights = weights

                    samples = torch.multinomial(self.__weights, num_samples=batch_size, replacement=True) + min_timestep
                    timestep = samples.to(dtype=torch.long, device=generator.device)
                    self.current_timestep = timestep.detach().clone()
                elif config.timestep_distribution == TimestepDistribution.SIGMOID:
                    # if self.__weights is None:
                    #     bias = config.noising_bias + 0.5
                    #     weight = config.noising_weight

                    #     weights = linspace / (shift - shift * linspace + linspace)
                    #     weights = 1 / (1 + torch.exp(-weight * (weights - bias)))  # Sigmoid
                    #     weights *= linspace_derivative
                    #     self.__weights = weights

                    # testando um schedule pra sigmoid dos timesteps, começa em 0 e fecha no valor definido na UI
                    current_epoch = getattr(train_progress, "epoch", 1)
                    total_epochs = config.epochs
                    
                    if self.__weights is None or self.__weights_epoch != current_epoch:
                        bias = config.noising_bias + 0.5
                        weight = config.noising_weight

                        # Recalcula somente uma vez por época
                        scaled = linspace / (shift - shift * linspace + linspace)
                        
                        # Ajusta 'weight' de 0 até config.noising_weight linearmente
                        prog = float(current_epoch) / float(total_epochs)
                        scheduled_weight = weight * prog

                        weights = 1 / (1 + torch.exp(-scheduled_weight * (scaled - bias)))  # Sigmoid
                        weights *= linspace_derivative

                        self.__weights = weights
                        self.__weights_epoch = current_epoch

                    samples = torch.multinomial(self.__weights, num_samples=batch_size, replacement=True) + min_timestep
                    timestep = samples.to(dtype=torch.long, device=generator.device)
                    self.current_timestep = timestep.detach().clone()

            return timestep.int()

    def _get_timestep_continuous(
            self,
            deterministic: bool,
            generator: Generator,
            batch_size: int,
            config: TrainConfig,
    ) -> Tensor:
        if deterministic:
            return torch.full(
                size=(batch_size,),
                fill_value=0.5,
                device=generator.device,
            )
        else:
            discrete_timesteps = 10000  # Discretize to 10000 timesteps
            discrete = self._get_timestep_discrete(
                num_train_timesteps=discrete_timesteps,
                deterministic=False,
                generator=generator,
                batch_size=batch_size,
                config=config,
            ) + 1

            continuous = (discrete.float() / discrete_timesteps)
            return continuous
