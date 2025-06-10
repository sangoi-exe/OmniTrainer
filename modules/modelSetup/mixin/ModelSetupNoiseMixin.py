import math
from abc import ABCMeta

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TimestepDistribution import TimestepDistribution

import torch
from torch import Generator, Tensor


class ModelSetupNoiseMixin(metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

        self.__weights = None

    def _create_noise(
            self,
            source_tensor: Tensor,
            config: TrainConfig,
            generator: Generator
    ) -> Tensor:
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

    def _get_timestep_discrete(
            self,
            num_train_timesteps: int,
            deterministic: bool,
            generator: Generator,
            batch_size: int,
            config: TrainConfig,
            latent_width: int | None = None,
            latent_height: int | None = None,
    ) -> Tensor:
        if deterministic:
            # -1 is for zero-based indexing
            return torch.tensor(
                int(num_train_timesteps * 0.5) - 1,
                dtype=torch.long,
                device=generator.device,
            ).unsqueeze(0)
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
                # continuous implementations
                if config.timestep_distribution == TimestepDistribution.UNIFORM:
                    timestep = min_timestep + (max_timestep - min_timestep) \
                               * torch.rand(batch_size, generator=generator, device=generator.device)
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

                timestep = num_train_timesteps * shift * timestep / ((shift - 1) * timestep + num_train_timesteps)
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
                        self.__weights = weights.to(device=generator.device)

                    samples = torch.multinomial(self.__weights, num_samples=batch_size, replacement=True, generator=generator) + min_timestep
                    timestep = samples.to(dtype=torch.long, device=generator.device)
                elif config.timestep_distribution == TimestepDistribution.SIGMOID:
                    if self.__weights is None:
                        bias = config.noising_bias + 0.5
                        weight = config.noising_weight

                        weights = linspace / (shift - shift * linspace + linspace)
                        weights = 1 / (1 + torch.exp(-weight * (weights - bias)))  # Sigmoid
                        weights *= linspace_derivative
                        self.__weights = weights.to(device=generator.device)

                    samples = torch.multinomial(self.__weights, num_samples=batch_size, replacement=True, generator=generator) + min_timestep
                    timestep = samples.to(dtype=torch.long, device=generator.device)
                elif config.timestep_distribution == TimestepDistribution.PRIORITY_SAMPLING:
                                    device = generator.device
                                    
                                    # --- Inicialização na primeira chamada ---
                                    if self._priority is None:
                                        # Inicializa as prioridades com 1.0 para garantir amostragem uniforme no início.
                                        self._priority = torch.ones(num_train_timesteps, device=device)
                                        
                                        # Guardamos uma cópia das prioridades iniciais para a fase de "boot-strap"
                                        self._unseen = torch.arange(num_train_timesteps, device=device)

                                    # --- Fase 1: Boot-strap (amostrar cada um pelo menos uma vez) ---
                                    # Garante que o modelo veja todo o espectro de timesteps no início.
                                    if self._unseen.numel() > 0:
                                        # Se o batch for maior que o número de timesteps restantes, pegue todos.
                                        k = min(batch_size, self._unseen.numel())
                                        
                                        # Amostra k índices aleatórios da lista de não vistos.
                                        perm = torch.randperm(self._unseen.numel(), generator=generator, device=device)[:k]
                                        timesteps = self._unseen[perm]
                                        
                                        # Remove os timesteps amostrados da lista de não vistos.
                                        # Esta é uma maneira eficiente de fazer isso sem reconstruir a lista toda.
                                        mask = torch.ones(self._unseen.numel(), dtype=torch.bool, device=device)
                                        mask[perm] = False
                                        self._unseen = self._unseen[mask]
                                        
                                        # Se o batch for maior, preencha o restante com amostragem aleatória simples.
                                        if k < batch_size:
                                            remaining = batch_size - k
                                            # Amostra aleatória de todo o range (fallback)
                                            random_timesteps = torch.randint(0, num_train_timesteps, (remaining,), generator=generator, device=device, dtype=torch.long)
                                            timesteps = torch.cat([timesteps, random_timesteps])
                                            
                                        return timesteps.long() # Retorna como long
                                    
                                    # --- Fase 2: Amostragem por Prioridade ---
                                    # Usa as prioridades atualizadas pela função `update_priorities`.
                                    # Softmax com temperatura para converter prioridades em probabilidades.
                                    # A temperatura ajusta o quão "gananciosa" é a amostragem.
                                    # Temp alta -> mais uniforme. Temp baixa -> mais focada nos picos.
                                    probs = torch.softmax(self._priority / config.priority_temperature, dim=0)

                                    # Amostra com base nas probabilidades calculadas.
                                    timesteps = torch.multinomial(
                                        probs,
                                        num_samples=batch_size,
                                        replacement=True,
                                        generator=generator
                                    )
                                    return timesteps.long() # Retorna como long
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

    @torch.no_grad()
    def update_priorities(self,
                          timesteps: torch.Tensor,   # 1-D, shape (bs,), dtype long
                          batch_loss: torch.Tensor,  # 1-D, shape (bs,), dtype float
                          config: TrainConfig):      # Passa a config para pegar os hiperparâmetros
        """
        Atualiza as prioridades de forma vetorial usando EMA e um kernel de espalhamento.
        `batch_loss` deve ser um tensor 1-D com a loss para cada item no batch.
        """
        if self._priority is None or self.config.timestep_distribution != TimestepDistribution.PRIORITY_SAMPLING:
            return  # Sampler não inicializado ou não está em uso

        if batch_loss.dim() == 0:
            batch_loss = batch_loss.expand(timesteps.shape[0])

        # --- Etapa 1: Calcular o valor de atualização da EMA para cada timestep no batch ---
        # `target_priority` é o valor que queremos que a prioridade se aproxime: loss * learning_rate
        # Adicionamos 1.0 como base para garantir exploração.
        target_priority = 1.0 + (batch_loss * config.priority_lr)

        # --- Etapa 2: Aplicar a atualização da EMA de forma vetorial ---
        # Pega as prioridades atuais para os timesteps amostrados.
        old_priorities = self._priority[timesteps]
        
        # Fórmula da EMA: β * old + (1 - β) * new
        # O `new` aqui é o nosso `target_priority`.
        new_priorities = config.priority_beta * old_priorities + (1 - config.priority_beta) * target_priority
        
        # Atualiza o tensor de prioridades principal nos locais corretos.
        # `scatter_` é bom para isso, mas uma simples indexação é mais clara e igualmente eficiente aqui.
        self._priority[timesteps] = new_priorities

        # --- Etapa 3 (Opcional, mas recomendado): Espalhamento (Smearing) Vetorizado ---
        if config.priority_radius > 0:
            radius = config.priority_radius
            
            # Precisamos operar sobre os timesteps únicos para evitar interferência.
            unique_ts = timesteps.unique()
            
            # Cria a matriz de deslocamentos do kernel (de -radius a +radius)
            kernel_offsets = torch.arange(-radius, radius + 1, device=self._priority.device)
            
            # Pega as prioridades atualizadas dos timesteps únicos
            updated_priorities_at_unique_ts = self._priority[unique_ts].unsqueeze(1) # Shape: [num_unique, 1]

            # Calcula os pesos do kernel (triangular)
            # Shape: [1, 2*radius+1]
            kernel_weights = (1.0 - kernel_offsets.abs() / (radius + 1)).unsqueeze(0)
            
            # Calcula as prioridades a serem espalhadas
            # Shape: [num_unique, 2*radius+1]
            priorities_to_spread = updated_priorities_at_unique_ts * kernel_weights

            # Calcula os índices de destino no tensor de prioridade principal
            # Shape: [num_unique, 2*radius+1]
            target_indices = unique_ts.unsqueeze(1) + kernel_offsets.unsqueeze(0)

            # --- Clipping de segurança para evitar erros de índice out-of-bounds ---
            # mascara valores fora do range [0, num_timesteps-1]
            valid_mask = (target_indices >= 0) & (target_indices < self._priority.numel())
            
            # Aplica a máscara para pegar apenas os valores e índices válidos
            flat_target_indices = target_indices[valid_mask]
            flat_priorities_to_spread = priorities_to_spread[valid_mask]

            # --- Operação final de espalhamento ---
            # Usamos `torch.max` para garantir que o espalhamento só aumente as prioridades,
            # nunca diminuindo uma prioridade que já era alta.
            # `scatter_reduce_` com 'amax' é a operação perfeita e mais eficiente para isso.
            self._priority.scatter_reduce_(
                dim=0,
                index=flat_target_indices,
                src=flat_priorities_to_spread,
                reduce="amax", # amax = maximum
                include_self=False # não inclui o valor original no cálculo do max
            )        