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

        # Atributos para o novo sistema de amostragem de timestep
        self._timestep_array: Tensor | None = None
        self._timestep_idx_pointer: int = 0
        # self.__weights e self.__weights_epoch não são mais necessários com a nova lógica
        
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

    def _get_timestep_discrete(
            self,
            num_train_timesteps: int,
            deterministic: bool,
            generator: Generator,
            batch_size: int,
            config: TrainConfig,
            train_progress: TrainProgress, # Removido 'Optional' para garantir que sempre estará presente
    ) -> Tensor:
        if deterministic:
            # -1 é para indexação zero-based
            return torch.tensor(
                int(num_train_timesteps * 0.5) - 1,
                dtype=torch.long,
                device=generator.device,
            ).unsqueeze(0).expand(batch_size) # Expande para o batch size
            
        # --- NOVA LÓGICA DE CRIAÇÃO E AMOSTRAGEM ---

        # 1. GERA O ARRAY DE TIMESTEPS APENAS UMA VEZ NO INÍCIO DO TREINO
        if self._timestep_array is None:
            print("[Timestep Sampler] Creating a globally balanced timestep array for the entire training...")
            total_steps = train_progress.total_steps
            
            # Se total_steps não for válido, usa fallback
            if not total_steps or total_steps <= 0:
                print("[Timestep Sampler] Warning: total_steps is invalid. Using fallback random sampling.")
                return self._fallback_random_timestep(num_train_timesteps, generator, batch_size, config)

            min_timestep = int(num_train_timesteps * config.min_noising_strength)
            max_timestep = int(num_train_timesteps * config.max_noising_strength)

            # Lista de timesteps possíveis
            possible_timesteps = torch.arange(min_timestep, max_timestep, device=generator.device)

            # --- LÓGICA DE VIÉS COM SIGMOID ---
            # Normaliza os timesteps para [0, 1] para a sigmoid
            normalized_timesteps = (possible_timesteps.float() - min_timestep) / (max_timestep - min_timestep)
            
            # Hiperparâmetros para controlar a curva sigmoid
            # Podem ser movidos para TrainConfig para fácil ajuste
            sigmoid_bias = config.noising_bias + 0.5
            sigmoid_weight = config.noising_weight
            
            # A sigmoid(x) cresce. Para dar mais peso a valores baixos, usamos sigmoid(-x).
            # Um valor negativo no weight inverte a curva, dando peso a timesteps menores.
            # `(normalized_timesteps - sigmoid_bias)` centraliza a curva antes de aplicar o peso.
            weights = 1 / (1 + torch.exp(sigmoid_weight * (normalized_timesteps - sigmoid_bias)))
            
            # Adiciona um pequeno epsilon para garantir que nenhum peso seja zero
            weights += 1e-8

            # Amostra 'total_steps' índices da distribuição de pesos
            sampled_indices = torch.multinomial(
                weights, 
                num_samples=total_steps, 
                replacement=True, 
                generator=generator
            )
            
            # Cria o array de timesteps com base nos índices amostrados
            arr_tensor = possible_timesteps[sampled_indices]
            
            # Embaralha o array final para garantir aleatoriedade na ordem de aparição
            perm = torch.randperm(total_steps, generator=generator, device=generator.device)
            self._timestep_array = arr_tensor[perm]
            
            # Inicializa o ponteiro
            self._timestep_idx_pointer = 0

            print(f"[Timestep Sampler] Successfully created a biased, globally balanced array of {self._timestep_array.numel()} timesteps.")


        # 2. SELECIONA O PRÓXIMO TIMESTEP USANDO O PONTEIRO
        if self._timestep_array is not None and self._timestep_idx_pointer < self._timestep_array.numel():
            sel_timestep = self._timestep_array[self._timestep_idx_pointer]
            self._timestep_idx_pointer += 1

            # Retorna o mesmo timestep para todas as amostras no batch (se bs > 1)
            return sel_timestep.expand(batch_size)
        
        # 3. FALLBACK: Se o array acabar ou falhar na criação
        print("[Timestep Sampler] Warning: Timestep array exhausted or not created. Using fallback random sampling.")
        return self._fallback_random_timestep(num_train_timesteps, generator, batch_size, config)


    def _fallback_random_timestep(
            self,
            num_train_timesteps: int,
            generator: Generator,
            batch_size: int,
            config: TrainConfig,
    ) -> Tensor:
        # Lógica original como um método de segurança
        min_timestep = int(num_train_timesteps * config.min_noising_strength)
        max_timestep = int(num_train_timesteps * config.max_noising_strength)
        
        # Amostragem aleatória simples como fallback
        timesteps = torch.randint(
            min_timestep,
            max_timestep,
            (batch_size,),
            generator=generator,
            device=generator.device,
            dtype=torch.long
        )
        return timesteps

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