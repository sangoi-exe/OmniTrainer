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
        self._priority:  torch.Tensor | None = None
        self._visited:   torch.Tensor | None = None
        self._unseen:    torch.Tensor | None = None
        self._steps_seen: int = 0

        # Atributos para o novo sistema de amostragem de timestep
        self._timestep_array: Tensor | None = None
        self._timestep_idx_pointer: int = 0
        
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
            generator: torch.Generator,
            batch_size: int,
            config: TrainConfig,
            train_progress: TrainProgress,
    ) -> torch.Tensor:
        """
        Seleciona timesteps:
            1. Boot-strap (sem replacement) até visitar todos.
            2. Depois, amostra com prioridade (softmax+temperatura) corrigida.
        """

        device = generator.device

        # --- Fase determinística -------------------------------------------------
        if deterministic:
            centre = (num_train_timesteps // 2) - 1
            return torch.full((batch_size,), centre, dtype=torch.long, device=device)

        # --- Inicialização -------------------------------------------------------
        if self._priority is None:
            self._priority = torch.ones(num_train_timesteps, device=device)
            self._visited  = torch.zeros(num_train_timesteps, dtype=torch.bool, device=device)
            self._unseen   = torch.arange(num_train_timesteps, device=device)

        # --- Boot-strap sem replacement -----------------------------------------
        if self._unseen.numel() > 0:
            # Amostra SEM replacement dos timesteps ainda não vistos
            perm = torch.randperm(self._unseen.numel(), generator=generator, device=device)
            selected = self._unseen[perm[:batch_size]]
            t = selected

            # Marca visitados e remove da lista
            self._visited[t] = True
            mask_unseen      = ~self._visited[self._unseen]
            self._unseen     = self._unseen[mask_unseen]

            return t.int()

        # --- Amostragem por prioridade ------------------------------------------
        pri = self._priority.clone()

        # Temperatura base (pode subir dinamicamente)
        temperature = 2.0

        # Limite duro para uma única timestep dominar
        P_MAX = 0.20              # ≤ 20 % do batch

        # Laço tenta achar uma temperatura que respeite P_MAX
        for _ in range(8):        # no máximo 8 ajustes (barato)
            prob = torch.softmax(pri / temperature, dim=0)
            if prob.max() <= P_MAX:
                break
            temperature *= 1.25   # achata mais

        # Piso mínimo de exploração
        EPS = 0.30
        prob = (1 - EPS) * prob + EPS / num_train_timesteps

        t = torch.multinomial(prob, batch_size, replacement=True, generator=generator)

        return t.int()

    # Lembre-se que sua função update_priorities precisa ser chamada no trainer.
    # A lógica dela parece ok, mas ela acumula loss indefinidamente.
    # Considere adicionar um fator de decaimento para que as prioridades antigas percam força:
    # self._priority *= 0.999 # Um decaimento lento a cada step de atualização.

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

    @torch.no_grad()
    def update_priorities(self,
                          timesteps: torch.Tensor,   # 1-D, shape (bs,)
                          batch_loss: float,
                          radius: int = 10):
        """
        Soma a loss aos timesteps amostrados e espalha para vizinhos
        (kernel triangular de raio `radius`).
        Chamar depois de cada backward.
        """
        if self._priority is None:
            return  # sampler ainda não inicializado

        for t in timesteps.unique():
            t_int = int(t)
            lo = max(t_int - radius, 0)
            hi = min(t_int + radius + 1, self._priority.numel())
            # peso triangular decresce |Δ|
            distances = torch.arange(lo, hi, device=t.device) - t_int
            weights = 1.0 - (distances.abs() / (radius + 1))
            self._priority[lo:hi] += batch_loss * weights        