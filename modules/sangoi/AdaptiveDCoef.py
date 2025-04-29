### START AdaptiveDCoef.py #################################################
import math
# START: Import torch
import torch
# END: Import torch
import gzip
import json


class AdaptiveDCoef:
    """
    Ajusta d_coef por módulo na Run-2 usando o perfil dumpado pela ModuleDynRecorder.
    Espera encontrar no JSON:
      • 'conv_step'    → passo de convergência
      • 'd_hat_final'  → valor de d_hat no fim (opcional)
      • 'd_coef_base'  → valor inicial de d_coef

    Modificado para suportar cálculos vetorizados em GPU com torch.
    """
    def __init__(self, profile_path: str,
                 gamma: float = 2.0, alpha: float = 1.0, min_scale: float = 1e-3):
        import gzip, json, math
        with gzip.open(profile_path, "rt", encoding="utf-8") as f:
            p = json.load(f)
        # essas chaves vêm do dump() do ModuleDynRecorder
        self.conv_step    = {k: int(v) for k, v in p.get("conv_step", {}).items()}
        self.d_hat_final  = p.get("d_hat_final", {})      # caso queira usar depois
        self.d_coef_base  = {k: float(v) for k, v in p.get("d_coef_base", {}).items()} # Store as float
        self.gam          = gamma
        self.alpha        = alpha
        self.min_scale    = min_scale
        # START: Remove math.pi, use torch.pi
        # self.pi           = math.pi
        # END: Remove math.pi, use torch.pi

    def scale(self, name: str, step: int, total_steps: int) -> float:
        """ Método original (CPU) - mantido para referência ou fallback. """
        # pega t0 = passo em que esse módulo convergiu (ou total_steps)
        t0 = self.conv_step.get(name, total_steps) * self.alpha
        # normaliza em [0,1]
        # Ensure denominator is not zero, minimum 1
        denominator = max(1.0, float(total_steps) - t0)
        x = (float(step) - t0) / denominator
        x = min(max(x, 0.0), 1.0) # Clamp between 0 and 1
        # cosine fall-off de 1→0
        s = 0.5 * (1 + math.cos(math.pi * x))
        return max(self.min_scale, s ** self.gam)

    # START: Add vectorized scale method using torch
    def scale_vectorized(self,
                         names: list[str],
                         step: int,
                         total_steps: int,
                         device: torch.device,
                         dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
        """
        Calcula os fatores de escala para múltiplos módulos de forma vetorizada em um device específico.

        Args:
            names: Lista de nomes de módulos para calcular a escala.
            step: Passo de treino atual.
            total_steps: Total de passos de treino estimados.
            device: Device torch (e.g., 'cuda:0') para realizar os cálculos.
            dtype: Dtype torch para os cálculos.

        Returns:
            Dicionário mapeando nome do módulo para seu fator de escala (tensor escalar no device).
        """
        if not names:
            return {}

        # Coleta dados necessários para os módulos solicitados
        conv_steps_list = [self.conv_step.get(name, total_steps) for name in names]

        # Cria tensores no device alvo
        t0_tensor = torch.tensor(conv_steps_list, device=device, dtype=dtype) * self.alpha
        step_tensor = torch.tensor(float(step), device=device, dtype=dtype)
        total_steps_tensor = torch.tensor(float(total_steps), device=device, dtype=dtype)

        # Evita divisão por zero ou valores negativos no denominador
        denominator = torch.clamp(total_steps_tensor - t0_tensor, min=1.0)

        # Normaliza x em [0, 1]
        x = (step_tensor - t0_tensor) / denominator
        x = torch.clamp(x, min=0.0, max=1.0)

        # Cosine fall-off 1 -> 0 usando torch
        # torch.pi requires torch 1.8+
        s = 0.5 * (1.0 + torch.cos(torch.pi * x))

        # Aplica gamma e garante escala mínima
        scales = torch.clamp(s.pow(self.gam), min=self.min_scale)

        # Retorna dicionário mapeando nome -> tensor escalar de escala
        return {name: scales[i] for i, name in enumerate(names)}
    # END: Add vectorized scale method using torch

### END AdaptiveDCoef.py ###################################################