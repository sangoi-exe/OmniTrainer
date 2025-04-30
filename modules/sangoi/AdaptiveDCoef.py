import math
import torch
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
    def __init__(self,
                 profile_path: str,
                 gamma: float = 2.0,
                 alpha: float = 1.0,
                 min_scale: float = 1e-3,
                 debug: bool = True,
                 verbose: bool = True):
        import gzip, json  # Keep local imports if preferred style
        import os
        if not os.path.isfile(profile_path):
            raise FileNotFoundError(f"AdaptiveDCoef profile not found at: {profile_path}")

        with gzip.open(profile_path, "rt", encoding="utf-8") as f:
            p = json.load(f)
        # essas chaves vêm do dump() do ModuleDynRecorder
        self.conv_step    = {k: int(v) for k, v in p.get("conv_step", {}).items()}
        self.d_hat_final  = p.get("d_hat_final", {})      # caso queira usar depois
        self.d_coef_base  = {k: float(v) for k, v in p.get("d_coef_base", {}).items()}

        self.gam          = gamma
        self.alpha        = alpha
        self.min_scale    = min_scale

        self.debug       = debug
        self.verbose     = verbose

    def scale(self, name: str, step: int, total_steps: int) -> float:
        """ Método original (CPU) - mantido para referência ou fallback. """
        import math  # Local import for math fallback

        # --- START: debug entry log ---
        if self.debug:
            cs = self.conv_step.get(name, total_steps)
            print(f"[AdaptiveDCoef][DEBUG] scale() name={name}, step={step}, total_steps={total_steps}, conv_step={cs}")
        # --- END: debug entry log ---

        # pega t0 = passo em que esse módulo convergiu (ou total_steps)
        t0 = self.conv_step.get(name, total_steps) * self.alpha
        # normaliza em [0,1]
        denominator = max(1.0, float(total_steps) - t0)
        x = (float(step) - t0) / denominator
        x = min(max(x, 0.0), 1.0)  # Clamp between 0 and 1
        # cosine fall-off de 1→0
        s = 0.5 * (1 + math.cos(math.pi * x))
        result = max(self.min_scale, s ** self.gam)

        # --- START: debug exit log ---
        if self.debug:
            print(f"[AdaptiveDCoef][DEBUG] scale() result for '{name}' = {result:.6f}")
        # --- END: debug exit log ---

        # --- START: verbose log ---
        if self.verbose:
            print(f"[AdaptiveDCoef] scale('{name}', step={step}) -> {result:.6f}")
        # --- END: verbose log ---

        return result

    # START: Add vectorized scale method using torch
    def scale_vectorized(self,
                         names: list[str],
                         step: int,
                         total_steps: int,
                         device: torch.device,
                         dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
        """
        Calcula os fatores de escala para múltiplos módulos de forma vetorizada em um device específico.
        """
        # --- START: debug entry log ---
        if self.debug:
            print(f"[AdaptiveDCoef][DEBUG] scale_vectorized() names={names}, step={step}, total_steps={total_steps}, device={device}")
        # --- END: debug entry log ---

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

        # Normaliza x em [0,1]
        x = (step_tensor.expand_as(t0_tensor) - t0_tensor) / denominator
        x = torch.clamp(x, min=0.0, max=1.0)

        # Cosine fall-off 1→0 usando torch
        s = 0.5 * (1.0 + torch.cos(torch.pi * x))

        # Aplica gamma e garante escala mínima
        scales = torch.clamp(s.pow(self.gam), min=self.min_scale)

        # --- START: debug tensor scales ---
        if self.debug:
            print(f"[AdaptiveDCoef][DEBUG] vectorized scales: {scales}")
        # --- END: debug tensor scales ---

        # --- START: verbose summary ---
        if self.verbose:
            for i, name in enumerate(names):
                print(f"[AdaptiveDCoef] scale_vectorized {name} -> {scales[i].item():.6f}")
        # --- END: verbose summary ---

        return {name: scales[i] for i, name in enumerate(names)}
    # END: Add vectorized scale method using torch
