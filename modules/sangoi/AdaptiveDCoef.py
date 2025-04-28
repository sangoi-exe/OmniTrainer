### START AdaptiveDCoef.py #################################################
import math


class AdaptiveDCoef:
    """
    Ajusta d_coef por módulo na Run-2 usando o perfil dumpado pela ModuleDynRecorder.
    Espera encontrar no JSON:
      • 'conv_step'    → passo de convergência
      • 'd_hat_final'  → valor de d_hat no fim (opcional)
      • 'd_coef_base'  → valor inicial de d_coef
    """
    def __init__(self, profile_path: str,
                 gamma: float = 2.0, alpha: float = 1.0, min_scale: float = 1e-3):
        import gzip, json, math
        with gzip.open(profile_path, "rt", encoding="utf-8") as f:
            p = json.load(f)
        # essas chaves vêm do dump() do ModuleDynRecorder
        self.conv_step    = {k: int(v) for k, v in p.get("conv_step", {}).items()}
        self.d_hat_final  = p.get("d_hat_final", {})      # caso queira usar depois
        self.d_coef_base  = p.get("d_coef_base", {})      # nome exato do recorder
        self.gam          = gamma
        self.alpha        = alpha
        self.min_scale    = min_scale
        self.pi           = math.pi

    def scale(self, name: str, step: int, total_steps: int) -> float:
        # pega t0 = passo em que esse módulo convergiu (ou total_steps)
        t0 = self.conv_step.get(name, total_steps) * self.alpha
        # normaliza em [0,1]
        x = (step - t0) / max(1, total_steps - t0)
        x = min(max(x, 0.0), 1.0)
        # cosine fall-off de 1→0
        s = 0.5 * (1 + math.cos(self.pi * x))
        return max(self.min_scale, s ** self.gam)
### END AdaptiveDCoef.py ###################################################
