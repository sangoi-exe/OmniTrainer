import math
import traceback
import torch
import gzip
import json
import os
from typing import Dict, List, Tuple, Any, Optional


class AdaptiveDCoef:
    """
    Ajusta d_coef por módulo na Run-2 usando o perfil dumpado.
    Pode também expor dados carregados do perfil para outros módulos.
    """

    # Manter sincronizado com ConvergeControl
    ConvergencePoint = Tuple[
        int,
        float,
        float,
        float,
        float,
        float,
        Optional[float],
        Optional[float],
        Optional[float],
        bool,
    ]

    def __init__(
        self,
        profile_path: str,
        gamma: float = 2.0,
        alpha: float = 1.0,
        min_scale: float = 1e-3,
        debug: bool = True,
        verbose: bool = True,
    ):

        if not os.path.isfile(profile_path):
            raise FileNotFoundError(
                f"AdaptiveDCoef profile not found at: {profile_path}"
            )

        self.profile_path = profile_path
        self.gam = gamma
        self.alpha = alpha
        self.min_scale = min_scale
        self.debug = debug
        self.verbose = verbose

        # --- Dados Carregados ---
        self.conv_step: Dict[str, int] = {}
        self.d_hat_final: Dict[str, float] = {}
        self.d_coef_base: Dict[str, float] = {}
        self.convergence_history: Dict[str, List[AdaptiveDCoef.ConvergencePoint]] = {}
        self.convergence_history_schema: List[str] = []

        self._load_profile()

        if self.debug:
            print(
                f"[AdaptiveDCoef] Perfil '{profile_path}' carregado. "
                f"d_coef_base: {len(self.d_coef_base)} mods, "
                f"conv_hist: {len(self.convergence_history)} mods, "
                f"conv_step: {len(self.conv_step)} mods."
            )  # Adicionado conv_step ao log

    def _load_profile(self):
        """Carrega todos os dados relevantes do arquivo de perfil."""
        try:
            with gzip.open(self.profile_path, "rt", encoding="utf-8") as f:
                p = json.load(f)

            self.conv_step = {k: int(v) for k, v in p.get("conv_step", {}).items()}
            self.d_hat_final = p.get("d_hat_final", {})
            self.d_coef_base = {
                k: float(v) for k, v in p.get("d_coef_base", {}).items()
            }
            self.convergence_history_schema = p.get("convergence_history_schema", [])
            raw_history = p.get("convergence_history", {})
            if raw_history:
                # TODO: Validar schema se necessário
                self.convergence_history = {
                    name: [tuple(point) for point in points]
                    for name, points in raw_history.items()
                }
        except Exception as e:
            print(
                f"[AdaptiveDCoef] ERRO ao carregar perfil de '{self.profile_path}': {e}"
            )
            traceback.print_exc()
            self.conv_step = {}
            self.d_hat_final = {}
            self.d_coef_base = {}
            self.convergence_history = {}
            self.convergence_history_schema = []

    def scale(self, name: str, step: int, total_steps: int) -> float:
        """Método original (CPU) - requer `conv_step`."""
        if not self.conv_step:
            if self.debug:
                print(
                    f"[AdaptiveDCoef Scale CPU] Aviso: conv_step não carregado, retornando 1.0 para '{name}'"
                )
            return 1.0  # Fallback se conv_step não estiver disponível

        t0 = self.conv_step.get(name, total_steps) * self.alpha
        denominator = max(1.0, float(total_steps) - t0)
        x = (float(step) - t0) / denominator
        x = min(max(x, 0.0), 1.0)
        s = 0.5 * (1 + math.cos(math.pi * x))
        result = max(self.min_scale, s**self.gam)

        # Logs opcionais como antes
        # if self.debug: ...
        # if self.verbose: ...
        return result

    def scale_vectorized(
        self,
        names: list[str],
        step: int,
        total_steps: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """Versão Vetorizada (GPU) - requer `conv_step`."""
        if not names:
            return {}
        if not self.conv_step:
            if self.debug:
                print(
                    f"[AdaptiveDCoef Scale Vec] Aviso: conv_step não carregado, retornando 1.0 para todos os módulos."
                )
            scales_tensor = torch.ones(len(names), device=device, dtype=dtype)
            return {name: scales_tensor[i] for i, name in enumerate(names)}

        conv_steps_list = [self.conv_step.get(name, total_steps) for name in names]

        t0_tensor = (
            torch.tensor(conv_steps_list, device=device, dtype=dtype) * self.alpha
        )
        step_tensor = torch.tensor(float(step), device=device, dtype=dtype)
        total_steps_tensor = torch.tensor(
            float(total_steps), device=device, dtype=dtype
        )

        denominator = torch.clamp(total_steps_tensor - t0_tensor, min=1.0)
        x = (step_tensor.expand_as(t0_tensor) - t0_tensor) / denominator
        x = torch.clamp(x, min=0.0, max=1.0)
        s = 0.5 * (1.0 + torch.cos(torch.pi * x))
        scales = torch.clamp(s.pow(self.gam), min=self.min_scale)

        # Logs opcionais como antes
        # if self.debug: ...
        # if self.verbose: ...
        return {name: scales[i] for i, name in enumerate(names)}

    def get_convergence_history_run1(self) -> Dict[str, List[ConvergencePoint]]:
        return self.convergence_history

    def get_conv_step_run1(self) -> Dict[str, int]:
        return self.conv_step

    def get_d_coef_base_run1(self) -> Dict[str, float]:
        return self.d_coef_base
