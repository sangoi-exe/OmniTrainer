import math
import traceback
import torch
import gzip
import json
import os
from typing import Dict, List, Tuple, Any, Optional

from modules.sangoi.logFun import logFun


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
        gamma: float = 2.0,
        alpha: float = 1.0,
        min_scale: float = 1e-3,
        debug: bool = True,
        verbose: bool = True,
    ):

        self.gam = gamma
        self.alpha = alpha
        self.min_scale = min_scale
        self.debug = debug
        self.verbose = verbose

        self.d_hat_final: Dict[str, float] = {}
        self.d_coef_base: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Escala dinâmica **apenas** com base no delta_L2 atual
    #   • delta_now  : Δ-L2 corrente do módulo (precisa ser fornecido)
    #   • ref_delta  : d_hat_final gravado na run-1  (fallback 1.0)
    #   • Formula    : scale = max(min_scale, (delta_now / ref_delta)^gamma)
    # ------------------------------------------------------------------
    def scale(self, name: str, delta_now: float) -> float:
        ref = max(1e-12, self.d_hat_final.get(name, 1.0))
        ratio = (delta_now / ref) ** self.gam
        s = max(self.min_scale, ratio)

        if self.debug:
            logFun(f"[AdaptiveDCoef] {name}: Δ_now={delta_now:.4g} / ref={ref:.4g} "
                  f"→ scale={s:.4g}", lvl="debug")
        return s

    def scale_vectorized(
        self,
        names: list[str],
        delta_now: torch.Tensor,   # tensor [len(names)] de Δ-L2 atuais
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        if not names:
            return {}

        ref = torch.tensor(
            [max(1e-12, self.d_hat_final.get(n, 1.0)) for n in names],
            device=device, dtype=dtype
        )
        scales = torch.clamp((delta_now / ref) ** self.gam, min=self.min_scale)

        if self.debug:
            logFun(f"[AdaptiveDCoef] vector scale Δ={delta_now[:5].tolist()} "
                   f"→ {scales[:5].tolist()}", lvl="debug")
        return {n: scales[i] for i, n in enumerate(names)}
