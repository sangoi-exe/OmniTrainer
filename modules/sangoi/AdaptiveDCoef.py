import math
import traceback
import torch
import gzip
import json
import os
from typing import Dict, List, Tuple, Any, Optional

from modules.sangoi.logFun import logFun

class AdaptiveDCoef:
    def __init__(
        self,
        gamma: float = 2.0,
        min_scale: float = 0.1,
        max_scale: float = 3.0,
        debug: bool = True,
        verbose: bool = True,
    ):
        
        self.gam = gamma
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.debug = debug
        self.verbose = verbose

        self.module_prodigy_d_final_run1: Dict[str, float] = {} 

    def load_prodigy_d_final_run1(self, data_recorder_dump_path: Optional[str]):
        if data_recorder_dump_path and os.path.exists(data_recorder_dump_path):
            try:
                with gzip.open(data_recorder_dump_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
                self.module_prodigy_d_final_run1 = data.get("d_pdgy_final", {})
                if self.verbose and self.module_prodigy_d_final_run1:
                    logFun(f"[AdaptiveDCoef] 'd_pdgy_final' da Run 1 carregado para {len(self.module_prodigy_d_final_run1)} módulos.", lvl="info")
                elif not self.module_prodigy_d_final_run1:
                    logFun(f"[AdaptiveDCoef] Perfil '{data_recorder_dump_path}' carregado, mas 'd_pdgy_final' está vazio ou ausente.", lvl="warning")
            except Exception as e:
                logFun(f"[AdaptiveDCoef] Falha ao carregar 'd_pdgy_final' de '{data_recorder_dump_path}': {e}", lvl="error")
                traceback.print_exc()
        else:
            logFun(f"[AdaptiveDCoef] Caminho do perfil 'd_pdgy_final' da Run 1 ('{data_recorder_dump_path}') não encontrado ou não fornecido. Scales individuais podem usar fallback.", lvl="info")

    def load_module_scores_run1(self, path: str) -> Dict[str, float]:
        data = json.load(open(path, "r"))
        return {
            prefix: metrics["snr"]  # ou escolha entre "snr", "gd", "gns", etc.
            for prefix, metrics in data["modules"].items()
        }

    def calculate_individual_d_coef(
            self,
            module_scores_run1: Dict[str, float],
            current_module_scores: Dict[str, float],
            gamma: float,
            min_scale: float,
            max_scale: float,
        ) -> Dict[str, float]:
        
        final_d_coefs: Dict[str, float] = {}

        if not self.module_prodigy_d_final_run1 and self.verbose:
            logFun(f"[AdaptiveDCoef] Calculando d_coefs, mas 'd_pdgy_final' da Run 1 não foi carregado. Todos os módulos usarão fallback para ref_prodigy_d_run1.", lvl="warning")

        for prefix in self.module_prefixes:
            # 1) Score atual e referência (fallback 1.0)
            score_now = current_module_scores.get(prefix, 1.0)
            ref_score = module_scores_run1.get(prefix, 1.0)  # se não existir, assume 1.0

            # 2) Cálculo da escala individual
            individual_scale = (score_now / ref_score) ** gamma

            # 3) Clamping
            clamped = torch.clamp(individual_scale, min_scale, max_scale)

            # 4) Dcoef final = escala (sem global)
            final_d_coefs[prefix] = clamped

            # Log de debug
            logFun(f"{prefix:40s} | score_now={score_now:.4f} | "
                        f"ref={ref_score:.4f} | scale={individual_scale:.4f} | "
                        f"clamped={clamped:.4f}", lvl="debug")
        return final_d_coefs