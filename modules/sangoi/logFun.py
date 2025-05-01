import traceback
import json, gzip, os
from typing import Dict, Set, Optional, List, Tuple, Any


class ModuleDynRecorder:
    """
    Guarda perfil de cada módulo durante o treino:
      • d_hat_final
      • d_coef_base     (1º valor visto)
      • Dados de Convergência (histórico detalhado do ConvergeControl da run atual)
    Dumpa em .json.gz no final.
    """

    def __init__(self, k_stride: int = 10, debug: bool = True, verbose: bool = True):
        import collections

        # self.k_stride     = k_stride # k_stride não parece mais ser usado
        self.d_hat_final: Dict[str, float] = {}
        self.d_coef_base: Dict[str, float] = {}
        self.debug = debug
        self.verbose = verbose

    def log_step(
        self,
        name: str,
        step: int,
        uwr: float,
        d_hat: float,
        d_coef: float,
        converged: bool,
    ):  # converged é ignorado
        """Registra d_coef inicial e atualiza d_hat final."""
        # Log apenas para debug
        if self.debug and step % 500 == 0:  # Log bem menos frequente
            print(
                f"[ModuleDynRecorder][DEBUG@{step}] log_step for '{name}': "
                f"uwr={uwr:.4f}, d_hat={d_hat:.4f}, d_coef={d_coef:.4f}"
            )

        # Manter lógica para d_hat_final e d_coef_base
        self.d_hat_final[name] = d_hat
        self.d_coef_base.setdefault(name, d_coef)

    def dump(
        self,
        path: str,
        convergence_data: Optional[Dict[str, List[Tuple[Any, ...]]]] = None,
    ):
        """Salva os dados coletados (d_coef, d_hat, histórico de convergência)."""
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if self.debug:
            conv_data_summary = (
                {k: len(v) for k, v in convergence_data.items()}
                if convergence_data
                else {}
            )
            print(
                f"[ModuleDynRecorder][DEBUG] Dumping para '{path}': "
                f"Base d_coef={len(self.d_coef_base)} mods, "
                f"Final d_hat={len(self.d_hat_final)} mods, "
                f"Convergence Data Points={conv_data_summary}"
            )

        data_to_dump = {
            "d_hat_final": self.d_hat_final,
            "d_coef_base": self.d_coef_base,
            # Adiciona o schema para o histórico de convergência
            "convergence_history_schema": [
                "step",
                "uwr_latest",
                "uwr_abs_latest",
                "median_uwr",
                "std_dev_uwr",
                "z_robust",
                "global_median_uwr",
                "global_mad_uwr",
                "delta_median_uwr",
                "was_suspect_flag",
            ],
            "convergence_history": convergence_data if convergence_data else {},
            # Opcional: Adicionar conv_step aqui se for útil tê-lo salvo
            # "conv_step": {name: step for name, step in self.get_final_conv_steps().items()} # Exigiria lógica extra
        }

        try:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(data_to_dump, f, indent=2)
            if self.verbose:
                print(f"[ModuleDynRecorder] Perfil salvo com sucesso em {path}")
        except Exception as e:
            print(f"[ModuleDynRecorder] ERRO ao salvar perfil em {path}: {e}")
            traceback.print_exc()
