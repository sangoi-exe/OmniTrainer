import traceback
import json, gzip, os
from modules.sangoi.logFun import logFun
from typing import Dict, Optional, List, Tuple, Any


class DataRecorder:
    """
    Guarda perfil de cada módulo durante o treino:
      • d_hat_final
      • Dados de Convergência (histórico detalhado do ConvergeControl da run atual)
    Dumpa em .json.gz no final.
    """

    def __init__(self, debug: bool = True, verbose: bool = True):
        import collections

        self.d_hat_final: Dict[str, float] = {}
        self.debug = debug
        self.verbose = verbose

    def log_step(
        self,
        name: str,
        d_hat: float,
    ):  # converged é ignorado
        """Registra d_coef inicial e atualiza d_hat final."""
        # Log apenas para debug
        # if self.debug and step % 500 == 0:  # Log bem menos frequente
        #     logFun(
        #         f"[DataRecorder][DEBUG@{step}] log_step for '{name}': "
        #         f"uwr={uwr:.4f}, d_hat={d_hat:.4f}, d_coef={d_coef:.4f}",
        #         lvl="debug"
        #     )

        # Manter lógica para d_hat_final e d_coef_base
        self.d_hat_final[name] = d_hat

    def dump(
        self,
        path: str,
    ):
        """Salva os dados coletados (d_coef, d_hat, histórico de convergência)."""
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        data_to_dump = {
            "d_hat_final": self.d_hat_final,
        }

        try:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(data_to_dump, f, indent=2)
            if self.verbose:
                logFun(f"[DataRecorder] Perfil salvo com sucesso em {path}", lvl="success")
        except Exception as e:
            print(f"[DataRecorder] ERRO ao salvar perfil em {path}: {e}", lvl="error")
            traceback.print_exc()
