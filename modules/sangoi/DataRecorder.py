import traceback
import json, gzip, os
from modules.sangoi.logFun import logFun
from typing import Dict, Optional, List, Tuple, Any
from collections import defaultdict


class DataRecorder:

    def __init__(self, debug: bool = True, verbose: bool = True):
        # Estrutura: module_name -> metric_name -> list_of_values
        self.metrics_history: Dict[str, Dict[str, List[Optional[float]]]] = defaultdict(lambda: defaultdict(list))
        self.current_step = 0  # Para possivelmente registrar o step junto com as métricas
        self.debug = debug
        self.verbose = verbose

    def set_current_step(self, step: int):
        self.current_step = step

    def log_metrics_step(
        self,
        name: str,  # Nome do módulo
        d_num_pdgy: Optional[float] = None,
        d_den_pdgy: Optional[float] = None,
        dlr_pdgy: Optional[float] = None,
        grad_norm: Optional[float] = None,
        # E as próprias condições primárias e de estabilidade se quiser (bools)
    ):
        """Registra todas as métricas relevantes para o módulo no step atual."""
        if d_num_pdgy is not None:
            self.metrics_history[name]["d_num_pdgy"].append(d_num_pdgy)
        if d_den_pdgy is not None:
            self.metrics_history[name]["d_den_pdgy"].append(d_den_pdgy)
        if dlr_pdgy is not None:
            self.metrics_history[name]["dlr_pdgy"].append(dlr_pdgy)
        if grad_norm is not None:
            self.metrics_history[name]["grad_norm"].append(grad_norm)

        # if self.debug and step >= 5 and step % 5 == 0: # Use a flag de debug do DataRecorder
        #     debug_log_message = f"DataRecorder for '{name}': "
        #     if d_num_pdgy is not None: debug_log_message += f"d_num_pdgy={d_num_pdgy:.2e} "
        #     if d_den_pdgy is not None: debug_log_message += f"d_den_pdgy={d_den_pdgy:.2e} "

        # Alternativa: sempre adicionar algo para cada métrica para manter o comprimento das listas igual
        # self.metrics_history[name]["gd"].append(gd if gd is not None else float('nan'))
        # self.metrics_history[name]["snr"].append(snr if snr is not None else float('nan'))
        # etc.

    def dump(self, path: str):
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Converter defaultdicts para dicts normais para serialização JSON
        data_to_dump = {"metrics_history": {k: dict(v) for k, v in self.metrics_history.items()}}

        try:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(data_to_dump, f)  # indent=2 pode ser muito grande para dados de série temporal
            if self.verbose:
                logFun(f"[DataRecorder] Histórico de métricas salvo com sucesso em {path}", lvl="success")
        except Exception as e:
            logFun(f"[DataRecorder] ERRO ao salvar histórico de métricas em {path}: {e}", lvl="error")
            traceback.print_exc()
            raise
