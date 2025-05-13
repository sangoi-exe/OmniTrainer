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
        gd: Optional[float] = None,
        snr: Optional[float] = None,
        gns_t: Optional[float] = None,
        d_pdgy: Optional[float] = None,
        gd_ewma: Optional[float] = None,
        gd_std_ewma_var: Optional[float] = None,  # EWMA da variância do GD
        snr_median: Optional[float] = None,
        snr_iqr: Optional[float] = None,
        gns_t_median: Optional[float] = None,
        gns_t_iqr: Optional[float] = None,
        latest_mu_temporal_norm_sq: Optional[float] = None,
        # E as próprias condições primárias e de estabilidade se quiser (bools)
    ):
        """Registra todas as métricas relevantes para o módulo no step atual."""
        if gd is not None:
            self.metrics_history[name]["gd"].append(gd)
        if snr is not None:
            self.metrics_history[name]["snr"].append(snr)
        if gns_t is not None:
            self.metrics_history[name]["gns_t"].append(gns_t)
        if d_pdgy is not None:
            self.metrics_history[name]["d_pdgy"].append(d_pdgy)
        if gd_ewma is not None:
            self.metrics_history[name]["gd_ewma"].append(gd_ewma)
        if gd_std_ewma_var is not None:
            self.metrics_history[name]["gd_std_ewma_var"].append(gd_std_ewma_var)

        # if self.debug and step >= 5 and step % 5 == 0: # Use a flag de debug do DataRecorder
        #     debug_log_message = f"DataRecorder for '{name}': "
        #     if gd is not None: debug_log_message += f"GD={gd:.2e} "
        #     if snr is not None: debug_log_message += f"SNR={snr:.2f} "
        #     if gns_t is not None: debug_log_message += f"GNS_T={gns_t:.2f} "
        #     if d_pdgy is not None: debug_log_message += f"dPDGY={d_pdgy:.2e} "
        #     if gd_ewma is not None: debug_log_message += f"GD_EWMA={gd_ewma:.2e} "
        #     if gd_std_ewma_var is not None: debug_log_message += f"GD_EWMA_VAR={gd_std_ewma_var:.2e} "
        #     # Adicione outras métricas que você passa para log_metrics_step
        #     logFun(debug_log_message, lvl="RECORDER_DEBUG") # Use um lvl específico se quiser

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
