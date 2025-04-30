import collections
import statistics
import math
import traceback
from typing import Dict, Set, List, Tuple, Optional # Adicionar Optional

import torch

class ConvergeControl:
    """
    Coleta métricas de convergência por módulo ao longo do tempo.
    Não toma decisões de congelamento, apenas registra dados para análise posterior.
    """
    # Atualizar a estrutura da Tupla no tipo do histórico
    ConvergencePoint = Tuple[
        int,    # 0: epoch
        int,    # 1: step
        float,  # 2: uwr_latest
        float,  # 3: uwr_abs_latest     <-- NOVO
        float,  # 4: median_uwr         (local do módulo)
        float,  # 5: std_dev_uwr        <-- NOVO (local do módulo)
        float,  # 6: z_robust
        Optional[float], # 7: global_median_uwr  <-- NOVO (pode ser None no início)
        Optional[float], # 8: global_mad_uwr     <-- NOVO (pode ser None no início)
        Optional[float], # 9: delta_median_uwr   <-- NOVO (pode ser None no primeiro registro)
        bool    # 10: was_suspect_flag
    ]

    def __init__(
        self,
        z_thresh: float = 2.0,
        u_abs_thresh: float = 1e-4,
        warmup_frac: float = 0.5,
        total_epochs: int = 100,
        debug: bool = False,
        verbose: bool = False,
        min_history: int = 30,
        min_buffer_epochs: float = .1,
        buffer_size: int = 100 # Adicionar tamanho do buffer como param
    ):
        self.z_thresh = z_thresh
        self.u_abs_thresh = u_abs_thresh
        self.warmup_frac = warmup_frac
        self.total_epochs = total_epochs
        self.debug = debug
        self.verbose = verbose
        self.min_history = min_history
        self.min_buffer_epochs = min_buffer_epochs
        # Usar max(buffer_size, min_history) para garantir que o buffer seja grande o suficiente
        self._buffer_maxlen = max(buffer_size, min_history)

        self.uwr_buffers: Dict[str, collections.deque[float]] = collections.defaultdict(
            lambda: collections.deque(maxlen=self._buffer_maxlen)
        )
        self.convergence_history: Dict[str, List[ConvergeControl.ConvergencePoint]] = collections.defaultdict(list)
        # Guardar a mediana anterior para calcular delta
        self.previous_medians: Dict[str, float] = {}

        self._current_epoch: int = 0
        self._current_step: int = 0

        if self.debug:
             print(f"[ConvergeControl] Inicializado em modo de coleta de métricas (buffer_size={self._buffer_maxlen}).")

    def set_current_time(self, epoch: int, step: int):
        self._current_epoch = epoch
        self._current_step = step

    def ingest_and_process(self, stats_list: List[Dict]):
        if not stats_list:
            return

        # --- Preparação: Atualizar buffers ---
        latest_uwrs = {}
        for stat in stats_list:
            name = stat.get("name")
            uwr = stat.get("uwr")
            if name is None or uwr is None:
                continue
            self.uwr_buffers[name].append(float(uwr))
            latest_uwrs[name] = float(uwr)

        # --- Calcular Estatísticas Globais ---
        module_names_with_history = [
            name for name, buf in self.uwr_buffers.items() if len(buf) >= self.min_history
        ]

        global_median_current: Optional[float] = None
        global_mad_current: Optional[float] = None
        current_medians: Dict[str, float] = {} # Medianas locais deste passo
        current_std_devs: Dict[str, float] = {} # Desvios padrão locais deste passo

        if len(module_names_with_history) >= 3:
            try:
                # Calcular medianas e desvios padrão locais
                for name in module_names_with_history:
                    buffer_list = list(self.uwr_buffers[name])
                    if len(buffer_list) >= 2: # Precisa de >= 2 para stddev
                        current_medians[name] = statistics.median(buffer_list)
                        current_std_devs[name] = statistics.stdev(buffer_list)
                    elif len(buffer_list) == 1: # Se tiver apenas 1, mediana é o valor, stddev é 0
                         current_medians[name] = buffer_list[0]
                         current_std_devs[name] = 0.0
                    # Se len < 1, não adiciona ao dict

                if current_medians:
                     global_median_current = statistics.median(current_medians.values())
                     abs_devs = [abs(m - global_median_current) for m in current_medians.values()]
                     if abs_devs:
                          global_mad_current = statistics.median(abs_devs) + 1e-12
                     else:
                          global_mad_current = 1e-12
            except statistics.StatisticsError:
                 if self.debug: print("[ConvergeControl][DEBUG] StatisticsError ao calcular globais.")
                 global_median_current = None
                 global_mad_current = None
                 current_medians = {}
                 current_std_devs = {} # Limpa também

        if self.debug and global_median_current is not None:
             print(f"[ConvergeControl][DEBUG] Step {self._current_step} - Global median: {global_median_current:.6f}, Global MAD: {global_mad_current:.6f}")
             # print(f"[ConvergeControl][DEBUG] Step {self._current_step} - Medians: { {k: f'{v:.6f}' for k,v in current_medians.items()} }")
             # print(f"[ConvergeControl][DEBUG] Step {self._current_step} - StdDevs: { {k: f'{v:.6f}' for k,v in current_std_devs.items()} }")

        # --- Calcular e Armazenar Métricas por Módulo ---
        progress = self._current_epoch / max(self.total_epochs, 1)
        in_warmup = progress < max(self.warmup_frac, self.min_buffer_epochs)

        for name, uwr_latest in latest_uwrs.items():
            uwr_abs_latest = abs(uwr_latest)

            # Calcular métricas apenas se não estiver em warmup, tiver histórico e stats globais
            if not in_warmup and name in module_names_with_history and global_median_current is not None and global_mad_current is not None:
                median_uwr = current_medians.get(name)
                std_dev_uwr = current_std_devs.get(name)

                # Verifica se conseguiu calcular mediana e stddev local
                if median_uwr is None: # Se não conseguiu mediana, pula este módulo neste passo
                     if self.debug: print(f"[ConvergeControl][DEBUG] Skipped storing metrics for '{name}' at step {self._current_step}. Reason: local median calculation failed.")
                     continue
                if std_dev_uwr is None: # Se não conseguiu stddev, usa 0.0
                    std_dev_uwr = 0.0

                z_robust = (median_uwr - global_median_current) / global_mad_current if global_mad_current > 1e-11 else 0.0
                was_suspect_flag = (z_robust > self.z_thresh) and (uwr_abs_latest < self.u_abs_thresh)

                # Calcular delta da mediana
                delta_median_uwr: Optional[float] = None
                if name in self.previous_medians:
                    delta_median_uwr = median_uwr - self.previous_medians[name]

                # Armazena o ponto de dados completo
                data_point: ConvergeControl.ConvergencePoint = (
                    self._current_epoch,
                    self._current_step,
                    uwr_latest,
                    uwr_abs_latest,          # Novo
                    median_uwr,
                    std_dev_uwr,             # Novo
                    z_robust,
                    global_median_current,   # Novo
                    global_mad_current,      # Novo
                    delta_median_uwr,        # Novo
                    was_suspect_flag
                )
                self.convergence_history[name].append(data_point)

                # Atualiza a mediana anterior para o próximo cálculo de delta
                self.previous_medians[name] = median_uwr

                if self.debug and self._current_step % 50 == 0: # Log menos frequente para não poluir muito
                    log_str = (
                        f"[CC Store][{name}@{self._current_step}]: "
                        f"uwr={uwr_latest:.4f}, |uwr|={uwr_abs_latest:.4f}, med={median_uwr:.4f}, "
                        f"std={std_dev_uwr:.4f}, z={z_robust:.2f}, g_med={global_median_current:.4f}, "
                        f"g_mad={global_mad_current:.4f}, d_med={delta_median_uwr if delta_median_uwr is not None else 'N/A':.4f}, "
                        f"susp={was_suspect_flag}"
                    )
                    print(log_str)
            elif self.debug:
                 # (Log de motivo de skip mantido)
                 status = []
                 if in_warmup: status.append("in warmup")
                 if name not in module_names_with_history: status.append("insufficient history")
                 if global_median_current is None or global_mad_current is None: status.append("insufficient global stats")
                 print(f"[ConvergeControl][DEBUG] Skipped storing metrics for '{name}' at step {self._current_step}. Reason: {', '.join(status)}")

    def get_convergence_data(self) -> Dict[str, List[ConvergencePoint]]: # Atualiza tipo de retorno
        """Retorna o dicionário com o histórico de métricas de convergência."""
        return self.convergence_history