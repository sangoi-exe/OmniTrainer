import collections
import statistics
import math
import traceback
import gzip
import json
import os
from typing import Dict, Set, List, Tuple, Optional, Any


class ConvergeControl:
    """
    Coleta métricas de convergência (Run 1 ou 2) e, opcionalmente (Run >= 2),
    toma decisões de congelamento usando dados históricos (externos) e atuais.
    """

    # Schema: step, uwr, |uwr|, med, std, z, g_med, g_mad, d_med, susp_flag
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
        run_number: int,
        # --- Dados Históricos (Opcionais, passados pelo Trainer da Run 1 via ADC) ---
        history_run1_data: Optional[Dict[str, List[ConvergencePoint]]] = None,
        conv_step_run1_data: Optional[Dict[str, int]] = None,
        # --- Parâmetros para cálculo/coleta ---
        z_thresh: float = 2.0,
        u_abs_thresh: float = 1e-4,
        warmup_frac: float = 0.5,
        total_epochs: int = 100,
        debug: bool = False,
        verbose: bool = False,
        min_history: int = 30,  # Histórico local para stats atuais
        min_buffer_epochs: float = 0.1,  # Warmup em épocas
        buffer_size: int = 100,
        # --- Parâmetros para DECISÃO (Run >= 2) ---
        k_confirm: int = 3,  # Número de checks para congelar
        freeze_logic: str = "original",  # 'original' ou 'custom_run1_aware', etc.
        # Adicione outros parâmetros que sua nova lógica possa precisar
        # Ex: conv_step_factor: float = 1.1
    ):
        self.run_number = run_number
        # Armazena os dados históricos recebidos
        self.history_run1: Dict[str, List[ConvergeControl.ConvergencePoint]] = (
            history_run1_data if history_run1_data else {}
        )
        self.conv_step_run1: Dict[str, int] = (
            conv_step_run1_data if conv_step_run1_data else {}
        )

        # --- Resto dos parâmetros e inicialização de estado atual/decisão ---
        self.z_thresh = z_thresh
        self.u_abs_thresh = u_abs_thresh
        self.warmup_frac = warmup_frac
        self.total_epochs = total_epochs
        self.debug = debug
        self.verbose = verbose
        self.min_history = min_history
        self.min_buffer_epochs = min_buffer_epochs
        self._buffer_maxlen = max(buffer_size, min_history)
        self.k_confirm = k_confirm
        self.freeze_logic = freeze_logic

        self.uwr_buffers: Dict[str, collections.deque[float]] = collections.defaultdict(
            lambda: collections.deque(maxlen=self._buffer_maxlen)
        )
        self.current_run_history: Dict[str, List[ConvergeControl.ConvergencePoint]] = (
            collections.defaultdict(list)
        )
        self.previous_medians: Dict[str, float] = {}
        self._current_epoch: int = 0
        self._current_step: int = 0

        self.suspect_counter: Dict[str, int] = collections.defaultdict(
            int
        )  # Contador para k_confirm
        self.perma_frozen: Set[str] = set()  # Módulos congelados NESTA run
        self.last_decisions: Dict[str, bool] = {}  # Última decisão tomada

        if self.debug:
            print(
                f"[ConvergeControl] Inicializado para Run {self.run_number} (logic='{self.freeze_logic}'). "
                f"Dados Históricos Run 1 recebidos: {bool(self.history_run1)}"
            )

    def set_current_time(self, epoch: int, step: int):
        """Atualiza o epoch e step atuais."""
        self._current_epoch = epoch
        self._current_step = step

    def ingest_and_process(self, stats_list: List[Dict]):
        """Coleta UWRs atuais, calcula métricas e armazena no histórico da run atual."""
        if not stats_list:
            return

        latest_uwrs = {}
        for stat in stats_list:
            # Assumindo que 'name' já está no dict stat vindo do Trainer
            name = stat.get("name")
            uwr = stat.get("uwr")
            if name is None or uwr is None:
                continue
            self.uwr_buffers[name].append(float(uwr))
            latest_uwrs[name] = float(uwr)

        # --- Calcular Estatísticas Globais ---
        module_names_with_history = [
            name
            for name, buf in self.uwr_buffers.items()
            if len(buf) >= self.min_history
        ]

        global_median_current: Optional[float] = None
        global_mad_current: Optional[float] = None
        current_medians: Dict[str, float] = {}
        current_std_devs: Dict[str, float] = {}

        if len(module_names_with_history) >= 3:
            try:
                for name in module_names_with_history:
                    buffer_list = list(self.uwr_buffers[name])
                    if len(buffer_list) >= 2:
                        current_medians[name] = statistics.median(buffer_list)
                        current_std_devs[name] = statistics.stdev(buffer_list)
                    elif len(buffer_list) == 1:
                        current_medians[name] = buffer_list[0]
                        current_std_devs[name] = 0.0

                if current_medians:
                    global_median_current = statistics.median(current_medians.values())
                    abs_devs = [
                        abs(m - global_median_current) for m in current_medians.values()
                    ]
                    if abs_devs:
                        global_mad_current = statistics.median(abs_devs) + 1e-12
                    else:
                        global_mad_current = 1e-12
            except statistics.StatisticsError:
                if self.debug:
                    print(
                        f"[ConvergeControl][DEBUG] {self._current_step}: StatisticsError ao calcular globais."
                    )
                global_median_current = None
                global_mad_current = None
                current_medians = {}
                current_std_devs = {}

        # --- Calcular e Armazenar Métricas por Módulo ---
        progress = (
            self._current_epoch / max(self.total_epochs, 1)
            if self.total_epochs > 0
            else 0
        )
        in_warmup_epochs = progress < self.min_buffer_epochs
        # Verifica se ALGUM buffer relevante (que tem stats no passo atual) é insuficiente
        relevant_buffers = [
            buf for name, buf in self.uwr_buffers.items() if name in latest_uwrs
        ]
        in_warmup_history = any(len(buf) < self.min_history for buf in relevant_buffers)

        # Não calcular/armazenar se em warmup ou sem stats globais
        if (
            in_warmup_epochs
            or in_warmup_history
            or global_median_current is None
            or global_mad_current is None
        ):
            if self.debug and self._current_step % 50 == 0:  # Log menos frequente
                status = []
                if in_warmup_epochs:
                    status.append("in epoch warmup")
                if in_warmup_history:
                    status.append("insufficient global history")
                if global_median_current is None:
                    status.append("no global stats")
                print(
                    f"[ConvergeControl][DEBUG] Step {self._current_step}: Skipping metric storage. Reason: {', '.join(status)}"
                )
            return

        # Armazenar métricas para módulos individuais que têm histórico
        for name, uwr_latest in latest_uwrs.items():
            if (
                name not in module_names_with_history
            ):  # Pula se o módulo específico não tem histórico local suficiente
                if self.debug:
                    print(
                        f"[ConvergeControl][DEBUG] Skipped storing metrics for '{name}'@{self._current_step}. Reason: insufficient local history ({len(self.uwr_buffers[name])}/{self.min_history})"
                    )
                continue

            uwr_abs_latest = abs(uwr_latest)
            median_uwr = current_medians.get(name)
            std_dev_uwr = current_std_devs.get(name, 0.0)

            if median_uwr is None:  # Segurança extra
                if self.debug:
                    print(
                        f"[ConvergeControl][DEBUG] Skipped storing metrics for '{name}'@{self._current_step}. Reason: local median calculation failed."
                    )
                continue

            z_robust = (
                (median_uwr - global_median_current) / global_mad_current
                if global_mad_current > 1e-11
                else 0.0
            )
            was_suspect_flag = (z_robust > self.z_thresh) and (
                uwr_abs_latest < self.u_abs_thresh
            )

            delta_median_uwr: Optional[float] = None
            if name in self.previous_medians:
                delta_median_uwr = median_uwr - self.previous_medians[name]

            # Armazena no histórico da RUN ATUAL
            data_point: ConvergeControl.ConvergencePoint = (
                self._current_step,
                uwr_latest,
                uwr_abs_latest,
                median_uwr,
                std_dev_uwr,
                z_robust,
                global_median_current,
                global_mad_current,
                delta_median_uwr,
                was_suspect_flag,
            )
            self.current_run_history[name].append(data_point)
            self.previous_medians[name] = median_uwr

            # Log de Debug (opcional)
            if self.debug and self._current_step % 50 == 0:
                log_str = (
                    f"[CC Store][{name}@{self._current_step}]: "
                    f"uwr={uwr_latest:.4f}, |uwr|={uwr_abs_latest:.4f}, med={median_uwr:.4f}, "
                    f"std={std_dev_uwr:.4f}, z={z_robust:.2f}, g_med={global_median_current:.4f}, "
                    f"g_mad={global_mad_current:.4f}, d_med={delta_median_uwr if delta_median_uwr is not None else 'N/A':.4f}, "
                    f"susp={was_suspect_flag}"
                )
                print(log_str)

    def decide(self) -> Dict[str, bool]:
        """
        Decide quais módulos congelar NESTA run (Run >= 2).
        Usa lógica selecionada (ex: original, customizada) combinando
        dados históricos da Run 1 (se disponíveis) e métricas atuais.
        Retorna {module_name: should_freeze}.
        """
        if self.run_number < 2:
            return {}  # Nenhuma decisão na Run 1

        # Verificar warmup e histórico mínimo ANTES de decidir
        progress = (
            self._current_epoch / max(self.total_epochs, 1)
            if self.total_epochs > 0
            else 0
        )
        if progress < self.min_buffer_epochs:
            if self.debug:
                print(
                    f"[ConvergeControl Decide@{self._current_step}] Skipping: In epoch warmup ({progress:.2f} < {self.min_buffer_epochs:.2f})"
                )
            return {name: False for name in self.uwr_buffers}
        # Verifica se TODOS os módulos com buffer já atingiram o histórico mínimo
        # Isso previne decisões baseadas em estatísticas globais imaturas
        if not all(
            len(buf) >= self.min_history for buf in self.uwr_buffers.values() if buf
        ):
            if self.debug:
                print(
                    f"[ConvergeControl Decide@{self._current_step}] Skipping: Insufficient global history."
                )
            return {name: False for name in self.uwr_buffers}

        if self.debug:
            print(
                f"[ConvergeControl Decide@{self._current_step}] Running decision logic (logic='{self.freeze_logic}')"
            )

        decisions: Dict[str, bool] = {}
        newly_frozen_count = 0

        # Iterar sobre os módulos que têm buffer (estado atual)
        for name in list(
            self.uwr_buffers.keys()
        ):  # Usar list() para evitar erro de tamanho mutável
            if name in self.perma_frozen:
                decisions[name] = True
                continue

            latest_point = (
                self.current_run_history[name][-1]
                if self.current_run_history[name]
                else None
            )
            should_freeze_module = False

            if latest_point:
                (step, uwr, uwr_abs, med, std, z, g_med, g_mad, d_med, was_susp) = (
                    latest_point
                )

                # --- Selecionar Lógica de Congelamento ---
                is_suspect_current = False  # Flag para lógicas que usam k_confirm

                if self.freeze_logic == "original":
                    is_suspect_current = (z > self.z_thresh) and (
                        uwr_abs < self.u_abs_thresh
                    )
                    # A decisão final com k_confirm é feita abaixo

                elif self.freeze_logic == "custom_run1_aware":
                    # --- SUA LÓGICA CUSTOMIZADA AQUI ---
                    # Exemplo: Congelar se passo atual > conv_step da Run 1 (se disponível)
                    conv_step_r1 = self.conv_step_run1.get(name, float("inf"))
                    is_past_conv_step = self._current_step > conv_step_r1

                    # Exemplo: Congelar se Z-score atual for consistentemente baixo
                    # (Precisaria olhar mais pontos no history_run1 ou current_run_history)
                    is_z_stable_low = z < 0.5  # Simplificação

                    # Exemplo: Combinar condições
                    should_freeze_module = is_past_conv_step  # Lógica mais simples: congelar após conv_step da Run 1
                    # Não usar k_confirm para esta lógica específica
                    is_suspect_current = False  # Desativa k_confirm para esta lógica

                # Adicione mais 'elif self.freeze_logic == "..."'

                # --- Lógica k_confirm (se aplicável pela freeze_logic) ---
                if (
                    is_suspect_current
                ):  # Se a lógica definida acima marcou como suspeito AGORA
                    self.suspect_counter[name] += 1
                    if self.suspect_counter.get(name, 0) >= self.k_confirm:
                        should_freeze_module = True  # Atingiu confirmação
                else:
                    # Resetar contador se não for suspeito AGORA (pela lógica atual)
                    if self.suspect_counter.get(name, 0) > 0 and self.debug:
                        print(
                            f"[ConvergeControl Decide@{self._current_step}] Resetando contador suspeita para '{name}'"
                        )
                    self.suspect_counter[name] = 0

            # -- Aplica a decisão final para este módulo --
            if should_freeze_module:
                if name not in self.perma_frozen:
                    newly_frozen_count += 1
                    if self.verbose or self.debug:
                        reason = (
                            f"k_confirm={self.k_confirm}"
                            if self.freeze_logic == "original"
                            else f"logic={self.freeze_logic}"
                        )
                        print(
                            f"[ConvergeControl Decide@{self._current_step}] Módulo '{name}' congelado ({reason}). Run {self.run_number}."
                        )
                self.perma_frozen.add(name)
                decisions[name] = True
            else:
                decisions[name] = False

        # --- Log de Resumo ---
        if newly_frozen_count > 0 or (self.debug and self._current_step % 100 == 0):
            total_modules = len(self.uwr_buffers)
            frozen_now = len(self.perma_frozen)
            active_now = total_modules - frozen_now
            if self.verbose or self.debug:
                print(
                    f"[ConvergeControl Decide@{self._current_step}] Estado: {frozen_now}/{total_modules} congelados | {active_now}/{total_modules} ativos"
                )

        self.last_decisions = decisions
        return decisions

    def get_current_run_convergence_data(self) -> Dict[str, List[ConvergencePoint]]:
        """Retorna o dicionário com o histórico de métricas da run atual."""
        return self.current_run_history

    def get_frozen_set(self) -> Set[str]:
        """Retorna o conjunto de módulos congelados nesta run."""
        return self.perma_frozen
