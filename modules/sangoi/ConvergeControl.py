import math
import torch
import collections
import traceback
import numpy as np
from rich.columns import Columns
from rich.console import Console # Manter, mas o Live Display usará o console do Trainer
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Group
from rich.text import Text
from rich import box
from scipy.stats import linregress
from modules.sangoi.logFun import logFun
from typing import Deque, Dict, Set, List, Optional, Tuple, Union # Adicionado Union
from dataclasses import dataclass
from river.stats import Quantile              # quantis streaming P² puro-Python
from collections import defaultdict  
import bisect
from collections import deque

# console = Console() # Pode ser removido se o Live display for o principal meio de output

# ------------------- Config dataclass -------------------
@dataclass
class ConvCfg:
    lambda_: float = 0.02       # fator EWMA  (≈ janela 1/λ)
    eps_slope: float = 1e-4     # |slope| limiar
    eps_width: float = 0.02     # (P95-P5)/|P50|
    z_lim: float = 1.64         # α = 10 %  (Mann-Kendall)
    k_confirm: int = 4
    k_confirm_steps: int = 50
    min_quantile_samples: int = 20

# ------------------- MK Streaming ----------------------
class MKStream:
    def __init__(self, maxlen: int = 200):
        # buffer de chegada para remoção de itens antigos
        self.buffer = deque()
        # lista sempre ordenada para cdf(x)
        self.sorted = []
        self.maxlen = maxlen
        self.S = 0.0
        self.n = 0
    def update(self, x: float):
        # 1) calcule rank aproximado usando sorted list
        if self.n > 0:
            idx = bisect.bisect_left(self.sorted, x)
            rank = idx / self.n
            self.S += (rank - (1 - rank)) * self.n

        # 2) insira em sorted e buffer
        bisect.insort(self.sorted, x)
        self.buffer.append(x)
        self.n += 1

        # 3) descarte item mais antigo se exceder maxlen
        if len(self.buffer) > self.maxlen:
            oldest = self.buffer.popleft()
            pos = bisect.bisect_left(self.sorted, oldest)
            # remove a primeira ocorrência
            self.sorted.pop(pos)
            self.n -= 1
    def z(self) -> float:
        if self.n < 8: return 0.0 # MK test typically needs n >= 8 or 10
        # Var(S) = n(n-1)(2n+5)/18 - Original formula, no ties
        # For streaming with ranks from TDigest, ties are implicitly handled to some extent by CDF.
        # Using the original variance formula for S calculated with ranks.
        varS = self.n*(self.n-1)*(2*self.n+5)/18
        if varS == 0: return 0.0 # Avoid division by zero if n is small leading to varS=0
        
        # Standard MK test statistic definition
        if self.S > 0:
            return (self.S - 1) / math.sqrt(varS)
        elif self.S < 0:
            return (self.S + 1) / math.sqrt(varS)
        else: # S == 0
            return 0.0


class ConvergeControl:
    """
    Coleta métricas de delta-L2 (para logging/score), d-max e SNR (para decisão
    de congelamento). Utiliza limiares dinâmicos para d-max e SNR.
    O congelamento só é aplicável a partir da Run 2 (ou se explicitamente habilitado).
    """

    def __init__(
        self,
        run_number: int,
        *,
        cfg: ConvCfg,
        verbose: bool = True,
        debug: bool = True,
        # Parâmetros que permanecem pois não foram mencionados para remoção e são usados
        # ou são relacionados ao Δ-L2 que ainda pode ser exibido.
        delta_buffer_size: int | None = None, # Usado para _buffer_maxlen_delta_l2
        # Os seguintes parâmetros de Δ-L2 são mantidos para a funcionalidade de display do Δ-L2
        delta_abs_thresh_fallback: float = 2e-3, 
        hist_size_delta_l2_display: int = 10, # Renomeado para clareza, afeta display de Δ-L2
        cv_thresh_fallback_display: float = 0.8, # Renomeado
        slope_thresh_fallback_display: float = 4e-4, # Renomeado
        dynamic_k_delta_l2_display: float = 2, # Renomeado
    ):

        self.run_number = run_number
        self.cfg = cfg
        self.k_confirm = cfg.k_confirm # Usando k_confirm da Cfg
        self.verbose = verbose
        self.debug = debug
        
        # Mantendo configurações de Δ-L2 para display e logging
        self.win_size_delta_l2 = 10 # Janela para cálculo de stats de Δ-L2
        self._hist_size_delta_l2_display = max(1, hist_size_delta_l2_display)
        self.metric_hist_delta_l2: Dict[str, Dict[str, Deque[float]]] = collections.defaultdict(
            lambda: {
                'mean': collections.deque(maxlen=self._hist_size_delta_l2_display),
                'cv': collections.deque(maxlen=self._hist_size_delta_l2_display),
                'slope': collections.deque(maxlen=self._hist_size_delta_l2_display),})
        self.dyn_thresh_delta_l2: Dict[str, Dict[str, float]] = {} # Limiares dinâmicos para Δ-L2 display

        # Fallbacks para display de Δ-L2
        self._cv_thresh_fallback_display = cv_thresh_fallback_display
        self._abs_thresh_fallback_display = delta_abs_thresh_fallback
        self._slope_thresh_fallback_display = slope_thresh_fallback_display
        self._dynamic_k_delta_l2_display = dynamic_k_delta_l2_display

        self.freeze_step_run2: Dict[str, int] = {}
        
        hint = delta_buffer_size or 0
        self._buffer_maxlen_delta_l2 = max(hint, self.win_size_delta_l2)
        self.delta_buffers = collections.defaultdict(lambda: collections.deque(maxlen=self._buffer_maxlen_delta_l2))
        self._rolling_stats = collections.defaultdict(lambda: RollingStats(self.win_size_delta_l2))
        
        self.perma_frozen: Set[str] = set()
        self.suspect_counter: Dict[str, int] = collections.defaultdict(int)
        self._current_epoch: int = 0
        self._current_step: int = 0
        self.last_ewma_slope: Dict[str,float] = {}
        
        # Freeze é habilitado por padrão, decisão efetiva depende da run_number e k_confirm
        self.enable_freeze_action = run_number >= 2 # run_number >= 2 testado em `decide` se necessário, ou sempre ativo
                                         # Se a intenção era disable para run 1, isso pode ser self.run_number >=2
        
        # Atributos relacionados a D-max e SNR para display (não mais para decisão primária)
        self.grad_eps = 1e-9
        self.thr_d_max: Dict[str, float] = {} # Para display
        self.thr_slope_d_max: Dict[str, float] = {} # Para display
        self.last_slope_d_max: Dict[str, float] = {} # Para display
        self.thr_snr: Dict[str, float] = {} # Para display
        
        self.win_snr_calc = 2 # Janela para cálculo de SNR instantâneo
        self.g_hist_for_snr_baseline = collections.defaultdict(lambda: collections.deque(maxlen=200)) # Warmup steps para SNR baseline
        self.g_vector_hist_for_snr_calc = collections.defaultdict(lambda: collections.deque(maxlen=self.win_snr_calc))
        self.d_max_hist = collections.defaultdict(lambda: collections.deque(maxlen=200)) # Warmup steps para D-max hist
        
        self.base_d_max: Dict[str, float] = {} # Para display
        self.base_snr: Dict[str, float] = {} # Para display

        # --- novos estados MK / quantis (River P²) -------------
        self.mk_stream: Dict[str, MKStream] = defaultdict(MKStream)
        self.ewma_slope: Dict[str, float]    = defaultdict(float)
        # quantis P² para 5%, 50%, 95%
        self.q5:    Dict[str, Quantile] = defaultdict(lambda: Quantile(0.05))
        self.q50:   Dict[str, Quantile] = defaultdict(lambda: Quantile(0.50))
        self.q95:   Dict[str, Quantile] = defaultdict(lambda: Quantile(0.95))
        self.q_count: Dict[str, int]    = defaultdict(int)

        if self.debug:
            try:
                # Log simplificado com a nova config
                logFun(
                    f"[ConvergeControl] Init: Run {self.run_number}, FreezeAction: {self.enable_freeze_action}. "
                    f"ConvCfg: lambda={self.cfg.lambda_}, eps_slope={self.cfg.eps_slope}, "
                    f"eps_width={self.cfg.eps_width}, z_lim={self.cfg.z_lim}, k_confirm={self.cfg.k_confirm}. "
                    f"Δ-L2 Display params: hist_size={self._hist_size_delta_l2_display}, dynamic_k={self._dynamic_k_delta_l2_display}.",
                    lvl="CONVCTRL")
            except Exception as e:
                print(f"[ConvergeControl __init__] Error in logFun: {e}")
                traceback.print_exc()

    def _robust_stats(self, seq):
        arr = np.asarray(seq, dtype=np.float32)
        if arr.size < 2:
            return float(np.median(arr)) if arr.size == 1 else 0.0, 0.0
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        return float(med), float(q3 - q1)

    def _calculate_snr(self, g_vector_deque: Deque[torch.Tensor]) -> Optional[float]:
        if len(g_vector_deque) < 2:
            return None
        valid_grads = [g for g in list(g_vector_deque) if g is not None and g.numel() > 0]
        if len(valid_grads) < 2:
            return None
        try:
            g_stack = torch.stack(valid_grads)
            mu_vec = g_stack.mean(dim=0)
            std_vec = g_stack.std(dim=0, unbiased=True) # use unbiased=True for sample std
            mu_norm = mu_vec.norm().item()
            std_norm = std_vec.norm().item()
            if std_norm < self.grad_eps: # Check against grad_eps
                return float('inf')
            return mu_norm / std_norm
        except Exception as e:
            logFun(f"[ConvergeControl] Error calculating SNR: {e}", lvl="error")
            return None

    def update_step_metrics(self, name: str, module: torch.nn.Module):
        try:
            grads = [
                p.grad.detach().flatten() for p in module.parameters() if p.grad is not None and p.grad.numel() > 0]
            if grads:
                grad_vec = torch.cat(grads)
                self.g_vector_hist_for_snr_calc[name].append(grad_vec.clone()) # MANTIDO PARA SNR DISPLAY
                if len(self.g_vector_hist_for_snr_calc[name]) >= self.win_snr_calc:
                    snr_val = self._calculate_snr(self.g_vector_hist_for_snr_calc[name])
                    if snr_val is not None:
                        self.g_hist_for_snr_baseline[name].append(snr_val) # MANTIDO PARA SNR DISPLAY

            # --- d-max para MK + quantis River P² -----
            if name in self.d_max_hist and self.d_max_hist[name]:
                d_val = self.d_max_hist[name][-1]
                # Mann–Kendall (igual)
                self.mk_stream[name].update(d_val)
                # P² quantis
                self.q_count[name] += 1
                self.q5[name].update(d_val)
                self.q50[name].update(d_val)
                self.q95[name].update(d_val)
                
        except Exception as e:
            logFun(f"[ConvergeControl] Exception during update_step_metrics for '{name}': {e}", lvl="error")

    def snapshot_epoch_weights(self):
        """
        CALCULA estatísticas de Delta-L2 e atualiza limiares dinâmicos (PARA DISPLAY).
        """
        for module_name in list(self.delta_buffers.keys()):
            try:
                buf_delta_l2 = self.delta_buffers[module_name]
                if len(buf_delta_l2) < self.win_size_delta_l2:
                    continue

                rs = self._rolling_stats[module_name]
                mean_dl2 = rs.mean
                std_dl2 = rs.std
                slope_dl2 = self.stats_window(list(buf_delta_l2)[-self.win_size_delta_l2:])[2]
                cv_dl2 = std_dl2 / (mean_dl2 + 1e-12) if mean_dl2 > 1e-12 else 0.0

                mh_dl2 = self.metric_hist_delta_l2[module_name]
                mh_dl2['mean'].append(mean_dl2)
                mh_dl2['cv'].append(cv_dl2)
                mh_dl2['slope'].append(abs(slope_dl2))

                if len(mh_dl2['mean']) == self._hist_size_delta_l2_display: # Usar o tamanho de histórico de display
                    for met_key in ('mean', 'cv', 'slope'):
                        metric_values_for_dyn_thresh = list(mh_dl2[met_key])
                        med_met, iqr_met = self._robust_stats(metric_values_for_dyn_thresh)
                        base, var = med_met, iqr_met
                        # Usar _dynamic_k_delta_l2_display para cálculo de threshold de display
                        calculated_thresh = base + self._dynamic_k_delta_l2_display * var
                        self.dyn_thresh_delta_l2.setdefault(module_name, {})[met_key] = calculated_thresh
            except Exception as e:
                logFun(
                    f"[ConvergeControl.snapshot_epoch_weights] Error processing CALCS for module '{module_name}': {e}",
                    lvl="error")

    def _generate_module_card(self, module_name: str) -> Optional[Panel]:
        """Gera um Panel (card) para um único módulo com suas métricas, estilo tabela tradicional."""
        try:
            # --- Lógica para short_module_name (mantida) ---
            short_module_name = module_name
            parts = module_name.split('_')
            if len(parts) > 2:
                third_segment_from_start = parts[2]
                last_meaningful_index = len(parts)
                while last_meaningful_index > 0 and parts[last_meaningful_index - 1].isdigit():
                    last_meaningful_index -= 1
                temp_end_segments = []
                idx = last_meaningful_index - 1
                segments_to_take_from_end = 0
                while idx >= 0 and segments_to_take_from_end < 3:
                    if parts[idx] in ["blocks", "attentions"] and idx < 5:
                        break
                    if idx <= 2 and len(parts) < 6:
                        break
                    temp_end_segments.insert(0, parts[idx])
                    segments_to_take_from_end += 1
                    idx -= 1
                if temp_end_segments:
                    end_part = "_".join(temp_end_segments)
                    short_module_name = f"{third_segment_from_start}_{end_part}"
                else:
                    short_module_name = third_segment_from_start
            elif len(module_name) > 30:
                short_module_name = module_name[:15] + "..." + module_name[-12:]
            # --- Fim da lógica short_module_name ---

            # Δ-L2 display (mantido)
            buf_delta_l2 = self.delta_buffers.get(module_name)
            mean_dl2_text, cv_dl2_text, slope_dl2_text, score_dl2_text = ("N/A",) * 4
            if buf_delta_l2 is not None and len(buf_delta_l2) >= self.win_size_delta_l2:
                rs = self._rolling_stats[module_name]
                mean_dl2 = rs.mean
                std_dl2 = rs.std
                slope_dl2 = self.stats_window(list(buf_delta_l2)[-self.win_size_delta_l2:])[2]
                cv_dl2 = std_dl2 / (mean_dl2 + 1e-12) if mean_dl2 > 1e-12 else 0.0
                score_dl2_val = self.stability_score(module_name) # stability_score usa fallbacks corretos
                mean_dl2_text = f"{mean_dl2:.2e}"
                cv_dl2_text = f"{cv_dl2:.2e}"
                slope_dl2_text = f"{slope_dl2:.2e}"
                score_dl2_text = f"{score_dl2_val*100:.0f}"


            # --- Métricas Novas/Reformuladas para display ---
            # D-max e seus derivados (EWMA slope, Quantiles, MK-z)
            d_max_current_text = Text("N/A")
            ewma_slope_text = Text("N/A")
            quantile_width_text = Text("N/A")
            mk_z_text = Text("N/A")

            if module_name in self.d_max_hist and self.d_max_hist[module_name]:
                d_max_current_val = self.d_max_hist[module_name][-1]
                d_max_current_text = Text(f"{d_max_current_val:.3e}")

                # EWMA Slope Display
                # A ewma_slope é atualizada em _ready_to_freeze. Aqui apenas lemos.
                # O 'slope' calculado em _ready_to_freeze é (ewma_slope[name] - prev_ewma_slope)
                # Para mostrar o valor do slope EWMA, precisamos do valor calculado lá.
                # Temporariamente, vamos mostrar o ewma_slope atual e o 'prev' se disponível.
                # Idealmente, o 'slope' calculado em _ready_to_freeze deveria ser armazenado se precisarmos dele aqui.
                # Por agora, vamos exibir o valor EWMA atual
                # EWMA-slope: exibimos o delta guardado em self.last_ewma_slope
                slope = self.last_ewma_slope.get(module_name, None)
                if slope is not None:
                    ewma_slope_text = Text(f"{slope:.2e}")
                    style = "green" if abs(slope) < self.cfg.eps_slope else "red"
                    ewma_slope_text.stylize(style)
                    ewma_slope_text.append(f" <{self.cfg.eps_slope:.0e}", style="dim")


            # exibição dos quantis P²
            if self.q_count.get(module_name, 0) >= self.cfg.min_quantile_samples:
                p5  = self.q5[module_name].get()
                p50 = self.q50[module_name].get()
                p95 = self.q95[module_name].get()
                if abs(p50) > 1e-12 : # Evitar divisão por zero
                    width = (p95 - p5) / abs(p50)
                    quantile_width_text = Text(f"{width:.3f}")
                    if width < self.cfg.eps_width:
                        quantile_width_text.stylize("green")
                    else:
                        quantile_width_text.stylize("red")
                    quantile_width_text.append(f" <{self.cfg.eps_width:.2f}", style="dim")
                else:
                    quantile_width_text = Text(f"P50 near zero ({p50:.2e})")


            mk_s = self.mk_stream.get(module_name)
            if mk_s and mk_s.n >= 8:
                z_val = mk_s.z()
                mk_z_text = Text(f"{z_val:.2f}")
                if abs(z_val) < self.cfg.z_lim:
                    mk_z_text.stylize("green")
                else:
                    mk_z_text.stylize("red")
                mk_z_text.append(f" |Z|<{self.cfg.z_lim:.2f}", style="dim")


            # Status de congelamento (mantido)
            suspect_count = self.suspect_counter.get(module_name, 0)
            is_permafrozen = module_name in self.perma_frozen
            freeze_status_combined_text = Text()
            if is_permafrozen:
                freeze_status_combined_text.append("PERMA", style="bold white on red")
            else:
                if suspect_count > 0:
                    freeze_status_combined_text.append(str(suspect_count), style="yellow")
                else:
                    freeze_status_combined_text.append(str(suspect_count))
                freeze_status_combined_text.append(f"/{self.cfg.k_confirm_steps}", style="dim")


            table = Table(title=None,
                          show_header=True,
                          header_style="bold white",
                          box=box.SIMPLE_HEAD,
                          padding=(0, 1),
                          show_edge=False,
                          expand=True)
            table.add_column("Stat", style="cyan", ratio=9, overflow="fold", no_wrap=False)
            table.add_column("Value", style="white", ratio=10)

            table.add_row("Δ-L2 Mean", mean_dl2_text)
            table.add_row("Δ-L2 CV", cv_dl2_text)
            table.add_row("Δ-L2 Slope", slope_dl2_text)
            table.add_row("Δ-L2 Score (%)", score_dl2_text)

            # Novas métricas de decisão
            table.add_row("D-max (last)", d_max_current_text) # Último valor de D-max observado
            table.add_row("D-max EWMA Slope", ewma_slope_text) # Usa cfg.eps_slope
            table.add_row("D-max Q-Width", quantile_width_text) # Usa cfg.eps_width
            table.add_row("D-max MK-Z", mk_z_text) # Usa cfg.z_lim

            # SNR (mantido para display, mas não mais na decisão primária do _ready_to_freeze)
            snr_value_text = Text("N/A")
            snr_from_calc_win = self._calculate_snr(
                self.g_vector_hist_for_snr_calc.get(module_name, collections.deque()))
            if snr_from_calc_win is not None: snr_value_text = Text(f"{snr_from_calc_win:.2f}")
            elif module_name in self.g_hist_for_snr_baseline and self.g_hist_for_snr_baseline[module_name]:
                snr_value_text = Text(f"{self.g_hist_for_snr_baseline[module_name][-1]:.2f}")
            table.add_row("SNR (info only)", snr_value_text)

            table.add_row("Freeze Status", freeze_status_combined_text)

            return Panel(table,
                        title=f"[magenta]{short_module_name}[/] (E{self._current_epoch})",
                        border_style="blue",
                        padding=(0, 0),
                        width=40) # Largura ajustada para acomodar novas métricas
        except Exception as e:
            logFun(f"Error generating card for {module_name}: {e}\n{traceback.format_exc()}", lvl="error")
            return Panel(Text(f"Error for {module_name}.", style="red"),
                        title=f"[red]{module_name[:20]}... - ERR[/red]",
                        width=40)


    def generate_status_renderable(self) -> Union[Group, Columns, Text]:  # Adicionado Columns
        """
            Gera um objeto Rich (Group de Panels) para ser exibido pelo Live display.
            Cada Panel é um "card" para um módulo.
            Usa Columns se houver muitos módulos.
            """
        if not self.delta_buffers and not self.d_max_hist and not self.g_hist_for_snr_baseline: # Check old and new data sources
            return Text("ConvergeControl: Aguardando os primeiros dados...", justify="center")

        # Consider all modules that have any form of data
        all_module_names = set(self.delta_buffers.keys()) | \
                           set(self.d_max_hist.keys()) | \
                           set(self.g_hist_for_snr_baseline.keys()) | \
                           set(self.mk_stream.keys()) | \
                           set(self.q_count.keys())


        if not all_module_names:
            return Text("ConvergeControl: Sem módulos ativos.", justify="center")

        module_cards = []
        sorted_module_names = sorted(list(all_module_names))

        for module_name in sorted_module_names:
            card = self._generate_module_card(module_name)
            if card:
                module_cards.append(card)

        if not module_cards:
            return Text("ConvergeControl: Coletando dados iniciais...", justify="center")

        if len(module_cards) > 1:
            return Columns(
                module_cards,
                expand=True,
                equal=False, 
                padding=0,
                column_first=True
            )
        elif module_cards: # Single card
             return module_cards[0]
        else: # Should not happen if all_module_names had items and _generate_module_card worked
             return Text("ConvergeControl: Nenhum card gerado.", justify="center")


    def _ready_to_freeze(self, name: str) -> bool:
        try:
            # warm-up de quantis P²
            if self.q_count[name] < self.cfg.min_quantile_samples:
                return False
            # --- largura relativa 5/95 via P² ----
            p5, p50, p95 = self.q5[name].get(), self.q50[name].get(), self.q95[name].get()
            if abs(p50) < 1e-12:
                width = float('inf')
            else:
                width = (p95 - p5) / abs(p50)


            # --- slope EWMA ----------------------------
            lam = self.cfg.lambda_
            if not self.d_max_hist[name]: return False                 
            last = self.d_max_hist[name][-1]
        
            prev_ewma_slope = self.ewma_slope[name]
            new_ewma = (1-lam)*prev_ewma_slope + lam*last
            slope = new_ewma - prev_ewma_slope
            self.ewma_slope[name]      = new_ewma
            self.last_ewma_slope[name] = slope
            self.ewma_slope[name] = new_ewma
            self.last_ewma_slope[name] = slope

            # --- MK z-score ----------------------------
            # self.mk_stream[name].z() já retorna 0.0 se n < 8
            z = abs(self.mk_stream[name].z())

            stable = (
                z      < self.cfg.z_lim    and
                width  < self.cfg.eps_width and
                abs(slope) < self.cfg.eps_slope
            )
            slope = new_ewma - prev_ewma_slope
            self.last_ewma_slope[name] = slope
            # Debug log opcional
            # if self.debug and self._current_step % 10 == 0:
            #     logFun(
            #         f"[_ready_to_freeze] '{name}': MK-Z={z:.2f} (Thresh <{self.cfg.z_lim}), "
            #         f"QWidth={width:.3f} (Thresh <{self.cfg.eps_width}), "
            #         f"EWMASlope={slope:.2e} (Thresh <{self.cfg.eps_slope}) -> Stable={stable}",
            #         lvl="CONVCTRL")
            return stable
        except Exception as e:
            if hasattr(self, 'debug') and self.debug:
                logFun(f"[ConvergeControl._ready_to_freeze] Error for module '{name}': {e}\n{traceback.format_exc()}", lvl="error")
            return False


    @staticmethod
    def stats_window(arr: List[float]) -> Tuple[float, float, float]:
        try:
            y = np.asarray(arr, dtype=np.float32)
            if len(y) == 0:
                return 0.0, 0.0, 0.0
            mean = float(y.mean())
            std = float(y.std(ddof=1)) if len(y) > 1 else 0.0
            slope = 0.0
            if len(y) >= 2:
                x = np.arange(len(y), dtype=np.float32)
                try:
                    # Filtra NaNs e Infs que podem vir de cálculos problemáticos
                    finite_mask = np.isfinite(y)
                    if np.sum(finite_mask) >= 2:
                        y_finite = y[finite_mask]
                        x_finite = x[finite_mask]
                        if len(y_finite) >=2:
                             slope = float(linregress(x_finite, y_finite).slope)
                        else: #Not enough finite points for regression
                             slope = 0.0
                    else: #Not enough finite points for regression
                        slope = 0.0
                except (ValueError, Exception):
                    slope = 0.0
            return mean, std, slope
        except Exception:
            return 0.0, 0.0, 0.0

    def set_current_time(self, epoch: int, step: int):
        self._current_epoch = epoch
        self._current_step = step

    def ingest_and_process(self, stats_list: List[Dict]):
        if not stats_list:
            return
        
        for stat in stats_list:
            name, delta_val, d_max_val = stat.get("name"), stat.get("delta_L2"), stat.get("d_max")
            if name is None:
                continue
            try:
                if delta_val is not None:
                    self.delta_buffers[name].append(float(delta_val))
                    self._rolling_stats[name].add(float(delta_val))
                if d_max_val is not None:
                    self.d_max_hist[name].append(float(d_max_val))
                    # A atualização de mk_stream e q_digest foi movida para update_step_metrics
                    # para garantir que ocorra após d_max_hist ser atualizado e ANTES de _ready_to_freeze
                    # ser chamado (se a ordem de chamada for ingest -> update_step_metrics -> decide).
                    # Se update_step_metrics é chamada antes de ingest_and_process, então d_max_hist[-1]
                    # não seria o valor mais recente.
                    # A instrução original coloca a atualização em update_step_metrics.
            except (ValueError, TypeError) as e:
                if self.debug: logFun(f"[ConvergeControl] Error processing stats for '{name}': {e}. ΔL2: {delta_val}, Dhat: {d_max_val}", lvl="warning")
                pass
            

    def decide(self) -> Dict[str, bool]:
        try:
            if not self.enable_freeze_action:
                return {}
            
            # A lógica de 'progress' e 'min_buffer_epochs' foi removida.
            # O warm-up agora é tratado por `qd.centroids < 20` em `_ready_to_freeze`.

            decisions: Dict[str, bool] = {}
            # Decide on modules that have data for new criteria (MK/Quantile)
            # These are populated via d_max_hist -> update_step_metrics
            # só aqueles com quantis, MK e d_max já iniciados
            module_names_to_decide = list(
                set(self.q_count.keys()) &
                set(self.mk_stream.keys()) &
                set(self.d_max_hist.keys())
            )


            for name in module_names_to_decide:
                try:
                    if name in self.perma_frozen:
                        decisions[name] = True
                        continue
                    
                    # _ready_to_freeze agora usa os novos critérios
                    is_module_stable = self._ready_to_freeze(name)
                    
                    if is_module_stable: self.suspect_counter[name] += 1
                    else: self.suspect_counter[name] = 0

                    should_freeze_eval = ( self.suspect_counter[name] >= self.cfg.k_confirm_steps )
                    decisions[name] = should_freeze_eval
                    
                    if should_freeze_eval and name not in self.perma_frozen:
                        self.perma_frozen.add(name)
                        self.freeze_step_run2[name] = self._current_step # Registra o passo do congelamento
                        if self.verbose or self.debug:
                            logFun(
                                f"[ConvergeControl] FREEZE ACTION '{name}' @ step {self._current_step} "
                                f"(Trigger: Stable for {self.suspect_counter.get(name, 0)} checks >= {self.k_confirm}) "
                                f"Criteria (cfg): z<{self.cfg.z_lim}, Qw<{self.cfg.eps_width}, Slp<{self.cfg.eps_slope}",
                                lvl="CONVCTRL")
                except Exception as e_module:
                    logFun(f"[ConvergeControl.decide] Error processing decision for module '{name}': {e_module}\n{traceback.format_exc()}",
                           lvl="error")
                    decisions[name] = False # Default to not freezing on error
            return decisions
        except Exception as e_outer:
            logFun(f"[ConvergeControl.decide] Outer error in decide method: {e_outer}\n{traceback.format_exc()}", lvl="error")
            return {}

    def get_frozen_set(self) -> Set[str]:
        return self.perma_frozen

    def get_final_conv_steps(self) -> Dict[str, int]:
        # A lógica original era: return {} if self.run_number == 1 and not self.enable_freeze_action else dict(self.freeze_step_run2)
        # Simplificando, se enable_freeze_action controla tudo:
        if not self.enable_freeze_action : # Se o congelamento nunca esteve ativo.
             return {}
        # Se a intenção é retornar vazio para run 1 especificamente:
        # if self.run_number == 1 and not self.cfg.allow_freeze_run1 (se tal config existisse)
        return dict(self.freeze_step_run2)


    def _delta_stable(self, name: str) -> bool: # Usado por stability_score
        try:
            win, buf = self.win_size_delta_l2, self.delta_buffers[name]
            if len(buf) < win:
                return False
            mean, std, slope = self.stats_window(list(buf)[-win:])
            cv = std / (mean + 1e-12)
            
            # Usando fallbacks de display para Δ-L2
            dyn_m_thresh = self.dyn_thresh_delta_l2.get(name, {}).get('mean', self._abs_thresh_fallback_display)
            dyn_cv_thresh = self.dyn_thresh_delta_l2.get(name, {}).get('cv', self._cv_thresh_fallback_display)
            dyn_s_thresh = self.dyn_thresh_delta_l2.get(name, {}).get('slope', self._slope_thresh_fallback_display)
            
            mean_ok, cv_ok, slope_ok = mean < dyn_m_thresh, cv < dyn_cv_thresh, abs(slope) < dyn_s_thresh
            return (mean_ok and cv_ok) or (cv_ok and slope_ok)
        except Exception:
            return False

    def stability_score(self, name: str) -> float: # Usado por _generate_module_card
        try:
            buf = self.delta_buffers.get(name)
            if buf is None or len(buf) < self.win_size_delta_l2:
                return 0.0
            mean, std, slope = self.stats_window(list(buf)[-self.win_size_delta_l2:])
            cv = std / (mean + 1e-12)
            
            # Usando fallbacks de display para Δ-L2
            thr_mean = self.dyn_thresh_delta_l2.get(name, {}).get('mean', self._abs_thresh_fallback_display)
            thr_cv = self.dyn_thresh_delta_l2.get(name, {}).get('cv', self._cv_thresh_fallback_display)
            thr_slope = self.dyn_thresh_delta_l2.get(name, {}).get('slope', self._slope_thresh_fallback_display)
            
            mean_score = max(0.0, min(1.0, (thr_mean - mean) / (thr_mean + 1e-12))) if thr_mean > 1e-12 else (1.0 if mean < 1e-12 else 0.0)
            cv_score = max(0.0, min(1.0, (thr_cv - cv) / (thr_cv + 1e-12))) if thr_cv > 1e-12 else (1.0 if cv < 1e-12 else 0.0)
            slope_score = max(0.0, min(1.0, (thr_slope - abs(slope)) / (thr_slope + 1e-12))) if thr_slope > 1e-12 else (1.0 if abs(slope) < 1e-12 else 0.0)
            return (mean_score + cv_score + slope_score) / 3.0
        except Exception:
            return 0.0

    def _end_epoch_cleanup(self):
        try:
            # Limpa buffers que são por época ou passo, mas mantém estado de longo prazo.
            # Buffers de Δ-L2 e D-max/SNR para display são tipicamente limpos ou gerenciados por maxlen.
            # Os novos mk_stream, q_digest, ewma_slope são de streaming, não devem ser limpos por época.
            
            # Os buffers a seguir são tipicamente baseados em janelas (maxlen) e não precisam de clear por época
            # self.delta_buffers
            # self.d_max_hist
            # self.g_hist_for_snr_baseline
            # self.g_vector_hist_for_snr_calc

            # Histórico para cálculo de threshold dinâmico de Δ-L2 (display) é limpo.
            self.metric_hist_delta_l2.clear()
            self.dyn_thresh_delta_l2.clear()
            
            # Contadores e flags de estado por época
            # self.suspect_counter.clear() # Não limpar suspect_counter, pois k_confirm é mult-época

        except Exception as e:
            logFun(f"[ConvergeControl._end_epoch_cleanup] Error during cleanup: {e}", lvl="error")

class RollingStats:

    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError("Window size must be positive.")
        self.window = deque(maxlen=window_size)
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self._n = 0 # Número de elementos atualmente na janela

    def add(self, x: float):
        if self._n == self.window.maxlen: # Se a janela está cheia
            old = self.window[0] # self.window.popleft() implicitamente feito por maxlen
            self.sum_x -= old
            self.sum_x2 -= old * old
            # _n não muda aqui, pois um entra e um sai
        else: # Janela ainda não está cheia
            self._n +=1

        self.window.append(x)
        self.sum_x += x
        self.sum_x2 += x * x

    @property
    def mean(self) -> float:
        return self.sum_x / self._n if self._n else 0.0

    @property
    def variance(self) -> float:
        if self._n < 2:
            return 0.0
        # Usa a fórmula E[X^2] - (E[X])^2 para variância da amostra (dividido por n-1)
        # Var = (sum_x2 - (sum_x^2)/n) / (n-1)
        variance = (self.sum_x2 - (self.sum_x * self.sum_x) / self._n) / (self._n - 1)
        return max(0.0, variance) # Garante não negatividade devido a erros de ponto flutuante

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def count(self) -> int:
        return self._n