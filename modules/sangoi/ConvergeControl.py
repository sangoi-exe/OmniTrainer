import math
import torch
import collections
import traceback
import numpy as np
from rich.columns import Columns
from rich.console import Console  # Manter, mas o Live Display usará o console do Trainer
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Group
from rich.text import Text
from rich import box
from scipy.stats import linregress
from modules.sangoi.logFun import logFun
from typing import Deque, Dict, Set, List, Optional, Tuple, Union  # Adicionado Union

# console = Console() # Pode ser removido se o Live display for o principal meio de output


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
        delta_buffer_size: int | None = None,
        debug: bool = True,
        verbose: bool = True,
        total_epochs: int = 100,
        delta_abs_thresh_fallback: float = 2e-3,
        hist_size_delta_l2: int = 10,
        min_buffer_epochs: float = 0,
        enable_freeze_action: bool = True,
        dhat_slope_norm_factor: float = 1e-5,
        min_overall_gamma_dhat: float = 0.1,
        max_overall_gamma_dhat: float = 2.5,

        k_confirm: int = 3, # reduzir facilita
        delta_snr: float = 0.5, # reduzir facilita
        base_gamma_d_max: float = 0.5, # reduzir facilita
        dynamic_k_delta_l2: float = 2, # aumentar facilita
        cv_thresh_fallback: float = 0.8, # aumentar facilita
        warm_steps_dhat_snr: int = 200, # reduzir facilita
        slope_thresh_fallback: float = 4e-4, # aumentra facilita
        slope_sensitivity_dhat: float = 1, # aumentar facilita
        min_stable_metrics_freeze: int = 3, # quantas métricas exigir pra considerar stable (1~3)
    ):

        self.k_confirm = max(2, k_confirm)
        self.win_size_delta_l2 = 10
        self.cv_thresh_fallback = cv_thresh_fallback
        self.abs_thresh_fallback = delta_abs_thresh_fallback
        self.slope_thresh_fallback = slope_thresh_fallback
        self.dynamic_k_delta_l2 = dynamic_k_delta_l2
        self.hist_size_delta_l2 = max(1, hist_size_delta_l2)
        self.metric_hist_delta_l2: Dict[str, Dict[str, Deque[float]]] = collections.defaultdict(
            lambda: {
                'mean': collections.deque(maxlen=self.hist_size_delta_l2),
                'cv': collections.deque(maxlen=self.hist_size_delta_l2),
                'slope': collections.deque(maxlen=self.hist_size_delta_l2),})
        self.dyn_thresh_delta_l2: Dict[str, Dict[str, float]] = {}
        self._last_epoch_incr: Dict[str, int] = {}
        self.freeze_step_run2: Dict[str, int] = {}
        self.min_buffer_epochs = min_buffer_epochs
        hint = delta_buffer_size or 0
        self._buffer_maxlen_delta_l2 = max(hint, self.win_size_delta_l2)
        self.delta_buffers = collections.defaultdict(lambda: collections.deque(maxlen=self._buffer_maxlen_delta_l2))
        self._rolling_stats = collections.defaultdict(lambda: RollingStats(self.win_size_delta_l2))
        self.perma_frozen: Set[str] = set()
        self.suspect_counter: Dict[str, int] = collections.defaultdict(int)
        self._current_epoch: int = 0
        self._current_step: int = 0
        self.debug = debug
        self.verbose = verbose
        self.run_number = run_number
        self.total_epochs = total_epochs
        self.enable_freeze_action = (enable_freeze_action if run_number >= 2 else False)
        self.warm_steps = warm_steps_dhat_snr
        self.base_gamma_d_max_config = base_gamma_d_max
        self.delta_snr = delta_snr
        self.min_stable_metrics_freeze = min_stable_metrics_freeze  # Requer d-max, slope d-max E SNR estáveis
        self.grad_eps = 1e-9
        self.thr_d_max: Dict[str, float] = {}
        self.thr_slope_d_max: Dict[str, float] = {}
        self.last_slope_d_max: Dict[str, float] = {}
        self.thr_snr: Dict[str, float] = {}
        self.win_snr_calc = 2
        self.g_hist_for_snr_baseline = collections.defaultdict(lambda: collections.deque(maxlen=self.warm_steps))
        self.g_vector_hist_for_snr_calc = collections.defaultdict(lambda: collections.deque(maxlen=self.win_snr_calc))
        self.d_max_hist = collections.defaultdict(lambda: collections.deque(maxlen=self.warm_steps))
        self.base_d_max: Dict[str, float] = {}
        self.base_snr: Dict[str, float] = {}
        self.slope_sensitivity_dhat = slope_sensitivity_dhat
        self.dhat_slope_norm_factor = dhat_slope_norm_factor
        self.min_overall_gamma_dhat = min_overall_gamma_dhat
        self.max_overall_gamma_dhat = max_overall_gamma_dhat

        if self.debug:
            try:
                logFun(
                    f"[ConvergeControl] Init: Run {self.run_number}, FreezeAction: {self.enable_freeze_action}. "
                    f"Δ-L2 params: dynamic_k={self.dynamic_k_delta_l2}, hist_size={self.hist_size_delta_l2}. "
                    f"Freeze Decision (d-max, SNR): Warmup={self.warm_steps} steps, "
                    f"base_gamma_d_max_config={self.base_gamma_d_max_config:.2f} "
                    f"(dyn_slope_sens={self.slope_sensitivity_dhat:.2f}, dyn_slope_norm={self.dhat_slope_norm_factor:.1e}), "
                    f"delta_snr={self.delta_snr}, MinStableMetrics={self.min_stable_metrics_freeze}",
                    lvl="CONVCTRL")  # Use CONVCTRL log level
            except Exception as e:
                print(f"[ConvergeControl __init__] Error in logFun: {e}")  # Fallback print
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
            # if self.debug: logFun(f"[DEBUG_CALC_SNR] Not enough valid_grads, returning None.", lvl="CONVCTRL")
            return None
        try:
            g_stack = torch.stack(valid_grads)
            mu_vec = g_stack.mean(dim=0)
            std_vec = g_stack.std(dim=0, unbiased=True)
            mu_norm = mu_vec.norm().item()
            std_norm = std_vec.norm().item()
            if std_norm < self.grad_eps:
                # if self.debug: logFun(f"[DEBUG_CALC_SNR] std_norm too small, returning inf.", lvl="CONVCTRL")
                return float('inf')
            return mu_norm / std_norm
        except Exception as e:
            logFun(f"[ConvergeControl] Error calculating SNR: {e}", lvl="error")
            # traceback.print_exc() # Covered by logFun or top level
            return None

    def update_step_metrics(self, name: str, module: torch.nn.Module):
        try:
            grads = [
                p.grad.detach().flatten() for p in module.parameters() if p.grad is not None and p.grad.numel() > 0]
            if grads:
                grad_vec = torch.cat(grads)
                self.g_vector_hist_for_snr_calc[name].append(grad_vec.clone())
                if len(self.g_vector_hist_for_snr_calc[name]) >= self.win_snr_calc:
                    snr_val = self._calculate_snr(self.g_vector_hist_for_snr_calc[name])
                    if snr_val is not None:
                        self.g_hist_for_snr_baseline[name].append(snr_val)
        except Exception as e:
            logFun(f"[ConvergeControl] Exception during update_step_metrics for '{name}': {e}", lvl="error")

    def snapshot_epoch_weights(self):
        """
        CALCULA estatísticas de Delta-L2 e atualiza limiares dinâmicos.
        NÃO IMPRIME MAIS NADA DIRETAMENTE. A geração do renderizável é feita por
        generate_status_renderable().
        """
        for module_name in list(self.delta_buffers.keys()):
            try:
                buf_delta_l2 = self.delta_buffers[module_name]
                if len(buf_delta_l2) < self.win_size_delta_l2:
                    # if self.debug and self._current_step % 10 == 0:
                    #     logFun(
                    #         f"[ConvergeControl.snapshot_epoch_weights] Mod '{module_name}' ({len(buf_delta_l2)}/{self.win_size_delta_l2}) Δ-L2 buf too short for stats.",
                    #         lvl="CONVCTRL")
                    continue

                rs = self._rolling_stats[module_name]
                mean_dl2 = rs.mean
                std_dl2 = rs.std
                slope_dl2 = self.stats_window(list(buf_delta_l2)[-self.win_size_delta_l2:])[2]
                cv_dl2 = std_dl2 / (mean_dl2 + 1e-12) if mean_dl2 > 1e-12 else 0.0

                # Atualiza histórico e limiares dinâmicos para Delta-L2
                mh_dl2 = self.metric_hist_delta_l2[module_name]
                mh_dl2['mean'].append(mean_dl2)
                mh_dl2['cv'].append(cv_dl2)
                mh_dl2['slope'].append(abs(slope_dl2))

                if len(mh_dl2['mean']) == self.hist_size_delta_l2:
                    for met_key in ('mean', 'cv', 'slope'):
                        metric_values_for_dyn_thresh = list(mh_dl2[met_key])
                        med_met, iqr_met = self._robust_stats(metric_values_for_dyn_thresh)
                        base, var = med_met, iqr_met
                        calculated_thresh = base + self.dynamic_k_delta_l2 * var
                        self.dyn_thresh_delta_l2.setdefault(module_name, {})[met_key] = calculated_thresh
                        # if self.debug:
                        #     logFun(f"[ConvergeControl.snapshot_epoch_weights] Mod '{module_name}' DynThresh Δ-L2 for '{met_key}' updated to: {calculated_thresh:.3e} "
                        #           f"(based on hist_len={len(metric_values_for_dyn_thresh)}, med={med_met:.3e}, iqr={iqr_met:.3e})", lvl="CONVCTRL")
            except Exception as e:
                logFun(
                    f"[ConvergeControl.snapshot_epoch_weights] Error processing CALCS for module '{module_name}': {e}",
                    lvl="error")
                # traceback.print_exc() # Covered by logFun or top level

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

            buf_delta_l2 = self.delta_buffers.get(module_name)
            if buf_delta_l2 is None or len(buf_delta_l2) < self.win_size_delta_l2:
                return Panel(Text(f"Aguardando dados ({len(buf_delta_l2 or [])}/{self.win_size_delta_l2})",
                                  justify="center",
                                  style="dim white"),
                             title=f"[magenta]{short_module_name}[/]",
                             border_style="dim magenta",
                             padding=(0, 1),
                             width=55)

            rs = self._rolling_stats[module_name]
            mean_dl2 = rs.mean
            std_dl2 = rs.std
            slope_dl2 = self.stats_window(list(buf_delta_l2)[-self.win_size_delta_l2:])[2]
            cv_dl2 = std_dl2 / (mean_dl2 + 1e-12) if mean_dl2 > 1e-12 else 0.0
            score_dl2_val = self.stability_score(module_name)

            # --- Preparar valores e status para D-hat, D-hat Slope, SNR ---
            # D-hat
            d_hat_value_text = Text("N/A")  # Texto para o valor numérico
            d_hat_status_suffix = Text("")  # Texto para " (OK) of <threshold"
            if module_name in self.d_max_hist and self.d_max_hist[module_name]:
                curr_d_max_val = self.d_max_hist[module_name][-1]
                d_hat_value_text = Text(f"{curr_d_max_val:.2e}")
                if module_name in self.thr_d_max:
                    threshold_val = self.thr_d_max[module_name]
                    if curr_d_max_val < threshold_val:
                        d_hat_value_text.stylize("green")
                        d_hat_status_suffix = Text.assemble(Text(" (", style="dim"), Text("OK", style="green"),
                                                            Text(f") of <{threshold_val:.2e}", style="dim"))
                    else:
                        d_hat_value_text.stylize("red")
                        d_hat_status_suffix = Text.assemble(Text(" (", style="dim"), Text("NO", style="red"),
                                                            Text(f") of <{threshold_val:.2e}", style="dim"))
                elif module_name in self.base_d_max:
                    d_hat_status_suffix = Text(f" (warmup, base <{self.base_d_max[module_name]:.2e})", style="dim")

            # D-hat Slope
            d_hat_slope_value_text = Text("N/A")
            d_hat_slope_status_suffix = Text("")
            if module_name in self.last_slope_d_max:
                curr_slope_val = self.last_slope_d_max[module_name]
                d_hat_slope_value_text = Text(f"{curr_slope_val:.2e}")
                if module_name in self.thr_slope_d_max:
                    threshold_val = self.thr_slope_d_max[module_name]
                    if abs(curr_slope_val) < threshold_val:
                        d_hat_slope_value_text.stylize("green")
                        d_hat_slope_status_suffix = Text.assemble(Text(" (", style="dim"), Text("OK", style="green"),
                                                                  Text(f") of Abs <{threshold_val:.2e}", style="dim"))
                    else:
                        d_hat_slope_value_text.stylize("red")
                        d_hat_slope_status_suffix = Text.assemble(Text(" (", style="dim"), Text("NO", style="red"),
                                                                  Text(f") of Abs <{threshold_val:.2e}", style="dim"))

            # SNR
            snr_value_text = Text("N/A")
            snr_status_suffix = Text("")
            last_snr_val = None
            snr_from_calc_win = self._calculate_snr(
                self.g_vector_hist_for_snr_calc.get(module_name, collections.deque()))
            if snr_from_calc_win is not None:
                last_snr_val = snr_from_calc_win
            elif module_name in self.g_hist_for_snr_baseline and self.g_hist_for_snr_baseline[module_name]:
                last_snr_val = self.g_hist_for_snr_baseline[module_name][-1]
            if last_snr_val is not None:
                snr_value_text = Text(f"{last_snr_val:.2f}")
                if module_name in self.thr_snr:
                    threshold_val = self.thr_snr[module_name]
                    if last_snr_val < threshold_val:
                        snr_value_text.stylize("green")
                        snr_status_suffix = Text.assemble(Text(" (", style="dim"), Text("OK", style="green"),
                                                          Text(f") of <{threshold_val:.2f}", style="dim"))
                    else:
                        snr_value_text.stylize("red")
                        snr_status_suffix = Text.assemble(Text(" (", style="dim"), Text("NO", style="red"),
                                                          Text(f") of <{threshold_val:.2f}", style="dim"))
                elif module_name in self.base_snr:
                    snr_status_suffix = Text(f" (warmup, base <{self.base_snr[module_name]:.2f})", style="dim")

            suspect_count = self.suspect_counter.get(module_name, 0)
            is_permafrozen = module_name in self.perma_frozen
            freeze_status_combined_text = Text()
            if is_permafrozen:
                freeze_status_combined_text.append("PERMA", style="bold white on red")
            else:
                if suspect_count > 0:
                    freeze_status_combined_text.append(str(suspect_count), style="yellow")
                else:
                    freeze_status_combined_text.append(str(suspect_count))  # Sem cor se 0
                freeze_status_combined_text.append(f"/{self.k_confirm}", style="dim")

            table = Table(title=None,
                          show_header=True,
                          header_style="bold white",
                          box=box.SIMPLE_HEAD,
                          padding=(0, 1),
                          show_edge=False,
                          expand=True)
            table.add_column("Métrica", style="cyan", ratio=2, overflow="fold", no_wrap=False)
            table.add_column("Valor", justify="left", style="white", ratio=3)

            table.add_row("Δ-L2 Mean", f"{mean_dl2:.2e}")
            table.add_row("Δ-L2 CV", f"{cv_dl2:.2e}")
            table.add_row("Δ-L2 Slope", f"{slope_dl2:.2e}")
            table.add_row("Score (%)", f"{score_dl2_val*100:.0f}")  # Sem casas decimais para score

            # Usar Text.assemble para combinar o valor (já estilizado) com o sufixo de status
            table.add_row("D-max", Text.assemble(d_hat_value_text,
                                                 d_hat_status_suffix))  # Mudança para d-max como na imagem
            table.add_row("d-max Slope", Text.assemble(d_hat_slope_value_text,
                                                       d_hat_slope_status_suffix))  # Mudança para d-max Slope
            table.add_row("SNR", Text.assemble(snr_value_text, snr_status_suffix))
            table.add_row("Freeze Status", freeze_status_combined_text)

            return Panel(table,
                         title=f"[magenta]{short_module_name}[/] (E{self._current_epoch})",
                         border_style="magenta",
                         padding=(0, 0),
                         width=30)
        except Exception as e:
            logFun(f"Error generating card for {module_name}: {e}", lvl="error")
            return Panel(Text(f"Error for {module_name}.", style="red"),
                         title=f"[red]{module_name[:20]}... - ERR[/red]",
                         width=30)

    def generate_status_renderable(self) -> Union[Group, Columns, Text]:  # Adicionado Columns
        """
            Gera um objeto Rich (Group de Panels) para ser exibido pelo Live display.
            Cada Panel é um "card" para um módulo.
            Usa Columns se houver muitos módulos.
            """
        if not self.delta_buffers and not self.d_max_hist and not self.g_hist_for_snr_baseline:
            return Text("ConvergeControl: Aguardando os primeiros dados...", justify="center")

        all_module_names = set(self.delta_buffers.keys()) | \
                          set(self.d_max_hist.keys()) | \
                          set(self.g_hist_for_snr_baseline.keys())

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

        # Se houver mais de, por exemplo, 3 cards, use Columns
        if len(module_cards) > 1:  # Ajuste este número conforme o tamanho da sua tela
            return Columns(
                module_cards,
                expand=True,  # Tenta usar toda a largura disponível
                equal=False,  # Permite que as colunas tenham larguras baseadas no conteúdo (ou width fixo do Panel)
                padding=0,  # REDUZIR padding entre as colunas para 0 ou 1
                column_first=True  # Organiza os itens preenchendo colunas primeiro, depois linhas (pode ajudar no layout)
            )
        else:
            # Para poucos cards, Group (empilhamento vertical) pode ser ok, ou Columns com 1 coluna
            return Group(*module_cards)

    def _ready_to_freeze(self, name: str) -> bool:
        try:
            if self._current_step < self.warm_steps:
                # if self.debug and self._current_step % (self.warm_steps // 4 if self.warm_steps > 0 else 10) == 0:
                #     logFun(
                #         f"[_ready_to_freeze] Mod '{name}' in Dhat/SNR warmup ({self._current_step}/{self.warm_steps} steps).",
                #         lvl="CONVCTRL")
                return False

            if name in self.d_max_hist and len(self.d_max_hist[name]) >= 2:
                med_d_max, iqr_d_max = self._robust_stats(self.d_max_hist[name])
                self.base_d_max[name] = med_d_max
                slope_d_max = 0.0
                d_max_history_list = list(self.d_max_hist[name])
                if len(d_max_history_list) >= 2:
                    _, _, slope_d_max = self.stats_window(d_max_history_list)

                adjustment_value = 0.0
                if self.dhat_slope_norm_factor != 0:
                    adjustment_value = -(slope_d_max / self.dhat_slope_norm_factor) * self.slope_sensitivity_dhat
                current_gamma_d_max = self.base_gamma_d_max_config * (1 + adjustment_value)
                current_gamma_d_max = max(self.min_overall_gamma_dhat,
                                          min(self.max_overall_gamma_dhat, current_gamma_d_max))
                self.thr_d_max[name] = max(0.0, med_d_max - current_gamma_d_max * iqr_d_max)
                self.thr_slope_d_max[name] = self.slope_sensitivity_dhat * iqr_d_max  # Threshold para |slope|
                self.last_slope_d_max[name] = slope_d_max
            elif name in self.thr_d_max:
                del self.thr_d_max[name]
                del self.thr_slope_d_max[name]
                if name in self.base_d_max:
                    del self.base_d_max[name]
                if name in self.last_slope_d_max:
                    del self.last_slope_d_max[name]

            if name in self.g_hist_for_snr_baseline and len(self.g_hist_for_snr_baseline[name]) >= 2:
                med_snr, iqr_snr = self._robust_stats(self.g_hist_for_snr_baseline[name])
                self.base_snr[name] = med_snr
                self.thr_snr[name] = max(0.0, med_snr - self.delta_snr * iqr_snr)
            elif name in self.thr_snr:
                del self.thr_snr[name]
                if name in self.base_snr:
                    del self.base_snr[name]

            if not (name in self.thr_d_max and name in self.thr_snr and name in self.thr_slope_d_max):
                # if self.debug and self._current_step % 10 == 0:
                #     logFun(f"[_ready_to_freeze] Mod '{name}' waiting for D/S/Slope thresholds.", lvl="CONVCTRL")
                return False

            d_max_ok, slope_d_max_ok, snr_ok = False, False, False
            stable_metrics_count = 0

            curr_d_max_val = self.d_max_hist[name][-1] if name in self.d_max_hist and self.d_max_hist[name] else float(
                'inf')
            if curr_d_max_val < self.thr_d_max[name]:
                d_max_ok = True
                stable_metrics_count += 1

            curr_slope_d_max_val = self.last_slope_d_max.get(name, float('inf'))
            if abs(curr_slope_d_max_val) < self.thr_slope_d_max[name]:  # Check absolute slope
                slope_d_max_ok = True
                stable_metrics_count += 1

            current_snr_calculated = self._calculate_snr(self.g_vector_hist_for_snr_calc[name])
            curr_snr_val = current_snr_calculated if current_snr_calculated is not None else float('inf')
            if curr_snr_val < self.thr_snr[name]:
                snr_ok = True
                stable_metrics_count += 1

            is_overall_stable = stable_metrics_count >= self.min_stable_metrics_freeze
            # is_overall_stable = d_max_ok and slope_d_max_ok and snr_ok  # Exigir todos os 3 OK

            # if self.debug and self._current_step % 10 == 0:
            #     log_msg = (
            #         f"[_ready_to_freeze] '{name}': "
            #         f"Dhat={curr_d_max_val:.2e}({d_max_ok}), Dslp={curr_slope_d_max_val:.2e}({slope_d_max_ok}), SNR={curr_snr_val:.2e}({snr_ok}) -> Stable={is_overall_stable}"
            #     )
            #     logFun(log_msg, lvl="CONVCTRL")
            return is_overall_stable
        except Exception as e:
            if hasattr(self, 'debug') and self.debug:
                logFun(f"[ConvergeControl._ready_to_freeze] Error for module '{name}': {e}", lvl="error")
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
                    slope = float(linregress(x, y).slope)
                except (ValueError, Exception):
                    slope = 0.0  # Catch more general linregress errors
            return mean, std, slope
        except Exception:  # Catch-all for unexpected issues
            # logFun(f"[ConvergeControl.stats_window] Generic error: {e_general}. Input (first 5): {str(arr[:5])}", lvl="error")
            return 0.0, 0.0, 0.0

    def set_current_time(self, epoch: int, step: int):
        self._current_epoch = epoch
        self._current_step = step

    def ingest_and_process(self, stats_list: List[Dict]):
        if not stats_list:
            return
        processed_deltas, processed_d_maxs = 0, 0
        for stat in stats_list:
            name, delta_val, d_max_val = stat.get("name"), stat.get("delta_L2"), stat.get("d_max")
            if name is None:
                continue
            try:
                if delta_val is not None:
                    self.delta_buffers[name].append(float(delta_val))
                    self._rolling_stats[name].add(float(delta_val))
                    processed_deltas += 1
                if d_max_val is not None:
                    self.d_max_hist[name].append(float(d_max_val))
                    processed_d_maxs += 1
            except (ValueError, TypeError):  # Minor logging for conversion errors
                # if self.debug: logFun(f"[ConvergeControl] Error processing stats for '{name}': {e}. ΔL2: {delta_val}, Dhat: {d_max_val}", lvl="warning")
                pass
        # if self.debug and (processed_deltas == 0 or processed_d_maxs == 0) and self._current_step % 10 == 0:
        # logFun(f"[ConvergeControl] Step {self._current_step}: No ΔL2 ({processed_deltas}) or Dhat ({processed_d_maxs}) ingested.", lvl="CONVCTRL")
            pass

    def decide(self) -> Dict[str, bool]:
        try:
            if not self.enable_freeze_action:
                return {}
            progress = self._current_epoch / max(self.total_epochs, 1) if self.total_epochs > 0 else 0
            if progress < self.min_buffer_epochs:
                # if self.debug and self._current_step % (self.total_epochs // 10 if self.total_epochs > 10 else 10) == 0:
                # logFun(f"[ConvergeControl] Decide@{self._current_step}: Epoch Warmup ({progress:.2f} < {self.min_buffer_epochs:.2f}).", lvl="CONVCTRL")
                return {}

            decisions: Dict[str, bool] = {}
            module_names_to_decide = list(set(self.g_vector_hist_for_snr_calc.keys()) | set(self.d_max_hist.keys()))
            # if self.debug and not module_names_to_decide and self._current_step % 10 == 0:
            # logFun(f"[ConvergeControl] Decide@{self._current_step}: No modules with d-max/SNR history for eval.", lvl="CONVCTRL")

            for name in module_names_to_decide:
                try:
                    if name in self.perma_frozen:
                        decisions[name] = True
                        continue
                    is_module_stable = self._ready_to_freeze(name)
                    current_epoch, last_incr_epoch = self._current_epoch, self._last_epoch_incr.get(name, -1)
                    if is_module_stable:
                        if current_epoch > last_incr_epoch:
                            self.suspect_counter[name] += 1
                            self._last_epoch_incr[name] = current_epoch
                    elif current_epoch > last_incr_epoch:
                        self.suspect_counter[name] = 0
                        self._last_epoch_incr[name] = current_epoch

                    should_freeze_eval = self.suspect_counter.get(name, 0) >= self.k_confirm
                    decisions[name] = should_freeze_eval
                    if should_freeze_eval and name not in self.perma_frozen:
                        self.perma_frozen.add(name)
                        self.freeze_step_run2[name] = self._current_step
                        if self.verbose or self.debug:
                            logFun(
                                f"[ConvergeControl] FREEZE ACTION '{name}' @ step {self._current_step} "
                                f"(Trigger: Stable for {self.suspect_counter.get(name, 0)} checks >= {self.k_confirm})",
                                lvl="CONVCTRL")
                except Exception as e_module:
                    logFun(f"[ConvergeControl.decide] Error processing decision for module '{name}': {e_module}",
                           lvl="error")
                    decisions[name] = False
            return decisions
        except Exception as e_outer:
            logFun(f"[ConvergeControl.decide] Outer error in decide method: {e_outer}", lvl="error")
            return {}

    def get_frozen_set(self) -> Set[str]:
        return self.perma_frozen

    def get_final_conv_steps(self) -> Dict[str, int]:
        return {} if self.run_number == 1 and not self.enable_freeze_action else dict(self.freeze_step_run2)

    def _delta_stable(self, name: str) -> bool:
        try:
            win, buf = self.win_size_delta_l2, self.delta_buffers[name]
            if len(buf) < win:
                return False
            mean, std, slope = self.stats_window(list(buf)[-win:])
            cv = std / (mean + 1e-12)
            dyn_m_thresh = self.dyn_thresh_delta_l2.get(name, {}).get('mean', self.abs_thresh_fallback)
            dyn_cv_thresh = self.dyn_thresh_delta_l2.get(name, {}).get('cv', self.cv_thresh_fallback)
            dyn_s_thresh = self.dyn_thresh_delta_l2.get(name, {}).get('slope', self.slope_thresh_fallback)
            mean_ok, cv_ok, slope_ok = mean < dyn_m_thresh, cv < dyn_cv_thresh, abs(slope) < dyn_s_thresh
            return (mean_ok and cv_ok) or (cv_ok and slope_ok)
        except Exception:  # Simplified error handling
            # if hasattr(self, 'debug') and self.debug: logFun(f"[ConvergeControl._delta_stable] Error for module '{name}': {e}", lvl="error")
            return False

    def stability_score(self, name: str) -> float:
        try:
            buf = self.delta_buffers.get(name)
            if buf is None or len(buf) < self.win_size_delta_l2:
                return 0.0
            mean, std, slope = self.stats_window(list(buf)[-self.win_size_delta_l2:])
            cv = std / (mean + 1e-12)
            thr_mean = self.dyn_thresh_delta_l2.get(name, {}).get('mean', self.abs_thresh_fallback)
            thr_cv = self.dyn_thresh_delta_l2.get(name, {}).get('cv', self.cv_thresh_fallback)
            thr_slope = self.dyn_thresh_delta_l2.get(name, {}).get('slope', self.slope_thresh_fallback)
            mean_score = max(0.0, min(1.0, (thr_mean - mean) / (thr_mean + 1e-12)))
            cv_score = max(0.0, min(1.0, (thr_cv - cv) / (thr_cv + 1e-12)))
            slope_score = max(0.0, min(1.0, (thr_slope - abs(slope)) / (thr_slope + 1e-12)))
            return (mean_score + cv_score + slope_score) / 3.0
        except Exception:  # Simplified error handling
            # logFun(f"[ConvergeControl.stability_score] Error for module '{name}': {e}", lvl="error")
            return 0.0

    def _end_epoch_cleanup(self):
        try:
            for buf_list in [
                    self.delta_buffers, self.d_max_hist, self.g_hist_for_snr_baseline, self.g_vector_hist_for_snr_calc]:
                for buf in buf_list.values():
                    buf.clear()
            self.metric_hist_delta_l2.clear()
            self.dyn_thresh_delta_l2.clear()
            self._last_epoch_incr.clear()
            self.suspect_counter.clear()
        except Exception as e:
            logFun(f"[ConvergeControl._end_epoch_cleanup] Error during cleanup: {e}", lvl="error")


from collections import deque


class RollingStats:

    def __init__(self, window_size: int):
        self.window = deque(maxlen=window_size)
        self.sum_x = 0.0
        self.sum_x2 = 0.0

    def add(self, x: float):
        if len(self.window) == self.window.maxlen:
            old = self.window[0]
            self.sum_x -= old
            self.sum_x2 -= old * old
        self.window.append(x)
        self.sum_x += x
        self.sum_x2 += x * x

    @property
    def mean(self) -> float:
        n = len(self.window)
        return self.sum_x / n if n else 0.0

    @property
    def variance(self) -> float:
        n = len(self.window)
        if n < 2:
            return 0.0
        return (self.sum_x2 - (self.sum_x**2) / n) / (n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)
