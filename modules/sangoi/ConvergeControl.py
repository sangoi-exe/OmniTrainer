import math
import torch
import collections
import traceback
import numpy as np
from scipy.stats import linregress
from modules.sangoi.logFun import logFun
from typing import Deque, Dict, Set, List, Optional, Tuple




class ConvergeControl:
    """
    Coleta métricas de delta-L2 (para logging/score), D-hat e SNR (para decisão
    de congelamento). Utiliza limiares dinâmicos para D-hat e SNR.
    O congelamento só é aplicável a partir da Run 2 (ou se explicitamente habilitado).
    """

    def __init__(
        self,
        run_number: int,
        *,
        k_confirm: int = 5,
        delta_buffer_size: int | None = None,
        debug: bool = True,
        verbose: bool = True,
        total_epochs: int = 100,

        cv_thresh_fallback: float = 0.5,
        slope_thresh_fallback: float = 1e-4,
        delta_abs_thresh_fallback: float = 2e-3,
        dynamic_k_delta_l2: float = 1.5,
        hist_size_delta_l2: int = 10,

        warm_steps_dhat_snr: int = 200,
        # Renomeado para base_gamma_d_max no construtor para clareza
        # O valor passado aqui será o ponto de partida para o gamma dinâmico.
        base_gamma_d_max: float = 1.0,
        delta_snr: float = 1.0,
        min_stable_metrics_freeze: int = 2,
        min_buffer_epochs: float = 0,
        enable_freeze_action: bool = True,

        # Novos parâmetros para gamma_d_max dinâmico com slope
        # A janela para o slope do D-hat usará self.warm_steps (maxlen de d_max_hist)
        slope_sensitivity_dhat: float = 0.5,
        dhat_slope_norm_factor: float = 1e-5, # <<< AJUSTAR ISTO EXPERIMENTALMENTE
        min_overall_gamma_dhat: float = 0.1,
        max_overall_gamma_dhat: float = 2.5,
    ):

        self.k_confirm = max(2, k_confirm)
        self.win_size_delta_l2 = 10  # Janela para stats de delta-L2 (mean, cv, slope)

        # Thresholds fixos para delta-L2 (fallback para stability_score)
        self.cv_thresh_fallback = cv_thresh_fallback
        self.abs_thresh_fallback = delta_abs_thresh_fallback
        self.slope_thresh_fallback = slope_thresh_fallback

        # Parâmetros para limiares dinâmicos de delta-L2 (para stability_score)
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
        # Armazena o gamma_d_max base fornecido no init
        self.base_gamma_d_max_config = base_gamma_d_max
        self.delta_snr = delta_snr
        # agora exigimos D-hat, slope e SNR estáveis
        self.min_stable_metrics_freeze = 3

        self.grad_eps = 1e-9

        # thresholds dinâmicos para D-hat e seu slope
        self.thr_d_max: Dict[str, float] = {}
        self.thr_slope_d_max: Dict[str, float] = {}
        # último slope calculado de D-hat, para comparação na decisão
        self.last_slope_d_max: Dict[str, float] = {}
        self.thr_snr: Dict[str, float] = {}

        self.win_snr_calc = 2
        self.g_hist_for_snr_baseline = collections.defaultdict(
            lambda: collections.deque(maxlen=self.warm_steps)
        )
        self.g_vector_hist_for_snr_calc = collections.defaultdict(
            lambda: collections.deque(maxlen=self.win_snr_calc)
        )
        self.d_max_hist = collections.defaultdict(lambda: collections.deque(maxlen=self.warm_steps))

        self.base_d_max: Dict[str, float] = {} # Mediana de D-hat
        self.base_snr: Dict[str, float] = {} # Mediana de SNR

        # Atributos para o gamma dinâmico
        self.slope_sensitivity_dhat = slope_sensitivity_dhat
        self.dhat_slope_norm_factor = dhat_slope_norm_factor
        self.min_overall_gamma_dhat = min_overall_gamma_dhat
        self.max_overall_gamma_dhat = max_overall_gamma_dhat
        
        # Parâmetro gamma_d_max original agora é o 'base' para o cálculo dinâmico
        # Não precisa mais de self.gamma_d_max como um atributo fixo se ele será sempre dinâmico
        # dentro de _ready_to_freeze. O valor configurado é self.base_gamma_d_max_config.

        if self.debug:
            try:
                logFun(
                    f"[ConvergeControl] Init: Run {self.run_number}, FreezeAction: {self.enable_freeze_action}. "
                    f"Δ-L2 params: dynamic_k={self.dynamic_k_delta_l2}, hist_size={self.hist_size_delta_l2}. "
                    f"Freeze Decision (D-hat, SNR): Warmup={self.warm_steps} steps, "
                    f"base_gamma_d_max_config={self.base_gamma_d_max_config:.2f} " # Log do gamma base
                    f"(dyn_slope_sens={self.slope_sensitivity_dhat:.2f}, dyn_slope_norm={self.dhat_slope_norm_factor:.1e}), "
                    f"delta_snr={self.delta_snr}, MinStableMetrics={self.min_stable_metrics_freeze}",
                    lvl="debug")
            except Exception as e:
                print(f"[ConvergeControl __init__] Error in logFun: {e}")
                traceback.print_exc()

    def _robust_stats(self, seq):
        arr = np.asarray(seq, dtype=np.float32)
        if arr.size < 2:
            return float(np.median(arr)) if arr.size == 1 else 0.0, 0.0
        q1, med, q3 = np.percentile(arr, [25,50,75])
        return float(med), float(q3 - q1)

    def _calculate_snr(self, g_vector_deque: Deque[torch.Tensor]) -> Optional[float]:
        if len(g_vector_deque) < 2:
            return None

        valid_grads = [g for g in list(g_vector_deque) if g is not None and g.numel() > 0]

        if len(valid_grads) < 2:
            if self.debug:
                try:
                    logFun(f"[DEBUG_CALC_SNR] Not enough valid_grads, returning None.", lvl="debug")
                except Exception as log_e:
                    print(f"Logging error in _calculate_snr: {log_e}")  # Basic fallback
            return None

        try:
            g_stack = torch.stack(valid_grads)
            mu_vec = g_stack.mean(dim=0)
            std_vec = g_stack.std(dim=0, unbiased=True)

            mu_norm = mu_vec.norm().item()
            std_norm = std_vec.norm().item()

            if std_norm < self.grad_eps:
                if self.debug:
                    try:
                        logFun(f"[DEBUG_CALC_SNR] std_norm too small, returning inf.", lvl="debug")
                    except Exception as log_e:
                        print(f"Logging error in _calculate_snr: {log_e}")
                return float('inf')

            current_snr = mu_norm / std_norm
            return current_snr
        except Exception as e:
            try:
                logFun(f"[ConvergeControl] Error calculating SNR: {e}", lvl="error")
                traceback.print_exc()
            except Exception as log_e:
                print(f"Logging error in _calculate_snr during exception handling: {log_e}")
                print(f"Original error: {e}")
                traceback.print_exc()  # Print original traceback too
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
            else:
                pass

        except Exception as e:
            try:
                logFun(f"[ConvergeControl] Exception during update_step_metrics for '{name}': {e}", lvl="error")
                traceback.print_exc()
            except Exception as log_e:
                print(f"Logging error in update_step_metrics during exception handling: {log_e}")
                print(f"Original error for '{name}': {e}")
                traceback.print_exc()
        # finally:
        #     logFun(f"[DEBUG] update_step_metrics chamado para '{name}': "
        #           f"g_vec_len={len(self.g_vector_hist_for_snr_calc[name])}, "
        #           f"g_hist_len={len(self.g_hist_for_snr_baseline[name])}",
        #           lvl="debug")                    

    def snapshot_epoch_weights(self): # << NÃO RECEBE MAIS "deltas_by_module"
        """
        Chamado no final de cada época.
        Calcula estatísticas de Delta-L2 (mean, cv, slope) para cada módulo
        usando os dados acumulados em self.delta_buffers e self._rolling_stats
        (que foram populados por ingest_and_process com deltas da Run 2).
        Atualiza os limiares dinâmicos para o stability_score e logs.
        """
        for module_name in list(self.delta_buffers.keys()): # Itera sobre os módulos que têm dados
            try:
                buf_delta_l2 = self.delta_buffers[module_name]
                if len(buf_delta_l2) < self.win_size_delta_l2:
                    if self.debug and self._current_step % 10 == 0:
                        logFun(
                            f"[ConvergeControl.snapshot_epoch_weights] Mod '{module_name}' ({len(buf_delta_l2)}/{self.win_size_delta_l2}) Δ-L2 buf too short for stats.",
                            lvl="debug")
                    continue

                # Obtém média e desvio padrão direto do RollingStats
                # _rolling_stats foi populado por ingest_and_process ao longo da época
                rs = self._rolling_stats[module_name]
                mean_dl2 = rs.mean
                std_dl2  = rs.std
                
                # Slope é calculado sobre a janela mais recente do buffer completo
                # (RollingStats não mantém a sequência para cálculo de slope)
                slope_dl2 = self.stats_window(list(buf_delta_l2)[-self.win_size_delta_l2:])[2]
                cv_dl2 = std_dl2 / (mean_dl2 + 1e-12) if mean_dl2 > 1e-12 else 0.0 # Evitar divisão por zero se média for muito pequena
                
                score_dl2 = self.stability_score(module_name) # stability_score usará os mesmos buffers

                # Atualiza histórico e limiares dinâmicos para Delta-L2
                mh_dl2 = self.metric_hist_delta_l2[module_name]
                mh_dl2['mean'].append(mean_dl2)
                mh_dl2['cv'].append(cv_dl2)
                mh_dl2['slope'].append(abs(slope_dl2)) # Armazena valor absoluto do slope

                if len(mh_dl2['mean']) == self.hist_size_delta_l2:
                    for met_key in ('mean', 'cv', 'slope'):
                        metric_values_for_dyn_thresh = list(mh_dl2[met_key])
                        # Usar média e std do histórico da métrica para o limiar dinâmico
                        # (Não mais mediana e IQR aqui, para consistência com o que foi discutido para Prodigy)
                        # Ou manter mediana e IQR se preferir robustez contra outliers no histórico da métrica.
                        # Vamos manter _robust_stats por enquanto, como estava antes.
                        med_met, iqr_met = self._robust_stats(metric_values_for_dyn_thresh)
                        base, var = med_met, iqr_met

                        calculated_thresh = base + self.dynamic_k_delta_l2 * var
                        self.dyn_thresh_delta_l2.setdefault(module_name, {})[met_key] = calculated_thresh
                        if self.debug:
                            logFun(f"[ConvergeControl.snapshot_epoch_weights] Mod '{module_name}' DynThresh Δ-L2 for '{met_key}' updated to: {calculated_thresh:.3e} "
                                  f"(based on hist_len={len(metric_values_for_dyn_thresh)}, med={med_met:.3e}, iqr={iqr_met:.3e})", lvl="debug")

                # --- Logging Detalhado (como estava antes, adaptado para module_name) ---
                curr_d_max_val_str = "N/A"
                thr_d_max_str = "N/A (no base)"
                d_max_ok_str = "-"
                if module_name in self.d_max_hist and self.d_max_hist[module_name]:
                    curr_d_max_val = self.d_max_hist[module_name][-1]
                    curr_d_max_val_str = f"{curr_d_max_val:.2e}"
                    if module_name in self.thr_d_max:
                        thr_d_max_str = f"<{self.thr_d_max[module_name]:.2e}"
                        d_max_ok_str = "OK" if curr_d_max_val < self.thr_d_max[module_name] else "NO"
                    else:
                        base_d_max_val = self.base_d_max.get(module_name, float('inf'))
                        thr_d_max_str = f"<{base_d_max_val:.2e} (warmup)"


                curr_snr_val_str = "N/A"
                thr_snr_str = "N/A (no base)"
                snr_ok_str = "-"
                last_snr_from_baseline_hist = None
                if module_name in self.g_hist_for_snr_baseline and self.g_hist_for_snr_baseline[module_name]:
                    last_snr_from_baseline_hist = self.g_hist_for_snr_baseline[module_name][-1]

                if last_snr_from_baseline_hist is not None:
                    curr_snr_val_str = f"{last_snr_from_baseline_hist:.2f}"

                if module_name in self.thr_snr:
                    thr_snr_str = f"<{self.thr_snr[module_name]:.2f}"
                    if last_snr_from_baseline_hist is not None:
                        snr_ok_str = "OK" if last_snr_from_baseline_hist < self.thr_snr[module_name] else "NO"
                    else:
                        snr_ok_str = "?"
                elif module_name in self.base_snr:
                    thr_snr_str = f"(base={self.base_snr[module_name]:.2f} no_thr_yet)"
                else:
                    thr_snr_str = "N/A (no base)"                    

                log_line1 = (
                    f"[CONV_STAT] Época {self._current_epoch:03d} — '{module_name}':\n"
                    f"  Δ-L2 Score: {score_dl2 * 100:5.1f}% (Mean={mean_dl2:.2e}, CV={cv_dl2:.2e}, Slope={slope_dl2:.2e})"
                )
                thr_dl2_mean = self.dyn_thresh_delta_l2.get(module_name, {}).get('mean', self.abs_thresh_fallback)
                thr_dl2_cv = self.dyn_thresh_delta_l2.get(module_name, {}).get('cv', self.cv_thresh_fallback)
                thr_dl2_slope = self.dyn_thresh_delta_l2.get(module_name, {}).get('slope', self.slope_thresh_fallback)
                log_line2 = (f"  Δ-L2 Thresh: DynUsed=({module_name in self.dyn_thresh_delta_l2}), "
                             f"Mean<{thr_dl2_mean:.2e}, CV<{thr_dl2_cv:.2f}, Slope<{thr_dl2_slope:.2e}")
                log_line3 = (f"  D-hat: Val={curr_d_max_val_str} ({d_max_ok_str} {thr_d_max_str}) | "
                             f"SNR: Val={curr_snr_val_str} ({snr_ok_str} {thr_snr_str})")
                suspect_count = self.suspect_counter.get(module_name, 0)
                is_permafrozen = module_name in self.perma_frozen
                freeze_status_str = "PERMA" if is_permafrozen else f"{suspect_count}/{self.k_confirm}"
                log_line4 = (f"  Freeze Status: {freeze_status_str}")
                full_log_message = f"{log_line1}\n{log_line2}\n{log_line3}\n{log_line4}"
                logFun(full_log_message, lvl="warning")

            except Exception as e:
                try:
                    logFun(
                        f"[ConvergeControl.snapshot_epoch_weights] Error processing/logging for module '{module_name}': {e}",
                        lvl="error")
                    traceback.print_exc()
                except Exception as log_e:
                    print(f"Logging error in snapshot_epoch_weights (main loop): {log_e}")
                    print(f"Original error for module '{module_name}': {e}")
                    traceback.print_exc()

    def _ready_to_freeze(self, name: str) -> bool:
        try:
            if self._current_step < self.warm_steps:
                if self.debug and self._current_step % (self.warm_steps // 4 if self.warm_steps > 0 else 10) == 0: # Ajuste na frequência do log
                    logFun(
                        f"[_ready_to_freeze] Mod '{name}' in Dhat/SNR warmup ({self._current_step}/{self.warm_steps} steps).",
                        lvl="debug")
                return False

            # --- Recalcular Baselines e Thresholds ---
            
            # D-hat: Recalcula baseline (mediana, iqr) e threshold (com gamma dinâmico)
            # A condição len >= self.warm_steps é implicitamente verdadeira se o deque está cheio (maxlen=warm_steps)
            # mas precisamos de pelo menos 2 pontos para as estatísticas.
            if name in self.d_max_hist and len(self.d_max_hist[name]) >= 2:
                med_d_max, iqr_d_max = self._robust_stats(self.d_max_hist[name])
                self.base_d_max[name] = med_d_max # Atualiza a mediana base da janela atual

                slope_d_max = 0.0
                d_max_history_list = list(self.d_max_hist[name])
                
                # A janela para o slope do D-hat será todo o histórico atual (máx. self.warm_steps)
                # desde que tenha pelo menos 2 pontos.
                window_for_dhat_slope_len = 0
                if len(d_max_history_list) >= 2:
                    # Para o slope, usamos uma janela de até self.warm_steps (que é o maxlen de d_max_hist)
                    # Não há um self.win_size_dhat_slope separado, usamos o tamanho do histórico de D-hat.
                    window_for_dhat_slope_len = len(d_max_history_list)
                    _, _, slope_d_max = self.stats_window(d_max_history_list) # Usa todo o histórico atual
                
                # Lógica para gamma_d_max dinâmico
                # Se d_max está caindo (slope_d_max < 0), queremos aumentar gamma (ser mais agressivo).
                # O sinal do 'adjustment' deve ser o mesmo do efeito em gamma (aumento/diminuição).
                # Então, se slope < 0, adjustment > 0.
                adjustment_value = 0.0
                if self.dhat_slope_norm_factor != 0: # Evitar divisão por zero
                    adjustment_value = - (slope_d_max / self.dhat_slope_norm_factor) * self.slope_sensitivity_dhat
                
                current_gamma_d_max = self.base_gamma_d_max_config * (1 + adjustment_value)
                current_gamma_d_max = max(self.min_overall_gamma_dhat, min(self.max_overall_gamma_dhat, current_gamma_d_max))

                self.thr_d_max[name] = max(0.0, med_d_max - current_gamma_d_max * iqr_d_max)
                # novo: define limiar de slope de D-hat como k×IQR
                self.thr_slope_d_max[name] = self.slope_sensitivity_dhat * iqr_d_max
                # guarda o slope atual
                self.last_slope_d_max[name] = slope_d_max
                
                if self.debug: # Log movido para dentro da condição de cálculo bem-sucedido
                    logFun(
                        f"[BASELINE d_max] '{name}': med={med_d_max:.3e}, IQR={iqr_d_max:.3e}, \n"
                        f"slope(win={window_for_dhat_slope_len})={slope_d_max:.3e}, \n"
                        f"dyn_gamma={current_gamma_d_max:.2f} (base={self.base_gamma_d_max_config:.2f}, adj={adjustment_value:.3f}) -> thr_d_max={self.thr_d_max[name]:.3e} \n",
                        lvl="debug")
            elif name in self.thr_d_max: # Histórico insuficiente, remove limiar antigo
                del self.thr_d_max[name]
                if name in self.base_d_max: del self.base_d_max[name]

            # SNR: Recalcula baseline e threshold (fecha a “avenida”)
            if name in self.g_hist_for_snr_baseline and len(self.g_hist_for_snr_baseline[name]) >= 2:
                med_snr, iqr_snr = self._robust_stats(self.g_hist_for_snr_baseline[name])
                self.base_snr[name] = med_snr
                # Agora só considera SN R bem abaixo da mediana
                self.thr_snr[name] = max(0.0, med_snr - self.delta_snr * iqr_snr)
                if self.debug:
                    logFun(
                        f"[BASELINE SNR] '{name}': med={med_snr:.3e}, IQR={iqr_snr:.3e} -> "
                        f"thr_snr={self.thr_snr[name]:.3e} (med - {self.delta_snr}×IQR)",
                        lvl="debug"
                    )
            elif name in self.thr_snr:
                # histórico insuficiente, remove limiar antigo
                del self.thr_snr[name]
                if name in self.base_snr:
                    del self.base_snr[name]

            # Verificação se os limiares foram calculados (após a tentativa de recálculo)
            if not (name in self.thr_d_max and name in self.thr_snr):
                if self.debug and self._current_step % 10 == 0: # Ajuste na frequência do log
                    log_msg_wait = (f"[_ready_to_freeze] Mod '{name}' waiting for D/S thresholds. \n"
                                    f"DThrExists: {name in self.thr_d_max}, SThrExists: {name in self.thr_snr}. \n"
                                    f"DHist: {len(self.d_max_hist.get(name,[]))}/{self.warm_steps}, \n"
                                    f"SHist: {len(self.g_hist_for_snr_baseline.get(name,[]))}/{self.warm_steps}")
                    logFun(log_msg_wait, lvl="debug")
                return False

            # --- Avaliação de Estabilidade ---
            stable_metrics_count = 0
            d_max_ok, slope_ok, snr_ok = False, False, False

            curr_d_max_val = float('inf') 
            if name in self.d_max_hist and self.d_max_hist[name]: # Verifica se o histórico tem algo
                curr_d_max_val = self.d_max_hist[name][-1]
                if name in self.thr_d_max: # Verifica se o limiar foi calculado
                    if curr_d_max_val < self.thr_d_max[name]:
                        d_max_ok = True
                        stable_metrics_count += 1

            # verifica slope de D-hat
            curr_slope_val = self.last_slope_d_max.get(name, float('nan'))
            if name in self.thr_slope_d_max and abs(curr_slope_val) < self.thr_slope_d_max[name]:
                slope_ok = True
                stable_metrics_count += 1                        
            
            curr_snr_val = float('inf')
            current_snr_calculated = self._calculate_snr(self.g_vector_hist_for_snr_calc[name])
            if current_snr_calculated is not None:
                curr_snr_val = current_snr_calculated
                if curr_snr_val < self.thr_snr[name]:
                    snr_ok = True
                    stable_metrics_count += 1

            # Lógica de decisão de estabilidade geral
            # agora contamos 3 métricas possíveis
            num_metrics_evaluable = 0
            if name in self.thr_d_max     and not math.isnan(curr_d_max_val):   num_metrics_evaluable += 1
            if name in self.thr_slope_d_max and not math.isnan(curr_slope_val): num_metrics_evaluable += 1
            if name in self.thr_snr       and not math.isnan(curr_snr_val):     num_metrics_evaluable += 1
            
            is_overall_stable = False
            if num_metrics_evaluable >= self.min_stable_metrics_freeze:
                # Contar quantas das métricas *avaliáveis* estão OK
                # exigimos as 3 métricas OK
                actual_ok_count = 0
                if name in self.thr_d_max       and d_max_ok:   actual_ok_count += 1
                if name in self.thr_slope_d_max and slope_ok:   actual_ok_count += 1
                if name in self.thr_snr         and snr_ok:     actual_ok_count += 1
                
                if actual_ok_count >= self.min_stable_metrics_freeze:
                    is_overall_stable = True
            
            if self.debug and self._current_step % 10 == 0: # Ajuste na frequência do log
                log_msg = (
                    f"[_ready_to_freeze] '{name}': \n"
                    f"Dhat={curr_d_max_val:.2e}({d_max_ok}, thr={self.thr_d_max.get(name, float('nan')):.2e}), \n"
                    f"SNR={curr_snr_val:.2e}({snr_ok}, thr={self.thr_snr.get(name, float('nan')):.2e}) | \n"
                    f"Evaluable={num_metrics_evaluable}, ActualOKs={actual_ok_count if 'actual_ok_count' in locals() else 'N/A'}>={self.min_stable_metrics_freeze} -> Stable={is_overall_stable} \n")
                logFun(log_msg, lvl="debug")

            return is_overall_stable
        except Exception as e:
            if hasattr(self, 'debug') and self.debug: # Verifica se debug existe
                try:
                    logFun(f"[ConvergeControl._ready_to_freeze] Error for module '{name}': {e}", lvl="error")
                    traceback.print_exc()
                except Exception as log_e_inner: # Fallback extremo
                    print(f"CRITICAL LOGGING ERROR in _ready_to_freeze for '{name}': {log_e_inner}")
                    print(f"Original error in _ready_to_freeze for '{name}': {e}")
                    traceback.print_exc()
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
                    res = linregress(x, y)
                    slope = float(res.slope)
                except ValueError:
                    slope = 0.0
                except Exception as e_linreg:
                    try:
                        logFun(f"[ConvergeControl.stats_window] Error in linregress: {e_linreg}", lvl="error")
                        traceback.print_exc()
                    except Exception as log_e:
                        print(f"Logging error in stats_window (linregress): {log_e}")
                        print(f"Original linregress error: {e_linreg}")
                        traceback.print_exc()
                    slope = 0.0
            return mean, std, slope
        except (ValueError, TypeError) as e_conversion:
            try:
                logFun(
                    f"[ConvergeControl.stats_window] Error converting input or basic stats: {e_conversion}. Input (first 5): {str(arr[:5])}", lvl="error")
                traceback.print_exc()
            except Exception as log_e:
                print(f"Logging error in stats_window (conversion): {log_e}")
                print(f"Original conversion error: {e_conversion}")
                traceback.print_exc()
            return 0.0, 0.0, 0.0
        except Exception as e_general:
            try:
                logFun(f"[ConvergeControl.stats_window] Generic error: {e_general}. Input (first 5): {str(arr[:5])}",
                       lvl="error")
                traceback.print_exc()
            except Exception as log_e:
                print(f"Logging error in stats_window (general): {log_e}")
                print(f"Original general error: {e_general}")
                traceback.print_exc()
            return 0.0, 0.0, 0.0

    def set_current_time(self, epoch: int, step: int):
        self._current_epoch = epoch
        self._current_step = step

    def ingest_and_process(self, stats_list: List[Dict]):
        if not stats_list:
            return

        processed_deltas = 0
        processed_d_maxs = 0
        for stat in stats_list:
            name = stat.get("name")
            delta_val = stat.get("delta_L2")
            d_max_val = stat.get("d_max")

            if name is None:
                continue

            try:
                if delta_val is not None:
                    f_delta_val = float(delta_val)
                    self.delta_buffers[name].append(f_delta_val)
                    self._rolling_stats[name].add(f_delta_val) # ATUALIZA ROLLING STATS AQUI
                    processed_deltas += 1

                if d_max_val is not None:
                    self.d_max_hist[name].append(float(d_max_val))
                    processed_d_maxs += 1
            except (ValueError, TypeError) as e:
                if self.debug:
                    try:
                        logFun(
                            f"[ConvergeControl] Error processing stats for '{name}': {e}. ΔL2: {delta_val}, Dhat: {d_max_val}",
                            lvl="warning")
                    except Exception as log_e:
                        print(f"Logging error in ingest_and_process for '{name}': {log_e}")
                        print(f"Original error for '{name}': {e}")

        if self.debug and (processed_deltas == 0 or processed_d_maxs == 0) and self._current_step % 10 == 0:
            try:
                logFun(
                    f"[ConvergeControl] Step {self._current_step}: No ΔL2 ({processed_deltas}) or Dhat ({processed_d_maxs}) ingested.",
                    lvl="debug")
            except Exception as log_e:
                print(f"Logging error in ingest_and_process (summary log): {log_e}")

    def decide(self) -> Dict[str, bool]:
        try:
            if not self.enable_freeze_action:
                return {}

            progress = self._current_epoch / max(self.total_epochs, 1) if self.total_epochs > 0 else 0
            if progress < self.min_buffer_epochs:
                if self.debug and self._current_step % (self.total_epochs // 10 if self.total_epochs > 10 else 10) == 0:
                    logFun(
                        f"[ConvergeControl] Decide@{self._current_step}: Epoch Warmup ({progress:.2f} < {self.min_buffer_epochs:.2f}).",
                        lvl="debug")
                return {}

            decisions: Dict[str, bool] = {}
            module_names_to_decide = list(set(self.g_vector_hist_for_snr_calc.keys()) | set(self.d_max_hist.keys()))

            if self.debug and not module_names_to_decide and self._current_step % 10 == 0:
                logFun(f"[ConvergeControl] Decide@{self._current_step}: No modules with D-hat/SNR history for eval.", lvl="debug")

            for name in module_names_to_decide:
                try:
                    if name in self.perma_frozen:
                        decisions[name] = True
                        continue

                    is_module_stable = self._ready_to_freeze(name)

                    current_epoch = self._current_epoch
                    last_incr_epoch = self._last_epoch_incr.get(name, -1)

                    if is_module_stable:
                        if current_epoch > last_incr_epoch:
                            self.suspect_counter[name] += 1
                            self._last_epoch_incr[name] = current_epoch
                    else:
                        if current_epoch > last_incr_epoch:
                            self.suspect_counter[name] = 0
                            self._last_epoch_incr[name] = current_epoch

                    should_freeze_eval = self.suspect_counter.get(name, 0) >= self.k_confirm
                    decisions[name] = should_freeze_eval

                    if should_freeze_eval:
                        if name not in self.perma_frozen:
                            self.perma_frozen.add(name)
                            self.freeze_step_run2[name] = self._current_step
                            if self.verbose or self.debug:
                                logFun(
                                    f"[ConvergeControl] FREEZE ACTION '{name}' @ step {self._current_step} "
                                    f"(Trigger: D-hat/SNR stable for {self.suspect_counter.get(name, 0)} checks >= {self.k_confirm})",
                                    lvl="verbose" if self.verbose else "debug")
                except Exception as e_module:
                    logFun(f"[ConvergeControl.decide] Error processing decision for module '{name}': {e_module}", lvl="error")
                    traceback.print_exc()
                    decisions[name] = False  # Default to not freezing on error for this specific module
            return decisions
        except Exception as e_outer:  # Catch errors in the setup of decide method itself
            try:
                logFun(f"[ConvergeControl.decide] Outer error in decide method: {e_outer}", lvl="error")
                traceback.print_exc()
            except Exception as log_e:
                print(f"Logging error in decide (outer): {log_e}")
                print(f"Original decide outer error: {e_outer}")
                traceback.print_exc()
            return {}  # Return empty decisions on major failure

    def get_frozen_set(self) -> Set[str]:
        return self.perma_frozen

    def get_final_conv_steps(self) -> Dict[str, int]:
        if self.run_number == 1 and not self.enable_freeze_action:
            return {}
        return dict(self.freeze_step_run2)

    def _delta_stable(self, name: str) -> bool:
        try:
            win = self.win_size_delta_l2
            buf = self.delta_buffers[name]
            if len(buf) < win:
                return False

            window = list(buf)[-win:]
            mean, std, slope = self.stats_window(window)
            cv = std / (mean + 1e-12)

            dyn_m_thresh = self.dyn_thresh_delta_l2.get(name, {}).get('mean', self.abs_thresh_fallback)
            dyn_cv_thresh = self.dyn_thresh_delta_l2.get(name, {}).get('cv', self.cv_thresh_fallback)
            dyn_s_thresh = self.dyn_thresh_delta_l2.get(name, {}).get('slope', self.slope_thresh_fallback)

            mean_ok = mean < dyn_m_thresh
            cv_ok = cv < dyn_cv_thresh
            slope_ok = abs(slope) < dyn_s_thresh
            stable = (mean_ok and cv_ok) or (cv_ok and slope_ok)
            return stable
        except Exception as e:
            try:
                if hasattr(self, 'debug') and self.debug:
                    logFun(f"[ConvergeControl._delta_stable] Error for module '{name}': {e}", lvl="error")
                    traceback.print_exc()
            except Exception as log_e:
                print(f"Logging error in _delta_stable for '{name}': {log_e}")
                print(f"Original error for '{name}': {e}")
                traceback.print_exc()
            return False

    def stability_score(self, name: str) -> float:
        try:
            buf = self.delta_buffers.get(name, None)
            if buf is None or len(buf) < self.win_size_delta_l2:
                return 0.0

            window = list(buf)[-self.win_size_delta_l2:]
            mean, std, slope = self.stats_window(window)
            cv = std / (mean + 1e-12)

            thr_mean = self.dyn_thresh_delta_l2.get(name, {}).get('mean', self.abs_thresh_fallback)
            thr_cv = self.dyn_thresh_delta_l2.get(name, {}).get('cv', self.cv_thresh_fallback)
            thr_slope = self.dyn_thresh_delta_l2.get(name, {}).get('slope', self.slope_thresh_fallback)

            mean_score = max(0.0, min(1.0, (thr_mean - mean) / (thr_mean + 1e-12)))
            cv_score = max(0.0, min(1.0, (thr_cv - cv) / (thr_cv + 1e-12)))
            slope_score = max(0.0, min(1.0, (thr_slope - abs(slope)) / (thr_slope + 1e-12)))

            return (mean_score + cv_score + slope_score) / 3.0
        except Exception as e:
            try:
                logFun(f"[ConvergeControl.stability_score] Error for module '{name}': {e}", lvl="error")
                traceback.print_exc()
            except Exception as log_e:
                print(f"Logging error in stability_score for '{name}': {log_e}")
                print(f"Original error for '{name}': {e}")
                traceback.print_exc()
            return 0.0

    def _end_epoch_cleanup(self):
        try:
            for buf in self.delta_buffers.values():
                buf.clear()
            for hist in self.d_max_hist.values():
                hist.clear()
            for hist in self.g_hist_for_snr_baseline.values():
                hist.clear()
            for hist in self.g_vector_hist_for_snr_calc.values():
                hist.clear()
            self.metric_hist_delta_l2.clear()
            self.dyn_thresh_delta_l2.clear()
            self._last_epoch_incr.clear()
            self.suspect_counter.clear()
        except Exception as e:
            try:
                logFun(f"[ConvergeControl._end_epoch_cleanup] Error during cleanup: {e}", lvl="error")
                traceback.print_exc()
            except Exception as log_e:
                print(f"Logging error in _end_epoch_cleanup: {log_e}")
                print(f"Original cleanup error: {e}")
                traceback.print_exc()

from collections import deque

class RollingStats:
    """
    Estatísticas (média, variância) para uma janela móvel de valores.
    Complexidade de inserção O(1).
    """
    def __init__(self, window_size: int):
        self.window = deque(maxlen=window_size)
        self.sum_x = 0.0
        self.sum_x2 = 0.0

    def add(self, x: float):
        # Se a janela já estiver cheia, remova o mais antigo
        if len(self.window) == self.window.maxlen:
            old = self.window[0]
            self.sum_x  -= old
            self.sum_x2 -= old * old

        # Insere o novo
        self.window.append(x)
        self.sum_x  += x
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
        # var = (Σx² – Σx²/n) / (n-1)
        return (self.sum_x2 - (self.sum_x ** 2) / n) / (n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)
