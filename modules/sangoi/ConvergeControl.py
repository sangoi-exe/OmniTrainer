import math
import torch
import traceback
import numpy as np
from rich.columns import Columns
from rich.console import Group, Text
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich import box
from typing import Deque, Dict, Optional, Set, Union
from dataclasses import dataclass
from collections import defaultdict, deque
from rich.text import Text


from modules.sangoi.DataRecorder import DataRecorder
from modules.sangoi.logFun import logFun

# ------------------- Config -------------------
@dataclass
class ConvCfg:
    # k_confirm: int = 4 # Este será substituído/reinterpretado
    # GD
    gd_std_k: float = 1.5
    gd_hist_len: int = 32
    gd_ewma_alpha: float = 0.1
    gd_std_ewma_alpha: float = 0.1
    gd_stable_confirm_steps: int = 10 # Novo: k individual para GD
    gd_stability_std_thresh_factor: float = 0.1 # Novo: Fator da EWMA_GD para o limiar de std

    # SNR
    snr_k: float = 1.0
    snr_calc_win: int = 16
    snr_hist_len: int = 32
    snr_stable_confirm_steps: int = 10 # Novo: k individual para SNR
    snr_stability_window_len: int = 10 # Novo: Janela para calcular std do SNR
    snr_stability_std_thresh: float = 0.01 # Novo: Limiar absoluto para std do SNR

    # GNS Temporal
    gns_temporal_k_window: int = 10

    gns_temporal_thresh: float = 12.0 # Corrigido para float
    gns_temporal_hist_len: int = 32
    gns_max_value_clamp: float = 1000.0
    gns_temporal_k_iqr_factor: float = 1.0
    gns_mu_norm_sq_conv_thresh: float = 1e-8
    gns_t_stable_confirm_steps: int = 10 # Novo: k individual para GNS-T
    gns_t_stability_window_len: int = 10 # Novo: Janela para calcular std do GNS-T
    gns_t_stability_std_thresh: float = 0.5 # Novo: Limiar absoluto para std do GNS-T (GNS-T tem escala maior)


    # Warm-up
    min_gd_hist_eval: int = 16
    min_snr_hist_eval: int = 16
    min_hist_gns_temporal_eval: int = 12

def module_type(name: str) -> str:
    if "attention" in name or "attn" in name: return "attn"
    elif "conv" in name or "resnet" in name or "res_block" in name: return "resnet"
    return "resnet"


TYPE_CFG_DEFAULT_FALLBACK = ConvCfg()

TYPE_CFG = ConvCfg(
    gd_stable_confirm_steps=20, gd_stability_std_thresh_factor=0.15,
    snr_stable_confirm_steps=20, snr_stability_window_len=15, snr_stability_std_thresh=0.02,
    gns_t_stable_confirm_steps=20, gns_t_stability_window_len=15, gns_t_stability_std_thresh=0.75
)

TYPE_CFG_DEFAULT_FALLBACK = ConvCfg()

TYPE_CFG = {
    "attn":
        ConvCfg(
            # Original params from your snippet
            snr_k=1.5,
            gd_std_k=2.0,
            gd_hist_len=40,
            gd_ewma_alpha=0.05,
            gd_std_ewma_alpha=0.05,
            gns_temporal_thresh=10.0, # Corrected from 10.0,
            gns_temporal_k_window=12,
            gns_temporal_k_iqr_factor=1.00,
            min_gd_hist_eval=20,
            min_snr_hist_eval=15,
            min_hist_gns_temporal_eval=10,

            # Default ConvCfg params not in original "attn" snippet, but good to specify
            snr_calc_win=16, # Default ConvCfg value
            snr_hist_len=32, # Default ConvCfg value
            gns_temporal_hist_len=32, # Default ConvCfg value
            gns_max_value_clamp=1000.0, # Default ConvCfg value
            gns_mu_norm_sq_conv_thresh=1e-8, # Default ConvCfg value

            # New stability parameters for "attn"
            gd_stable_confirm_steps=20,
            gd_stability_std_thresh_factor=0.1,
            snr_stable_confirm_steps=20,
            snr_stability_window_len=15, # Window for SNR stability std calc
            snr_stability_std_thresh=0.015, # Std threshold for SNR stability
            gns_t_stable_confirm_steps=20,
            gns_t_stability_window_len=15, # Window for GNS-T stability std calc
            gns_t_stability_std_thresh=0.6 # Std threshold for GNS-T stability
        ),
    "resnet":
        ConvCfg(
            # Original params from your snippet
            snr_k=1.0,
            gd_hist_len=30,
            gd_std_k=1.5,
            gd_ewma_alpha=0.1,
            gd_std_ewma_alpha=0.1,
            gns_temporal_thresh=5.0, # Corrected from 5.0,
            gns_temporal_k_window=8,
            gns_temporal_k_iqr_factor=1.00,
            min_gd_hist_eval=15,
            min_snr_hist_eval=12,
            min_hist_gns_temporal_eval=8,

            # Default ConvCfg params not in original "resnet" snippet
            snr_calc_win=16, # Default ConvCfg value
            snr_hist_len=32, # Default ConvCfg value
            gns_temporal_hist_len=32, # Default ConvCfg value
            gns_max_value_clamp=1000.0, # Default ConvCfg value
            gns_mu_norm_sq_conv_thresh=1e-8, # Default ConvCfg value

            # New stability parameters for "resnet"
            gd_stable_confirm_steps=15,
            gd_stability_std_thresh_factor=0.15,
            snr_stable_confirm_steps=15,
            snr_stability_window_len=12, # Window for SNR stability std calc
            snr_stability_std_thresh=0.02,  # Std threshold for SNR stability
            gns_t_stable_confirm_steps=15,
            gns_t_stability_window_len=10, # Window for GNS-T stability std calc
            gns_t_stability_std_thresh=0.8  # Std threshold for GNS-T stability
        ),
}

# ------------------- ConvergeControl -------------------
class ConvergeControl:
    def __init__(
        self,
        run_number: int,
        *,
        verbose: bool = True,
        debug: bool = True,
        data_recorder: Optional[DataRecorder] = None,
    ):
        self.data_recorder = data_recorder
        self.run_number = run_number
        self.cfg_by_type = TYPE_CFG
        self.verbose = verbose
        self.debug = debug

        self.grad_prev_step: Dict[str, torch.Tensor] = {}
        self.gradient_disparity_hist: Dict[str, deque] = {}
        self.gd_ewma: Dict[str, float] = {}
        self.gd_std_ewma: Dict[str, float] = {} # Este é EWMA da variância do GD

        self.temporal_grad_history: Dict[str, deque] = {}
        self.gns_temporal_hist: Dict[str, deque] = {}
        self.latest_mu_temporal_norm_sq: Dict[str, float] = {}

        self._snr_window_grads: Dict[str, deque] = {} # Para cálculo do SNR
        self.snr_hist: Dict[str, deque] = {} # Histórico dos valores de SNR calculados

        # Novos históricos para cálculo de estabilidade de SNR e GNS-T
        self.snr_value_hist_for_stability: Dict[str, deque] = {}
        self.gns_t_value_hist_for_stability: Dict[str, deque] = {}

        self.perma_frozen: Set[str] = set()
        # self.suspect_counter: Dict[str, int] = defaultdict(int) # Não mais usado da mesma forma

        # Novos contadores individuais para passos "bons e estáveis"
        self.stable_green_gd_steps: Dict[str, int] = defaultdict(int)
        self.stable_green_snr_steps: Dict[str, int] = defaultdict(int)
        self.stable_green_gns_t_steps: Dict[str, int] = defaultdict(int)

        self._current_epoch = 0
        self._current_step = 0
        self.enable_freeze_action = run_number >= 2

        self._deques_initialized_for_module: Set[str] = set()

        if self.debug:
            logFun(f"[ConvergeControl] Init run={run_number}. Base CFG (fallback): {TYPE_CFG_DEFAULT_FALLBACK}", lvl="CONVCTRL")

    def get_global_convergence_score(self) -> float:
        """
        Calcula um score global de convergência (0.0 a 1.0).
        1.0 significa que todos os módulos monitorados e com histórico suficiente
        atendem aos critérios de 'pronto para congelar'.
        0.0 significa que nenhum atende.
        """
        monitored_modules_with_sufficient_history = []
        met_criteria_count = 0

        # Considera módulos que têm dados em todos os históricos relevantes
        # e passaram do período de warm-up para todas as métricas usadas em _ready_to_freeze.
        active_modules = list(self.grad_prev_step.keys()) # Módulos que receberam gradientes

        for name in active_modules:
            if name in self.perma_frozen: # Módulos já congelados contam como "convergidos"
                met_criteria_count +=1
                monitored_modules_with_sufficient_history.append(name)
                continue

            cfg = self._get_cfg_for_module(name)
            gd_hist_vals = self.gradient_disparity_hist.get(name)
            snr_hist_vals = self.snr_hist.get(name)
            gns_temporal_hist_vals = self.gns_temporal_hist.get(name)

            # Checagem de warm-up similar à _ready_to_freeze
            min_gd_for_ewma_stable = max(cfg.min_gd_hist_eval, int(1 / cfg.gd_ewma_alpha * 0.5)) if cfg.gd_ewma_alpha > 0 else cfg.min_gd_hist_eval

            has_sufficient_history = (
                gd_hist_vals and len(gd_hist_vals) >= min_gd_for_ewma_stable and
                snr_hist_vals and len(snr_hist_vals) >= cfg.min_snr_hist_eval and
                gns_temporal_hist_vals and len(gns_temporal_hist_vals) >= cfg.min_hist_gns_temporal_eval
            )

            if has_sufficient_history:
                monitored_modules_with_sufficient_history.append(name)
                if self._ready_to_freeze(name): 
                    met_criteria_count += 1
        
        if not monitored_modules_with_sufficient_history:
            if self.debug and active_modules:
                if self._current_step % 50 == 0:
                    logFun("[ConvergeControl] GlobalScore: Nenhum módulo com histórico suficiente para avaliação.", lvl="DEBUG")
            return 0.0

        score = met_criteria_count / len(monitored_modules_with_sufficient_history)
        if self.debug:
            logFun(f"[ConvergeControl] GlobalScore: {score:.3f} ({met_criteria_count}/{len(monitored_modules_with_sufficient_history)} módulos atenderam critérios)", lvl="CONVCTRL_DEBUG")
        return score

    def _get_cfg_for_module(self, name: str) -> ConvCfg:
        m_type = module_type(name)
        return self.cfg_by_type.get(m_type, TYPE_CFG_DEFAULT_FALLBACK)

    def _initialize_deques_for_module(self, name: str, cfg: ConvCfg):
            """Inicializa os deques para um módulo na primeira vez que é visto."""
            if name not in self._deques_initialized_for_module:
                self.gradient_disparity_hist[name] = deque(maxlen=cfg.gd_hist_len)
                self.gd_ewma[name] = np.nan
                self.gd_std_ewma[name] = np.nan # EWMA da variância

                self._snr_window_grads[name] = deque(maxlen=cfg.snr_calc_win)
                self.snr_hist[name] = deque(maxlen=cfg.snr_hist_len)
                # Novo deque para valores de SNR para cálculo de std de estabilidade
                self.snr_value_hist_for_stability[name] = deque(maxlen=cfg.snr_stability_window_len)


                self.temporal_grad_history[name] = deque(maxlen=cfg.gns_temporal_k_window)
                self.gns_temporal_hist[name] = deque(maxlen=cfg.gns_temporal_hist_len)
                self.latest_mu_temporal_norm_sq[name] = np.nan
                # Novo deque para valores de GNS-T para cálculo de std de estabilidade
                self.gns_t_value_hist_for_stability[name] = deque(maxlen=cfg.gns_t_stability_window_len)


                self._deques_initialized_for_module.add(name)


    def update_step_metrics(self, name: str, module: torch.nn.Module):
        if name in self.perma_frozen:
            return

        cfg = self._get_cfg_for_module(name)
        self._initialize_deques_for_module(name, cfg) # Garante inicialização na primeira vez

        # 1.b) Usar no_grad ao redor de cálculos que não precisam de gradientes
        with torch.no_grad():
            try:
                grads = [
                    p.grad.detach().flatten() for p in module.parameters() if p.grad is not None and p.grad.numel() > 0
                ]
                if not grads: return
                current_grad_vec = torch.cat(grads)
                if current_grad_vec.numel() == 0: return

                # --- SNR ---
                snr_grad_window = self._snr_window_grads[name]
                snr_grad_window.append(current_grad_vec.clone())
                calculated_snr_this_step = np.nan # Para popular o _for_stability
                if len(snr_grad_window) >= cfg.snr_calc_win:
                    grad_stack = torch.stack(list(snr_grad_window))
                    mu_g = grad_stack.mean(dim=0)
                    signal_power = mu_g.norm().pow(2).item()
                    noise_power = grad_stack.var(dim=0, unbiased=True).sum().item()
                    epsilon_snr = torch.finfo(mu_g.dtype).tiny
                    snr = float("inf") if noise_power < epsilon_snr else signal_power / (noise_power + epsilon_snr)
                    self.snr_hist[name].append(snr)
                    calculated_snr_this_step = snr # Guardar SNR calculado
                    snr_grad_window.clear()
                
                # Popular histórico de SNR para cálculo de estabilidade, mesmo se não calculado a cada passo
                # Ou apenas se calculado:
                if not np.isnan(calculated_snr_this_step):
                     self.snr_value_hist_for_stability[name].append(calculated_snr_this_step)

                # --- GD ---
                prev_grad_vec = self.grad_prev_step.get(name)
                if prev_grad_vec is not None and prev_grad_vec.numel() == current_grad_vec.numel():
                    gd_val = (current_grad_vec - prev_grad_vec).norm().item()
                    self.gradient_disparity_hist[name].append(gd_val)

                    if np.isnan(self.gd_ewma[name]): 
                        self.gd_ewma[name] = gd_val
                        self.gd_std_ewma[name] = 0.0 
                    else:
                        self.gd_ewma[name] = (1 - cfg.gd_ewma_alpha) * self.gd_ewma[name] + cfg.gd_ewma_alpha * gd_val
                        current_variance = (gd_val - self.gd_ewma[name])**2 # variância em relação à EWMA atual
                        self.gd_std_ewma[name] = (1 - cfg.gd_std_ewma_alpha) * self.gd_std_ewma[name] + cfg.gd_std_ewma_alpha * current_variance
                self.grad_prev_step[name] = current_grad_vec.clone()

                # --- GNS-Temporal ---
                self.temporal_grad_history[name].append(current_grad_vec.clone())
                calculated_gns_t_this_step = np.nan
                if len(self.temporal_grad_history[name]) >= cfg.gns_temporal_k_window:
                    grads_over_time = list(self.temporal_grad_history[name])
                    try:
                        M = torch.stack(grads_over_time, dim=0)
                        mu_temporal = M.mean(dim=0)
                        var_components_temporal = M.var(dim=0, unbiased=True)

                        trace_of_temporal_covariance = var_components_temporal.sum().item()
                        squared_norm_of_mean_temporal_grad = mu_temporal.norm().pow(2).item()
                        self.latest_mu_temporal_norm_sq[name] = squared_norm_of_mean_temporal_grad

                        epsilon_gns = torch.finfo(mu_temporal.dtype).tiny
                        gns_val_raw = trace_of_temporal_covariance / (squared_norm_of_mean_temporal_grad + epsilon_gns)
                        
                        gns_val_clamped = np.clip(gns_val_raw, -cfg.gns_max_value_clamp, cfg.gns_max_value_clamp)
                        if math.isinf(gns_val_raw) and gns_val_raw > 0: gns_val_final = float('inf')
                        elif math.isinf(gns_val_raw) and gns_val_raw < 0: gns_val_final = -float('inf')
                        else: gns_val_final = gns_val_clamped

                        self.gns_temporal_hist[name].append(gns_val_final)
                        calculated_gns_t_this_step = gns_val_final # Guardar GNS-T calculado
                    except RuntimeError as e:
                        logFun(f"[CC GNS-Temporal] Erro ao calcular GNS para '{name}': {e}", lvl="error")
                
                if not np.isnan(calculated_gns_t_this_step):
                    self.gns_t_value_hist_for_stability[name].append(calculated_gns_t_this_step)

            except Exception as e:
                logFun(f"[ConvergeControl] Error in update_step_metrics for '{name}': {e}", lvl="error")
                traceback.print_exc()

    def _ready_to_freeze(self, name: str) -> bool:
            cfg = self._get_cfg_for_module(name)

            gd_hist_vals = self.gradient_disparity_hist.get(name)
            snr_hist_vals = self.snr_hist.get(name) # Histórico de SNRs calculados
            gns_temporal_hist_vals = self.gns_temporal_hist.get(name)

            # Históricos de valores para cálculo de std de estabilidade
            snr_stability_hist = self.snr_value_hist_for_stability.get(name)
            gns_t_stability_hist = self.gns_t_value_hist_for_stability.get(name)

            # --- Warm-up checks ---
            # Para GD com EWMA
            min_gd_for_ewma_stable = max(cfg.min_gd_hist_eval, int(1 / cfg.gd_ewma_alpha * 0.5) if cfg.gd_ewma_alpha > 0 else cfg.min_gd_hist_eval)
            
            # Warm-up para os históricos de valores usados para calcular std de estabilidade
            snr_stability_hist_ready = snr_stability_hist and len(snr_stability_hist) >= cfg.snr_stability_window_len
            gns_t_stability_hist_ready = gns_t_stability_hist and len(gns_t_stability_hist) >= cfg.gns_t_stability_window_len
            
            # Warm-up para os históricos principais das métricas
            gd_main_hist_ready = gd_hist_vals and len(gd_hist_vals) >= min_gd_for_ewma_stable
            snr_main_hist_ready = snr_hist_vals and len(snr_hist_vals) >= cfg.min_snr_hist_eval
            gns_t_main_hist_ready = gns_temporal_hist_vals and len(gns_temporal_hist_vals) >= cfg.min_hist_gns_temporal_eval

            if not (gd_main_hist_ready and snr_main_hist_ready and gns_t_main_hist_ready and \
                    snr_stability_hist_ready and gns_t_stability_hist_ready):
                return False # Pelo menos um dos históricos não está pronto

            # --- Condições Primárias e de Estabilidade ---

            # 1. Gradient Disparity (GD)
            gd_cond_primary_met = False
            gd_is_stable = False
            current_gd = gd_hist_vals[-1]
            current_ewma_gd = self.gd_ewma.get(name, np.nan)
            current_ewma_var_gd = self.gd_std_ewma.get(name, np.nan) # EWMA da variância

            if not np.isnan(current_ewma_gd) and not np.isnan(current_ewma_var_gd):
                current_std_gd_from_ewma_var = math.sqrt(max(0, current_ewma_var_gd))
                # Condição Primária GD (<=, conforme discutido)
                gd_target_threshold = current_ewma_gd - cfg.gd_std_k * current_std_gd_from_ewma_var
                gd_cond_primary_met = current_gd <= gd_target_threshold
                # Condição de Estabilidade GD
                # O desvio padrão do GD (sqrt da EWMA da variância) deve ser baixo em relação à média do GD
                gd_is_stable = current_std_gd_from_ewma_var < (abs(current_ewma_gd) * cfg.gd_stability_std_thresh_factor + 1e-9) # Adiciona epsilon para evitar div por zero se ewma_gd for 0


            # 2. SNR
            snr_cond_primary_met = False
            snr_is_stable = False
            current_snr = snr_hist_vals[-1] # Último SNR calculado do histórico principal
            
            median_snr = float(np.median(list(snr_hist_vals)))
            q75_snr, q25_snr = np.percentile(list(snr_hist_vals), [75, 25])
            iqr_snr = q75_snr - q25_snr
            # Condição Primária SNR (<=)
            snr_target_cond_val = median_snr - cfg.snr_k * iqr_snr if iqr_snr > 1e-9 else median_snr
            snr_cond_primary_met = current_snr <= snr_target_cond_val
            
            # Condição de Estabilidade SNR (usando snr_stability_hist)
            # Verifica se snr_stability_hist está cheio (já feito no warm-up, mas redundância segura)
            if len(snr_stability_hist) >= cfg.snr_stability_window_len:
                std_snr_stability = np.std(list(snr_stability_hist))
                snr_is_stable = std_snr_stability < cfg.snr_stability_std_thresh
                
            # 3. GNS Temporal
            gns_t_cond_primary_met = False
            gns_t_is_stable = False
            current_gns_temporal = gns_temporal_hist_vals[-1]
            mu_norm_sq = self.latest_mu_temporal_norm_sq.get(name, float('inf'))
            
            gns_inf_is_valid_convergence = (math.isinf(current_gns_temporal) and \
                                                current_gns_temporal > 0 and \
                                                mu_norm_sq < cfg.gns_mu_norm_sq_conv_thresh)
            if gns_inf_is_valid_convergence:
                gns_t_cond_primary_met = True
                gns_t_is_stable = True # GNS Infinito válido é considerado estável para este propósito
            elif not math.isinf(current_gns_temporal):
                median_gns_t = float(np.median(list(gns_temporal_hist_vals)))
                q75_gns_t, q25_gns_t = np.percentile(list(gns_temporal_hist_vals), [75, 25])
                iqr_gns_t = q75_gns_t - q25_gns_t
                # Condição Primária GNS-T (>=)
                dynamic_gns_thresh = median_gns_t + cfg.gns_temporal_k_iqr_factor * iqr_gns_t
                if iqr_gns_t <= 1e-9 : dynamic_gns_thresh = median_gns_t 
                gns_t_cond_primary_met = current_gns_temporal >= dynamic_gns_thresh
                
                # Condição de Estabilidade GNS-T (usando gns_t_stability_hist)
                if len(gns_t_stability_hist) >= cfg.gns_t_stability_window_len:
                    std_gns_t_stability = np.std(list(gns_t_stability_hist))
                    gns_t_is_stable = std_gns_t_stability < cfg.gns_t_stability_std_thresh

            # --- Atualizar Contadores Individuais ---
            # Os contadores SÓ incrementam se AMBAS as condições (primária e estabilidade) forem verdadeiras.
            # Eles NÃO resetam se a condição falhar, acumulando o total de passos "bons e estáveis".
            if gd_cond_primary_met and gd_is_stable:
                self.stable_green_gd_steps[name] += 1
            
            if snr_cond_primary_met and snr_is_stable:
                self.stable_green_snr_steps[name] += 1
                
            if gns_t_cond_primary_met and gns_t_is_stable:
                self.stable_green_gns_t_steps[name] += 1

            # --- Decisão Final de Congelamento ---
            # Verifica se todos os contadores individuais atingiram seus respectivos limiares
            gd_ready = self.stable_green_gd_steps[name] >= cfg.gd_stable_confirm_steps
            snr_ready = self.stable_green_snr_steps[name] >= cfg.snr_stable_confirm_steps
            gns_t_ready = self.stable_green_gns_t_steps[name] >= cfg.gns_t_stable_confirm_steps

            return gd_ready and snr_ready and gns_t_ready

    # Métodos decide, get_frozen_set, get_final_conv_steps, set_current_time permanecem os mesmos

    def decide(self) -> Dict[str, bool]:
            decisions = {}
            if not self.enable_freeze_action: return decisions
            
            active_modules = list(self._deques_initialized_for_module) # Módulos que tiveram deques inicializados
                                                                      # Ou self.grad_prev_step.keys() se preferir módulos com gradientes recentes

            for name in active_modules:
                if name in self.perma_frozen:
                    decisions[name] = True
                    continue

                # _ready_to_freeze agora contém toda a lógica de contagem e limiares individuais
                is_module_ready_to_freeze = self._ready_to_freeze(name)

                if is_module_ready_to_freeze:
                    self.perma_frozen.add(name)
                    if not hasattr(self, "freeze_info"): self.freeze_info = {}
                    self.freeze_info[name] = {"step": self._current_step, "epoch": self._current_epoch}
                    decisions[name] = True
                    if self.verbose or self.debug:
                        logFun(f"[ConvergeControl] FREEZE '{name}' @epoch {self._current_epoch}, step {self._current_step} "
                              f"(GD steps: {self.stable_green_gd_steps[name]}, "
                              f"SNR steps: {self.stable_green_snr_steps[name]}, "
                              f"GNS-T steps: {self.stable_green_gns_t_steps[name]})", lvl="CONVCTRL")
                else:
                    decisions[name] = False
            return decisions

    def get_frozen_set(self) -> Set[str]: return self.perma_frozen

    def get_final_conv_steps(self) -> Dict[str, Dict]:
        if not self.enable_freeze_action: return {}
        return getattr(self, "freeze_info", {})
    
    def set_current_time(self, epoch: int, step: int):
        self._current_epoch, self._current_step = epoch, step

    def get_latest_scores(self) -> Dict[str, float]:
        return {name: self.snr_hist[name][-1] for name in self.snr_hist if self.snr_hist[name]}

    def _format_metric_cell(self, value, current_steps, target_steps, primary_cond_text, 
                            stability_cond_text, is_primary_met, is_stable, 
                            value_format="{:.2e}", nan_text="N/A") -> Text:
        """Helper para formatar o conteúdo de uma célula de métrica em uma única linha."""
        
        val_disp = value_format.format(value) if not (isinstance(value, str) or np.isnan(value)) else nan_text
        
        color = "green" if is_primary_met and is_stable else "red"
        # Se for N/A e as condições não forem atendidas (is_primary_met e is_stable são False),
        # não pintar explicitamente de vermelho; deixar a cor padrão ou uma cor neutra.
        if val_disp == nan_text and not (is_primary_met and is_stable):
            color = "default" # Ou "dim white" ou outra cor neutra

        count_disp = f"({current_steps}/{target_steps})"

        text_elements = []
        text_elements.append((f"{val_disp} {count_disp} ", color if val_disp != nan_text else "default")) # Cor no valor e contador
        
        # Adicionar um separador visual sutil se não for N/A para as condições
        # Ou simplesmente um espaço
        # text_elements.append(("| ", "dim")) if primary_cond_text != f"P:hist<{target_steps}" else (" ", "dim") # Exemplo de separador
        
        text_elements.append((f"{primary_cond_text} ", "dim"))
        text_elements.append((stability_cond_text, "dim"))
        
        return Text.assemble(*text_elements)

    def generate_status_renderable(self) -> Union[Columns, Group, Text]:
        monitored_modules = sorted(list(self._deques_initialized_for_module))

        if not monitored_modules:
            return Text("Sem módulos ativos para convergência.", justify="center")

        # Dividir módulos para duas tabelas
        split_point = (len(monitored_modules) + 1) // 2
        modules_table1 = monitored_modules[:split_point]
        modules_table2 = monitored_modules[split_point:]

        tables = []

        for i, module_list in enumerate([modules_table1, modules_table2]):
            if not module_list:
                continue

            table = Table(box=box.ROUNDED, show_edge=True, expand=True,
                          header_style="bold magenta", show_header=True, title=f"Convergence Status {i+1}")
            
            # Ajustar min_width para acomodar uma única linha mais longa nas células de métrica
            table.add_column("Module", style="cyan", min_width=18, no_wrap=True) # Pode manter ou reduzir um pouco
            table.add_column("GD", justify="left", min_width=30) # Aumentado (era ~17)
            table.add_column("SNR", justify="left", min_width=28) # Aumentado (era ~17)
            table.add_column("GNS-T", justify="left", min_width=32) # Aumentado (era ~20)
            table.add_column("SNR*GNS", justify="right", min_width=7) # Mantém
            table.add_column("Status", justify="left", min_width=15) # Mantém

            for name in module_list:
                cfg = self._get_cfg_for_module(name)
                is_frozen = name in self.perma_frozen

                # Coletar dados (similar à versão anterior, mas para preencher células da tabela)
                gd_hist_vals = self.gradient_disparity_hist.get(name, deque())
                snr_hist_vals = self.snr_hist.get(name, deque())
                gns_temporal_hist_vals = self.gns_temporal_hist.get(name, deque())
                snr_stability_hist = self.snr_value_hist_for_stability.get(name, deque())
                gns_t_stability_hist = self.gns_t_value_hist_for_stability.get(name, deque())

                # --- GD Cell ---
                current_gd, gd_met_primary, gd_is_stable_disp = np.nan, False, False
                gd_prim_cond_text, gd_stab_cond_text = f"P:hist<{cfg.min_gd_hist_eval}", "S:N/A"
                if gd_hist_vals and len(gd_hist_vals) >= cfg.min_gd_hist_eval:
                    current_gd = gd_hist_vals[-1]
                    ewma_gd = self.gd_ewma.get(name, np.nan)
                    ewma_var_gd = self.gd_std_ewma.get(name, np.nan)
                    if not np.isnan(ewma_gd) and not np.isnan(ewma_var_gd):
                        std_gd_from_ewma_var = math.sqrt(max(0, ewma_var_gd))
                        gd_target_threshold = ewma_gd - cfg.gd_std_k * std_gd_from_ewma_var
                        gd_met_primary = current_gd <= gd_target_threshold
                        gd_prim_cond_text = f"P:<={gd_target_threshold:.1e}"
                        
                        stab_thresh_gd = abs(ewma_gd) * cfg.gd_stability_std_thresh_factor + 1e-9
                        gd_is_stable_disp = std_gd_from_ewma_var < stab_thresh_gd
                        gd_stab_cond_text = f"S:σ<{stab_thresh_gd:.1e} ({std_gd_from_ewma_var:.1e})"
                    else:
                        gd_prim_cond_text = "P:EWMA warm"
                        gd_stab_cond_text = "S:EWMA warm"
                
                gd_cell = self._format_metric_cell(current_gd, self.stable_green_gd_steps.get(name,0),
                                                   cfg.gd_stable_confirm_steps, gd_prim_cond_text,
                                                   gd_stab_cond_text, gd_met_primary, gd_is_stable_disp)

                # --- SNR Cell ---
                current_snr, snr_met_primary, snr_is_stable_disp = np.nan, False, False
                snr_prim_cond_text, snr_stab_cond_text = f"P:hist<{cfg.min_snr_hist_eval}", "S:N/A"
                if snr_hist_vals and len(snr_hist_vals) >= cfg.min_snr_hist_eval:
                    current_snr = snr_hist_vals[-1]
                    median_snr = float(np.median(list(snr_hist_vals)))
                    q75_snr, q25_snr = np.percentile(list(snr_hist_vals), [75, 25])
                    iqr_snr = q75_snr - q25_snr
                    snr_target_value = median_snr - cfg.snr_k * iqr_snr if iqr_snr > 1e-9 else median_snr
                    snr_met_primary = current_snr <= snr_target_value
                    snr_prim_cond_text = f"P:<={snr_target_value:.2f}"
                    if snr_stability_hist and len(snr_stability_hist) >= cfg.snr_stability_window_len:
                        std_snr_stability = np.std(list(snr_stability_hist))
                        snr_is_stable_disp = std_snr_stability < cfg.snr_stability_std_thresh
                        snr_stab_cond_text = f"S:σ<{cfg.snr_stability_std_thresh:.2f} ({std_snr_stability:.2f})"
                    else:
                        snr_stab_cond_text = f"S:hist<{len(snr_stability_hist)}/{cfg.snr_stability_window_len}"
                
                snr_cell = self._format_metric_cell(current_snr, self.stable_green_snr_steps.get(name,0),
                                                    cfg.snr_stable_confirm_steps, snr_prim_cond_text,
                                                    snr_stab_cond_text, snr_met_primary, snr_is_stable_disp, value_format="{:.2f}")
                
                # --- GNS-T Cell ---
                current_gns_t, gns_t_met_primary, gns_t_is_stable_disp = np.nan, False, False
                gns_t_prim_cond_text, gns_t_stab_cond_text = f"P:hist<{cfg.min_hist_gns_temporal_eval}", "S:N/A"
                if gns_temporal_hist_vals and len(gns_temporal_hist_vals) >= cfg.min_hist_gns_temporal_eval:
                    current_gns_t = gns_temporal_hist_vals[-1]
                    mu_norm_sq_gns = self.latest_mu_temporal_norm_sq.get(name, float('inf'))
                    gns_inf_is_valid = (math.isinf(current_gns_t) and current_gns_t > 0 and mu_norm_sq_gns < cfg.gns_mu_norm_sq_conv_thresh)

                    if gns_inf_is_valid:
                        gns_t_met_primary = True
                        gns_t_is_stable_disp = True
                        gns_t_prim_cond_text = "P:inf valid"
                        gns_t_stab_cond_text = "S:inf valid"
                    elif not math.isinf(current_gns_t):
                        median_gns_t = float(np.median(list(gns_temporal_hist_vals)))
                        q75_gns_t, q25_gns_t = np.percentile(list(gns_temporal_hist_vals), [75, 25])
                        iqr_gns_t = q75_gns_t - q25_gns_t
                        dynamic_gns_thresh_display = median_gns_t + cfg.gns_temporal_k_iqr_factor * iqr_gns_t
                        if iqr_gns_t <= 1e-9: dynamic_gns_thresh_display = median_gns_t
                        gns_t_met_primary = current_gns_t >= dynamic_gns_thresh_display
                        gns_t_prim_cond_text = f"P:>={dynamic_gns_thresh_display:.2f}"
                        if gns_t_stability_hist and len(gns_t_stability_hist) >= cfg.gns_t_stability_window_len:
                            std_gns_t_stability = np.std(list(gns_t_stability_hist))
                            gns_t_is_stable_disp = std_gns_t_stability < cfg.gns_t_stability_std_thresh
                            gns_t_stab_cond_text = f"S:σ<{cfg.gns_t_stability_std_thresh:.2f} ({std_gns_t_stability:.2f})"
                        else:
                             gns_t_stab_cond_text = f"S:hist<{len(gns_t_stability_hist)}/{cfg.gns_t_stability_window_len}"
                    else:
                        gns_t_prim_cond_text = "P:invalid GNS"
                        gns_t_stab_cond_text = "S:invalid GNS"
                
                gns_t_cell = self._format_metric_cell(current_gns_t, self.stable_green_gns_t_steps.get(name,0),
                                                      cfg.gns_t_stable_confirm_steps, gns_t_prim_cond_text,
                                                      gns_t_stab_cond_text, gns_t_met_primary, gns_t_is_stable_disp, value_format="{:.2f}")

                # --- SNR*GNS Cell ---
                snr_gns_product_str = "N/A"
                # (Lógica para calcular snr_gns_product_str como antes)
                actual_current_snr_for_prod = snr_hist_vals[-1] if snr_hist_vals else np.nan
                actual_current_gns_t_for_prod = gns_temporal_hist_vals[-1] if gns_temporal_hist_vals else np.nan
                if not np.isnan(actual_current_snr_for_prod) and \
                    not np.isnan(actual_current_gns_t_for_prod) and \
                    not math.isinf(actual_current_gns_t_for_prod) and \
                    abs(actual_current_gns_t_for_prod) > 1e-9:
                    snr_gns_product = actual_current_snr_for_prod * actual_current_gns_t_for_prod
                    snr_gns_product_str = f"{snr_gns_product:.2f}"


                # --- Status Cell ---
                status_text_obj = Text()
                if is_frozen:
                    status_text_obj = Text("✓ FROZEN", style="green")
                else:
                    total_target_steps = cfg.gd_stable_confirm_steps + cfg.snr_stable_confirm_steps + cfg.gns_t_stable_confirm_steps
                    current_total_steps = self.stable_green_gd_steps.get(name,0) + \
                                          self.stable_green_snr_steps.get(name,0) + \
                                          self.stable_green_gns_t_steps.get(name,0)
                    prog_percent = 0
                    if total_target_steps > 0:
                        prog_percent = (current_total_steps / total_target_steps) * 100
                    status_text_obj = Text(f"✗ Active ({prog_percent:.0f}%)", style="yellow" if prog_percent > 0 else "red")


                # --- Module Name Cell (shortened) ---
                # (Lógica de encurtamento de nome como antes)
                original_module_name = name
                base_short_name = original_module_name 
                parts = original_module_name.split('_')
                if len(parts) > 2:
                    third_segment_from_start = parts[2]
                    last_meaningful_index = len(parts)
                    while last_meaningful_index > 0 and parts[last_meaningful_index - 1].isdigit():
                        last_meaningful_index -= 1
                    temp_end_segments = []
                    current_idx_name = last_meaningful_index - 1 
                    segments_taken_from_end = 0 
                    while current_idx_name >= 0 and segments_taken_from_end < 3:
                        if parts[current_idx_name] in ["blocks", "attentions"] and current_idx_name < 5 and len(parts) > 7: break
                        if current_idx_name <= 2 and len(parts) < 6: break
                        temp_end_segments.insert(0, parts[current_idx_name])
                        segments_taken_from_end += 1
                        current_idx_name -= 1
                    if temp_end_segments:
                        end_part = "_".join(temp_end_segments)
                        base_short_name = f"{third_segment_from_start}_{end_part}"
                    else: 
                        base_short_name = third_segment_from_start
                elif len(original_module_name) > 25 : # Shorten very long names if not fitting pattern
                    base_short_name = original_module_name[:12] + "..." + original_module_name[-10:]
                
                table.add_row(base_short_name, gd_cell, snr_cell, gns_t_cell, snr_gns_product_str, status_text_obj)
            
            tables.append(table)

        if not tables: # Should not happen if monitored_modules is not empty
             return Text("No data to display.", justify="center")
        if len(tables) == 1:
            return tables[0] # Se apenas uma tabela for populada (ex: poucos módulos)
            
        return Columns(tables, expand=True, equal=False, padding=1)