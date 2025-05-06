import collections
import traceback
import numpy as np
from scipy.stats import linregress
from typing import Deque, Dict, Set, List, Optional, Tuple

import torch

from modules.sangoi.logFun import logFun


class ConvergeControl:
    """
    Coleta métricas de delta-L2 e toma decisões de congelamento de módulos
    com base na estabilidade dessas métricas. O congelamento só é aplicável
    a partir da Run 2 (ou se explicitamente habilitado).
    """

    def __init__(
        self,
        run_number: int,
        *,
        k_confirm: int = 5,  # Número de checks para congelar
        delta_buffer_size: int | None = None,  # Novo parâmetro para buffer de delta
        debug: bool = True,
        verbose: bool = True,
        total_epochs: int = 100,
        # Parâmetros para DECISÃO
        cv_thresh: float = 0.5,        # ← coef. de variação limite (fallback)
        slope_thresh: float = 1e-4,    # ← limiar de inclinação (fallback)
        delta_abs_thresh: float = 2e-3,# ← limiar de Δ-L2 (fallback)
        dynamic_k: float = 1.5,        # ← multiplica IQR para threshold dinâmico
        hist_size: int = 10,  
        
        min_buffer_epochs: float = 0,  # Warmup em épocas (mantido)
        enable_freeze_action: bool = True,
    ):

        self.k_confirm = max(2, k_confirm)  # segurança (≥2)
        self.win_size = 50  # ← regra única

        self.cv_thresh = cv_thresh
        self.abs_thresh = delta_abs_thresh
        self.slope_thresh = slope_thresh
        
        # parâmetros para limiares dinâmicos
        self.dynamic_k = dynamic_k
        self.hist_size = hist_size

        # histórico de métricas para thresholds dinâmicos
        # cada módulo terá deques de 'mean', 'cv' e 'slope'
        self.metric_hist: Dict[str, Dict[str, Deque[float]]] = collections.defaultdict(
            lambda: {
                'mean':  collections.deque(maxlen=self.hist_size),
                'cv':    collections.deque(maxlen=self.hist_size),
                'slope': collections.deque(maxlen=self.hist_size),
            }
        )
        # thresholds dinâmicos calculados
        self.dyn_thresh: Dict[str, Dict[str, float]] = {}				

        self._last_epoch_incr: Dict[str, int] = {}
        self.freeze_step_run2: Dict[str, int] = {}  # Mantido para Run >= 2
        self.min_buffer_epochs = min_buffer_epochs
        self._prev_epoch_state: Dict[str, torch.Tensor] = {}

        hint = delta_buffer_size or 0
        self._buffer_maxlen = max(hint, self.win_size)
        self.delta_buffers = collections.defaultdict(lambda: collections.deque(maxlen=self._buffer_maxlen))

        # Buffer de Δ-L2 por módulo (mantido)
        self.perma_frozen: Set[str] = set()  # Módulos congelados NESTA run (mantido)
        self.suspect_counter: Dict[str, int] = collections.defaultdict(int)  # Contador para k_confirm (mantido)

        self._current_epoch: int = 0
        self._current_step: int = 0

        self.debug = debug
        self.verbose = verbose
        self.run_number = run_number
        self.total_epochs = total_epochs
        self.enable_freeze_action = (
            enable_freeze_action if run_number >= 2 else False
        )  # Congelamento só na Run 2+ por padrão

        self.warm_steps = 420  # coleta só estatísticas
        self.alpha_rur = 2.0  # k × IQR  →  thr_rur
        self.beta_grad = 2.0  # k × IQR  →  thr_grad
        self.win_snr = 20  # janela SNR (steps)
        self.grad_eps = 1e-9
        self.thr_rur: Dict[str, float] = {}
        self.thr_grad: Dict[str, float] = {}

        # ▼ buffers por módulo
        self.rur_hist = collections.defaultdict(  # deque de floats (steps)
            lambda: collections.deque(maxlen=self.warm_steps)
        )
        self.g_hist = collections.defaultdict(  # deque de tensores (steps)
            lambda: collections.deque(maxlen=self.win_snr)
        )
        self.prev_W = {}  # peso salvo fim de época

        # ▼ baselines mediana+IQR aprendidos no warm-up
        self.base_rur = {}
        self.base_grad = {}

        if self.debug:
            logFun(
                f"[ConvergeControl] Initializing for Run {self.run_number}. "
                f"Freeze Action Effective: {self.enable_freeze_action}. "
                # Mensagem sobre dados históricos removida
                f"Delta Buffer Maxlen: {self._buffer_maxlen}, k_confirm: {self.k_confirm}, win_size: {self.win_size}",
                lvl="debug")
        if self.debug:
            logFun(
                f"[ConvergeControl] Init finished for Run {self.run_number}. Using Delta-L2 stability logic.",
                # Mensagem sobre dados históricos removida
                lvl="debug")

    def update_step_metrics(self, name: str, module: torch.nn.Module):
        """
        Captura métricas por step a partir de um módulo:
          • grad_norm (concat de todos os grads) → SNR
          • RUR vs snapshot anterior → Relative-Update-Ratio
        """
        # AQUI TÁ SUAVE
        try:
            # 1) Grad norm para SNR
            grads = [
                p.grad.detach().flatten()
                for p in module.parameters()
                if p.grad is not None
            ]
            if grads:
                grad_vec = torch.cat(grads)
                self.g_hist[name].append(grad_vec.clone())

            # 2) Captura o vetor de peso corrente
            if hasattr(module, "flat_params"):
                curr_w = module.flat_params().detach()
            else:
                params = [p.detach().flatten() for p in module.parameters()]
                curr_w = torch.cat(params) if params else None
            if curr_w is None:
                return

            # 3) RUR em relação ao snapshot anterior
            prev_w = self.prev_W.get(name, curr_w)
            rur = (curr_w - prev_w).norm() / (prev_w.norm() + self.grad_eps)
            self.rur_hist[name].append(rur.item())

            # 4) Atualiza snapshot para o próximo step
            self.prev_W[name] = curr_w.clone()

            # Debug opcional
            #logFun(f"[ConvergeControl] Step RUR '{name}': {rur.item():.4e}", lvl="debug")
        except Exception as e:
            logFun(f"[ConvergeControl] Exception during update_step_metrics processing: {e}", lvl="error")
            traceback.print_exc()                        

    def snapshot_epoch_weights(self, deltas_by_module: Dict[str, float]):
        """
        Atualiza os buffers de Δ-L2 por módulo usando os deltas pré-calculados.
        """
        for module, delta in deltas_by_module.items():
            self.delta_buffers[module].append(delta)
        # Para imprimir tudo formatado
        # Imprime porcentagem de estabilidade de cada módulo ao fim da época
        # Imprime porcentagem de estabilidade e métricas absolutas de cada módulo
        for module in self.delta_buffers.keys():
            # re-calcula estatísticas para exibir
            buf = self.delta_buffers[module]
            window = list(buf)[-self.win_size:]
            mean, std, slope = self.stats_window(window)
            cv = std / (mean + 1e-12)
            # ❶ Atualiza histórico de métricas
            mh = self.metric_hist[module]
            mh['mean'].append(mean)
            mh['cv'].append(cv)
            mh['slope'].append(abs(slope))

            # ❷ Se houver hist_size pontos, calcula thresholds dinâmicos
            if len(mh['mean']) == self.hist_size:
                import numpy as np
                # para cada métrica, usa median + k * IQR
                for met in ('mean','cv','slope'):
                    arr = np.array(mh[met])
                    q1, q3 = np.percentile(arr, [25, 75])
                    med = np.median(arr)
                    self.dyn_thresh.setdefault(module, {})[met] = med + self.dynamic_k * (q3 - q1)

            # logs dos thresholds dinâmicos/fixos e da estabilidade
            thr = self.dyn_thresh.get(module, {})
            logFun(
                f"[THR] '{module}': abs<{thr.get('mean', self.abs_thresh):.3e}, "
                f"cv<{thr.get('cv', self.cv_thresh):.3f}, "
                f"slope<{thr.get('slope', self.slope_thresh):.3e}",
                lvl="debug"
            )

            # ❸ Exibe estabilidade com métricas absolutas
            score = self.stability_score(module)
            logFun(
                f"[DEBUG] Época {self._current_epoch:03d} — '{module}': "
                f"estab={score*100:5.1f}% │ "
                f"mean={mean:.3e}, cv={cv:.3e}, slope={slope:.3e}",
                lvl="warning"
            )
            
    def _ready_to_freeze(self, name: str) -> bool:
        """Usa RUR e grad_norm com limiares aprendidos no warm-up."""
        # warm-up: ainda aprendendo baselines
        if self._current_step < self.warm_steps or name not in self.prev_W:
            return False

        # calcula baselines se ainda não existem
        if name not in self.base_rur:
            def _robust(arr):
                q1, q3 = np.percentile(arr, [25, 75])
                return np.median(arr), (q3 - q1)
            self.base_rur[name] , spread_rur  = _robust(self.rur_hist[name])
            self.base_grad[name], spread_grad = _robust([x.norm().item() for x in self.g_hist[name]])

            self.thr_rur[name]  = max(0.0, self.base_rur[name]  - self.alpha_rur * spread_rur)
            self.thr_grad[name] = max(0.0, self.base_grad[name] - self.beta_grad * spread_grad)

        # métricas correntes
        curr_rur   = self.rur_hist[name][-1]
        curr_gnorm = self.g_hist[name][-1].norm().item()

        rur_ok   = curr_rur   < self.thr_rur[name]
        grad_ok  = curr_gnorm < self.thr_grad[name]

        # SNR opcional
        if len(self.g_hist[name]) >= self.win_snr:
            g_stack = torch.stack(list(self.g_hist[name]))
            mu  = g_stack.mean(dim=0)
            std = g_stack.std(dim=0, unbiased=False) + self.grad_eps
            snr = mu.norm() / std.norm()
            snr_ok = snr < 0.05   # limiar fixo ou adaptativo
        else:
            snr_ok = False

        stable = (rur_ok + grad_ok + snr_ok) >= 2
        return stable                        

    @staticmethod
    def stats_window(arr: List[float]) -> Tuple[float, float, float]:
        """Retorna média, desvio padrão e inclinação (OLS) de uma sequência."""
        y = np.asarray(arr, dtype=np.float32)
        mean = float(y.mean())
        std = float(y.std(ddof=1))
        x = np.arange(len(y), dtype=np.float32)
        slope, *_ = linregress(x, y)  # small slope → ~estável
        return mean, std, float(slope)

    def set_current_time(self, epoch: int, step: int):
        """Atualiza o epoch e step atuais."""
        self._current_epoch = epoch
        self._current_step = step

    def ingest_and_process(self, stats_list: List[Dict]):
        """Coleta apenas delta_L2 dos módulos para o buffer."""
        if not stats_list:
            return

        processed_deltas = 0
        for stat in stats_list:
            # AQUI TÁ SUAVE
            name = stat.get("name")
            delta_val = stat.get("delta_L2")  # Única métrica relevante aqui

            if name is None or delta_val is None:
                continue

            try:
                # AQUI TA SUAVE
                # Registra Δ-L2 no buffer correspondente
                self.delta_buffers[name].append(float(delta_val))
                processed_deltas += 1
            except (ValueError, TypeError) as e:
                if self.debug:
                    logFun(f"[ConvergeControl] Error processing delta_L2 for '{name}': {e}. Value: {delta_val}", lvl="warning")

        if self.debug and processed_deltas == 0:  # Log menos frequente
            logFun(f"[ConvergeControl] Step {self._current_step}: Nenhum delta_L2 válido recebido para processar.", lvl="debug")
        # elif self.debug and self._current_step % 50 == 0:
        #     logFun(f"[ConvergeControl] Step {self._current_step}: Ingested {processed_deltas} delta_L2 values.", lvl="debug") # Log opcional de sucesso

    def decide(self) -> Dict[str, bool]:
        """
        Decide quais módulos devem ser congelados com base na estabilidade
        do delta-L2. Retorna um dicionário {nome_modulo: deve_congelar}.
        A ação de congelamento só ocorre se enable_freeze_action for True.
        """
        # A decisão de congelar só é relevante/possível a partir da Run 2
        # ou se enable_freeze_action foi forçado para True no init.
        if not self.enable_freeze_action:
            # Se o congelamento não está habilitado globalmente, retorna vazio.
            # Nota: perma_frozen ainda pode ser preenchido se enable_freeze_action
            # for ligado durante o treino, mas decide() não retornará True aqui.
            return {}

        # Warm-up global baseado em épocas
        progress = self._current_epoch / max(self.total_epochs, 1) if self.total_epochs > 0 else 0
        if progress < self.min_buffer_epochs:
            logFun("dentro do if progress < self.min_buffer_epochs", lvl="warning")
            # Ainda em warm-up de épocas, não toma decisões
            if self.debug:  # Log menos frequente
                logFun(
                    f"[ConvergeControl] Decide@{self._current_step}: Skipping decisions (Epoch Warmup {self._current_epoch}/{self.total_epochs}, progress {progress:.2f} < {self.min_buffer_epochs:.2f}).",
                    lvl="debug")
            return {}  # Retorna dict vazio em warmup

        decisions: Dict[str, bool] = {}
        # Decide para todos os módulos que possuem algum histórico de delta_L2
        module_names_to_decide = list(self.delta_buffers.keys())

        if self.debug and not module_names_to_decide:
            logFun(f"[ConvergeControl] Decide@{self._current_step}: Nenhum módulo com buffer delta para avaliar.", lvl="debug")

        for name in module_names_to_decide:
            # Pula avaliação se já congelado *permanentemente* NESTA run
            if name in self.perma_frozen:
                decisions[name] = True  # Mantém a decisão como True se já congelado permanentemente
                continue

            # Avalia convergência/estabilidade via Δ-L2
            is_stable = self._ready_to_freeze(name)

            # Incrementa contador de checks consecutivos estáveis por módulo
            # Resetar contador se instável ou se a época mudou (contagem por época)
            current_epoch = self._current_epoch
            last_incr_epoch = self._last_epoch_incr.get(name, -1)

            if is_stable:
                # Se estável e é uma nova época desde a última atualização do contador para este módulo
                if current_epoch > last_incr_epoch:
                    self.suspect_counter[name] += 1
                    self._last_epoch_incr[name] = current_epoch
                # Se estável e ainda na mesma época, não incrementa de novo, mas mantém o valor
            else:
                # Se instável e é uma nova época
                if current_epoch > last_incr_epoch:
                    self.suspect_counter[name] = 0  # Reset se instável
                    self._last_epoch_incr[name] = current_epoch
                # Se instável e na mesma época, mantém o contador (pode ter sido > 0 antes)

            # A *avaliação* de congelamento é True se for estável por k_confirm épocas consecutivas
            should_freeze_eval = self.suspect_counter.get(name, 0) >= self.k_confirm

            # Registra a *avaliação* (se o critério foi atingido)
            decisions[name] = should_freeze_eval

            # A *ação* de congelamento só ocorre se a avaliação for True
            # E a ação estiver habilitada globalmente (já checado no início)
            if should_freeze_eval:  # Implicitamente, self.enable_freeze_action é True aqui
                # Adiciona ao conjunto de congelados permanentes desta run
                # e registra o step do congelamento efetivo
                if name not in self.perma_frozen:  # Congela apenas uma vez
                    self.perma_frozen.add(name)
                    self.freeze_step_run2[name] = self._current_step
                    if self.verbose or self.debug:
                        logFun(
                            f"[ConvergeControl] FREEZE ACTION '{name}' @ step {self._current_step} "
                            f"(Triggered by: Delta-L2 stability for {self.suspect_counter.get(name, 0)} consecutive checks >= {self.k_confirm})",
                            lvl="verbose" if self.verbose else "debug")
            # Não há 'else' aqui para logar skip, pois a avaliação já é False se não deve congelar.
            # O skip por enable_freeze_action=False é tratado no início da função.

        return decisions

    def get_frozen_set(self) -> Set[str]:
        """Retorna o conjunto de módulos permanentemente congelados nesta run."""
        return self.perma_frozen

    def get_final_conv_steps(self) -> Dict[str, int]:
        """
        Retorna os steps onde o congelamento *efetivo* ocorreu (Run >= 2).
        Para Run 1, retorna um dicionário vazio, pois o congelamento não é aplicado.
        """
        if self.run_number == 1:
            logFun("[ConvergeControl] get_final_conv_steps (Run 1): Returning empty dict (no freeze action in Run 1).", lvl="debug")
            return {}  # Congelamento não aplicável na Run 1 por padrão
        else:  # Run 2 ou maior
            # Retorna os steps onde o congelamento *efetivo* ocorreu
            logFun(
                f"[ConvergeControl] get_final_conv_steps (Run {self.run_number}): Returning effective freeze_step_run2 ({len(self.freeze_step_run2)} items).",
                lvl="debug")
            return dict(self.freeze_step_run2)

    def _delta_stable(self, name: str) -> bool:
        """
        Considera o módulo estável se, na última 'win' amostras:
          • (mean  < abs_thresh  AND  cv < cv_thresh)        OU
          • (cv    < cv_thresh   AND  |slope| < slope_thresh)

        Onde cv = std / (mean + ε).  
        Isso captura 3 caminhos:
          1) pequeno + pouco ruído
          2) pouco ruído + sem tendência
          3) pequeno + sem tendência   (é o subconjunto dos dois acima)
        """

        win = self.win_size  # p.ex. 10
        buf = self.delta_buffers[name]  # type: Deque[float]
        if len(buf) < win:
            return False  # ainda sem histórico

        window = list(buf)[-win:]  # fatia mais recente
        mean, std, slope = self.stats_window(window)

        cv = std / (mean + 1e-12)
        mean_ok = mean < self.abs_thresh
        cv_ok = cv < self.cv_thresh
        slope_ok = abs(slope) < self.slope_thresh

        stable = (mean_ok and cv_ok) or (cv_ok and slope_ok)
        # try:
        #     if self.debug:
        #         # Último delta-L2 observado
        #         last_delta_val = f"{window[-1]:.3e}" if window else "N/A"
        #         logFun(
        #             f"[Δ-STABLE CHECK] {name}@{self._current_step} (win={len(window)}): "
        #             f"Δ_last={last_delta_val} | "
        #             f"mean={mean:.3e}  ({'OK' if mean_ok  else 'NO'})< {self.abs_thresh:.1e} | "
        #             f"CV={cv:.3e}     ({'OK' if cv_ok    else 'NO'})< {self.cv_thresh:.2f} | "
        #             f"slope={slope:.3e} ({'OK' if slope_ok else 'NO'})< {self.slope_thresh:.1e} || "
        #             f"STABLE={stable}",
        #             lvl="debug")
        # except KeyError as e:
        #     logFun(f"[Trainer] KeyError ao acessar stat['group_idx'] ou mapeamento. Chave ausente? Erro: {e}", lvl="error")
        #     traceback.print_exc()
        # except IndexError as e:
        #     logFun(f"[Trainer] IndexError ao acessar self.model.param_group_mapping. group_idx fora do range? Erro: {e}", lvl="error")
        #     traceback.print_exc()
        # except Exception as e:
        #     logFun(f"[Trainer@{self._current_step}] Exception during post-optimizer step processing (ConvergeControl/Recorder): {e}", lvl="error")
        #     traceback.print_exc()

        return stable

    def stability_score(self, name: str) -> float:
        """
        Retorna um score [0.0, 1.0] baseado em:
          • mean < abs_thresh
          • cv   < cv_thresh
          • |slope| < slope_thresh
        Média simples das 3 componentes.
        """
        buf = self.delta_buffers.get(name, None)
        if buf is None or len(buf) < self.win_size:
            return 0.0  # ainda sem histórico suficiente

        # calcula estatísticas dos últimos win_size deltas
        window = list(buf)[-self.win_size:]
        mean, std, slope = self.stats_window(window)
        cv = std / (mean + 1e-12)
        
        # escolhe thresholds: dinâmico se calculado, senão o fixo
        thr_mean   = self.dyn_thresh.get(name, {}).get('mean',   self.abs_thresh)
        thr_cv     = self.dyn_thresh.get(name, {}).get('cv',     self.cv_thresh)
        thr_slope  = self.dyn_thresh.get(name, {}).get('slope',  self.slope_thresh)

        # pontua cada critério (0 = falhou, 1 = perfeição)
        mean_score  = max(0.0, min(1.0, (thr_mean   - mean)  / thr_mean))
        cv_score    = max(0.0, min(1.0, (thr_cv     - cv)    / thr_cv))
        slope_score = max(0.0, min(1.0, (thr_slope  - abs(slope)) / thr_slope))

        # média simples como score final
        return (mean_score + cv_score + slope_score) / 3.0