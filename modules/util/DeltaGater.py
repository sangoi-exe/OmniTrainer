import collections
import statistics
import math
import traceback
from typing import Dict, Set

import torch


class DeltaGater:
    """
    Early-stop modular (freeze-once) usando UWR.
    Acumula histórico completo e só congela módulos após confirmação consistente.
    """

    def __init__(
        self,
        z_thresh: float = 2.0,
        u_abs_thresh: float = 1e-4,
        k_confirm: int = 3,
        warmup_frac: float = 0.5,
        total_epochs: int = 100,
    ):
        self.z_thresh = z_thresh
        self.u_abs_thresh = u_abs_thresh
        self.k_confirm = k_confirm
        self.warmup_frac = warmup_frac
        self.total_epochs = total_epochs

        self.buffers: Dict[str, list[float]] = collections.defaultdict(list)
        self.suspect_counter: Dict[str, int] = collections.defaultdict(int)
        self.perma_frozen: Set[str] = set()

        self._current_epoch: int = 0
        self.last_decisions: Dict[str, bool] = {}

    def ingest(self, group_name: str, uwr: float):
        """Registra UWR para o módulo."""
        self.buffers[group_name].append(float(uwr))

    def set_current_epoch(self, epoch: int):
        """Atualiza o epoch atual (deve ser chamado antes de decidir)."""
        self._current_epoch = epoch

    def decide(self) -> Dict[str, bool]:
        """
        Decide congelamento baseado nos UWR acumulados.
        Retorna {module_name: should_freeze}.
        """
        try:
            # WARMUP: não age ainda
            progress = self._current_epoch / max(self.total_epochs, 1)
            if progress < self.warmup_frac:
                return {}

            # 1. Calcula estatísticas globais
            medians = {k: statistics.median(v) for k, v in self.buffers.items() if v}
            if len(medians) < 3:
                return {}

            global_median = statistics.median(medians.values()) + 1e-12  # proteção

            # 2. Avalia quem está suspeito
            decisions = {}
            for name, med in medians.items():
                if name in self.perma_frozen:
                    decisions[name] = True
                    continue

                uwr_latest = self.buffers[name][-1] if self.buffers[name] else 0.0
                z = (med - global_median) / (global_median + 1e-12)

                is_suspect = (z > self.z_thresh) and (uwr_latest < self.u_abs_thresh)

                if is_suspect:
                    self.suspect_counter[name] += 1
                else:
                    self.suspect_counter[name] = 0

                if self.suspect_counter[name] >= self.k_confirm:
                    self.perma_frozen.add(name)
                    decisions[name] = True
                else:
                    decisions[name] = False

            previous_decisions = getattr(self, "last_decisions", {})
            # 3. Logging compacto APENAS se mudou algo
            changed = False
            for name, frozen_now in decisions.items():
                if previous_decisions.get(name, False) != frozen_now:
                    changed = True
                    break

            if changed:
                total_modules = len(medians)
                frozen_now = sum(1 for v in decisions.values() if v)
                active_now = total_modules - frozen_now
                print(
                    f"[DeltaGater] Epoch {self._current_epoch} | "
                    f"Triggers {frozen_now}/{total_modules} | "
                    f"Active {active_now}/{total_modules}"
                )
            self.last_decisions = decisions  # salva agora!                

            return decisions

        except Exception as e:
            print(f"[DeltaGater] Deu merda na decisão: {e}")
            traceback.print_exc()
            return {}
