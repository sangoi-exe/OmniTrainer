import gzip
import json
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple


class ModuleDynRecorder:
    """
    Guarda perfil de cada módulo durante a Run-1
      • hist UWR        (sparse)
      • passo conv      (1ª vez que ConvergeControl congela)
      • d_hat_final
      • d_coef_base     (1º valor visto)
    Dumpa em .json.gz no final.
    """
    def __init__(self, k_stride: int = 10, debug: bool = True, verbose: bool = True):
        import collections
        self.k_stride     = k_stride
        self.uwr_hist     = collections.defaultdict(list)
        self.conv_step    = {}
        self.d_hat_final  = {}
        self.d_coef_base  = {}
        # --- START: Modificação solicitada - Store debug/verbose flags ---
        self.debug   = debug
        self.verbose = verbose
        # --- END: Modificação solicitada - Store debug/verbose flags ---

    def log_step(self, name:str, step:int, uwr:float,
                d_hat:float, d_coef:float, converged:bool):
        # --- START: Modificação solicitada - Add debug logging ---
        if self.debug:
            print(
                f"[ModuleDynRecorder][DEBUG] log_step para '{name}': "
                f"step={step}, uwr={uwr:.6f}, d_hat={d_hat:.6f}, "
                f"d_coef={d_coef:.6f}, converged={converged}"
            )
        # --- END: Modificação solicitada - Add debug logging ---
        if step % self.k_stride == 0:
            self.uwr_hist[name].append((step, uwr))
            # --- START: Modificação solicitada - Add debug logging ---
            if self.debug:
                print(f"[ModuleDynRecorder][DEBUG] uwr_hist['{name}'] appends. Size={len(self.uwr_hist[name])}")
            # --- END: Modificação solicitada - Add debug logging ---
        if converged and name not in self.conv_step:
            self.conv_step[name] = step
            # --- START: Modificação solicitada - Add debug logging ---
            if self.debug:
                print(f"[ModuleDynRecorder][DEBUG] conv_step['{name}'] = {step} (primeira vez)")
            # --- END: Modificação solicitada - Add debug logging ---
        self.d_hat_final[name] = d_hat
        self.d_coef_base.setdefault(name, d_coef) # Use setdefault for base_d_coef
        # --- START: Modificação solicitada - Add debug logging ---
        if self.debug:
            print(f"[ModuleDynRecorder][DEBUG] Atualizado d_hat_final['{name}']={d_hat:.6f}, d_coef_base['{name}']={self.d_coef_base[name]:.6f}")
        # --- END: Modificação solicitada - Add debug logging ---   

    # Assinatura usa Any para a tupla interna para simplificar
    def dump(self, path: str, convergence_data: Optional[Dict[str, List[Tuple[Any, ...]]]] = None):
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if self.debug:
            conv_data_summary = {k: len(v) for k,v in convergence_data.items()} if convergence_data else {}
            # Exemplo de log mostrando a primeira tupla para um módulo, se existir
            first_module = next(iter(conv_data_summary.keys()), None)
            first_point_example = convergence_data.get(first_module, [None])[0] if first_module else None
            print(f"[ModuleDynRecorder][DEBUG] Dumping para '{path}': Base d_coef={self.d_coef_base}, Convergence Data Points={conv_data_summary}")
            if first_point_example:
                print(f"[ModuleDynRecorder][DEBUG] Exemplo do 1º ponto de dados ('{first_module}'): {first_point_example}")

        data_to_dump = {
            "d_hat_final": self.d_hat_final,
            "d_coef_base": self.d_coef_base,
            # Adiciona cabeçalho para explicar os campos da tupla no JSON
            "convergence_history_schema": [
                "epoch", "step", "uwr_latest", "uwr_abs_latest", "median_uwr",
                "std_dev_uwr", "z_robust", "global_median_uwr", "global_mad_uwr",
                "delta_median_uwr", "was_suspect_flag"
            ],
            "convergence_history": convergence_data if convergence_data else {},
        }

        try:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(data_to_dump, f, indent=2)
            if self.verbose:
                print(f"[ModuleDynRecorder] Perfil salvo com sucesso em {path} (incluindo histórico de convergência detalhado)")
        except Exception as e:
             print(f"[ModuleDynRecorder] ERRO ao salvar perfil em {path}: {e}")
             traceback.print_exc()