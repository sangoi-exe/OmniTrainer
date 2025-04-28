class ModuleDynRecorder:
    """
    Guarda perfil de cada módulo durante a Run-1
      • hist UWR        (sparse)
      • passo conv      (1ª vez que DeltaGater congela)
      • d_hat_final
      • d_coef_base     (1º valor visto)
    Dumpa em .json.gz no final.
    """
    def __init__(self, k_stride:int=10):
        import collections
        self.k_stride     = k_stride
        self.uwr_hist     = collections.defaultdict(list)
        self.conv_step    = {}
        self.d_hat_final  = {}
        self.d_coef_base  = {}

    def log_step(self, name:str, step:int, uwr:float,
                 d_hat:float, d_coef:float, converged:bool):
        if step % self.k_stride == 0:
            self.uwr_hist[name].append((step, uwr))
        if converged and name not in self.conv_step:
            self.conv_step[name] = step
        self.d_hat_final[name] = d_hat
        self.d_coef_base.setdefault(name, d_coef)

    def dump(self, path:str):
        import json, gzip, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)
        print(f"[Recorder] perfil salvo em {path}")
