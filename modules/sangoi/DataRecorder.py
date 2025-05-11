import traceback
import json, gzip, os
from modules.sangoi.logFun import logFun
from typing import Dict, Optional, List, Tuple, Any


class DataRecorder:
    """
    Guarda perfil de cada módulo durante o treino:
      • d_pdgy_final (estimativa 'd' do Prodigy no final da Run 1)
    Dumpa em .json.gz no final.
    """

    def __init__(self, debug: bool = True, verbose: bool = True):
        self.d_pdgy_final: Dict[str, float] = {}
        self.debug = debug
        self.verbose = verbose

    def log_step(
            self,
            name: str,  # Nome do módulo/grupo de parâmetros
            d_pdgy: float,  # Valor 'd' do otimizador Prodigy para este módulo
    ):
        """Registra o d_pdgy mais recente para o módulo."""
        # O 'd_pdgy' aqui é o 'd' do grupo de params do Prodigy, que é a estimativa da distância ao ótimo.
        self.d_pdgy_final[name] = d_pdgy
        # if self.debug and step % 500 == 0: ...

    def dump(
        self,
        path: str,
    ):
        """Salva os dados coletados (d_pdgy_final)."""
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        data_to_dump = {
            "d_pdgy_final": self.d_pdgy_final,}

        try:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(data_to_dump, f, indent=2)
            if self.verbose:
                logFun(f"[DataRecorder] Perfil (d_pdgy_final) salvo com sucesso em {path}", lvl="success")
        except Exception as e:
            # IA_MODIFICACAO: Corrigido o log de erro para usar logFun e passar o nível corretamente.
            logFun(f"[DataRecorder] ERRO ao salvar perfil em {path}: {e}", lvl="error")
            traceback.print_exc()
