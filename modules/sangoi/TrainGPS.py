import json
import traceback
import os
import torch
from torch import Tensor
from typing import Iterable, Tuple, Dict, Optional, List, TYPE_CHECKING
from collections import defaultdict

from modules.sangoi.logFun import logFun

if TYPE_CHECKING:
    from modules.util.NamedParameterGroup import NamedParameterGroupCollection

class TrainGPS:
    """
    Orquestra o registro e a aplicação de padrões de treinamento (deltas de peso).
    Opera em três modos principais baseados nas flags 'train_gps_save_it' e 'train_gps_use_it':
    1.  Modo Coleta (save_it=True, use_it=False): Captura os pesos iniciais e registra os deltas
        acumulados por época. O objetivo é criar um padrão de referência.
    2.  Modo Aplicação (save_it=False, use_it=True): Carrega um padrão de referência e calcula uma
        penalidade de loss para guiar o treino atual a seguir esse padrão.
    3.  Modo Refinamento (save_it=True, use_it=True): Faz ambos. Usa um padrão existente como guia
        e, ao mesmo tempo, salva os deltas da run atual para uma futura análise ou para criar
        um novo padrão refinado.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        param_collection: "NamedParameterGroupCollection",
        penalty_metric: str = "cosine"
    ):
        if param_collection is None:
            raise ValueError("param_collection não pode ser None para TrainGPS.")

        self.model = model
        self.param_collection: "NamedParameterGroupCollection" = param_collection
        self.penalty_metric: str = penalty_metric.lower()
        if self.penalty_metric not in {"mse", "cosine"}:
            raise ValueError(f"penalty_metric inválida: '{self.penalty_metric}'. Use 'mse' ou 'cosine'.")

        # --- Armazenamento de dados ---
        # Usado para calcular os deltas a serem salvos (Modo Coleta/Refinamento)
        self.save_run_initial_weights: Dict[str, torch.Tensor] = {}
        # Usado para calcular a penalidade contra o padrão de referência (Modo Aplicação/Refinamento)
        self.use_run_initial_weights: Dict[str, torch.Tensor] = {}
        # Padrão de referência carregado de um arquivo
        self.reference_deltas: Dict[str, torch.Tensor] = {}
        # Log dos deltas da run atual, agrupados por época
        self.delta_log_by_module: Dict[str, Dict[str, float]] = defaultdict(dict)

        # --- Caching de normas para logging ---
        self.current_total_delta_norm: Optional[float] = None
        self.reference_delta_norm: Optional[float] = None

    # --- Métodos de Interface Pública ---

    def setup_for_save(self):
        """Prepara o GPS para o modo de Coleta ou Refinamento, capturando os pesos iniciais."""
        logFun("[TrainGPS] Configurando para salvar deltas (Run 1 ou Run N)...", lvl="TRAINGPS")
        self.save_run_initial_weights = self._capture_weights_generic(torch.device("cpu"))

    def setup_for_use(self, pattern_path: str):
        """Prepara o GPS para o modo de Aplicação ou Refinamento, carregando um padrão de referência."""
        logFun(f"[TrainGPS] Configurando para usar padrão de deltas de '{pattern_path}'...", lvl="TRAINGPS")
        if not pattern_path or not os.path.isfile(pattern_path):
            raise FileNotFoundError(f"Arquivo de padrão de delta não encontrado: {pattern_path}")

        self._load_reference_pattern_from_file(pattern_path)
        
        target_device = self._get_target_device()
        self.use_run_initial_weights = self._capture_weights_generic(target_device, torch.bfloat16)

    def log_epoch_deltas(self, epoch: int):
        """
        Calcula os deltas da época atual em relação aos pesos iniciais e os registra.
        Deve ser chamado no final de cada época no modo de Coleta/Refinamento.
        """
        if not self.save_run_initial_weights:
            logFun("[TrainGPS] Tentativa de logar deltas sem pesos iniciais capturados. Ignorando.", lvl="warning")
            return

        current_deltas = self._calculate_current_deltas(self.save_run_initial_weights)
        
        # Agrupa os deltas por prefixo e calcula a norma L2
        group_norms_sq: Dict[str, float] = defaultdict(float)
        for name, delta in current_deltas.items():
            # Define o prefixo de agrupamento (ajuste aqui para granularidade desejada)
            prefix = name.split(".")[0]
            group_norms_sq[prefix] += torch.norm(delta.float(), p=2).pow(2).item()

        epoch_key = f"epoch_{epoch}"
        for prefix, norm_sq in group_norms_sq.items():
            delta_norm = norm_sq ** 0.5
            self.delta_log_by_module[epoch_key][prefix] = delta_norm

        logFun(f"[TrainGPS] Deltas logados para {epoch_key}.", lvl="info")

    def compute_penalty(self, lambda_weight: float) -> torch.Tensor:
        """
        Calcula a penalidade da loss comparando os deltas atuais com o padrão de referência.
        Retorna um tensor escalar com a penalidade ponderada, ou zero se não aplicável.
        """
        target_device = self._get_target_device()

        if not self.reference_deltas or not self.use_run_initial_weights:
            self.current_total_delta_norm = None
            return torch.tensor(0.0, device=target_device)

        try:
            current_deltas_by_group = self._get_current_deltas_by_group(self.use_run_initial_weights)
            
            # Precisamos obter os deltas de referência para a época atual.
            current_epoch = self.model.train_progress.epoch
            reference_key_prefix = f"epoch_{current_epoch}"
            
            relevant_reference_deltas = {
                key.split("/", 1)[1]: val
                for key, val in self.reference_deltas.items()
                if key.startswith(reference_key_prefix)
            }

            if not relevant_reference_deltas:
                # Nenhuma referência para esta época, sem penalidade.
                return torch.tensor(0.0, device=target_device)

            common_keys = [k for k in relevant_reference_deltas if k in current_deltas_by_group]
            if not common_keys:
                return torch.tensor(0.0, device=target_device)

            # Empilha os vetores para cálculo em batch
            ref_vec = torch.stack([relevant_reference_deltas[k].to(device=target_device, dtype=torch.float32) for k in common_keys])
            cur_vec = torch.stack([current_deltas_by_group[k].to(dtype=torch.float32) for k in common_keys])

            # Calcula a norma do delta atual para logging
            self.current_total_delta_norm = float(torch.norm(cur_vec, p=2))

            if self.penalty_metric == "cosine":
                # 1 - similaridade = distância. Queremos minimizar a distância.
                penalty = 1.0 - torch.nn.functional.cosine_similarity(cur_vec.flatten(), ref_vec.flatten(), dim=0, eps=1e-8)
            else:  # "mse"
                penalty = torch.nn.functional.mse_loss(cur_vec, ref_vec)

            return (lambda_weight * penalty).to(dtype=self._get_target_dtype())

        except Exception as e:
            logFun(f"[TrainGPS] Erro em compute_penalty: {e}", lvl="error")
            traceback.print_exc()
            self.current_total_delta_norm = None
            return torch.tensor(0.0, device=target_device)

    def save_log_to_file(self, path: str):
        """Salva o dicionário de deltas logados em um arquivo JSON."""
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.delta_log_by_module, f, indent=2)
            logFun(f"[TrainGPS] Log de deltas salvo com sucesso em {path}", lvl="success")
        except Exception as e:
            logFun(f"[TrainGPS] ERRO ao salvar log de deltas: {e}", lvl="error")
            traceback.print_exc()

    def get_delta_norms_for_logging(self) -> Tuple[Optional[float], Optional[float]]:
        """Retorna as normas L2 cacheadas do delta atual e de referência para logging."""
        return self.current_total_delta_norm, self.reference_delta_norm

    # --- Métodos Privados de Lógica Interna ---

    def _iterate_params(self) -> Iterable[Tuple[str, torch.Tensor]]:
        """Iterador privado que abstrai a obtenção de parâmetros treináveis do modelo."""
        # Itera sobre os grupos de parâmetros que definimos no setup
        for group in self.param_collection.groups:
            # O nome do grupo já é o prefixo único do módulo LoRA
            prefix = group.unique_name
            for param in group.parameters:
                # O 'nome' completo do parâmetro não é tão importante quanto o prefixo do grupo.
                # Usamos o prefixo para consistência com como os deltas são agrupados.
                yield prefix, param.detach()

    def _get_target_device(self) -> torch.device:
        """Obtém o device do primeiro parâmetro do modelo."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu") # Fallback

    def _get_target_dtype(self) -> torch.dtype:
        """Obtém o dtype do primeiro parâmetro do modelo."""
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32 # Fallback

    def _capture_weights_generic(
        self, target_device: torch.device, target_dtype: Optional[torch.dtype] = None
    ) -> Dict[str, torch.Tensor]:
        """Helper genérico para capturar os pesos atuais do modelo."""
        storage_dict = {}
        processed_params = 0
        for name, param in self._iterate_params():
            captured_param = param.clone().to(device=target_device)
            if target_dtype:
                captured_param = captured_param.to(dtype=target_dtype)
            storage_dict[name] = captured_param
            processed_params += 1
        
        if processed_params == 0:
            logFun("[TrainGPS] Aviso: _capture_weights_generic não encontrou parâmetros.", lvl="warning")
        
        return storage_dict
    
    def _calculate_current_deltas(self, reference_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Helper privado para calcular deltas atuais contra um dicionário de referência."""
        current_deltas = {}
        for name, current_param in self._iterate_params():
            if name in reference_weights:
                initial_weight = reference_weights[name].to(device=current_param.device)
                current_deltas[name] = current_param - initial_weight
        return current_deltas
    
    def _get_current_deltas_by_group(self, reference_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calcula a norma L2 dos deltas atuais, agrupados por prefixo."""
        deltas = self._calculate_current_deltas(reference_weights)
        
        # Agrupa os deltas (ainda como tensores) pela norma quadrada
        deltas_sq: Dict[str, torch.Tensor] = defaultdict(lambda: torch.tensor(0.0, device=self._get_target_device()))
        for name, delta_tensor in deltas.items():
            # O 'name' aqui já é o prefixo do grupo
            deltas_sq[name] += torch.norm(delta_tensor.float(), p=2).pow(2)
            
        return {k: torch.sqrt(v) for k, v in deltas_sq.items()}

    def _load_reference_pattern_from_file(self, pattern_path: str):
        """Carrega o padrão de referência de um arquivo JSON e calcula a norma total."""
        try:
            with open(pattern_path, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)

            # Converte os dados carregados em tensores no device da CPU por enquanto
            self.reference_deltas = {
                key: torch.tensor(value, dtype=torch.float32, device="cpu")
                for epoch_data in loaded_data.values() for key, value in epoch_data.items()
            }
            
            # Recalcula a norma de referência total.
            if self.reference_deltas:
                total_norm_sq = sum(v.pow(2).sum() for v in self.reference_deltas.values())
                self.reference_delta_norm = torch.sqrt(total_norm_sq).item()
                logFun(f"[TrainGPS] Padrão de referência carregado com {len(self.reference_deltas)} deltas. Norma L2 total: {self.reference_delta_norm:.4f}", lvl="success")
            else:
                logFun("[TrainGPS] Padrão de referência carregado, mas está vazio.", lvl="warning")
        
        except Exception as e:
            logFun(f"[TrainGPS] Falha ao carregar ou processar padrão de referência: {e}", lvl="error")
            traceback.print_exc()
            self.reference_deltas = {}
            self.reference_delta_norm = None