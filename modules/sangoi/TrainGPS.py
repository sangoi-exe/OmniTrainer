import os
import json
import traceback
import torch
from typing import Iterable, Tuple, Dict, Optional, TYPE_CHECKING
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
        self.save_run_initial_weights: Dict[str, list[torch.Tensor]] = {}
        # Usado para calcular a penalidade contra o padrão de referência (Modo Aplicação/Refinamento)
        self.use_run_initial_weights: Dict[str, list[torch.Tensor]] = {}
        # Padrão de referência carregado de um arquivo
        self.reference_deltas: Dict[str, Dict[str, torch.Tensor]] = {}
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

        with torch.no_grad():
            current_deltas = self._calculate_current_deltas(self.save_run_initial_weights)

            group_norms_sq: Dict[str, float] = defaultdict(float)
            for group_name, delta_tensors in current_deltas.items():
                for delta_tensor in delta_tensors:
                    group_norms_sq[group_name] += torch.norm(delta_tensor.float(), p=2).pow(2).item()

        epoch_key = f"epoch_{epoch}"
        for group_name, norm_sq in group_norms_sq.items():
            delta_norm = norm_sq ** 0.5
            self.delta_log_by_module[epoch_key][group_name] = delta_norm

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

        current_deltas_by_group = self._get_current_deltas_by_group(self.use_run_initial_weights)

        current_epoch = self.model.train_progress.epoch
        reference_key_prefix = f"epoch_{current_epoch}"

        relevant_reference_deltas = self.reference_deltas.get(reference_key_prefix, {})

        if not relevant_reference_deltas:
            return torch.tensor(0.0, device=target_device)

        common_keys = [key for key in relevant_reference_deltas if key in current_deltas_by_group]
        if not common_keys:
            raise RuntimeError(f"TrainGPS reference epoch has no matching parameter groups: {reference_key_prefix}")

        ref_vec = torch.stack([
            relevant_reference_deltas[key].to(device=target_device, dtype=torch.float32)
            for key in common_keys
        ])
        cur_vec = torch.stack([
            current_deltas_by_group[key].to(dtype=torch.float32)
            for key in common_keys
        ])

        self.current_total_delta_norm = float(torch.norm(cur_vec.detach(), p=2))

        if self.penalty_metric == "cosine":
            penalty = 1.0 - torch.nn.functional.cosine_similarity(cur_vec.flatten(), ref_vec.flatten(), dim=0, eps=1e-8)
        else:
            penalty = torch.nn.functional.mse_loss(cur_vec, ref_vec)

        return (lambda_weight * penalty).to(dtype=self._get_target_dtype())

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
            raise

    def get_delta_norms_for_logging(self) -> Tuple[Optional[float], Optional[float]]:
        """Retorna as normas L2 cacheadas do delta atual e de referência para logging."""
        return self.current_total_delta_norm, self.reference_delta_norm

    # --- Métodos Privados de Lógica Interna ---

    def _iterate_param_groups(self, detach: bool) -> Iterable[Tuple[str, list[torch.Tensor]]]:
        """Iterador privado que abstrai a obtenção de parâmetros treináveis do modelo."""
        for unique_name in self.param_collection.unique_name_mapping:
            group = self.param_collection.by_unique_name(unique_name)
            if group is None:
                raise RuntimeError(f"Missing TrainGPS parameter group: {unique_name}")

            yield unique_name, [
                param.detach() if detach else param
                for param in group.parameters
            ]

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
    ) -> Dict[str, list[torch.Tensor]]:
        """Helper genérico para capturar os pesos atuais do modelo."""
        storage_dict = {}
        processed_params = 0
        for name, parameters in self._iterate_param_groups(detach=True):
            captured_params = []
            for param in parameters:
                captured_param = param.clone().to(device=target_device)
                if target_dtype:
                    captured_param = captured_param.to(dtype=target_dtype)
                captured_params.append(captured_param)
                processed_params += 1

            storage_dict[name] = captured_params
        
        if processed_params == 0:
            logFun("[TrainGPS] Aviso: _capture_weights_generic não encontrou parâmetros.", lvl="warning")
        
        return storage_dict
    
    def _calculate_current_deltas(self, reference_weights: Dict[str, list[torch.Tensor]]) -> Dict[str, list[torch.Tensor]]:
        """Helper privado para calcular deltas atuais contra um dicionário de referência."""
        current_deltas = {}
        for name, current_params in self._iterate_param_groups(detach=False):
            if name in reference_weights:
                initial_weights = reference_weights[name]
                if len(current_params) != len(initial_weights):
                    raise RuntimeError(f"TrainGPS parameter count changed for group: {name}")

                current_deltas[name] = [
                    current_param - initial_weight.to(device=current_param.device)
                    for current_param, initial_weight in zip(current_params, initial_weights, strict=True)
                ]
        return current_deltas
    
    def _get_current_deltas_by_group(self, reference_weights: Dict[str, list[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Calcula a norma L2 dos deltas atuais, agrupados por prefixo."""
        deltas = self._calculate_current_deltas(reference_weights)
        
        # Agrupa os deltas (ainda como tensores) pela norma quadrada.
        deltas_sq: Dict[str, torch.Tensor] = {}
        for name, delta_tensors in deltas.items():
            for delta_tensor in delta_tensors:
                delta_norm_sq = torch.norm(delta_tensor.float(), p=2).pow(2)
                if name in deltas_sq:
                    deltas_sq[name] = deltas_sq[name] + delta_norm_sq
                else:
                    deltas_sq[name] = delta_norm_sq
            
        return {k: torch.sqrt(v) for k, v in deltas_sq.items()}

    def _load_reference_pattern_from_file(self, pattern_path: str):
        """Carrega o padrão de referência de um arquivo JSON e calcula a norma total."""
        try:
            with open(pattern_path, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)

            reference_deltas: Dict[str, Dict[str, torch.Tensor]] = {}
            for epoch_key, epoch_data in loaded_data.items():
                if not isinstance(epoch_key, str) or not isinstance(epoch_data, dict):
                    raise ValueError("TrainGPS reference pattern must map epoch keys to group dictionaries")

                reference_deltas[epoch_key] = {}
                for group_name, value in epoch_data.items():
                    if not isinstance(group_name, str) or isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError("TrainGPS reference group values must be numeric")
                    reference_deltas[epoch_key][group_name] = torch.tensor(value, dtype=torch.float32, device="cpu")

            self.reference_deltas = reference_deltas
            
            # Recalcula a norma de referência total.
            if self.reference_deltas:
                total_norm_sq = sum(
                    value.pow(2).sum()
                    for epoch_data in self.reference_deltas.values()
                    for value in epoch_data.values()
                )
                self.reference_delta_norm = torch.sqrt(total_norm_sq).item()
                delta_count = sum(len(epoch_data) for epoch_data in self.reference_deltas.values())
                logFun(f"[TrainGPS] Padrão de referência carregado com {delta_count} deltas. Norma L2 total: {self.reference_delta_norm:.4f}", lvl="success")
            else:
                raise ValueError("TrainGPS reference pattern is empty")
        
        except Exception as e:
            logFun(f"[TrainGPS] Falha ao carregar ou processar padrão de referência: {e}", lvl="error")
            traceback.print_exc()
            self.reference_deltas = {}
            self.reference_delta_norm = None
            raise
