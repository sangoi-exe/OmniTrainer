import json
import traceback
import os
import warnings
import torch
from torch import Tensor

from typing import Iterable, Tuple, Dict, Optional, TYPE_CHECKING, List # Adicionado List

from modules.sangoi.logFun import logFun

if TYPE_CHECKING:
    from modules.util.NamedParameterGroup import NamedParameterGroupCollection

class TrainGPS:
    def __init__(
        self, model: torch.nn.Module, param_collection: "NamedParameterGroupCollection", penalty_metric: str = "cosine"
    ):
        if param_collection is None:
            raise ValueError(
                "param_collection não pode ser None para TrainGPS."
            )
        self.model = model
        self.param_collection: "NamedParameterGroupCollection" = param_collection
        self.delta_log_by_module: Dict[str, Dict[str, float]] = {}
        self.initial_weights_run1: Dict[str, torch.Tensor] = (
            {}
        )  # Pesos no início da Run 1 (para salvar) - Em CPU
        self.initial_weights_run2: Dict[str, torch.Tensor] = (
            {}
        )  # Pesos no início da Run 2 (para calcular penalidade) - Em CPU
        self.reference_deltas: Dict[str, torch.Tensor] = (
            {}
        )  # Deltas carregados da Run 1 - Em CPU
        self.current_total_delta_norm: Optional[float] = (
            None  # Cache da norma L2 do delta atual
        )
        self.reference_delta_norm: Optional[float] = (
            None  # Cache da norma L2 do delta de referência
        )
        self.penalty_metric: str = penalty_metric.lower()
        if self.penalty_metric not in {"mse", "cosine"}:
            raise ValueError("penalty_metric deve ser 'mse' ou 'cosine'")

    def _iterate_params(self) -> Iterable[Tuple[str, torch.Tensor, torch.device]]:
        """Iterates through tensors within the state_dicts of relevant LoRA wrappers."""
        # agora consideramos o wrapper que carrega os módulos LoRA
        wrappers_to_check = [
            getattr(self.model, "unet_lora", None),  # se você renomear assim
        ]
        processed_keys = (
            set()
        )  # Evita processar a mesma chave de diferentes wrappers (improvável mas seguro)

        for wrapper in wrappers_to_check:
            if wrapper is None:
                continue
            try:
                # state_dict() do wrapper deve retornar chaves de módulos reais/dummies
                wrapper_state_dict = wrapper.state_dict()
                for key, tensor in wrapper_state_dict.items():
                    # Verifica se é um tensor e ainda não foi processado
                    if isinstance(tensor, torch.Tensor) and key not in processed_keys:
                        # Assume que todos os tensores no state_dict são relevantes
                        # Retorna tensor destacado (detach) para evitar problemas de grafo
                        yield key, tensor.detach(), tensor.device
                        processed_keys.add(key)
            except Exception as e:
                logFun(f"[TrainGPS] Erro ao iterar state_dict para wrapper {getattr(wrapper, 'prefix', 'Unknown')}: {e}")

    def capture_weights(self):
        """Captura os pesos iniciais (usado no início da Run 1) e armazena em CPU."""
        self.initial_weights_run1 = {}  # Limpa antes de capturar
        count = 0
        try:
            for key, param, _ in self._iterate_params():
                self.initial_weights_run1[key] = param.detach().cpu()  # Armazena em CPU
                count += 1
        except Exception as e:
            logFun(f"[TrainGPS] Erro durante capture_weights: {e}", lvl="error")
            traceback.print_exc()
            raise  # Re-levanta a exceção para indicar falha

        if count == 0:
            logFun("[TrainGPS] capture_weights não encontrou parâmetros treináveis.", lvl="warning")
        else:
            logFun(f"[TrainGPS] Capturou pesos iniciais (Run 1) para {count} parâmetros treináveis.", lvl="success")
            # Opcional: Calcular e logar norma inicial aqui se desejado
            initial_norm = self._calculate_total_norm(self.initial_weights_run1)
            logFun(f"[TrainGPS] Norma L2 total dos pesos iniciais (Run 1): {initial_norm:.4f}", lvl="info")

    def capture_initial_weights_run2(self):
        """Captura os pesos iniciais da Run 2 (usado para cálculo da penalidade) e armazena em CPU."""
        # Armazena já no device do modelo (GPU) e em float16 para economizar memória/banda
        target_device = next(self._iterate_params())[2]
        self.initial_weights_run2 = {}
        count = 0
        try:
            for key, param, _ in self._iterate_params():
                # detach, mover para GPU e converter em half
                self.initial_weights_run2[key] = (
                    param.detach().clone() .to(device=target_device, dtype=torch.bfloat16, non_blocking=True)
                )
                count += 1
        except Exception as e:
            logFun(f"[TrainGPS] Erro durante capture_initial_weights_run2: {e}", lvl="error")
            traceback.print_exc()
            raise

        if count == 0:
            logFun("[TrainGPS] capture_initial_weights_run2 não encontrou parâmetros treináveis.", lvl="warning")
        else:
            logFun(f"[TrainGPS] Capturou pesos iniciais (Run 2) para {count} parâmetros treináveis.", lvl="success")
            # Opcional: Calcular e logar norma inicial aqui se desejado
            initial_norm_run2 = self._calculate_total_norm(self.initial_weights_run2)
            logFun(f"[TrainGPS] Norma L2 total dos pesos iniciais (Run 2): {initial_norm_run2:.4f}", lvl="info")

    def load_reference_pattern(self, pattern_path: str):
        """Carrega o padrão de delta de referência de um arquivo JSON."""
        self.reference_deltas = {}
        self.reference_delta_norm = None

        if not os.path.isfile(pattern_path):
            logFun(f"[TrainGPS] Arquivo JSON não encontrado: {pattern_path}", lvl="error")
            return

        try:
            with open(pattern_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            target_device = next(self._iterate_params())[2]
            flat_dict = {}
            for epoch_key, metrics in json_data.items():
                for param_name, value in metrics.items():
                    full_key = f"{epoch_key}/{param_name}"
                # cria no device do modelo e em float16
                flat_dict[full_key] = (
                    torch.tensor(value, dtype=torch.float32).to(device=target_device, dtype=torch.bfloat16, non_blocking=True)
                )

            if flat_dict:
                self.reference_deltas = flat_dict
                self.reference_delta_norm = self._calculate_total_norm(
                    self.reference_deltas
                )
                epoch_numbers = [
                    int(k.split("/")[0].replace("epoch_", ""))
                    for k in flat_dict.keys()
                    if k.startswith("epoch_")
                ]
                self.max_epoch_loaded = max(epoch_numbers) if epoch_numbers else None
                logFun(f"[TrainGPS] JSON '{pattern_path}' carregado com {len(flat_dict)} deltas.", lvl="success")
                logFun(f"[TrainGPS] Último epoch registrado no JSON: {self.max_epoch_loaded}", lvl="info")
                logFun(f"[TrainGPS] Norma L2 total do delta de referência: {self.reference_delta_norm:.4f}", lvl="info")
            else:
                logFun("[TrainGPS] JSON estava vazio ou mal formatado.", lvl="warning")
        except Exception as e:
            logFun(f"[TrainGPS] Falha ao carregar JSON de deltas: {e}", lvl="error")
            traceback.print_exc()
            self.reference_deltas = {}

    def compute_penalty(self, lambda_weight: float) -> torch.Tensor:
        """
        Calcula penalidade entre delta atual (por módulo) e delta de referência.
        Métrica definida em `self.penalty_metric` ("mse" ou "cosine").
        """
        with torch.no_grad():
            # ── descobrir device/dtype ───────────────────────────────────────────
            try:
                first_param = next(iter(self.param_collection.parameters()))
                target_device, target_dtype = first_param.device, first_param.dtype
            except StopIteration:
                target_device, target_dtype = torch.device("cpu"), torch.float32
                logFun("[TrainGPS] Parâmetros vazios; usando CPU/float32.", lvl="warning")

            # ── early‑exit se epoch > padrão carregado ───────────────────────────
            if hasattr(self, "max_epoch_loaded") and hasattr(self.model, "train_progress"):
                if self.max_epoch_loaded is not None and self.model.train_progress.epoch > self.max_epoch_loaded:
                    return torch.tensor(0.0, device=target_device, dtype=target_dtype)

            # ── pré‑condições ────────────────────────────────────────────────────
            if not self.reference_deltas or not self.initial_weights_run2:
                self.current_total_delta_norm = None
                return torch.tensor(0.0, device=target_device, dtype=target_dtype)

            try:
                # 1) Obtém deltas atuais por módulo
                cur_mod = self._get_current_module_deltas(target_device, target_dtype)
                # 2) Determina quais módulos têm referência e atual
                # (aqui criamos ref_mod para mapear “epoch_X/<module>” → tensor)
                ref_mod = {
                    full_key.split("/", 1)[1]: val.to(device=target_device, dtype=target_dtype, non_blocking=True)
                    for full_key, val in self.reference_deltas.items()
                }
                common_keys = [k for k in ref_mod if k in cur_mod]
                # Se não há nada em comum, já retorna zero
                if not common_keys:
                    self.current_total_delta_norm = 0.0
                    return torch.tensor(0.0, device=target_device, dtype=target_dtype)
                # 3) Empilha vetores em batch único (vetorização)
                ref_vec = torch.stack([ref_mod[k].float() for k in common_keys], dim=0)
                cur_vec = torch.stack([cur_mod[k].float() for k in common_keys], dim=0)

                if self.penalty_metric == "cosine":
                    cos_sim = torch.nn.functional.cosine_similarity(cur_vec, ref_vec, dim=0, eps=1e-8)
                    penalty = 1.0 - cos_sim  # distância angular
                else:  # "mse"
                    penalty = torch.nn.functional.mse_loss(cur_vec, ref_vec)

                self.current_total_delta_norm = float(torch.norm(cur_vec, p=2))
                return (lambda_weight * penalty).to(dtype=target_dtype)

            except Exception as e:
                logFun(f"[TrainGPS] Erro compute_penalty: {e}", lvl="error")
                traceback.print_exc()
                self.current_total_delta_norm = None
                return torch.tensor(0.0, device=target_device, dtype=target_dtype)

    def _calculate_total_norm(
        self, weight_dict: Dict[str, Tensor], device: torch.device = torch.device("cpu")
    ) -> float:
        """Calcula a norma L2 total sobre todos os tensores no dicionário."""
        if not weight_dict:
            return 0.0
        # Usa float64 para precisão na soma, no device especificado
        total_norm_sq = torch.tensor(0.0, dtype=torch.float32, device=device)
        for tensor in weight_dict.values():
            # Garante que o tensor está no device correto e calcula a norma quadrada
            # Usa .float() para norm que pode não suportar bf16 diretamente
            total_norm_sq += (
                torch.norm(tensor.to(device=device, non_blocking=True).float(), p=2)
                .pow(2)
                .double()
            )
        return torch.sqrt(total_norm_sq).item()  # .item() move para CPU

    def get_delta_norms(self) -> Tuple[Optional[float], Optional[float]]:
        """Retorna a norma L2 total cacheada do delta atual e do delta de referência."""
        return self.current_total_delta_norm, self.reference_delta_norm

    def log_group_deltas(self, epoch: int):
        """
        Salva norma L2 do delta para cada grupo (agrupados por prefixo simples) no epoch atual.
        Usa self.initial_weights_run1 como base.
        """
        if not self.initial_weights_run1:
            logFun("[TrainGPS] log_group_deltas: pesos iniciais não capturados.", lvl="warning")
            return

        epoch_key = f"epoch_{epoch}"
        self.delta_log_by_module[epoch_key] = {}
        # do_tensorboard = hasattr(self, "tensorboard") and self.tensorboard is not None

        group_norms: Dict[str, float] = {}
        param_counts: Dict[str, int] = {}

        for name, current_param, _ in self._iterate_params():
            if name not in self.initial_weights_run1:
                continue

            initial_weight = self.initial_weights_run1[name].to(
                dtype=torch.float32, device=current_param.device
            )
            delta = current_param.detach().to(dtype=torch.float32) - initial_weight

            # Define o prefixo de agrupamento (ajuste aqui para granularidade desejada)
            prefix = name.split(".")[
                0
            ]

            group_norms.setdefault(prefix, 0.0)
            param_counts.setdefault(prefix, 0)

            group_norms[prefix] += torch.norm(delta, p=2).pow(2).item()
            param_counts[prefix] += 1

        for prefix, norm_sq in group_norms.items():
            count = param_counts[prefix]
            delta_norm = norm_sq**0.5 if count > 0 else 0.0
            self.delta_log_by_module[epoch_key][prefix] = delta_norm
        # if do_tensorboard:
        #     self.tensorboard.add_scalar(f"delta/by_group/{prefix}", delta_norm, epoch)

        logFun(f"[TrainGPS] Deltas logados para epoch {epoch_key}.", lvl="success")

    def save_group_deltas(
        self, path: str = "./training_pattern/delta_log_by_module.json"
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.delta_log_by_module, f, indent=2)
        logFun(f"[TrainGPS] Delta por grupo salvo em {path}", lvl="success")

    def _get_current_module_deltas(
        self, target_device: torch.device, target_dtype: torch.dtype
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula e retorna {prefixo_módulo: norma_L2_delta_atual} no device/dtype alvos.
        """
        deltas_sq: Dict[str, torch.Tensor] = {}

        for name, current_param, _ in self._iterate_params():
            if name not in self.initial_weights_run2:
                continue
            prefix = name.split(".")[0]

            initial = self.initial_weights_run2[name].to(
                device=target_device, dtype=target_dtype, non_blocking=True
            )
            delta = (current_param.detach().to(dtype=target_dtype) - initial).float()
            deltas_sq.setdefault(
                prefix,
                torch.tensor(0.0, device=target_device, dtype=torch.float32),
            )
            deltas_sq[prefix] += torch.norm(delta, p=2).pow(2)

        return {k: torch.sqrt(v) for k, v in deltas_sq.items()}