import json
import math
import os
import traceback
from collections.abc import Iterable
from typing import TYPE_CHECKING

from modules.sangoi.logFun import logFun
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TrainGPSPenaltyMetric import TrainGPSPenaltyMetric
from modules.util.enum.TrainingMethod import TrainingMethod

import torch

if TYPE_CHECKING:
    from modules.util.NamedParameterGroup import NamedParameterGroupCollection


def _parse_penalty_metric(penalty_metric: TrainGPSPenaltyMetric | str) -> TrainGPSPenaltyMetric:
    if isinstance(penalty_metric, TrainGPSPenaltyMetric):
        return penalty_metric
    if isinstance(penalty_metric, str):
        normalized_metric = penalty_metric.upper()
        for metric in TrainGPSPenaltyMetric:
            if normalized_metric in {metric.name, metric.value.upper()}:
                return metric
    raise ValueError(f"invalid TrainGPS penalty metric: {penalty_metric!r}")


def validate_train_gps_support(config: TrainConfig):
    if not (config.train_gps_save_it or config.train_gps_use_it):
        return
    if not config.model_type.is_stable_diffusion_xl() or config.training_method != TrainingMethod.LORA:
        raise ValueError("TrainGPS is only supported for SDXL LoRA training")
    if config.train_gps_use_it and not config.train_gps_path:
        raise ValueError("TrainGPS use mode requires train_gps_path")


class TrainGPS:
    def __init__(
        self,
        model: torch.nn.Module,
        param_collection: "NamedParameterGroupCollection",
        penalty_metric: TrainGPSPenaltyMetric | str = TrainGPSPenaltyMetric.MSE,
    ):
        if param_collection is None:
            raise ValueError("param_collection cannot be None for TrainGPS")

        self.model = model
        self.param_collection: NamedParameterGroupCollection = param_collection
        self.penalty_metric = _parse_penalty_metric(penalty_metric)

        self.save_run_initial_weights: dict[str, list[torch.Tensor]] = {}
        self.use_run_initial_weights: dict[str, list[torch.Tensor]] = {}
        self.reference_deltas: dict[str, dict[str, torch.Tensor]] = {}
        self.delta_records: list[dict[str, int | str | float]] = []

        self.current_total_delta_norm: float | None = None
        self.reference_delta_norm: float | None = None

    def setup_for_save(self):
        logFun("[TrainGPS] Capturing initial weights for delta logging.", lvl="TRAINGPS")
        self.save_run_initial_weights = self._capture_weights_generic(torch.device("cpu"))

    def setup_for_use(self, pattern_path: str):
        logFun(f"[TrainGPS] Loading delta reference from '{pattern_path}'.", lvl="TRAINGPS")
        if not pattern_path or not os.path.isfile(pattern_path):
            raise FileNotFoundError(f"TrainGPS delta reference not found: {pattern_path}")

        self._load_reference_pattern_from_file(pattern_path)

        target_device = self._get_target_device()
        self.use_run_initial_weights = self._capture_weights_generic(target_device, torch.bfloat16)

    def log_epoch_deltas(self, epoch: int):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("TrainGPS epoch must be a non-negative integer")
        if not self.save_run_initial_weights:
            logFun("[TrainGPS] Skipping delta log because initial weights were not captured.", lvl="warning")
            return

        with torch.no_grad():
            current_deltas = self._calculate_current_deltas(self.save_run_initial_weights)
            for group_name in sorted(current_deltas):
                norm_squared = 0.0
                for delta_tensor in current_deltas[group_name]:
                    norm_squared += torch.linalg.vector_norm(delta_tensor.float()).pow(2).item()
                self.delta_records.append(
                    {
                        "epoch": epoch,
                        "group": group_name,
                        "delta_norm": norm_squared**0.5,
                    }
                )

        logFun(f"[TrainGPS] Logged deltas for epoch_{epoch}.", lvl="info")

    def compute_penalty(self, lambda_weight: float) -> torch.Tensor:
        target_device = self._get_target_device()

        if not self.reference_deltas or not self.use_run_initial_weights:
            self.current_total_delta_norm = None
            return torch.tensor(0.0, device=target_device)

        current_deltas_by_group = self._get_current_deltas_by_group(self.use_run_initial_weights)
        current_epoch = self.model.train_progress.epoch
        reference_key_prefix = f"epoch_{current_epoch}"
        relevant_reference_deltas = self.reference_deltas.get(reference_key_prefix, {})

        if not relevant_reference_deltas:
            raise ValueError(f"TrainGPS reference has no records for current epoch: {reference_key_prefix}")

        current_groups = set(current_deltas_by_group)
        reference_groups = set(relevant_reference_deltas)
        missing_reference_groups = sorted(current_groups - reference_groups)
        missing_current_groups = sorted(reference_groups - current_groups)
        if missing_reference_groups or missing_current_groups:
            raise ValueError(
                "TrainGPS reference groups must match live parameter groups; "
                f"missing_reference={missing_reference_groups}, missing_current={missing_current_groups}"
            )

        common_keys = sorted(reference_groups)
        if not common_keys:
            raise ValueError(f"TrainGPS reference epoch has no matching parameter groups: {reference_key_prefix}")
        if self.penalty_metric == TrainGPSPenaltyMetric.COSINE and len(common_keys) < 2:
            raise ValueError("TrainGPS COSINE metric requires at least two common reference groups")

        ref_vec = torch.stack(
            [relevant_reference_deltas[key].to(device=target_device, dtype=torch.float32) for key in common_keys]
        )
        cur_vec = torch.stack([current_deltas_by_group[key].to(dtype=torch.float32) for key in common_keys])

        self.current_total_delta_norm = float(torch.linalg.vector_norm(cur_vec.detach()))

        if self.penalty_metric == TrainGPSPenaltyMetric.COSINE:
            penalty = 1.0 - torch.nn.functional.cosine_similarity(cur_vec.flatten(), ref_vec.flatten(), dim=0, eps=1e-8)
        else:
            penalty = torch.nn.functional.mse_loss(cur_vec, ref_vec)

        return (lambda_weight * penalty).to(dtype=self._get_target_dtype())

    def save_log_to_file(self, path: str):
        if not self.delta_records:
            raise ValueError("TrainGPS save requested before any epoch delta records were logged")

        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        payload = {
            "schema_version": 1,
            "kind": "sangoi_train_gps_delta_log",
            "metric": self.penalty_metric.value,
            "records": sorted(self.delta_records, key=lambda record: (record["epoch"], record["group"])),
        }

        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            logFun(f"[TrainGPS] Saved delta log to {path}", lvl="success")
        except Exception as error:
            logFun(f"[TrainGPS] Failed to save delta log: {error}", lvl="error")
            traceback.print_exc()
            raise

    def get_delta_norms_for_logging(self) -> tuple[float | None, float | None]:
        return self.current_total_delta_norm, self.reference_delta_norm

    def _iterate_param_groups(self, detach: bool) -> Iterable[tuple[str, list[torch.Tensor]]]:
        for unique_name in self.param_collection.unique_name_mapping:
            group = self.param_collection.by_unique_name(unique_name)
            if group is None:
                raise RuntimeError(f"Missing TrainGPS parameter group: {unique_name}")

            yield unique_name, [param.detach() if detach else param for param in group.parameters]

    def _get_target_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _get_target_dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _capture_weights_generic(
        self, target_device: torch.device, target_dtype: torch.dtype | None = None
    ) -> dict[str, list[torch.Tensor]]:
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
            logFun("[TrainGPS] No parameters were captured.", lvl="warning")

        return storage_dict

    def _calculate_current_deltas(
        self, reference_weights: dict[str, list[torch.Tensor]]
    ) -> dict[str, list[torch.Tensor]]:
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

    def _get_current_deltas_by_group(self, reference_weights: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
        deltas = self._calculate_current_deltas(reference_weights)

        deltas_sq: dict[str, torch.Tensor] = {}
        for name, delta_tensors in deltas.items():
            for delta_tensor in delta_tensors:
                delta_norm_sq = torch.linalg.vector_norm(delta_tensor.float()).pow(2)
                if name in deltas_sq:
                    deltas_sq[name] = deltas_sq[name] + delta_norm_sq
                else:
                    deltas_sq[name] = delta_norm_sq

        return {name: torch.sqrt(norm_sq) for name, norm_sq in deltas_sq.items()}

    def _load_reference_pattern_from_file(self, pattern_path: str):
        try:
            with open(pattern_path, "r", encoding="utf-8") as handle:
                loaded_data = json.load(handle)

            if not isinstance(loaded_data, dict):
                raise ValueError("TrainGPS reference pattern must be a JSON object")
            expected_keys = {"schema_version", "kind", "metric", "records"}
            if set(loaded_data) != expected_keys:
                raise ValueError("TrainGPS reference pattern has an invalid schema")
            if loaded_data["schema_version"] != 1:
                raise ValueError("TrainGPS reference pattern has an unsupported schema_version")
            if loaded_data["kind"] != "sangoi_train_gps_delta_log":
                raise ValueError("TrainGPS reference pattern has an invalid kind")
            reference_metric = _parse_penalty_metric(loaded_data["metric"])
            if reference_metric != self.penalty_metric:
                raise ValueError(
                    "TrainGPS reference metric must match configured metric; "
                    f"reference={reference_metric.value}, configured={self.penalty_metric.value}"
                )

            records = loaded_data["records"]
            if not isinstance(records, list) or not records:
                raise ValueError("TrainGPS reference pattern records must be a non-empty list")

            reference_deltas: dict[str, dict[str, torch.Tensor]] = {}
            seen_records = set()
            for record in records:
                if not isinstance(record, dict) or set(record) != {"epoch", "group", "delta_norm"}:
                    raise ValueError("TrainGPS reference records must contain epoch, group, and delta_norm")

                epoch = record["epoch"]
                group_name = record["group"]
                delta_norm = record["delta_norm"]

                if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
                    raise ValueError("TrainGPS reference epoch must be a non-negative integer")
                if not isinstance(group_name, str) or not group_name:
                    raise ValueError("TrainGPS reference group must be a non-empty string")
                if isinstance(delta_norm, bool) or not isinstance(delta_norm, int | float):
                    raise ValueError("TrainGPS reference delta_norm must be numeric")
                delta_norm = float(delta_norm)
                if not math.isfinite(delta_norm) or delta_norm < 0.0:
                    raise ValueError("TrainGPS reference delta_norm must be finite and non-negative")

                record_key = (epoch, group_name)
                if record_key in seen_records:
                    raise ValueError("TrainGPS reference contains duplicate epoch/group records")
                seen_records.add(record_key)

                epoch_key = f"epoch_{epoch}"
                reference_deltas.setdefault(epoch_key, {})[group_name] = torch.tensor(
                    delta_norm, dtype=torch.float32, device="cpu"
                )

            self.reference_deltas = reference_deltas
            all_reference_values = [
                value for epoch_data in self.reference_deltas.values() for value in epoch_data.values()
            ]
            self.reference_delta_norm = torch.linalg.vector_norm(torch.stack(all_reference_values)).item()
            logFun(
                f"[TrainGPS] Loaded {len(all_reference_values)} reference deltas. "
                f"L2 norm: {self.reference_delta_norm:.4f}",
                lvl="success",
            )

        except Exception as error:
            logFun(f"[TrainGPS] Failed to load reference pattern: {error}", lvl="error")
            traceback.print_exc()
            self.reference_deltas = {}
            self.reference_delta_norm = None
            raise
