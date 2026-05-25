from collections import deque
from collections.abc import Mapping

import torch
from torch import Tensor


class LossTracker:
    def __init__(self, window_size: int = 100, use_mad: bool = False) -> None:
        self.window_size = window_size
        self.use_mad = use_mad
        self.mse_losses = deque(maxlen=window_size)
        self.mae_losses = deque(maxlen=window_size)
        self.log_cosh_losses = deque(maxlen=window_size)

    @property
    def sample_count(self) -> int:
        return len(self.mse_losses)

    def update(self, mse_loss: Tensor, mae_loss: Tensor, log_cosh_loss: Tensor) -> None:
        self.mse_losses.append(self.__as_scalar(mse_loss))
        self.mae_losses.append(self.__as_scalar(mae_loss))
        self.log_cosh_losses.append(self.__as_scalar(log_cosh_loss))

    def compute_z_scores(self, mse_loss: Tensor, mae_loss: Tensor, log_cosh_loss: Tensor) -> tuple[float, float, float]:
        mse_center, mse_scale = self.__compute_stats(self.mse_losses)
        mae_center, mae_scale = self.__compute_stats(self.mae_losses)
        log_cosh_center, log_cosh_scale = self.__compute_stats(self.log_cosh_losses)

        mse_z = (self.__as_scalar(mse_loss) - mse_center) / mse_scale
        mae_z = (self.__as_scalar(mae_loss) - mae_center) / mae_scale
        log_cosh_z = (self.__as_scalar(log_cosh_loss) - log_cosh_center) / log_cosh_scale

        return mse_z, mae_z, log_cosh_z

    @staticmethod
    def __as_scalar(value: Tensor) -> float:
        return value.detach().to(dtype=torch.float32).mean().item()

    def __compute_stats(self, values: deque) -> tuple[float, float]:
        if len(values) < 2:
            return 0.0, 1e-8

        tensor = torch.tensor(list(values), dtype=torch.float32)
        if self.use_mad:
            median = tensor.median()
            mad = torch.abs(tensor - median).median()
            return median.item(), max(mad.item(), 1e-8)

        return tensor.mean().item(), max(tensor.std(unbiased=False).item(), 1e-8)


class DynamicLossControl:
    def __init__(
        self,
        use_ema: bool = False,
        ema_decay: float = 0.9,
        outlier_threshold: float = 3.0,
        schedule_params: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.outlier_threshold = outlier_threshold
        self.schedule_params = self.__initialize_schedule_params(schedule_params)
        self.ema_weights = {"mse": 1.0, "mae": 1.0, "log_cosh": 1.0}
        self.initialized = False

    def adjust_weights(
        self,
        mse_z: float,
        mae_z: float,
        log_cosh_z: float,
        progress_fraction: float,
        mutate_ema: bool = True,
    ) -> tuple[float, float, float]:
        z_scores = {
            "mse": min(abs(mse_z), self.outlier_threshold),
            "mae": min(abs(mae_z), self.outlier_threshold),
            "log_cosh": min(abs(log_cosh_z), self.outlier_threshold),
        }

        total_z = max(sum(z_scores.values()), 1e-8)
        inverted_z = {loss_name: (total_z - z_score) / total_z for loss_name, z_score in z_scores.items()}
        total_inverted = max(sum(inverted_z.values()), 1e-8)
        base_weights = {loss_name: inverted / total_inverted for loss_name, inverted in inverted_z.items()}

        if self.use_ema:
            if mutate_ema:
                if not self.initialized:
                    self.ema_weights = dict(base_weights)
                    self.initialized = True
                else:
                    alpha = 1.0 - self.ema_decay
                    for loss_name in self.ema_weights:
                        self.ema_weights[loss_name] = (1.0 - alpha) * self.ema_weights[
                            loss_name
                        ] + alpha * base_weights[loss_name]

            if self.initialized:
                total_ema = max(sum(self.ema_weights.values()), 1e-8)
                base_weights = {loss_name: weight / total_ema for loss_name, weight in self.ema_weights.items()}

        progress_fraction = min(max(progress_fraction, 0.0), 1.0)
        scheduled_factors = {
            loss_name: params["start"] * (1.0 - progress_fraction) + params["end"] * progress_fraction
            for loss_name, params in self.schedule_params.items()
        }
        weighted = {loss_name: base_weights[loss_name] * scheduled_factors[loss_name] for loss_name in base_weights}
        total_weighted = max(sum(weighted.values()), 1e-8)
        final_weights = {loss_name: weight / total_weighted for loss_name, weight in weighted.items()}

        return final_weights["mse"], final_weights["mae"], final_weights["log_cosh"]

    @staticmethod
    def __initialize_schedule_params(
        schedule_params: Mapping[str, Mapping[str, float]] | None,
    ) -> dict[str, dict[str, float]]:
        defaults = {
            "mae": {"start": 0.6, "end": 0.0},
            "mse": {"start": 0.2, "end": 0.6},
            "log_cosh": {"start": 0.2, "end": 0.4},
        }

        if schedule_params is None:
            return defaults

        for loss_name, params in defaults.items():
            if loss_name in schedule_params:
                params.update(schedule_params[loss_name])

        return defaults
