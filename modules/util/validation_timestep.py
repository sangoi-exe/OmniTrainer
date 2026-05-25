from __future__ import annotations

import hashlib
import random
from typing import Any

import torch
from torch import Tensor

from modules.util.enum.ValidationTimestepMode import ValidationTimestepMode


TIMESTEP_TAG = "validation-timestep"
NOISE_TAG = "validation-noise"


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, Tensor):
        return int(value.detach().cpu().flatten()[0].item())
    return int(value)


def _stable_seed(seed: int, tag: str, index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{tag}:{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def parse_validation_timestep_values(values: str) -> list[int]:
    if values is None or str(values).strip() == "":
        raise ValueError("validation_timestep_values must not be empty")

    parsed_values: list[int] = []
    for raw_value in str(values).split(","):
        raw_value = raw_value.strip()
        if raw_value == "":
            raise ValueError("validation_timestep_values contains an empty value")
        parsed_values.append(int(raw_value))

    if any(value < 0 for value in parsed_values):
        raise ValueError("validation_timestep_values must be non-negative")
    return parsed_values


def apply_timestep_shift_unit(position: float, shift: float) -> float:
    if shift == 1.0:
        return position
    return shift * position / ((shift - 1.0) * position + 1.0)


def stratified_unit_position(index: int, count: int, seed: int) -> float:
    if count <= 0:
        raise ValueError("validation sample count must be positive")

    jitter = random.Random(_stable_seed(seed, TIMESTEP_TAG, index)).random()
    return (index + jitter) / count


def validation_noise_seed(index: int, seed: int) -> int:
    return _stable_seed(seed, NOISE_TAG, index) % (2**31)


def resolve_discrete_validation_timestep(
    config: TrainConfig,
    num_train_timesteps: int,
    batch_size: int,
    device: torch.device,
    shift: float,
    validation_index: Any | None,
    validation_count: Any | None,
) -> Tensor:
    if num_train_timesteps <= 0:
        raise ValueError("num_train_timesteps must be positive")

    match config.validation_timestep_mode:
        case ValidationTimestepMode.AUTO:
            timestep = int(num_train_timesteps * 0.5) - 1
        case ValidationTimestepMode.FIXED:
            timesteps = parse_validation_timestep_values(config.validation_timestep_values)
            index = _coerce_optional_int(validation_index) or 0
            timestep = timesteps[index % len(timesteps)]
            if timestep >= num_train_timesteps:
                raise ValueError(
                    f"fixed validation timestep {timestep} is outside scheduler range 0-{num_train_timesteps - 1}"
                )
        case ValidationTimestepMode.STRATIFIED:
            index = _coerce_optional_int(validation_index)
            count = _coerce_optional_int(validation_count)
            if index is None or count is None:
                raise ValueError("stratified validation timesteps require validation index and count")
            position = stratified_unit_position(index, count, config.validation_timestep_seed)
            timestep = int(apply_timestep_shift_unit(position, shift) * num_train_timesteps)
        case _:
            raise NotImplementedError(f"validation timestep mode {config.validation_timestep_mode} is not implemented")

    timestep = max(0, min(num_train_timesteps - 1, timestep))
    return torch.full((batch_size,), timestep, dtype=torch.long, device=device)


def resolve_continuous_validation_timestep(
    config: TrainConfig,
    batch_size: int,
    device: torch.device,
    validation_index: Any | None,
    validation_count: Any | None,
) -> Tensor:
    match config.validation_timestep_mode:
        case ValidationTimestepMode.AUTO:
            position = 0.5
        case ValidationTimestepMode.FIXED:
            timesteps = parse_validation_timestep_values(config.validation_timestep_values)
            index = _coerce_optional_int(validation_index) or 0
            timestep = timesteps[index % len(timesteps)]
            if timestep > 10000:
                raise ValueError("fixed continuous validation timestep must be between 0 and 10000")
            position = timestep / 10000.0
        case ValidationTimestepMode.STRATIFIED:
            index = _coerce_optional_int(validation_index)
            count = _coerce_optional_int(validation_count)
            if index is None or count is None:
                raise ValueError("stratified validation timesteps require validation index and count")
            position = stratified_unit_position(index, count, config.validation_timestep_seed)
        case _:
            raise NotImplementedError(f"validation timestep mode {config.validation_timestep_mode} is not implemented")

    position = max(0.0, min(1.0, position))
    return torch.full((batch_size,), position, device=device)
