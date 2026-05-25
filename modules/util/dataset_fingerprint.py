from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from enum import Enum
from typing import Any

from modules.util.config.ConceptConfig import ConceptConfig


def normalize_config_for_fingerprint(config: Any) -> Any:
    if hasattr(config, "to_dict"):
        return normalize_config_for_fingerprint(config.to_dict())
    if isinstance(config, dict):
        return {str(key): normalize_config_for_fingerprint(value) for key, value in sorted(config.items())}
    if isinstance(config, list):
        return [normalize_config_for_fingerprint(value) for value in config]
    if isinstance(config, tuple):
        return [normalize_config_for_fingerprint(value) for value in config]
    if isinstance(config, Enum):
        return str(config)
    return config


def compute_concept_fingerprint(
    concepts: Iterable[ConceptConfig] | Iterable[dict[str, Any]] | None,
    concept_file_name: str | None,
) -> tuple[str, int]:
    if concepts is not None:
        concept_items = list(concepts)
    elif concept_file_name is not None and os.path.exists(concept_file_name):
        with open(concept_file_name, "r") as concept_file:
            concept_items = json.load(concept_file) or []
    else:
        concept_items = []

    payload = json.dumps(
        [normalize_config_for_fingerprint(concept) for concept in concept_items],
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(concept_items)
