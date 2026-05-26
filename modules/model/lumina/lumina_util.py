from __future__ import annotations

from pathlib import Path

from modules.model.lumina.flux_autoencoder import AutoEncoderParams, LuminaAutoEncoder
from modules.model.lumina.lumina_models import NextDiT, NextDiT_2B_GQA_patch2_Adaln_Refiner

import torch

from transformers import Gemma2Config, Gemma2Model

from accelerate import init_empty_weights
from safetensors.torch import load_file

MODEL_VERSION_LUMINA_V2 = "lumina2"
GEMMA_TOKENIZER_ID = "google/gemma-2-2b"
LUMINA_NUM_TRAIN_TIMESTEPS = 1000

_GEMMA2_CONFIG = {
    "_name_or_path": "google/gemma-2-2b",
    "architectures": ["Gemma2Model"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_logit_softcapping": 50.0,
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "eos_token_id": 1,
    "final_logit_softcapping": 30.0,
    "head_dim": 256,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 2304,
    "initializer_range": 0.02,
    "intermediate_size": 9216,
    "max_position_embeddings": 8192,
    "model_type": "gemma2",
    "num_attention_heads": 8,
    "num_hidden_layers": 26,
    "num_key_value_heads": 4,
    "pad_token_id": 0,
    "query_pre_attn_scalar": 256,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000.0,
    "sliding_window": 4096,
    "torch_dtype": "float32",
    "transformers_version": "4.44.2",
    "use_cache": True,
    "vocab_size": 256000,
}

_LUMINA_AE_PARAMS = AutoEncoderParams(
    resolution=256,
    in_channels=3,
    ch=128,
    out_ch=3,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    z_channels=16,
    scale_factor=0.3611,
    shift_factor=0.1159,
)


def _require_checkpoint(path: str, label: str) -> str:
    if not path:
        raise ValueError(f"Lumina {label} checkpoint path is required")
    checkpoint_path = Path(path)
    if checkpoint_path.suffix != ".safetensors":
        raise ValueError(f"Lumina {label} checkpoint must use the .safetensors extension: {path}")
    if not checkpoint_path.is_file():
        raise ValueError(f"Lumina {label} checkpoint must be an existing .safetensors file: {path}")
    return str(checkpoint_path)


def _load_safetensors(path: str, device: torch.device | str, dtype: torch.dtype | None) -> dict[str, torch.Tensor]:
    state_dict = load_file(path, device=str(device))
    if dtype is None:
        return state_dict
    return {key: value.to(dtype=dtype) if value.is_floating_point() else value for key, value in state_dict.items()}


def _load_state_dict_or_raise(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    label: str,
    checkpoint_path: str,
) -> None:
    incompatible_keys = module.load_state_dict(state_dict, strict=False, assign=True)
    missing_keys = incompatible_keys.missing_keys
    unexpected_keys = incompatible_keys.unexpected_keys
    if not missing_keys and not unexpected_keys:
        return

    details = []
    if missing_keys:
        missing_preview = ", ".join(missing_keys[:8])
        if len(missing_keys) > 8:
            missing_preview += f", ... ({len(missing_keys)} total)"
        details.append(f"missing keys: {missing_preview}")
    if unexpected_keys:
        unexpected_preview = ", ".join(unexpected_keys[:8])
        if len(unexpected_keys) > 8:
            unexpected_preview += f", ... ({len(unexpected_keys)} total)"
        details.append(f"unexpected keys: {unexpected_preview}")

    raise ValueError(
        f"Lumina {label} checkpoint does not match the expected model keys: {checkpoint_path}. "
        + "; ".join(details)
    )


def load_lumina_transformer(
    checkpoint_path: str,
    dtype: torch.dtype | None,
    device: torch.device | str,
) -> NextDiT:
    checkpoint_path = _require_checkpoint(checkpoint_path, "transformer")
    with torch.device("meta"):
        model = NextDiT_2B_GQA_patch2_Adaln_Refiner()
        if dtype is not None:
            model = model.to(dtype=dtype)

    state_dict = _load_safetensors(checkpoint_path, device, dtype)
    if "model.diffusion_model.cap_embedder.0.weight" in state_dict:
        state_dict = {
            key.replace("model.diffusion_model.", ""): value
            for key, value in state_dict.items()
            if key.startswith("model.diffusion_model.")
        }

    _load_state_dict_or_raise(model, state_dict, "transformer", checkpoint_path)
    return model


def load_lumina_autoencoder(
    checkpoint_path: str,
    dtype: torch.dtype | None,
    device: torch.device | str,
) -> LuminaAutoEncoder:
    checkpoint_path = _require_checkpoint(checkpoint_path, "autoencoder")
    with torch.device("meta"):
        autoencoder = LuminaAutoEncoder(_LUMINA_AE_PARAMS)
        if dtype is not None:
            autoencoder = autoencoder.to(dtype=dtype)

    state_dict = _load_safetensors(checkpoint_path, device, dtype)
    if "vae.decoder.conv_in.bias" in state_dict:
        state_dict = {
            key.replace("vae.", ""): value for key, value in state_dict.items() if key.startswith("vae.")
        }

    _load_state_dict_or_raise(autoencoder, state_dict, "autoencoder", checkpoint_path)
    return autoencoder


def load_lumina_gemma2(
    checkpoint_path: str,
    dtype: torch.dtype | None,
    device: torch.device | str,
) -> Gemma2Model:
    checkpoint_path = _require_checkpoint(checkpoint_path, "text encoder")
    config = Gemma2Config(**_GEMMA2_CONFIG)
    with init_empty_weights():
        text_encoder = Gemma2Model._from_config(config)

    state_dict = _load_safetensors(checkpoint_path, device, dtype)
    for key in list(state_dict.keys()):
        new_key = key.replace("model.", "")
        if new_key == key:
            break
        state_dict[new_key] = state_dict.pop(key)

    if "text_encoders.gemma2_2b.logit_scale" in state_dict:
        state_dict = {
            key.replace("text_encoders.gemma2_2b.transformer.model.", ""): value
            for key, value in state_dict.items()
            if key.startswith("text_encoders.gemma2_2b.transformer.model.")
        }

    _load_state_dict_or_raise(text_encoder, state_dict, "text encoder", checkpoint_path)
    return text_encoder
