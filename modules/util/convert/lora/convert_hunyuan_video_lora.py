from modules.util.convert.lora.convert_clip import map_clip
from modules.util.convert.lora.convert_llama import map_llama
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet, convert_to_omi, map_prefix_range
from modules.util.convert_util import convert as convert_util
from modules.util.convert_util import lora_qkv_fusion, lora_qkv_mlp_fusion

import torch
from torch import Tensor


def __map_token_refiner_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("self_attn_qkv.0", "attn.to_q", parent=key_prefix)]
    keys += [LoraConversionKeySet("self_attn_qkv.1", "attn.to_k", parent=key_prefix)]
    keys += [LoraConversionKeySet("self_attn_qkv.2", "attn.to_v", parent=key_prefix)]

    keys += [LoraConversionKeySet("self_attn_proj", "attn.to_out.0", parent=key_prefix)]
    keys += [LoraConversionKeySet("mlp.fc0", "ff.net.0.proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("mlp.fc2", "ff.net.2", parent=key_prefix)]
    keys += [LoraConversionKeySet("adaLN_modulation.1", "norm_out.linear", parent=key_prefix)]
    keys += [LoraConversionKeySet("norm1", "norm1", parent=key_prefix)]
    keys += [LoraConversionKeySet("norm2", "norm2", parent=key_prefix)]

    return keys


def __map_double_transformer_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("img_attn_qkv.0", "attn.to_q", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_attn_qkv.1", "attn.to_k", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_attn_qkv.2", "attn.to_v", parent=key_prefix)]

    keys += [LoraConversionKeySet("txt_attn_qkv.0", "attn.add_q_proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_attn_qkv.1", "attn.add_k_proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_attn_qkv.2", "attn.add_v_proj", parent=key_prefix)]

    keys += [LoraConversionKeySet("img_attn_proj", "attn.to_out.0", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_mlp.fc0", "ff.net.0.proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_mlp.fc2", "ff.net.2", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_mod.linear", "norm1.linear", parent=key_prefix)]

    keys += [LoraConversionKeySet("txt_attn_proj", "attn.to_add_out", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_mlp.fc0", "ff_context.net.0.proj", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_mlp.fc2", "ff_context.net.2", parent=key_prefix)]
    keys += [LoraConversionKeySet("txt_mod.linear", "norm1_context.linear", parent=key_prefix)]

    return keys


def __map_single_transformer_block(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("linear1.0", "attn.to_q", parent=key_prefix)]
    keys += [LoraConversionKeySet("linear1.1", "attn.to_k", parent=key_prefix)]
    keys += [LoraConversionKeySet("linear1.2", "attn.to_v", parent=key_prefix)]
    keys += [LoraConversionKeySet("linear1.3", "proj_mlp", parent=key_prefix)]

    keys += [LoraConversionKeySet("linear2", "proj_out", parent=key_prefix)]
    keys += [LoraConversionKeySet("modulation.linear", "norm.linear", parent=key_prefix)]

    return keys


def __map_transformer(key_prefix: LoraConversionKeySet) -> list[LoraConversionKeySet]:
    keys = []

    keys += [
        LoraConversionKeySet(
            "txt_in.c_embedder.linear_1", "context_embedder.time_text_embed.text_embedder.linear_1", parent=key_prefix
        )
    ]
    keys += [
        LoraConversionKeySet(
            "txt_in.c_embedder.linear_2", "context_embedder.time_text_embed.text_embedder.linear_2", parent=key_prefix
        )
    ]
    keys += [
        LoraConversionKeySet(
            "txt_in.t_embedder.linear_1",
            "context_embedder.time_text_embed.timestep_embedder.linear_1",
            parent=key_prefix,
        )
    ]
    keys += [
        LoraConversionKeySet(
            "txt_in.t_embedder.linear_2",
            "context_embedder.time_text_embed.timestep_embedder.linear_2",
            parent=key_prefix,
        )
    ]
    keys += [LoraConversionKeySet("txt_in.input_embedder", "context_embedder.proj_in", parent=key_prefix)]
    keys += [
        LoraConversionKeySet("final_layer.adaLN_modulation.1", "norm_out.linear", parent=key_prefix, swap_chunks=True)
    ]
    keys += [LoraConversionKeySet("final_layer.linear", "proj_out", parent=key_prefix)]
    keys += [LoraConversionKeySet("guidance_in.mlp.0", "time_text_embed.guidance_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("guidance_in.mlp.2", "time_text_embed.guidance_embedder.linear_2", parent=key_prefix)]
    keys += [LoraConversionKeySet("vector_in.in_layer", "time_text_embed.text_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("vector_in.out_layer", "time_text_embed.text_embedder.linear_2", parent=key_prefix)]
    keys += [LoraConversionKeySet("time_in.mlp.0", "time_text_embed.timestep_embedder.linear_1", parent=key_prefix)]
    keys += [LoraConversionKeySet("time_in.mlp.2", "time_text_embed.timestep_embedder.linear_2", parent=key_prefix)]
    keys += [LoraConversionKeySet("img_in.proj", "x_embedder.proj", parent=key_prefix)]

    for k in map_prefix_range(
        "txt_in.individual_token_refiner.blocks", "context_embedder.token_refiner.refiner_blocks", parent=key_prefix
    ):
        keys += __map_token_refiner_block(k)

    for k in map_prefix_range("double_blocks", "transformer_blocks", parent=key_prefix):
        keys += __map_double_transformer_block(k)

    for k in map_prefix_range("single_blocks", "single_transformer_blocks", parent=key_prefix):
        keys += __map_single_transformer_block(k)

    return keys


def convert_hunyuan_video_lora_key_sets() -> list[LoraConversionKeySet]:
    keys = []

    keys += [LoraConversionKeySet("bundle_emb", "bundle_emb")]
    keys += __map_transformer(LoraConversionKeySet("transformer", "lora_transformer"))
    keys += map_llama(LoraConversionKeySet("llama", "lora_te1"))
    keys += map_clip(LoraConversionKeySet("clip_l", "lora_te2"))

    return keys


_COMFYUI_QKV_SPLIT_ATTRS = ("linear1", "img_attn_qkv", "txt_attn_qkv")
_LORA_SUFFIXES = (".lora_down.weight", ".lora_up.weight", ".lora_A.weight", ".lora_B.weight", ".alpha", ".dora_scale")

_COMFYUI_BLOCK_PATTERNS = [
    (
        "transformer.double_blocks.{i}",
        "transformer.double_blocks.{i}",
        lora_qkv_fusion("img_attn_qkv.0", "img_attn_qkv.1", "img_attn_qkv.2", "img_attn_qkv")
        + lora_qkv_fusion("txt_attn_qkv.0", "txt_attn_qkv.1", "txt_attn_qkv.2", "txt_attn_qkv")
        + [
            ("img_attn_proj", "img_attn_proj"),
            ("img_mlp.fc0", "img_mlp.fc1"),
            ("img_mlp.fc2", "img_mlp.fc2"),
            ("img_mod.linear", "img_mod.linear"),
            ("txt_attn_proj", "txt_attn_proj"),
            ("txt_mlp.fc0", "txt_mlp.fc1"),
            ("txt_mlp.fc2", "txt_mlp.fc2"),
            ("txt_mod.linear", "txt_mod.linear"),
        ],
    ),
    (
        "transformer.single_blocks.{i}",
        "transformer.single_blocks.{i}",
        lora_qkv_mlp_fusion("linear1.0", "linear1.1", "linear1.2", "linear1.3", "linear1")
        + [
            ("linear2", "linear2"),
            ("modulation.linear", "modulation.linear"),
        ],
    ),
]


def convert_hunyuan_video_lora_to_comfyui(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    omi_state_dict = convert_to_omi(state_dict, convert_hunyuan_video_lora_key_sets())
    block_state_dict = {
        key: value
        for key, value in omi_state_dict.items()
        if ".double_blocks." in key or ".single_blocks." in key
    }

    dora_scales = {key: value for key, value in block_state_dict.items() if key.endswith(".dora_scale")}
    main_state_dict = {key: value for key, value in block_state_dict.items() if not key.endswith(".dora_scale")}
    converted_state_dict: dict[str, Tensor] = convert_util(main_state_dict, _COMFYUI_BLOCK_PATTERNS, strict=False)

    qkv_dora_scales: dict[str, dict[int, Tensor]] = {}
    for key, value in dora_scales.items():
        matched_qkv_attr = False
        for attr in _COMFYUI_QKV_SPLIT_ATTRS:
            pattern = f".{attr}."
            if pattern not in key:
                continue

            pattern_position = key.index(pattern)
            index_start = pattern_position + len(pattern)
            index_end = key.index(".", index_start)
            base_key = key[: pattern_position + len(pattern) - 1]
            component_index = int(key[index_start:index_end])
            qkv_dora_scales.setdefault(base_key, {})[component_index] = value
            matched_qkv_attr = True
            break

        if not matched_qkv_attr:
            key = key.replace(".img_mlp.fc0.", ".img_mlp.fc1.")
            key = key.replace(".txt_mlp.fc0.", ".txt_mlp.fc1.")
            converted_state_dict[key] = value

    for base_key, components in qkv_dora_scales.items():
        converted_state_dict[f"{base_key}.dora_scale"] = torch.cat(
            [components[index] for index in sorted(components.keys())],
            dim=0,
        )

    return converted_state_dict
