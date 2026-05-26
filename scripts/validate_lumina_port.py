#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_TRANSFORMER_TARGET_CLASSES = ["JointTransformerBlock", "FinalLayer"]
EXPECTED_TEXT_ENCODER_TARGET_CLASSES = [
    "Gemma2Attention",
    "Gemma2FlashAttention2",
    "Gemma2SdpaAttention",
    "Gemma2MLP",
]
EXPECTED_TRANSFORMER_TARGET_FILTERS = ["context_refiner", "noise_refiner", "layers", "final_layer"]
EXPECTED_TEXT_ENCODER_TARGET_FILTERS = ["self_attn", "mlp"]

FORBIDDEN_TERMS = [
    "hunyuan_image",
    "HUNYUAN_IMAGE",
    "Hunyuan Image",
    "anima_",
    "ANIMA_",
    "Anima",
    "control_net_lllite",
    "ControlNet-LLLite",
    "network_args",
    "network_module",
    "module_algo_map",
    "name_algo_map",
    "deepspeed",
    "fp8_base_model",
    "blocks_to_swap",
    "enable_block_swap",
]

SCAN_EXCLUDED_PATHS = {
    "AGENTS.md",
    "scripts/validate_lumina_port.py",
}
SCAN_EXCLUDED_PREFIXES = (
    ".git/",
    ".refs/",
    ".sangoi/",
    ".uv/",
    ".venv/",
)
SCAN_SUFFIXES = {
    ".bat",
    ".cfg",
    ".css",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def fail(message: str) -> None:
    raise AssertionError(message)


def read_text(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)


def check_contains(path: str, terms: list[str]) -> None:
    source = read_text(path)
    for term in terms:
        if term not in source:
            fail(f"{path} is missing required term: {term}")


def literal_assignment(path: str, assignment_name: str):
    tree = ast.parse(read_text(path), filename=path)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == assignment_name:
                    return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == assignment_name:
            return ast.literal_eval(node.value)
    fail(f"{path} is missing assignment {assignment_name}")
    raise AssertionError("unreachable")


def changed_paths() -> list[Path]:
    tracked = run_git(["diff", "--name-only", "HEAD", "--"]).splitlines()
    untracked = run_git(["ls-files", "--others", "--exclude-standard"]).splitlines()
    paths = []
    for rel_path in sorted(set(tracked + untracked)):
        if rel_path in SCAN_EXCLUDED_PATHS:
            continue
        if rel_path.startswith(SCAN_EXCLUDED_PREFIXES):
            continue
        path = REPO_ROOT / rel_path
        if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
            continue
        paths.append(path)
    return paths


def check_forbidden_new_scope_terms() -> None:
    failures: list[str] = []
    for path in changed_paths():
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8", errors="ignore")
        for line_index, line in enumerate(source.splitlines(), start=1):
            failures.extend(
                f"{rel_path}:{line_index}: {term}"
                for term in FORBIDDEN_TERMS
                if term in line
            )

    if failures:
        fail("forbidden out-of-scope Lumina-slice terms introduced:\n" + "\n".join(failures))


def check_static_contracts() -> None:
    check_contains(
        "modules/util/enum/ModelType.py",
        [
            'LUMINA_2 = "LUMINA_2"',
            "def is_lumina(self):",
            "or self.is_lumina()",
        ],
    )
    check_contains(
        "modules/ui/TopBar.py",
        [
            '("Lumina 2", ModelType.LUMINA_2)',
            "elif self.train_config.model_type.is_lumina():",
            '("LoRA", TrainingMethod.LORA)',
        ],
    )
    top_bar_source = read_text("modules/ui/TopBar.py")
    lumina_branch = top_bar_source.split("elif self.train_config.model_type.is_lumina():", 1)[1].split("elif ", 1)[0]
    if "TrainingMethod.FINE_TUNE" in lumina_branch or "TrainingMethod.EMBEDDING" in lumina_branch:
        fail("Lumina UI advertises an unsupported training method")

    check_contains(
        "modules/ui/ModelTab.py",
        [
            'base_model_label="NextDiT Checkpoint"',
            "Lumina NextDiT .safetensors checkpoint",
            'text_encoder_label="Gemma2 Checkpoint"',
            "Gemma2 .safetensors checkpoint",
            "transformer_include_gguf=False",
        ],
    )
    model_tab_source = read_text("modules/ui/ModelTab.py")
    lumina_model_branch = model_tab_source.split("def __setup_lumina_ui", 1)[1].split(
        "def __setup_",
        1,
    )[0]
    if "include_gguf=True" in lumina_model_branch or "GGUF" in lumina_model_branch:
        fail("Lumina model tab exposes GGUF even though the Lumina loader only supports safetensors")

    check_contains(
        "modules/modelLoader/lumina/LuminaModelLoader.py",
        [
            "model_names.base_model",
            "model_names.text_encoder_model",
            "model_names.vae_model",
            "Lumina base_model_name must point to the NextDiT safetensors checkpoint",
            "Lumina text_encoder.model_name must point to the Gemma2 safetensors checkpoint",
            "Lumina vae.model_name must point to the AE safetensors checkpoint",
            "FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=6.0)",
        ],
    )
    check_contains(
        "modules/model/LuminaModel.py",
        [
            "<Prompt Start>",
            "max_length=256 if max_length is None else max_length",
            "output.hidden_states[-2]",
        ],
    )
    check_contains(
        "modules/dataLoader/LuminaBaseDataLoader.py",
        [
            "aspect_bucketing_quantization=16",
            "model.add_system_prompt(config.lumina_system_prompt, prompt)",
            'tokens_out_name="tokens"',
            'mask_out_name="tokens_mask"',
            'hidden_state_out_name="text_encoder_hidden_state"',
        ],
    )
    check_contains(
        "modules/model/lumina/lumina_util.py",
        [
            'GEMMA_TOKENIZER_ID = "google/gemma-2-2b"',
            "LUMINA_NUM_TRAIN_TIMESTEPS = 1000",
            "NextDiT_2B_GQA_patch2_Adaln_Refiner",
            'checkpoint_path = _require_checkpoint(checkpoint_path, "transformer")',
            'checkpoint_path = _require_checkpoint(checkpoint_path, "autoencoder")',
            'checkpoint_path = _require_checkpoint(checkpoint_path, "text encoder")',
            "checkpoint must use the .safetensors extension",
            "_load_state_dict_or_raise",
            "missing keys",
            "unexpected keys",
        ],
    )
    check_contains(
        "modules/modelSetup/BaseLuminaSetup.py",
        [
            "TimestepDistribution.NEXTDIT_SHIFT",
            "image_seq_len = (height // 2) * (width // 2)",
            "exp_mu = math.exp(mu)",
            "model_timestep = 1.0 - timestep / LUMINA_NUM_TRAIN_TIMESTEPS",
        ],
    )
    check_contains(
        "modules/ui/TrainUI.py",
        [
            "self.train_config.base_model_name == TrainConfig.default_values().base_model_name",
            'self.ui_state.get_var("base_model_name").set("")',
            "TimestepDistribution.NEXTDIT_SHIFT",
        ],
    )
    check_contains(
        "modules/modelSampler/LuminaSampler.py",
        [
            "factory.register(BaseModelSampler, LuminaSampler, ModelType.LUMINA_2)",
            "current_timestep = 1 - timestep / LUMINA_NUM_TRAIN_TIMESTEPS",
        ],
    )

    for path, architecture in [
        ("resources/sd_model_spec/lumina_2.json", "lumina2"),
        ("resources/sd_model_spec/lumina_2-lora.json", "lumina2/lora"),
    ]:
        data = json.loads(read_text(path))
        if data.get("modelspec.architecture") != architecture:
            fail(f"{path} has wrong modelspec.architecture")


def check_lora_source_contract() -> None:
    source = read_text(".refs/sd-scripts/networks/lora_lumina.py")
    for target in EXPECTED_TRANSFORMER_TARGET_CLASSES + EXPECTED_TEXT_ENCODER_TARGET_CLASSES:
        if target not in source:
            fail(f"pinned sd-scripts Lumina LoRA source missing {target}")

    if (
        literal_assignment("modules/modelSetup/LuminaLoRASetup.py", "LUMINA_TRANSFORMER_LORA_TARGET_CLASSES")
        != EXPECTED_TRANSFORMER_TARGET_CLASSES
    ):
        fail("Lumina transformer target classes drifted from source contract")
    if (
        literal_assignment("modules/modelSetup/LuminaLoRASetup.py", "LUMINA_TEXT_ENCODER_LORA_TARGET_CLASSES")
        != EXPECTED_TEXT_ENCODER_TARGET_CLASSES
    ):
        fail("Lumina text-encoder target classes drifted from source contract")
    if (
        literal_assignment("modules/modelSetup/LuminaLoRASetup.py", "LUMINA_TRANSFORMER_LORA_TARGET_FILTERS")
        != EXPECTED_TRANSFORMER_TARGET_FILTERS
    ):
        fail("Lumina transformer LoRA filters no longer match the source-target paths")
    if (
        literal_assignment("modules/modelSetup/LuminaLoRASetup.py", "LUMINA_TEXT_ENCODER_LORA_TARGET_FILTERS")
        != EXPECTED_TEXT_ENCODER_TARGET_FILTERS
    ):
        fail("Lumina text-encoder LoRA filters no longer match the source-target paths")


def check_config_roundtrip() -> None:
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.ModelType import ModelType, PeftType
    from modules.util.enum.TimestepDistribution import TimestepDistribution
    from modules.util.enum.TrainingMethod import TrainingMethod

    config = TrainConfig.default_values()
    config.model_type = ModelType.LUMINA_2
    config.training_method = TrainingMethod.LORA
    config.peft_type = PeftType.LORA
    config.base_model_name = "models/lumina-nextdit.safetensors"
    config.text_encoder.model_name = "models/gemma2.safetensors"
    config.vae.model_name = "models/lumina-ae.safetensors"
    config.text_encoder_sequence_length = 256
    config.timestep_distribution = TimestepDistribution.NEXTDIT_SHIFT
    config.lumina_system_prompt = "You are an assistant designed to generate images."

    restored = TrainConfig.default_values().from_dict(config.to_dict())
    if restored.model_type != ModelType.LUMINA_2:
        fail("Lumina model_type did not roundtrip")
    if restored.training_method != TrainingMethod.LORA:
        fail("Lumina training_method did not roundtrip")
    if restored.peft_type != PeftType.LORA:
        fail("Lumina peft_type did not roundtrip")
    if restored.base_model_name != config.base_model_name:
        fail("Lumina base_model_name did not roundtrip")
    if restored.text_encoder.model_name != config.text_encoder.model_name:
        fail("Lumina text_encoder.model_name did not roundtrip")
    if restored.vae.model_name != config.vae.model_name:
        fail("Lumina vae.model_name did not roundtrip")
    if restored.text_encoder_sequence_length != 256:
        fail("Lumina text_encoder_sequence_length did not roundtrip")
    if restored.timestep_distribution != TimestepDistribution.NEXTDIT_SHIFT:
        fail("Lumina NEXTDIT_SHIFT timestep distribution did not roundtrip")
    if restored.lumina_system_prompt != config.lumina_system_prompt:
        fail("Lumina system prompt did not roundtrip")

    model_names = restored.model_names()
    if model_names.base_model != config.base_model_name:
        fail("Lumina model_names base mapping is wrong")
    if model_names.text_encoder_model != config.text_encoder.model_name:
        fail("Lumina model_names text encoder mapping is wrong")
    if model_names.vae_model != config.vae.model_name:
        fail("Lumina model_names VAE mapping is wrong")

    unsupported = TrainConfig.default_values()
    unsupported.model_type = ModelType.LUMINA_2
    unsupported.training_method = TrainingMethod.FINE_TUNE
    try:
        unsupported.validate_for_training()
    except ValueError:
        pass
    else:
        fail("Lumina fine-tune config did not fail validation")


def check_lumina_timestep_preview() -> None:
    from modules.ui.TimestepDistributionWindow import TimestepGenerator
    from modules.util.enum.TimestepDistribution import TimestepDistribution

    generator = TimestepGenerator(
        timestep_distribution=TimestepDistribution.NEXTDIT_SHIFT,
        min_noising_strength=0.0,
        max_noising_strength=1.0,
        noising_weight=0.0,
        noising_bias=0.0,
        timestep_shift=6.0,
        resolution="512x768",
    )
    timesteps = generator.generate()
    if timesteps.shape != (1000000,):
        fail(f"Lumina NEXTDIT_SHIFT preview shape drifted: {tuple(timesteps.shape)}")
    if timesteps.min().item() < 0 or timesteps.max().item() > 999:
        fail("Lumina NEXTDIT_SHIFT preview generated timesteps outside scheduler range")


def check_lumina_loader_rejections() -> None:
    from modules.model.lumina.lumina_util import (
        load_lumina_autoencoder,
        load_lumina_gemma2,
        load_lumina_transformer,
    )

    import torch

    from safetensors.torch import save_file

    def expect_value_error(description: str, callback, required_terms: list[str]) -> None:
        try:
            callback()
        except ValueError as exc:
            message = str(exc)
            for term in required_terms:
                if term not in message:
                    fail(f"{description} failed with an unclear error: {message}")
        else:
            fail(f"{description} did not fail")

    loaders = [
        ("transformer", load_lumina_transformer),
        ("autoencoder", load_lumina_autoencoder),
        ("text encoder", load_lumina_gemma2),
    ]

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        wrong_suffix_path = temp_path / "lumina.pt"
        wrong_suffix_path.write_bytes(b"not a safetensors file")
        partial_checkpoint_path = temp_path / "partial.safetensors"
        save_file({"unexpected.weight": torch.zeros(1)}, str(partial_checkpoint_path))

        for label, loader in loaders:
            expect_value_error(
                f"Lumina {label} wrong suffix",
                lambda loader=loader: loader(str(wrong_suffix_path), None, "cpu"),
                [".safetensors", str(wrong_suffix_path)],
            )
            expect_value_error(
                f"Lumina {label} partial checkpoint",
                lambda loader=loader: loader(str(partial_checkpoint_path), None, "cpu"),
                ["does not match the expected model keys", "missing keys", "unexpected keys"],
            )


def check_factory_matrix() -> None:
    import modules.util.create as create
    from modules.dataLoader.BaseDataLoader import BaseDataLoader
    from modules.model.BaseModel import BaseModel
    from modules.modelLoader.BaseModelLoader import BaseModelLoader
    from modules.modelSampler.BaseModelSampler import BaseModelSampler
    from modules.modelSaver.BaseModelSaver import BaseModelSaver
    from modules.modelSetup.BaseModelSetup import BaseModelSetup
    from modules.util import factory
    from modules.util.enum.ModelType import ModelType
    from modules.util.enum.TrainingMethod import TrainingMethod

    import torch

    if create.create_model_loader(ModelType.LUMINA_2, TrainingMethod.LORA) is None:
        fail("Lumina LoRA model loader is not registered")
    if create.create_model_saver(ModelType.LUMINA_2, TrainingMethod.LORA) is None:
        fail("Lumina LoRA model saver is not registered")
    if create.create_model_setup(ModelType.LUMINA_2, torch.device("cpu"), torch.device("cpu"), TrainingMethod.LORA) is None:
        fail("Lumina LoRA model setup is not registered")

    if factory.get(BaseDataLoader, ModelType.LUMINA_2) is None:
        fail("Lumina data loader is not registered")

    class DummyModel(BaseModel):
        def __init__(self):
            super().__init__(ModelType.LUMINA_2)

        def adapters(self):
            return []

        def all_embeddings(self):
            return []

        def to(self, device):
            return None

        def eval(self):
            return None

    if (
        create.create_model_sampler(
            torch.device("cpu"), torch.device("cpu"), DummyModel(), ModelType.LUMINA_2, TrainingMethod.LORA
        )
        is None
    ):
        fail("Lumina sampler is not registered")

    for unsupported_method in [TrainingMethod.FINE_TUNE, TrainingMethod.EMBEDDING]:
        if factory.get(BaseModelLoader, ModelType.LUMINA_2, unsupported_method) is not None:
            fail(f"Lumina has unsupported loader for {unsupported_method}")
        if factory.get(BaseModelSaver, ModelType.LUMINA_2, unsupported_method) is not None:
            fail(f"Lumina has unsupported saver for {unsupported_method}")
        if factory.get(BaseModelSetup, ModelType.LUMINA_2, unsupported_method) is not None:
            fail(f"Lumina has unsupported setup for {unsupported_method}")
        if factory.get(BaseModelSampler, ModelType.LUMINA_2, unsupported_method) is not None:
            fail(f"Lumina has unsupported method-specific sampler for {unsupported_method}")


def check_lora_filters() -> None:
    from modules.module.LoRAModule import LoRAModuleWrapper
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.ModelType import ModelType, PeftType
    from modules.util.enum.TrainingMethod import TrainingMethod

    import torch
    from torch import nn

    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.context_refiner = nn.ModuleList([nn.Sequential(nn.Linear(4, 4))])
            self.noise_refiner = nn.ModuleList([nn.Sequential(nn.Linear(4, 4))])
            self.layers = nn.ModuleList([nn.Sequential(nn.Linear(4, 4))])
            self.final_layer = nn.Linear(4, 4)
            self.x_embedder = nn.Linear(4, 4)
            self.t_embedder = nn.Linear(4, 4)
            self.cap_embedder = nn.Linear(4, 4)

    class TinyTextEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "self_attn": nn.ModuleDict({"q_proj": nn.Linear(4, 4)}),
                            "mlp": nn.ModuleDict({"up_proj": nn.Linear(4, 4)}),
                            "input_layernorm": nn.Linear(4, 4),
                        }
                    )
                ]
            )

    config = TrainConfig.default_values()
    config.model_type = ModelType.LUMINA_2
    config.training_method = TrainingMethod.LORA
    config.peft_type = PeftType.LORA
    config.train_device = "cpu"
    config.lora_rank = 2
    config.lora_alpha = 2.0

    transformer_wrapper = LoRAModuleWrapper(
        TinyTransformer(), "lora_transformer", config, EXPECTED_TRANSFORMER_TARGET_FILTERS
    )
    transformer_keys = sorted(transformer_wrapper.lora_modules)
    for required_prefix in ["context_refiner", "noise_refiner", "layers", "final_layer"]:
        if not any(key.startswith(required_prefix) for key in transformer_keys):
            fail(f"Lumina transformer filter did not select {required_prefix}")
    for forbidden_prefix in ["x_embedder", "t_embedder", "cap_embedder"]:
        if any(key.startswith(forbidden_prefix) for key in transformer_keys):
            fail(f"Lumina transformer filter selected forbidden {forbidden_prefix}")

    text_encoder_wrapper = LoRAModuleWrapper(
        TinyTextEncoder(), "lora_te", config, EXPECTED_TEXT_ENCODER_TARGET_FILTERS
    )
    text_encoder_keys = sorted(text_encoder_wrapper.lora_modules)
    if not any("self_attn" in key for key in text_encoder_keys):
        fail("Lumina text encoder filter did not select self_attn")
    if not any("mlp" in key for key in text_encoder_keys):
        fail("Lumina text encoder filter did not select mlp")
    if any("input_layernorm" in key for key in text_encoder_keys):
        fail("Lumina text encoder filter selected non-source target input_layernorm")

    del torch


def check_synthetic_predict_and_loss() -> None:
    from modules.modelSetup.BaseLuminaSetup import BaseLuminaSetup
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.DataType import DataType
    from modules.util.enum.LossWeight import LossWeight
    from modules.util.enum.ModelType import ModelType, PeftType
    from modules.util.enum.TimestepDistribution import TimestepDistribution
    from modules.util.enum.TrainingMethod import TrainingMethod
    from modules.util.TrainProgress import TrainProgress

    import torch

    class SyntheticLuminaSetup(BaseLuminaSetup):
        def create_parameters(self, model, config):
            fail("synthetic Lumina validation should not create parameters")

        def setup_model(self, model, config):
            fail("synthetic Lumina validation should not setup model")

        def setup_train_device(self, model, config):
            fail("synthetic Lumina validation should not setup train device")

        def after_optimizer_step(self, model, config, train_progress):
            fail("synthetic Lumina validation should not run optimizer hooks")

    class SyntheticTransformer:
        def __call__(self, *, x, t, cap_feats, cap_mask):
            if x.shape != (2, 16, 4, 4):
                fail(f"unexpected Lumina transformer x shape: {tuple(x.shape)}")
            if t.shape != (2,):
                fail(f"unexpected Lumina transformer timestep shape: {tuple(t.shape)}")
            if cap_feats.shape != (2, 8, 2304):
                fail(f"unexpected Lumina cap_feats shape: {tuple(cap_feats.shape)}")
            if cap_mask.shape != (2, 8) or cap_mask.dtype != torch.int32:
                fail("unexpected Lumina cap_mask contract")
            return x * 0.25

    class SyntheticModel:
        def __init__(self):
            self.autocast_context = nullcontext()
            self.train_dtype = DataType.FLOAT_32
            self.transformer = SyntheticTransformer()
            self.noise_scheduler = SimpleNamespace(sigmas=torch.linspace(0.0, 1.0, 1000))

        def encode_text(
            self,
            *,
            train_device,
            batch_size,
            rand,
            tokens,
            attention_mask,
            text_encoder_output,
            text_encoder_dropout_probability,
        ):
            if batch_size != 2:
                fail("Lumina synthetic batch size drifted")
            if text_encoder_output is None:
                fail("Lumina synthetic path should use cached text encoder hidden states")
            return text_encoder_output.to(train_device), attention_mask.to(train_device)

    config = TrainConfig.default_values()
    config.model_type = ModelType.LUMINA_2
    config.training_method = TrainingMethod.LORA
    config.peft_type = PeftType.LORA
    config.train_device = "cpu"
    config.train_dtype = DataType.FLOAT_32
    config.batch_size = 2
    config.gradient_accumulation_steps = 1
    config.text_encoder.train = False
    config.text_encoder.dropout_probability = 0.0
    config.timestep_distribution = TimestepDistribution.NEXTDIT_SHIFT
    config.timestep_shift = 6.0
    config.noising_weight = 0.0
    config.noising_bias = 0.0
    config.loss_weight_fn = LossWeight.CONSTANT
    config.force_epsilon_prediction = False
    config.force_v_prediction = False
    config.masked_training = False
    config.k_noise_sampling = 1

    batch = {
        "latent_image": torch.randn(2, 16, 4, 4),
        "tokens": torch.ones(2, 8, dtype=torch.long),
        "tokens_mask": torch.ones(2, 8, dtype=torch.long),
        "text_encoder_hidden_state": torch.randn(2, 8, 2304),
        "loss_weight": torch.ones(2),
    }

    setup = SyntheticLuminaSetup(torch.device("cpu"), torch.device("cpu"), False)
    model = SyntheticModel()
    data = setup.predict(model, batch, config, TrainProgress(global_step=3))

    if data["loss_type"] != "target":
        fail("Lumina predict no longer returns target loss_type")
    if data["predicted"].shape != batch["latent_image"].shape:
        fail("Lumina predicted flow shape drifted")
    if data["target"].shape != batch["latent_image"].shape:
        fail("Lumina target flow shape drifted")
    if not torch.isfinite(data["predicted"]).all() or not torch.isfinite(data["target"]).all():
        fail("Lumina predict produced non-finite tensors")

    losses = setup.calculate_loss(model, batch, data, config)
    if losses.numel() == 0 or losses.ndim > 1:
        fail(f"Lumina loss shape drifted: {tuple(losses.shape)}")
    if not torch.isfinite(losses).all():
        fail("Lumina loss produced non-finite values")


def main() -> None:
    check_forbidden_new_scope_terms()
    check_static_contracts()
    check_lora_source_contract()
    check_config_roundtrip()
    check_lumina_timestep_preview()
    check_lumina_loader_rejections()
    check_factory_matrix()
    check_lora_filters()
    check_synthetic_predict_and_loss()
    print("Lumina port validation passed")


if __name__ == "__main__":
    main()
