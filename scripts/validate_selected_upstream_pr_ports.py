#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_TIMESTEP_SETUPS = [
    "modules/modelSetup/BaseStableDiffusionSetup.py",
    "modules/modelSetup/BaseStableDiffusionXLSetup.py",
    "modules/modelSetup/BaseStableDiffusion3Setup.py",
    "modules/modelSetup/BaseFluxSetup.py",
    "modules/modelSetup/BaseFlux2Setup.py",
    "modules/modelSetup/BaseChromaSetup.py",
    "modules/modelSetup/BaseErnieSetup.py",
    "modules/modelSetup/BaseHiDreamSetup.py",
    "modules/modelSetup/BaseHunyuanVideoSetup.py",
    "modules/modelSetup/BasePixArtAlphaSetup.py",
    "modules/modelSetup/BaseQwenSetup.py",
    "modules/modelSetup/BaseSanaSetup.py",
    "modules/modelSetup/BaseWuerstchenSetup.py",
    "modules/modelSetup/BaseZImageSetup.py",
]

GUARDED_PATH_PREFIXES = [
    "modules/cloud/",
    "resources/docker/",
]

GUARDED_EXACT_PATHS = [
    ".dockerignore",
    ".gitmodules",
    ".python-version",
    "install.sh",
    "lib.include.sh",
    "pyproject.toml",
    "run-cmd.sh",
    "scripts/train_remote.py",
    "start-ui.sh",
    "update.sh",
]

GUARDED_GLOBS = [
    "requirements*.txt",
]

ALLOWED_GUARDED_REQUIREMENTS_DIFFS = {
    "requirements-global.txt": ["+einops==0.7.0"],
}

BLOCKED_RUNTIME_TERMS = [
    "resolution_quantization",
    "cfg_distillation",
    "DistillationConfig",
    "DistillationCacheMode",
    "DistillationLossType",
    "DistillationTargetMode",
    "ParentModelWrapper",
    "diff2flow",
    "smart-disk-cache",
    "SmartDiskCache",
    "LoadSmartCache",
    "SaveSmartCache",
    "smartdiskcache",
    "native_split",
    "flash_split",
    "split_attention",
    "activation_quantization",
    "LinearA8",
    "delta_transfer",
    "validation_timesteps",
    "FLUX_2_EDIT",
    "caption_dropout",
    "fixed_resolution",
    "TheRock",
    "Windows ROCm",
    "attention_backend",
    "modules.ipex_to_cuda",
    "ipex_to_cuda",
]

BLOCKED_RUNTIME_TERM_EXEMPTIONS = {
    "attention_backend": ["set_attention_backend"],
}

BLOCKED_SCAN_EXCLUDED_PATHS = {
    "scripts/validate_selected_upstream_pr_ports.py",
}

NO_ALIAS_TERMS = [
    "legacy_",
    "alias_",
    "compat_",
    "dual_read",
    "dual_write",
    "fallback_adapter",
]

SANGOI_DEFAULTS = {
    "lora_modules_rank_rules": [],
    "lora_modules_alpha_rules": [],
    "lora_layers_blacklist": [],
    "lora_key_export_path": "",
    "aspect_bucketing_quantization_override": 0,
    "train_gps_save_it": False,
    "train_gps_use_it": False,
    "train_gps_weight": 0.0,
    "data_recorder": False,
}


def read_text(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)


def fail(message: str) -> None:
    raise AssertionError(message)


def check_validation_timesteps() -> None:
    from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.ValidationTimestepMode import ValidationTimestepMode
    from modules.util.validation_timestep import (
        resolve_continuous_validation_timestep,
        resolve_discrete_validation_timestep,
    )

    import torch

    util_source = read_text("modules/util/validation_timestep.py")
    for symbol in [
        "parse_validation_timestep_values",
        "resolve_discrete_validation_timestep",
        "resolve_continuous_validation_timestep",
        "validation_noise_seed",
    ]:
        if f"def {symbol}" not in util_source:
            fail(f"validation timestep utility missing {symbol}")

    trainer_source = read_text("modules/trainer/GenericTrainer.py")
    for key in [
        "__validation_timestep_index__",
        "__validation_timestep_count__",
        "__validation_noise_seed__",
    ]:
        if key not in trainer_source:
            fail(f"GenericTrainer does not stage {key}")
    generate_losses_source = read_text("modules/module/GenerateLossesModel.py")
    for key in [
        "for batch_index, batch in enumerate(self.data_loader.get_data_loader()):",
        "__validation_timestep_index__",
        "__validation_timestep_count__",
        "__validation_noise_seed__",
    ]:
        if key not in generate_losses_source:
            fail(f"GenerateLossesModel does not stage deterministic validation timestep key {key}")

    for setup_path in VALIDATION_TIMESTEP_SETUPS:
        source = read_text(setup_path)
        if "validation_index=batch.get(\"__validation_timestep_index__\")" not in source:
            fail(f"{setup_path} does not pass validation_index into timestep helper")
        if "validation_count=batch.get(\"__validation_timestep_count__\")" not in source:
            fail(f"{setup_path} does not pass validation_count into timestep helper")
        if "validation_timestep_values" in source:
            fail(f"{setup_path} parses validation timestep policy directly")

    config = TrainConfig.default_values()
    config.validation_timestep_mode = ValidationTimestepMode.FIXED
    config.validation_timestep_values = "9999"
    try:
        resolve_discrete_validation_timestep(
            config=config,
            num_train_timesteps=1000,
            batch_size=1,
            device=torch.device("cpu"),
            shift=1.0,
            validation_index=0,
            validation_count=1,
        )
    except ValueError:
        pass
    else:
        fail("fixed validation timestep outside scheduler range did not fail")

    config.validation_timestep_values = "10001"
    try:
        resolve_continuous_validation_timestep(
            config=config,
            batch_size=1,
            device=torch.device("cpu"),
            validation_index=0,
            validation_count=1,
        )
    except ValueError:
        pass
    else:
        fail("fixed continuous validation timestep outside range did not fail")

    config = TrainConfig.default_values()
    config.validation_timestep_mode = ValidationTimestepMode.STRATIFIED
    config.validation_timestep_seed = 11
    deterministic_batch = {
        "__validation_timestep_index__": 0,
        "__validation_timestep_count__": 3,
    }
    noise_mixin = object.__new__(ModelSetupNoiseMixin)
    generator = torch.Generator(device="cpu")
    discrete_timestep = noise_mixin._get_timestep_discrete(
        num_train_timesteps=1000,
        deterministic=True,
        generator=generator,
        batch_size=1,
        config=config,
        validation_index=deterministic_batch.get("__validation_timestep_index__"),
        validation_count=deterministic_batch.get("__validation_timestep_count__"),
    )
    continuous_timestep = noise_mixin._get_timestep_continuous(
        deterministic=True,
        generator=generator,
        batch_size=1,
        config=config,
        validation_index=deterministic_batch.get("__validation_timestep_index__"),
        validation_count=deterministic_batch.get("__validation_timestep_count__"),
    )
    if discrete_timestep.numel() != 1 or continuous_timestep.numel() != 1:
        fail("calculate-loss-style STRATIFIED deterministic timestep smoke returned wrong shape")


def check_mid_accum_resume() -> None:
    from modules.trainer.GenericTrainer import GenericTrainer
    from modules.util.config.ConceptConfig import ConceptConfig
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.TrainProgress import TrainProgress

    import torch

    source = read_text("modules/trainer/GenericTrainer.py")
    required = [
        "__resume_config_fingerprint",
        "compute_concept_fingerprint",
        "__stage_accumulator_state",
        "__restore_accumulator_state",
        "saved accumulator state does not match gradient_accumulation_steps",
        "saved accumulator state does not match the current dataset/config fingerprint",
        "saved accumulator state does not match the current global_step",
        "torch.get_rng_state",
        "np.random.get_state",
        "random.getstate",
        "model.accumulator_state",
    ]
    for term in required:
        if term not in source:
            fail(f"mid-accum resume missing {term}")

    for path in [
        "modules/modelLoader/mixin/InternalModelLoaderMixin.py",
        "modules/modelLoader/BaseModelLoader.py",
        "modules/modelSaver/mixin/InternalModelSaverMixin.py",
    ]:
        if "accumulator_state" not in read_text(path):
            fail(f"{path} does not handle accumulator_state")

    if "normalize_config_for_fingerprint" not in source:
        fail("accumulator fingerprint does not normalize material config fields")

    concept = ConceptConfig.default_values()
    concept.name = "concept"
    concept.path = "/tmp/concept"
    base_config = TrainConfig.default_values()
    base_config.concepts = [concept]

    def clone_config() -> TrainConfig:
        return TrainConfig.default_values().from_dict(base_config.to_dict())

    def fingerprint_for(config: TrainConfig) -> str:
        trainer = object.__new__(GenericTrainer)
        trainer.config = config
        return trainer._GenericTrainer__resume_config_fingerprint()

    base_fingerprint = fingerprint_for(base_config)

    class ModelFixture:
        pass

    def assert_material_mismatch(label: str, changed_config: TrainConfig) -> None:
        if fingerprint_for(changed_config) == base_fingerprint:
            fail(f"accumulator fingerprint ignores {label}")

        changed_trainer = object.__new__(GenericTrainer)
        changed_trainer.config = changed_config
        changed_trainer.model = ModelFixture()
        changed_trainer.model.accumulator_state = {
            "config_fingerprint": base_fingerprint,
            "global_step": 0,
            "gradient_accumulation_steps": changed_config.gradient_accumulation_steps,
        }
        try:
            changed_trainer._GenericTrainer__restore_accumulator_state(
                torch.tensor(0.0),
                torch.device("cpu"),
                None,
                TrainProgress(global_step=0),
            )
        except ValueError as exc:
            if "dataset/config fingerprint" not in str(exc):
                raise
        else:
            fail(f"accumulator restore accepted {label} fingerprint mismatch")

    changed_resolution = clone_config()
    changed_resolution.resolution = "768"
    assert_material_mismatch("resolution changes", changed_resolution)

    changed_learning_rate = clone_config()
    changed_learning_rate.learning_rate = 0.000123
    assert_material_mismatch("learning_rate changes", changed_learning_rate)

    changed_optimizer = clone_config()
    changed_optimizer.optimizer.beta1 = 0.123
    assert_material_mismatch("optimizer beta1 changes", changed_optimizer)

    changed_text_encoder = clone_config()
    changed_text_encoder.text_encoder.train = not changed_text_encoder.text_encoder.train
    assert_material_mismatch("text_encoder train flag changes", changed_text_encoder)

    changed_unet = clone_config()
    changed_unet.unet.train = not changed_unet.unet.train
    assert_material_mismatch("unet train flag changes", changed_unet)

    changed_concept = clone_config()
    changed_concept.concepts[0].image.enable_crop_jitter = not changed_concept.concepts[0].image.enable_crop_jitter
    assert_material_mismatch("material concept image changes", changed_concept)


def check_resume_actions() -> None:
    from modules.modelLoader.BaseModelLoader import BaseModelLoader
    from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
    from modules.util.enum.TimeUnit import TimeUnit
    from modules.util.TimedActionMixin import TimedActionMixin
    from modules.util.TrainProgress import TrainProgress

    progress_source = read_text("modules/util/TrainProgress.py")
    timed_source = read_text("modules/util/TimedActionMixin.py")
    loader_source = read_text("modules/modelLoader/BaseModelLoader.py")
    saver_source = read_text("modules/modelSaver/mixin/InternalModelSaverMixin.py")
    for source_path, source in [
        ("TrainProgress.py", progress_source),
        ("TimedActionMixin.py", timed_source),
        ("BaseModelLoader.py", loader_source),
        ("InternalModelSaverMixin.py", saver_source),
    ]:
        if "last_action_epoch" not in source:
            fail(f"{source_path} does not preserve last_action_epoch")

    if "{\"validate\": train_progress.epoch" in loader_source + read_text("modules/modelLoader/mixin/InternalModelLoaderMixin.py"):
        fail("loader backfills missing last_action_epoch instead of using TrainProgress default")

    timed_action = TimedActionMixin()
    progress = TrainProgress(epoch=2, epoch_step=0)
    if not timed_action.repeating_action_needed(
        "backup", 1, TimeUnit.EPOCH, progress, start_at_zero=False
    ):
        fail("epoch backup action was not due at an epoch boundary")
    if progress.last_action_epoch.get("backup") != 2:
        fail("epoch backup action did not persist last_action_epoch")
    if timed_action.repeating_action_needed("backup", 1, TimeUnit.EPOCH, progress, start_at_zero=False):
        fail("epoch backup action repeated after persisting last_action_epoch")

    resumed_progress = TrainProgress(epoch=2, epoch_step=0, last_action_epoch={"backup": 2})
    if TimedActionMixin().repeating_action_needed(
        "backup", 1, TimeUnit.EPOCH, resumed_progress, start_at_zero=False
    ):
        fail("epoch backup action repeated after resume with persisted last_action_epoch")

    class BaseLoaderFixture(BaseModelLoader):
        def load(self, model_type, model_names, weight_dtypes, quantization):
            return None

    class MixinLoaderFixture(InternalModelLoaderMixin):
        pass

    class ModelFixture:
        pass

    def write_meta(directory: str, last_action_epoch=None):
        meta = {
            "train_progress": {
                "epoch": 3,
                "epoch_step": 1,
                "epoch_sample": 4,
                "global_step": 5,
            },
            "tensorboard_subdir": "tb",
        }
        if last_action_epoch is not None:
            meta["last_action_epoch"] = last_action_epoch
        with open(Path(directory) / "meta.json", "w", encoding="utf-8") as meta_file:
            json.dump(meta, meta_file)

    for loader_name, load_method in [
        ("BaseModelLoader", BaseLoaderFixture()._load_internal_state),
        ("InternalModelLoaderMixin", MixinLoaderFixture()._load_internal_data),
    ]:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_meta(temp_dir)
            model = ModelFixture()
            load_method(model, temp_dir)
            if model.train_progress.last_action_epoch != {}:
                fail(f"{loader_name} backfilled missing last_action_epoch")

        persisted_actions = {"validate": 1, "sample": 2, "backup": 3, "save": 4}
        with tempfile.TemporaryDirectory() as temp_dir:
            write_meta(temp_dir, persisted_actions)
            model = ModelFixture()
            load_method(model, temp_dir)
            if model.train_progress.last_action_epoch != persisted_actions:
                fail(f"{loader_name} did not preserve persisted last_action_epoch")


def check_tensorboard_resume() -> None:
    for path in [
        "modules/model/BaseModel.py",
        "modules/modelLoader/mixin/InternalModelLoaderMixin.py",
        "modules/trainer/GenericTrainer.py",
        "modules/util/config/TrainConfig.py",
        "modules/ui/TrainUI.py",
    ]:
        source = read_text(path)
        if "tensorboard" not in source.lower():
            fail(f"{path} does not participate in tensorboard resume continuity")
    if "tensorboard_resume_run" not in read_text("modules/util/config/TrainConfig.py"):
        fail("TrainConfig missing tensorboard_resume_run")


def check_patience_contract() -> None:
    from modules.trainer.GenericTrainer import GenericTrainer, ValidationMetrics
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.TimeUnit import TimeUnit
    from modules.util.TimedActionMixin import TimedActionMixin
    from modules.util.TrainProgress import TrainProgress

    import torch

    trainer_source = read_text("modules/trainer/GenericTrainer.py")
    for term in [
        "class ValidationMetrics",
        "__build_validation_metrics",
        "__write_validation_metrics",
        "_patience_best_loss",
        "__update_patience",
        "patience-best",
        "total_average_loss",
        "Patience stopped training",
    ]:
        if term not in trainer_source:
            fail(f"patience implementation missing {term}")

    config_source = read_text("modules/util/config/TrainConfig.py")
    if "patience requires validation" not in config_source:
        fail("TrainConfig does not reject patience without validation")
    calculate_source = trainer_source.split("def __calculate_validation_metrics", 1)[1].split(
        "def __update_patience", 1
    )[0]
    if "__needs_validate" in calculate_source:
        fail("__calculate_validation_metrics consumes validation timing state")

    single = GenericTrainer._GenericTrainer__build_validation_metrics(
        {1: 6.0},
        {1: 3},
        {1: "single"},
    )
    if not isinstance(single, ValidationMetrics):
        fail("__build_validation_metrics did not return ValidationMetrics")
    if single.average_loss_per_concept != {1: 2.0} or single.total_average_loss != 2.0:
        fail("single-concept validation metrics are incorrect")

    multi = GenericTrainer._GenericTrainer__build_validation_metrics(
        {1: 4.0, 2: 30.0},
        {1: 2, 2: 3},
        {1: "a", 2: "b"},
    )
    expected_total = 34.0 / 5.0
    if multi.average_loss_per_concept != {1: 2.0, 2: 10.0} or multi.total_average_loss != expected_total:
        fail("multi-concept validation metrics are not weighted by validation batch counts")

    class ValidationDataset:
        def __init__(self):
            self.started = False

        def start_next_epoch(self):
            self.started = True

        def approximate_length(self):
            return 1

    class ValidationLoader:
        def __init__(self):
            self.dataset = ValidationDataset()

        def get_data_set(self):
            return self.dataset

        def get_data_loader(self):
            return [
                {
                    "concept_name": ["validation"],
                    "concept_path": ["/tmp/validation"],
                    "concept_seed": torch.tensor(7),
                }
            ]

    class Callbacks:
        def on_update_status(self, _status):
            pass

    class ModelSetup:
        def setup_train_device(self, _model, _config):
            pass

        def predict(self, _model, _batch, _config, _train_progress, deterministic=False):
            if not deterministic:
                fail("validation smoke did not request deterministic prediction")
            return {}

        def calculate_loss(self, _model, _batch, _model_output_data, _config):
            return torch.tensor(4.0)

    class Tensorboard:
        def __init__(self):
            self.scalars = []

        def add_scalar(self, tag, value, step):
            self.scalars.append((tag, value, step))

    trainer = object.__new__(GenericTrainer)
    TimedActionMixin.__init__(trainer)
    trainer.config = TrainConfig.default_values()
    trainer.config.validation = True
    trainer.config.validate_after = 1
    trainer.config.validate_after_unit = TimeUnit.EPOCH
    trainer.callbacks = Callbacks()
    trainer.model_setup = ModelSetup()
    trainer.model = object()
    trainer.validation_data_loader = ValidationLoader()
    trainer.tensorboard = Tensorboard()

    train_progress = TrainProgress(epoch=0, epoch_step=0, global_step=0)
    metrics = trainer._GenericTrainer__validate(train_progress)
    if not isinstance(metrics, ValidationMetrics):
        fail("epoch validation did not produce ValidationMetrics")
    if metrics.total_average_loss != 4.0:
        fail("epoch validation smoke returned wrong total_average_loss")
    if train_progress.last_action_epoch.get("validate") != 0:
        fail("epoch validation did not persist last_action_epoch after running")
    if "loss/validation_step/total_average" not in [tag for tag, _value, _step in trainer.tensorboard.scalars]:
        fail("epoch validation did not write total_average metric")


def check_component_stop_contract() -> None:
    trainer_source = read_text("modules/trainer/GenericTrainer.py")
    for term in [
        "All trainable components have reached their stop_training_after limit",
        "requires_grad",
        "get_stop_command",
    ]:
        if term not in trainer_source:
            fail(f"component stop implementation missing {term}")


def check_attention_backend() -> None:
    enum_source = read_text("modules/util/enum/AttentionMechanism.py")
    if "FLASH" not in enum_source or "SDP" not in enum_source:
        fail("AttentionMechanism enum missing SDP/FLASH")

    setup_source = read_text("modules/modelSetup/BaseModelSetup.py")
    for term in [
        "AttentionBackendName.FLASH",
        "AttentionBackendName.NATIVE",
        "set_attention_backend",
        "does not support FLASH attention",
    ]:
        if term not in setup_source:
            fail(f"attention backend contract missing {term}")
    if "_native_flash" in setup_source:
        fail("attention backend uses native flash instead of diffusers FLASH backend")


def check_timestep_distributions() -> None:
    from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.TimestepDistribution import TimestepDistribution

    import torch

    enum_source = read_text("modules/util/enum/TimestepDistribution.py")
    for term in ["BETA", "SPEED", "PRIORITY_SAMPLING"]:
        if term not in enum_source:
            fail(f"TimestepDistribution missing {term}")

    config = TrainConfig.default_values()
    config.timestep_distribution = TimestepDistribution.BETA
    config.validate_for_training()

    noise_mixin = object.__new__(ModelSetupNoiseMixin)

    def beta_timesteps(noising_weight: float, noising_bias: float, global_seed: int):
        torch.manual_seed(global_seed)
        beta_config = TrainConfig.default_values()
        beta_config.timestep_distribution = TimestepDistribution.BETA
        beta_config.noising_weight = noising_weight
        beta_config.noising_bias = noising_bias
        beta_config.validate_for_training()
        generator = torch.Generator(device="cpu").manual_seed(123)
        return noise_mixin._get_timestep_discrete(
            num_train_timesteps=1000,
            deterministic=False,
            generator=generator,
            batch_size=8,
            config=beta_config,
        )

    def sample_beta(noising_weight: float, noising_bias: float):
        timesteps = beta_timesteps(noising_weight, noising_bias, 1)
        repeated_timesteps = beta_timesteps(noising_weight, noising_bias, 999)
        if not torch.equal(timesteps, repeated_timesteps):
            fail("BETA timestep smoke ignored the supplied generator")
        if timesteps.shape != (8,) or timesteps.dtype != torch.int32:
            fail("BETA timestep smoke returned wrong shape or dtype")
        if int(timesteps.min()) < 0 or int(timesteps.max()) >= 1000:
            fail("BETA timestep smoke returned out-of-range timesteps")

    sample_beta(0.0, 0.0)
    sample_beta(0.5, 1.0)
    sample_beta(1.0, 2.0)
    sample_beta(0.7, 1.3)

    for noising_weight, noising_bias in [(-0.1, 1.0), (1.0, -0.1)]:
        invalid_config = TrainConfig.default_values()
        invalid_config.timestep_distribution = TimestepDistribution.BETA
        invalid_config.noising_weight = noising_weight
        invalid_config.noising_bias = noising_bias
        try:
            invalid_config.validate_for_training()
        except ValueError:  # noqa: PERF203
            pass
        else:
            fail("BETA timestep distribution accepted negative alpha/beta config")

    noise_source = read_text("modules/modelSetup/mixin/ModelSetupNoiseMixin.py")
    for term in ["TimestepDistribution.BETA", "TimestepDistribution.SPEED", "__sample_gamma_unit", "betas", "sigmas"]:
        if term not in noise_source:
            fail(f"noise mixin missing timestep distribution support for {term}")
    if "torch.distributions.Beta" in noise_source:
        fail("BETA timestep distribution bypasses the supplied generator")


def check_flow_timestep_rejection() -> None:
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.ModelType import ModelType
    from modules.util.enum.TimestepDistribution import TimestepDistribution

    for distribution in [TimestepDistribution.BETA, TimestepDistribution.SPEED]:
        config = TrainConfig.default_values()
        config.model_type = ModelType.FLUX_2
        config.timestep_distribution = distribution
        try:
            config.validate_for_training()
        except ValueError:  # noqa: PERF203
            pass
        else:
            fail(f"flow model accepted unsupported timestep distribution {distribution}")


def check_immiscible_contract() -> None:
    from modules.util.immiscible_diffusion import immiscible_oversampling

    import torch

    latents = torch.tensor([[[[0.0]]], [[[10.0]]]])
    noise = torch.tensor([[[[[10.0]]], [[[0.0]]]], [[[[0.0]]], [[[10.0]]]]])
    sampled = immiscible_oversampling(latents, noise)
    if not torch.equal(sampled, torch.tensor([[[[0.0]]], [[[10.0]]]])):
        fail("immiscible oversampling did not choose nearest noise assignments")

    config_source = read_text("modules/util/config/TrainConfig.py")
    if "k_noise_sampling must be at least 1" not in config_source:
        fail("TrainConfig does not validate k_noise_sampling")


def check_perturbations_contract() -> None:
    sdxl_source = read_text("modules/modelSetup/BaseStableDiffusionXLSetup.py")
    noise_source = read_text("modules/modelSetup/mixin/ModelSetupNoiseMixin.py")
    config_source = read_text("modules/util/config/TrainConfig.py")
    for term in ["cep_enabled", "cep_gamma", "ciop_noise_weight", "ciop_p"]:
        if term not in config_source:
            fail(f"TrainConfig missing perturbation field {term}")
    for term in ["_apply_conditional_embedding_perturbation", "_apply_ciop"]:
        if term not in noise_source:
            fail(f"noise mixin missing {term}")
    for term in ["_apply_conditional_embedding_perturbation", "_apply_ciop"]:
        if term not in sdxl_source:
            fail(f"SDXL setup does not apply {term}")


def check_flux2_edit_contract() -> None:
    from modules.util.config.SampleConfig import SampleConfig
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.ModelType import ModelType

    mixin_source = read_text("modules/dataLoader/mixin/DataLoaderText2ImageMixin.py")
    flux2_loader_source = read_text("modules/dataLoader/Flux2BaseDataLoader.py")
    flux2_setup_source = read_text("modules/modelSetup/BaseFlux2Setup.py")
    model_source = read_text("modules/model/Flux2Model.py")
    sampler_source = read_text("modules/modelSampler/Flux2Sampler.py")
    sample_config_source = read_text("modules/util/config/SampleConfig.py")
    sample_frame_source = read_text("modules/ui/SampleFrame.py")
    sample_window_source = read_text("modules/ui/SampleWindow.py")
    sampling_tab_source = read_text("modules/ui/SamplingTab.py")

    for term in [
        "config.model_type.is_flux_2() and config.custom_conditioning_image",
        "SelectFirstInput(in_names=[\"custom_conditioning_image\"], out_name=\"conditioning_image\")",
    ]:
        if term not in mixin_source:
            fail(f"Flux2 custom conditioning image mapping missing {term}")

    for term in ["latent_conditioning_image_distribution", "latent_conditioning_image"]:
        if term not in flux2_loader_source:
            fail(f"Flux2 dataloader missing {term}")

    for term in ["latent_conditioning_image", "prepare_latent_image_ids", "[:, :image_seq_len, :]"]:
        if term not in flux2_setup_source:
            fail(f"Flux2 setup missing edit-conditioning term {term}")

    if "FLUX_2_EDIT" in model_source + flux2_setup_source + flux2_loader_source + sampler_source:
        fail("Flux2 edit introduced forbidden duplicate model identity")
    if "index: int = 0" not in model_source:
        fail("Flux2Model.prepare_latent_image_ids does not expose conditioning image index")

    for term in [
        "custom_conditioning_image: bool",
        "self.custom_conditioning_image = train_config.custom_conditioning_image",
        "(\"custom_conditioning_image\", False, bool, False)",
    ]:
        if term not in sample_config_source:
            fail(f"SampleConfig does not mirror Flux2 edit switch: missing {term}")
    if "model_type.is_flux_2() and sample.custom_conditioning_image" not in sample_frame_source:
        fail("SampleFrame exposes Flux2 base image outside custom_conditioning_image")
    if "self.sample.from_train_config(train_config)" not in sample_window_source:
        fail("SampleWindow does not seed SampleConfig from TrainConfig before building Flux2 UI")
    if ".from_train_config(self.train_config)" not in sampling_tab_source:
        fail("SamplingTab does not seed SampleConfig from TrainConfig before building sample UI")
    for term in [
        "sample_config.base_image_path and not sample_config.custom_conditioning_image",
        "base_image_path=sample_config.base_image_path if sample_config.custom_conditioning_image else \"\"",
    ]:
        if term not in sampler_source:
            fail(f"Flux2Sampler does not gate base image sampling: missing {term}")

    train_config = TrainConfig.default_values()
    train_config.model_type = ModelType.FLUX_2
    sample_config = SampleConfig.default_values(ModelType.FLUX_2)
    if sample_config.from_train_config(train_config) is not sample_config:
        fail("SampleConfig.from_train_config does not return the mutated SampleConfig")
    if sample_config.custom_conditioning_image:
        fail("SampleConfig enables Flux2 custom conditioning image by default")
    train_config.custom_conditioning_image = True
    sample_config.from_train_config(train_config)
    if not sample_config.custom_conditioning_image:
        fail("SampleConfig does not mirror enabled Flux2 custom conditioning image")
    loaded_sample = SampleConfig.default_values(ModelType.FLUX_2).from_train_config(train_config).from_dict(
        sample_config.to_dict()
    )
    if not isinstance(loaded_sample, SampleConfig):
        fail("SampleConfig add/load chain does not return SampleConfig")


def check_flux2_dropout_block() -> None:
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.ModelType import ModelType

    config = TrainConfig.default_values()
    config.model_type = ModelType.FLUX_2
    config.text_encoder.dropout_probability = 0.1
    try:
        config.validate_for_training()
    except ValueError:
        pass
    else:
        fail("Flux2 text encoder dropout was not rejected before model execution")


def check_hidream_tokenizer_contract() -> None:
    source = read_text("modules/modelLoader/hiDream/HiDreamModelLoader.py")
    for term in ["tokenizer_2", "include_text_encoder_4"]:
        if term not in source:
            fail(f"HiDream tokenizer loader fix missing {term}")


def check_guarded_files() -> None:
    changed = run_git(["diff", "--name-only", "HEAD"]).splitlines()
    untracked = run_git(["ls-files", "--others", "--exclude-standard"]).splitlines()
    guarded_hits = []

    for path in changed + untracked:
        if path in ALLOWED_GUARDED_REQUIREMENTS_DIFFS:
            diff_lines = run_git(["diff", "HEAD", "--", path]).splitlines()
            material_diff_lines = [
                line
                for line in diff_lines
                if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
            ]
            if material_diff_lines == ALLOWED_GUARDED_REQUIREMENTS_DIFFS[path]:
                continue

        if path in GUARDED_EXACT_PATHS:
            guarded_hits.append(path)
        if any(path.startswith(prefix) for prefix in GUARDED_PATH_PREFIXES):
            guarded_hits.append(path)
        if any(Path(path).match(pattern) for pattern in GUARDED_GLOBS):
            guarded_hits.append(path)

    if guarded_hits:
        fail("guarded bootstrap/cloud/docker files changed: " + ", ".join(sorted(set(guarded_hits))))


def check_blocked_residue() -> None:
    def blocked_offenders(path: str, source: str) -> list[str]:
        file_offenders: list[str] = []
        for line_number, line in enumerate(source.splitlines(), start=1):
            lower_line = line.lower()
            for term in BLOCKED_RUNTIME_TERMS:
                lower_term = term.lower()
                if lower_term not in lower_line:
                    continue
                exemptions = BLOCKED_RUNTIME_TERM_EXEMPTIONS.get(lower_term, [])
                if any(exemption.lower() in lower_line for exemption in exemptions):
                    continue
                file_offenders.append(f"{path}:{line_number}:{term}")
        return file_offenders

    for term in BLOCKED_RUNTIME_TERMS:
        if not blocked_offenders("<validator-fixture>", f"forbidden {term}"):
            fail(f"blocked residue fixture did not catch {term}")
    if blocked_offenders("<validator-fixture>", "component.set_attention_backend(value)"):
        fail("blocked residue fixture rejects allowed set_attention_backend API use")

    pathspecs = ["modules", "scripts", "resources", "training_presets", ".sangoi/CHANGELOG.md"]
    tracked_paths = run_git(["ls-files", "--", *pathspecs]).splitlines()
    untracked_paths = run_git(["ls-files", "--others", "--exclude-standard", "--", *pathspecs]).splitlines()
    paths = sorted(set(tracked_paths + untracked_paths) - BLOCKED_SCAN_EXCLUDED_PATHS)
    offenders: list[str] = []
    for path in paths:
        full_path = REPO_ROOT / path
        if not full_path.is_file():
            continue
        source = full_path.read_text(encoding="utf-8", errors="ignore")
        offenders.extend(blocked_offenders(path, source))

    if offenders:
        fail("blocked/deferred PR residue found: " + ", ".join(offenders))


def check_strict_config_contract() -> None:
    from modules.util.config.TrainConfig import TrainConfig

    config = TrainConfig.default_values()
    config.validate_for_training()
    payload = config.to_dict()

    for stale_key in ["resolution_quantization", "distillation", "cfg_distillation"]:
        try:  # noqa: PERF203
            TrainConfig.default_values().from_dict(payload | {stale_key: True})
        except ValueError:  # noqa: PERF203
            pass
        else:
            fail(f"TrainConfig accepted stale key {stale_key}")

    invalid_values = {
        "tensorboard_resume_run": "not-bool",
        "patience": "not-bool",
        "patience_epochs": "not-int",
        "validation_timestep_mode": 123,
        "validation_timestep_values": "not-an-int",
        "validation_timestep_seed": "not-int",
        "prefetch_next_batch": "not-bool",
        "attention_mechanism": 123,
        "k_noise_sampling": "not-int",
        "cep_enabled": "not-bool",
        "cep_gamma": "not-float",
        "ciop_noise_weight": "not-float",
        "ciop_p": "not-float",
        "lora_te_scale": "not-float",
        "lora_unet_scale": "not-float",
        "scaled_oft": "not-bool",
        "dora_oft": "not-bool",
        "oft_coft": "not-bool",
        "coft_eps": "not-float",
        "lokr_dim": "not-int",
        "lokr_decompose_both": "not-bool",
        "lokr_decompose_factor": "not-int",
        "lokr_use_tucker": "not-bool",
        "lokr_weight_decompose": "not-bool",
        "lokr_dora_on_output": "not-bool",
        "lokr_full_matrix": "not-bool",
        "lokr_vec_trick": "not-bool",
    }
    for field_name, invalid_value in invalid_values.items():
        try:  # noqa: PERF203
            TrainConfig.default_values().from_dict(payload | {field_name: invalid_value})
        except ValueError:  # noqa: PERF203
            pass
        else:
            fail(f"TrainConfig accepted invalid present value for {field_name}")

    tree = ast.parse(read_text("modules/util/config/TrainConfig.py"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "setdefault":
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in payload:
                fail(f"TrainConfig uses setdefault for canonical key {node.args[0].value}")

    for path in [
        "modules/util/config/TrainConfig.py",
        "modules/ui/TrainUI.py",
        "modules/ui/TrainingTab.py",
        "modules/ui/LoraTab.py",
        "modules/trainer/GenericTrainer.py",
        "modules/modelSetup/StableDiffusionLoRASetup.py",
        "modules/modelSetup/StableDiffusionXLLoRASetup.py",
    ]:
        source = read_text(path)
        for term in NO_ALIAS_TERMS:
            if term in source:
                fail(f"{path} contains forbidden compatibility term {term}")


def check_sangoi_invariants() -> None:
    from modules.module.LoRAModule import build_lora_key_manifest
    from modules.util.config.TrainConfig import TrainConfig

    config = TrainConfig.default_values()
    for key, expected in SANGOI_DEFAULTS.items():
        if getattr(config, key) != expected:
            fail(f"Sangoi TrainConfig default drifted for {key}: {getattr(config, key)!r}")

    class Wrapper:
        def state_dict(self):
            return {
                "lora_unet_down_blocks_0_resnets_0.alpha": 1,
                "lora_te1_text_model_encoder_layers_0.alpha": 1,
            }

    manifest_1 = build_lora_key_manifest([Wrapper()])
    manifest_2 = build_lora_key_manifest([Wrapper()])
    if manifest_1 != manifest_2:
        fail("LoRA key manifest is not deterministic")
    if sorted(manifest_1["keys_by_block"]) != list(manifest_1["keys_by_block"]):
        fail("LoRA key manifest block order is not deterministic")

    if (REPO_ROOT / "modules/cloud").exists():
        status = run_git(["status", "--porcelain", "--", "modules/cloud", "resources/docker", "scripts/train_remote.py"])
        if status.strip():
            fail("cloud/docker guarded sources have pending changes")


def check_hunyuan_comfy_contract() -> None:
    from modules.modelLoader.hunyuanVideo.HunyuanVideoLoRALoader import HunyuanVideoLoRALoader
    from modules.util.convert.lora.convert_hunyuan_video_lora import convert_hunyuan_video_lora_to_comfyui
    from modules.util.enum.ModelFormat import ModelFormat

    import torch

    if not hasattr(ModelFormat, "COMFY_LORA"):
        fail("ModelFormat.COMFY_LORA is missing")

    convert_ui_source = read_text(REPO_ROOT / "modules/ui/ConvertModelUI.py")
    for required_snippet in [
        "ModelType.HUNYUAN_VIDEO",
        "TrainingMethod.LORA",
        "options.append((\"ComfyUI LoRA\", ModelFormat.COMFY_LORA))",
        "output_model_format_var.set(self.output_model_format_options[0][1])",
    ]:
        if required_snippet not in convert_ui_source:
            fail(f"ConvertModelUI does not gate ComfyUI LoRA output format: missing {required_snippet}")

    saver_source = read_text(REPO_ROOT / "modules/modelSaver/mixin/LoRASaverMixin.py")
    if "ModelFormat.COMFY_LORA" not in saver_source or "raise NotImplementedError" not in saver_source:
        fail("generic LoRA saver must fail loud for unsupported ComfyUI LoRA output")

    processed = HunyuanVideoLoRALoader()._preprocess_state_dict(
        {
            "transformer.double_blocks.0.img_mlp.fc1.lora_up.weight": object(),
            "lora_llama_x.lora_up.weight": object(),
            "lora_te1_x.lora_up.weight": object(),
        }
    )
    for key in [
        "lora_transformer.double_blocks.0.img_mlp.fc0.lora_up.weight",
        "lora_te1_x.lora_up.weight",
        "lora_te2_x.lora_up.weight",
    ]:
        if key not in processed:
            fail(f"HunyuanVideo LoRA preprocessing missing {key}")

    def lora_state(prefix: str) -> dict[str, torch.Tensor]:
        return {
            f"{prefix}.lora_up.weight": torch.ones(2, 1),
            f"{prefix}.lora_down.weight": torch.ones(1, 2),
            f"{prefix}.alpha": torch.tensor(1.0),
        }

    comfy_input = {}
    for prefix in [
        "transformer.double_blocks.0.img_attn_qkv.0",
        "transformer.double_blocks.0.img_attn_qkv.1",
        "transformer.double_blocks.0.img_attn_qkv.2",
        "transformer.single_blocks.0.linear1.0",
        "transformer.single_blocks.0.linear1.1",
        "transformer.single_blocks.0.linear1.2",
        "transformer.single_blocks.0.linear1.3",
        "llama.layers.0.self_attn.q_proj",
        "clip_l.text_model.encoder.layers.0.self_attn.q_proj",
        "other_model.block",
    ]:
        comfy_input |= lora_state(prefix)
    comfy_input["bundle_emb.weight"] = torch.ones(1)

    comfy_output = convert_hunyuan_video_lora_to_comfyui(comfy_input)
    for required_prefix in [
        "transformer.double_blocks.0.img_attn_qkv",
        "transformer.single_blocks.0.linear1",
    ]:
        if not any(key.startswith(required_prefix) for key in comfy_output):
            fail(f"Hunyuan Comfy output missing {required_prefix}")
    for forbidden_prefix in ["llama.", "clip_l.", "lora_llama", "lora_te", "bundle_emb", "other_model"]:
        if any(key.startswith(forbidden_prefix) for key in comfy_output):
            fail(f"Hunyuan Comfy output leaked forbidden key prefix {forbidden_prefix}")


def check_adapters_contract() -> None:
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.ModelType import PeftType

    config = TrainConfig.default_values()
    for field in [
        "scaled_oft",
        "dora_oft",
        "oft_coft",
        "coft_eps",
        "lokr_dim",
        "lokr_decompose_both",
        "lokr_decompose_factor",
        "lokr_use_tucker",
        "lokr_weight_decompose",
        "lokr_dora_on_output",
        "lokr_full_matrix",
        "lokr_vec_trick",
        "lora_te_scale",
        "lora_unet_scale",
    ]:
        if not hasattr(config, field):
            fail(f"TrainConfig missing adapter field {field}")
    if not hasattr(PeftType, "LOKR"):
        fail("PeftType missing LOKR")

    lora_source = read_text("modules/module/LoRAModule.py")
    for term in ["class LoKrModule", "PeftType.LOKR", "DoRAOFTModule", "export_lora_key_manifest"]:
        if term not in lora_source:
            fail(f"LoRA module adapter support missing {term}")


def check_prefetch_cache_contract() -> None:
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.PrefetchIterator import PrefetchIterator

    config = TrainConfig.default_values()
    config.prefetch_next_batch = True
    config.latent_caching = False
    try:
        config.validate_for_training()
    except ValueError:
        pass
    else:
        fail("prefetch_next_batch without latent_caching did not fail")

    config = TrainConfig.default_values()
    config.prefetch_next_batch = True
    config.only_cache = True
    try:
        config.validate_for_training()
    except ValueError:
        pass
    else:
        fail("prefetch_next_batch with only_cache did not fail")

    iterator = PrefetchIterator([1, 2, 3])
    if list(iterator) != [1, 2, 3]:
        fail("PrefetchIterator did not yield all producer items")

    def failing_iter():
        yield 1
        raise RuntimeError("prefetch producer failure")

    iterator = PrefetchIterator(failing_iter())
    next(iterator)
    try:
        next(iterator)
    except RuntimeError as exc:
        if "prefetch producer failure" not in str(exc):
            raise
    else:
        fail("PrefetchIterator did not propagate producer exception")


def check_prefetch_device_boundary_contract() -> None:
    import threading

    from modules.util.PrefetchIterator import PrefetchIterator

    trainer_source = read_text("modules/trainer/GenericTrainer.py")
    prefetch_source = read_text("modules/util/PrefetchIterator.py")
    for term in [
        "__can_prefetch_next_batch",
        "self.config.latent_caching",
        "not self.config.only_cache",
        "not self.config.train_text_encoder_or_embedding()",
        "prefetch_batches.wait_until_idle()",
    ]:
        if term not in trainer_source:
            fail(f"GenericTrainer missing prefetch device-boundary term {term}")
    for term in [
        "try:\n                for batch in batches:",
        "finally:\n                if prefetch_batches is not None:\n                    prefetch_batches.close()",
    ]:
        if term not in trainer_source:
            fail(f"GenericTrainer does not close prefetch worker on consumer-loop exit: missing {term}")
    for term in ["def wait_until_idle", "_idle_event", "def close", "def is_running"]:
        if term not in prefetch_source:
            fail(f"PrefetchIterator missing lifecycle term {term}")

    iterator_holder = {}
    try:
        iterator = PrefetchIterator(range(1000000), queue_size=1, stop_poll_interval=0.001)
        iterator_holder["iterator"] = iterator
        try:
            for _ in iterator:
                raise RuntimeError("prefetch consumer failure")
        finally:
            iterator.close()
    except RuntimeError as exc:
        if "prefetch consumer failure" not in str(exc):
            raise
    else:
        fail("prefetch consumer failure smoke did not raise")
    if iterator_holder["iterator"].is_running():
        fail("PrefetchIterator worker survived consumer failure cleanup")

    iterator = PrefetchIterator(range(1000000), queue_size=1, stop_poll_interval=0.001)
    if next(iterator) != 0:
        iterator.close()
        fail("PrefetchIterator queue-full smoke did not yield the first item")
    wait_returned = threading.Event()

    def wait_for_idle():
        iterator.wait_until_idle()
        wait_returned.set()

    wait_thread = threading.Thread(target=wait_for_idle)
    wait_thread.start()
    if not wait_returned.wait(timeout=1.0):
        iterator.close()
        wait_thread.join(timeout=1.0)
        fail("PrefetchIterator.wait_until_idle blocked while producer was queue-full")
    iterator.close()
    wait_thread.join(timeout=1.0)
    if wait_thread.is_alive():
        fail("PrefetchIterator.wait_until_idle smoke thread did not exit")


def check_source_guards() -> None:
    check_guarded_files()
    check_blocked_residue()


CHECKS = {
    "accumulator": check_mid_accum_resume,
    "accumulator-mid-step": check_mid_accum_resume,
    "adapters": check_adapters_contract,
    "attention-backend": check_attention_backend,
    "blocked-residue": check_blocked_residue,
    "component-stop": check_component_stop_contract,
    "flow-timestep-rejection": check_flow_timestep_rejection,
    "flux2-dropout-block": check_flux2_dropout_block,
    "flux2-edit": check_flux2_edit_contract,
    "guarded-files": check_guarded_files,
    "hidream-tokenizer": check_hidream_tokenizer_contract,
    "hunyuan-comfy": check_hunyuan_comfy_contract,
    "immiscible": check_immiscible_contract,
    "lora-manifest": check_sangoi_invariants,
    "mid-accum-resume": check_mid_accum_resume,
    "patience": check_patience_contract,
    "perturbations": check_perturbations_contract,
    "prefetch-cache": check_prefetch_cache_contract,
    "prefetch-device-boundary": check_prefetch_device_boundary_contract,
    "resume-actions": check_resume_actions,
    "sangoi-invariants": check_sangoi_invariants,
    "source-guards": check_source_guards,
    "strict-config": check_strict_config_contract,
    "tensorboard": check_tensorboard_resume,
    "timed-actions": check_resume_actions,
    "timestep-distributions": check_timestep_distributions,
    "validation-timesteps": check_validation_timesteps,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="append", choices=sorted(CHECKS), help="Run one check. Repeatable.")
    args = parser.parse_args()

    selected_checks = args.check or sorted(CHECKS)
    for check_name in selected_checks:
        CHECKS[check_name]()
        print(f"ok {check_name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
