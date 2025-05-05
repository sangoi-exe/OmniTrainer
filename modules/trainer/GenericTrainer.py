import os
import json
import copy
import time
import shutil
import traceback
import contextlib
import collections
from pathlib import Path
from datetime import datetime
from collections.abc import Callable
from typing import Dict

from numpy import dtype, inf

from modules.trainer.BaseTrainer import BaseTrainer
from modules.dataLoader.BaseDataLoader import BaseDataLoader

from modules.model.BaseModel import BaseModel
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput


from modules.util import create, path_util
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_grad_scaler, enable_grad_scaling
from modules.util.enum.FileType import FileType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.memory_util import TorchMemoryRecorder
from modules.util.time_util import get_string_timestamp
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils.hooks import RemovableHandle
from modules.util.TensorBoardManager import TensorBoardManager
from torchvision.transforms.functional import pil_to_tensor

import huggingface_hub
from tqdm import tqdm
from requests.exceptions import ConnectionError

from modules.sangoi.ConvergeControl import ConvergeControl
from modules.sangoi.AdaptiveDCoef import AdaptiveDCoef
from modules.sangoi.DataRecorder import DataRecorder
from modules.sangoi.TrainGPS import TrainGPS
from modules.sangoi.logFun import logFun

class GenericTrainer(BaseTrainer):
    model_loader: BaseModelLoader
    model_setup: BaseModelSetup
    data_loader: BaseDataLoader
    model_saver: BaseModelSaver
    model_sampler: BaseModelSampler
    model: BaseModel | None
    validation_data_loader: BaseDataLoader
    previous_sample_time: float
    sample_queue: list[Callable]
    parameters: list[Parameter]
    grad_hook_handles: list[RemovableHandle]

    tensorboard: TensorBoardManager
    adaptive_dcoef: AdaptiveDCoef | None
    recorder: DataRecorder | None
    converge_control: ConvergeControl | None
    is_run2: bool # Flag to easily check run mode
    _temp_recorder_data: collections.defaultdict # Temporary storage for recorder data before log_step

    # atributos para pause
    is_paused: bool
    pause_request_locked: bool # Para travar o switch da UI
    pause_requested_at_epoch_end: bool

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super().__init__(config, callbacks, commands)

        tensorboard_log_dir = os.path.join(config.workspace_dir, "tensorboard")
        os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)
        self.tensorboard = TensorBoardManager(
            log_dir=os.path.join(
                tensorboard_log_dir,
                f"{config.save_filename_prefix}{get_string_timestamp()}",
            )
        )
        if config.tensorboard:
            super()._start_tensorboard()

        self.model = None
        self.one_step_trained = False
        self.grad_hook_handles = []


        self.is_paused = False
        self.pause_request_locked = False
        self.pause_requested_at_epoch_end = False
        self.converge_control = None
        self.recorder = None
        self.adaptive_dcoef = None
        self.converge_control = None
        self.train_device =  torch.device(getattr(self.config, "train_device"))
        self.train_dtype: None

        # Component initialization flags (from config, with defaults)
        dcoef_debug = getattr(self.config, "dcoef_debug", True)
        dcoef_verbose = getattr(self.config, "dcoef_verbose", True)
        recorder_debug = getattr(self.config, "recorder_debug", True)
        recorder_verbose = getattr(self.config, "recorder_verbose", True)
        converge_debug = getattr(self.config, "converge_debug", True)
        converge_verbose = getattr(self.config, "converge_verbose", True)

        # Determine the current run number
        self.run_number = getattr(self.config, "run_number", 1) # Default to 1 if not set
        logFun(f"[Trainer] Configurando para Run {self.run_number}.", lvl="debug")

        self.recorder = None
        if getattr(self.config, "data_recorder", False):
            # Recorder is active based on its flag. Its behavior (recording) is mainly for Run 1,
            # but it might be optionally used in Run 2 for analysis if needed.
            # No specific check for run_number needed here for activation, just for the 'purpose' log maybe.
            self.recorder = DataRecorder(
                debug=recorder_debug,
                verbose=recorder_verbose
            )
            purpose = "coleta de dados" if self.run_number == 1 else "gravação opcional de dinâmica"
            logFun(f"[DataRecorder] Inicializado (flag data_recorder=True) para {purpose} na Run {self.run_number}.", lvl="debug")
        else:
            logFun("[DataRecorder] Desativado (flag data_recorder=False).", lvl="debug")

        # 2. AdaptiveDCoef
        self.adaptive_dcoef = None
        # AdaptiveDCoef only makes sense and should only be activated in Run 2 AND if its flag is True.
        if self.run_number == 2 and getattr(self.config, "adpt_dcoef_use_it", False):
            try:
                self.adaptive_dcoef = AdaptiveDCoef(
                    debug=dcoef_debug,
                    verbose=dcoef_verbose
                )
                logFun(f"[AdaptiveDCoef] Inicializado com sucesso.", lvl="success")
            except (FileNotFoundError, ValueError, Exception) as e:
                logFun(f"[Trainer Aviso] Falha ao inicializar AdaptiveDCoef para Run 2: {e}. d_coef dinâmico será desativado.", lvl="debug")
                self.adaptive_dcoef = None # Ensure it's None if init fails
        elif getattr(self.config, "adpt_dcoef_use_it", False):
            logFun("[AdaptiveDCoef] Desativado (não é Run 2 ou flag adpt_dcoef_use_it=False).", lvl="debug")

    def start(self):
        self.__save_config_to_workspace()

        if self.config.clear_cache_before_training and self.config.latent_caching:
            self.__clear_cache()

        if self.config.train_dtype.enable_tf():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model_loader = self.create_model_loader()
        self.model_setup = self.create_model_setup()

        self.callbacks.on_update_status("loading the model")

        model_names = self.config.model_names()

        if self.config.continue_last_backup:
            self.callbacks.on_update_status("searching for previous backups")
            last_backup_path = self.config.get_last_backup_path()

            if last_backup_path:
                if self.config.training_method == TrainingMethod.LORA:
                    model_names.lora = last_backup_path
                elif self.config.training_method == TrainingMethod.EMBEDDING:
                    model_names.embedding.model_name = last_backup_path
                else:  # fine-tunes
                    model_names.base_model = last_backup_path

                print(f"Continuing training from backup '{last_backup_path}'...")
            else:
                print("No backup found, continuing without backup...")

        if self.config.secrets.huggingface_token != "":
            self.callbacks.on_update_status("logging into Hugging Face")
            with contextlib.suppress(ConnectionError):
                huggingface_hub.login(
                    token=self.config.secrets.huggingface_token,
                    new_session=False,
                )

        self.callbacks.on_update_status("loading the model")
        self.model = self.model_loader.load(
            model_type=self.config.model_type,
            model_names=model_names,
            weight_dtypes=self.config.weight_dtypes(),
        )
        self.model.train_config = self.config
        self.train_dtype = self.model.train_dtype.torch_dtype()
        self.callbacks.on_update_status("running model setup")

        self.model_setup.setup_optimizations(self.model, self.config)
        self.model_setup.setup_train_device(self.model, self.config)
        self.model_setup.setup_model(self.model, self.config, self.tensorboard)
        self.model.to(self.temp_device)
        self.model.eval()
        torch_gc()


        self.callbacks.on_update_status("creating the data loader/caching")

        self.data_loader = self.create_data_loader(
            self.model, self.model.train_progress
        )
        self.model_saver = self.create_model_saver()

        self.model_sampler = self.create_model_sampler(self.model)
        self.previous_sample_time = -1
        self.sample_queue = []

        self.parameters = self.model.parameters.parameters()
        if self.config.validation:
            self.validation_data_loader = self.create_data_loader(
                self.model, self.model.train_progress, is_validation=True
            )

        # iniciando o convergecontrol aqui pra poder usar a função de calcular o tamanho do treino        
        if getattr(self.config, "convctrl_use_it", False):
            # Lê a flag da configuração, com padrão True se não existir
            enable_freeze_action_flag = getattr(self.config, "converge_control_enable_freeze", True)
            logFun(f"[Trainer] ConvergeControl freeze action flag from config: {enable_freeze_action_flag}", lvl="debug")
            # END: Modificação solicitada - Ler e passar a flag enable_freeze_action

            converge_control = ConvergeControl(
                run_number=self.run_number,
                k_confirm=getattr(self.config, "converge_control_k_confirm", 5),
                total_epochs=self.config.epochs,
                enable_freeze_action=enable_freeze_action_flag,  # Passa a flag lida da config
            )

            self.converge_control = converge_control
            logFun(f"[ConvergeControl] Inicializado com sucesso.", lvl="debug")
        else:
            logFun("[ConvergeControl] Desativado (flag convctrl_use_it=False).", lvl="debug")

            
    def __save_config_to_workspace(self):
        path = path_util.canonical_join(self.config.workspace_dir, "config")
        os.makedirs(Path(path).absolute(), exist_ok=True)
        path = path_util.canonical_join(path, f"{get_string_timestamp()}.json")
        with open(path, "w") as f:
            json.dump(self.config.to_pack_dict(secrets=False), f, indent=4)

    def __clear_cache(self):
        print(
            f"Clearing cache directory {self.config.cache_dir}!"
            f"You can disable this if you want to continue using the same cache."
        )
        if os.path.isdir(self.config.cache_dir):
            for filename in os.listdir(self.config.cache_dir):
                path = os.path.join(self.config.cache_dir, filename)
                if os.path.isdir(path) and (
                    filename.startswith("epoch-") or filename in ["image", "text"]
                ):
                    shutil.rmtree(path)

    def __prune_backups(self, backups_to_keep: int):
        backup_dirpath = os.path.join(self.config.workspace_dir, "backup")
        if os.path.exists(backup_dirpath):
            backup_directories = sorted(
                [
                    dirpath
                    for dirpath in os.listdir(backup_dirpath)
                    if os.path.isdir(os.path.join(backup_dirpath, dirpath))
                ],
                reverse=True,
            )

            for dirpath in backup_directories[backups_to_keep:]:
                dirpath = os.path.join(backup_dirpath, dirpath)
                try:
                    shutil.rmtree(dirpath)
                except Exception:
                    print(f"Could not delete old rolling backup {dirpath}")

        return

    def __enqueue_sample_during_training(self, fun: Callable):
        self.sample_queue.append(fun)

    def __execute_sample_during_training(self):
        for fun in self.sample_queue:
            fun()
        self.sample_queue = []

    def __sample_loop(
        self,
        train_progress: TrainProgress,
        train_device: torch.device,
        sample_config_list: list[SampleConfig],
        folder_postfix: str = "",
        is_custom_sample: bool = False,
    ):
        for i, sample_config in enumerate(sample_config_list):
            if sample_config.enabled:
                try:
                    safe_prompt = path_util.safe_filename(sample_config.prompt)

                    if is_custom_sample:
                        sample_dir = os.path.join(
                            self.config.workspace_dir,
                            "samples",
                            "custom",
                        )
                    else:
                        sample_dir = os.path.join(
                            self.config.workspace_dir,
                            "samples",
                            f"{str(i)} - {safe_prompt}{folder_postfix}",
                        )

                    sample_path = os.path.join(
                        sample_dir,
                        f"{get_string_timestamp()}-training-sample-{train_progress.filename_string()}",
                    )

                    def on_sample_default(sampler_output: ModelSamplerOutput):
                        if (
                            self.config.samples_to_tensorboard
                            and sampler_output.file_type == FileType.IMAGE
                        ):
                            self.tensorboard.add_image(
                                f"sample{str(i)} - {safe_prompt}",
                                pil_to_tensor(sampler_output.data),  # noqa: B023
                                train_progress.global_step,
                            )
                        self.callbacks.on_sample_default(sampler_output)

                    def on_sample_custom(sampler_output: ModelSamplerOutput):
                        self.callbacks.on_sample_custom(sampler_output)

                    on_sample = (
                        on_sample_custom if is_custom_sample else on_sample_default
                    )
                    on_update_progress = (
                        self.callbacks.on_update_sample_custom_progress
                        if is_custom_sample
                        else self.callbacks.on_update_sample_default_progress
                    )

                    self.model.to(self.temp_device)
                    self.model.eval()

                    sample_config = copy.copy(sample_config)
                    sample_config.from_train_config(self.config)

                    self.model_sampler.sample(
                        sample_config=sample_config,
                        destination=sample_path,
                        image_format=self.config.sample_image_format,
                        video_format=self.config.sample_video_format,
                        audio_format=self.config.sample_audio_format,
                        on_sample=on_sample,
                        on_update_progress=on_update_progress,
                    )
                except Exception:
                    traceback.print_exc()
                    print("Error during sampling, proceeding without sampling")

                torch_gc()

    def __sample_during_training(
        self,
        train_progress: TrainProgress,
        train_device: torch.device,
        sample_params_list: list[SampleConfig] = None,
    ):
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()
        torch_gc()

        self.callbacks.on_update_status("sampling")

        is_custom_sample = False
        if not sample_params_list:
            if self.config.samples is not None:
                sample_params_list = self.config.samples
            else:
                with open(self.config.sample_definition_file_name, "r") as f:
                    samples = json.load(f)
                    for i in range(len(samples)):
                        samples[i] = SampleConfig.default_values().from_dict(samples[i])
                    sample_params_list = samples
        else:
            is_custom_sample = True

        if self.model.ema:
            self.model.ema.copy_ema_to(self.parameters, store_temp=True)

        self.__sample_loop(
            train_progress=train_progress,
            train_device=train_device,
            sample_config_list=sample_params_list,
            is_custom_sample=is_custom_sample,
        )

        if self.model.ema:
            self.model.ema.copy_temp_to(self.parameters)

        # ema-less sampling, if an ema model exists
        if self.model.ema and not is_custom_sample and self.config.non_ema_sampling:
            self.__sample_loop(
                train_progress=train_progress,
                train_device=train_device,
                sample_config_list=sample_params_list,
                folder_postfix=" - no-ema",
            )

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def __validate(self, train_progress: TrainProgress):
        if self.__needs_validate(train_progress):
            self.validation_data_loader.get_data_set().start_next_epoch()
            current_epoch_length_validation = (
                self.validation_data_loader.get_data_set().approximate_length()
            )

            if current_epoch_length_validation == 0:
                return

            self.callbacks.on_update_status("calculating validation loss")
            self.model_setup.setup_train_device(self.model, self.config)

            torch_gc()

            step_tqdm_validation = tqdm(
                self.validation_data_loader.get_data_loader(),
                desc="validation_step",
                total=current_epoch_length_validation,
            )

            accumulated_loss_per_concept = {}
            concept_counts = {}
            mapping_seed_to_label = {}
            mapping_label_to_seed = {}

            for validation_batch in step_tqdm_validation:
                if self.__needs_gc(train_progress):
                    torch_gc()

                with torch.no_grad():
                    model_output_data = self.model_setup.predict(
                        self.model,
                        validation_batch,
                        self.config,
                        train_progress,
                        deterministic=True,
                    )
                    loss_validation = self.model_setup.calculate_loss(
                        self.model, validation_batch, model_output_data, self.config
                    )

                # since validation batch size = 1
                concept_name = validation_batch["concept_name"][0]
                concept_path = validation_batch["concept_path"][0]
                concept_seed = validation_batch["concept_seed"].item()
                loss = loss_validation.item()

                label = concept_name if concept_name else os.path.basename(concept_path)
                # check and fix collision to display both graphs in tensorboard
                if (
                    label in mapping_label_to_seed
                    and mapping_label_to_seed[label] != concept_seed
                ):
                    suffix = 1
                    new_label = f"{label}({suffix})"
                    while (
                        new_label in mapping_label_to_seed
                        and mapping_label_to_seed[new_label] != concept_seed
                    ):
                        suffix += 1
                        new_label = f"{label}({suffix})"
                    label = new_label

                if concept_seed not in mapping_seed_to_label:
                    mapping_seed_to_label[concept_seed] = label
                    mapping_label_to_seed[label] = concept_seed

                accumulated_loss_per_concept[concept_seed] = (
                    accumulated_loss_per_concept.get(concept_seed, 0) + loss
                )
                concept_counts[concept_seed] = concept_counts.get(concept_seed, 0) + 1

            for concept_seed, total_loss in accumulated_loss_per_concept.items():
                average_loss = total_loss / concept_counts[concept_seed]

                self.tensorboard.add_scalar(
                    f"loss/validation_step/{mapping_seed_to_label[concept_seed]}",
                    average_loss,
                    train_progress.global_step,
                )

            if len(concept_counts) > 1:
                total_loss = sum(
                    accumulated_loss_per_concept[key] for key in concept_counts
                )
                total_count = sum(concept_counts[key] for key in concept_counts)
                total_average_loss = total_loss / total_count

                self.tensorboard.add_scalar(
                    "loss/validation_step/total_average",
                    total_average_loss,
                    train_progress.global_step,
                )

    def __save_backup_config(self, backup_path):
        config_path = os.path.join(backup_path, "onetrainer_config")
        args_path = path_util.canonical_join(config_path, "args.json")
        concepts_path = path_util.canonical_join(config_path, "concepts.json")
        samples_path = path_util.canonical_join(config_path, "samples.json")

        os.makedirs(Path(config_path).absolute(), exist_ok=True)

        with open(args_path, "w") as f:
            json.dump(self.config.to_settings_dict(secrets=False), f, indent=4)
        if os.path.isfile(self.config.concept_file_name):
            shutil.copy2(self.config.concept_file_name, concepts_path)
        if os.path.isfile(self.config.sample_definition_file_name):
            shutil.copy2(self.config.sample_definition_file_name, samples_path)

    def backup(
        self,
        train_progress: TrainProgress,
        print_msg: bool = True,
        print_cb: Callable[[str], None] = print,
    ):
        torch_gc()

        self.callbacks.on_update_status("creating backup")

        backup_name = (
            f"{get_string_timestamp()}-backup-{train_progress.filename_string()}"
        )
        backup_path = os.path.join(self.config.workspace_dir, "backup", backup_name)

        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

        try:
            if print_msg:
                print_cb("Creating Backup " + backup_path)

            self.model_saver.save(
                self.model,
                self.config.model_type,
                ModelFormat.INTERNAL,
                backup_path,
                None,
            )

            self.__save_backup_config(backup_path)
        except Exception:
            traceback.print_exc()
            print("Could not save backup. Check your disk space!")
            try:
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
            except Exception:
                traceback.print_exc()
                print("Could not delete partial backup")
        finally:
            if self.config.rolling_backup:
                self.__prune_backups(self.config.rolling_backup_count)

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def save(
        self,
        train_progress: TrainProgress,
        print_msg: bool = True,
        print_cb: Callable[[str], None] = print,
    ):
        torch_gc()

        self.callbacks.on_update_status("saving")

        save_path = os.path.join(
            self.config.workspace_dir,
            "save",
            f"{self.config.save_filename_prefix}{get_string_timestamp()}-save-{train_progress.filename_string()}{self.config.output_model_format.file_extension()}",
        )
        if print_msg:
            print_cb("Saving " + save_path)

        try:
            if self.model.ema:
                self.model.ema.copy_ema_to(self.parameters, store_temp=True)

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()
            self.model_saver.save(
                model=self.model,
                model_type=self.config.model_type,
                output_model_format=self.config.output_model_format,
                output_model_destination=save_path,
                dtype=self.config.output_dtype.torch_dtype(),
            )
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()
        except Exception:
            traceback.print_exc()
            print("Could not save model. Check your disk space!")
            try:
                if os.path.isfile(save_path):
                    shutil.rmtree(save_path)
            except Exception:
                traceback.print_exc()
                print("Could not delete partial save")
        finally:
            if self.model.ema:
                self.model.ema.copy_temp_to(self.parameters)

        torch_gc()

    def __needs_sample(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "sample_skip_first",
            self.config.sample_skip_first,
            self.config.sample_after_unit,
            train_progress,
        ) and self.repeating_action_needed(
            "sample",
            self.config.sample_after,
            self.config.sample_after_unit,
            train_progress,
        )

    def __needs_backup(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "backup",
            self.config.backup_after,
            self.config.backup_after_unit,
            train_progress,
            start_at_zero=False,
        )

    def __needs_save(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "save_skip_first",
            self.config.save_skip_first,
            self.config.save_every_unit,
            train_progress,
        ) and self.repeating_action_needed(
            "save",
            self.config.save_every,
            self.config.save_every_unit,
            train_progress,
            start_at_zero=False,
        )

    def __needs_gc(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "gc", 5, TimeUnit.MINUTE, train_progress, start_at_zero=False
        )

    def __needs_validate(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "validate",
            self.config.validate_after,
            self.config.validate_after_unit,
            train_progress,
        )

    def __is_update_step(self, train_progress: TrainProgress) -> bool:
        return self.repeating_action_needed(
            "update_step",
            self.config.gradient_accumulation_steps,
            TimeUnit.STEP,
            train_progress,
            start_at_zero=False,
        )

    def __apply_fused_back_pass(self, scaler):
        if (
            self.config.optimizer.optimizer.supports_fused_back_pass()
            and self.config.optimizer.fused_back_pass
        ):
            if self.config.gradient_accumulation_steps > 1:
                print(
                    "Warning: activating fused_back_pass with gradient_accumulation_steps > 1 does not reduce VRAM usage."
                )

            for param_group in self.model.optimizer.param_groups:
                for i, parameter in enumerate(param_group["params"]):
                    # TODO: Find a better check instead of "parameter.requires_grad".
                    #       This will break if the some parameters don't require grad during the first training step.
                    if parameter.requires_grad:
                        if scaler:

                            def __grad_hook(
                                tensor: Tensor, param_group=param_group, i=i
                            ):
                                if self.__is_update_step(self.model.train_progress):
                                    scaler.unscale_parameter_(
                                        tensor, self.model.optimizer
                                    )
                                    if self.config.clip_grad_norm is not None:
                                        nn.utils.clip_grad_norm_(
                                            tensor, self.config.clip_grad_norm
                                        )
                                    scaler.maybe_opt_step_parameter(
                                        tensor, param_group, i, self.model.optimizer
                                    )
                                    tensor.grad = None
                        else:
                            def __grad_hook(
                                tensor: Tensor, param_group=param_group, i=i
                            ):
                                if self.__is_update_step(self.model.train_progress):
                                    if self.config.clip_grad_norm is not None:
                                        nn.utils.clip_grad_norm_(
                                            tensor, self.config.clip_grad_norm
                                        )
                                    self.model.optimizer.step_parameter(
                                        tensor, param_group, i
                                    )
                                    tensor.grad = None

                        handle = parameter.register_post_accumulate_grad_hook(
                            __grad_hook
                        )
                        self.grad_hook_handles.append(handle)

    def __before_eval(self):
        # Special case for schedule-free optimizers, which need eval()
        # called before evaluation. Can and should move this to a callback
        # during a refactoring.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

    def _handle_pause_logic(self):
        """Executa a lógica de pausa, movendo o modelo e esperando."""
        if not self.is_paused: # Segurança extra
            return

        logFun("[Trainer] Iniciando Pausa...", lvl="info")
        self.callbacks.on_update_status("Pausing... Moving model to CPU")
        try:
            self.model.to(self.temp_device) # Mover para CPU
            self.model.eval() # Garantir modo eval
            torch_gc() # Limpar VRAM
            logFun(f"[Trainer] Modelo movido para {self.temp_device}. VRAM liberada.", lvl="success")
            self.callbacks.on_update_status(f"Paused. Model on {self.temp_device}. Toggle switch to resume.")
            # Notificar UI que a pausa iniciou e o switch pode ser reativado (para desligar)
            if hasattr(self.callbacks, 'on_pause_initiated'):
                self.callbacks.on_pause_initiated()


            # Loop de espera pela retomada
            while self.is_paused:
                if self.commands.get_stop_command():
                    logFun("[Trainer] Comando STOP recebido durante a pausa. Interrompendo.", lvl="warning")
                    self.is_paused = False # Força a saída do loop de pausa
                    # Mantém o comando de stop ativo para o loop principal
                    break

                if self.commands.get_and_reset_resume_request():
                    logFun("[Trainer] Comando RESUME recebido.", lvl="info")
                    self.is_paused = False # Sinaliza para sair do loop
                    self.pause_request_locked = False # Desbloqueia a UI
                    # Notificar UI que o resume começou (switch ainda ativo)
                    if hasattr(self.callbacks, 'on_resume_started'):
                        self.callbacks.on_resume_started()
                    break # Sai do loop de espera

                time.sleep(0.5) # Evita busy-waiting, checa a cada 0.5s

            if not self.commands.get_stop_command(): # Só retoma se não for parar
                logFun("[Trainer] Retomando treinamento...", lvl="info")
                self.callbacks.on_update_status("Resuming... Moving model to GPU")
                try:
                    # Recarregar para o dispositivo de treino
                    self.model_setup.setup_train_device(self.model, self.config)
                    torch_gc() # Limpeza extra
                    logFun(f"[Trainer] Modelo movido de volta para {self.config.train_device}.", lvl="success")
                    self.callbacks.on_update_status("Training resumed.")
                    # Notificar UI que o resume foi concluído
                    if hasattr(self.callbacks, 'on_resume_completed'):
                        self.callbacks.on_resume_completed()

                except Exception as e:
                    logFun(f"[Trainer] Erro ao mover modelo de volta para GPU: {e}", lvl="error")
                    traceback.print_exc()
                    # Tentar continuar mesmo assim? Ou parar? Por segurança, parar.
                    self.commands.stop()
            else:
                logFun("[Trainer] Retomada cancelada devido ao comando STOP.", lvl="warning")


        except Exception as e:
            logFun(f"[Trainer] Erro durante o processo de pausa/retomada: {e}", lvl="error")
            traceback.print_exc()
            self.is_paused = False # Garante que não fique preso no estado pausado
            self.pause_request_locked = False
            # Considerar parar o treino em caso de erro grave aqui
            self.commands.stop()
            self.callbacks.on_update_status(f"Error during pause/resume: {e}")

    def train(self):
        scheduler_step_counter = 0
        # Determine target device and dtype from a model parameter if available

        def wrap_scheduler_step(orig_step):
            def wrapped(*args, **kwargs):
                nonlocal scheduler_step_counter
                scheduler_step_counter += 1
                print(
                    f"[DEBUG] scheduler.step() chamado {scheduler_step_counter} vezes"
                )
                print(f"[DEBUG] scheduler.last_epoch = {lr_scheduler.last_epoch}")
                return orig_step(*args, **kwargs)

            return wrapped

        train_device = torch.device(self.config.train_device)        

        train_progress = self.model.train_progress

        if self.config.only_cache:
            self.callbacks.on_update_status("caching")
            for _epoch in tqdm(
                range(train_progress.epoch, self.config.epochs, 1), desc="epoch"
            ):
                self.data_loader.get_data_set().start_next_epoch()
            return

        scaler = (
            create_grad_scaler()
            if enable_grad_scaling(self.config.train_dtype, self.parameters)
            else None
        )

        self.__apply_fused_back_pass(scaler)

        # False if the model gradients are all None, True otherwise
        # This is used to schedule sampling only when the gradients don't take up any space
        has_gradient = False

        accumulated_loss = 0.0
        ema_loss = None

        lr_scheduler = None

        for _epoch in tqdm(
            range(train_progress.epoch, self.config.epochs, 1), desc="epoch"
        ):

            if self.is_paused:
                logFun(f"[Trainer] Treino iniciado em estado PAUSADO (Epoch {train_progress.epoch}). Aguardando resume...", lvl="info")
                self._handle_pause_logic()
                if self.commands.get_stop_command(): # Se o stop foi dado durante a pausa inicial
                    logFun("[Trainer] Comando STOP ativo após pausa inicial. Encerrando.", lvl="warning")
                    break # Sai do loop de épocas

            self.callbacks.on_update_status(f"Starting Epoch {train_progress.epoch + 1}/{self.config.epochs}")

            if self.config.latent_caching:
                self.data_loader.get_data_set().start_next_epoch()
                self.model_setup.setup_train_device(self.model, self.config)
            else:
                self.model_setup.setup_train_device(self.model, self.config)
                self.data_loader.get_data_set().start_next_epoch()

            # Special case for schedule-free optimizers, which need train()
            # called before training. Can and should move this to a callback
            # during a refactoring.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()

            torch_gc()

            if lr_scheduler is None:
                lr_scheduler = create.create_lr_scheduler(
                    config=self.config,
                    optimizer=self.model.optimizer,
                    learning_rate_scheduler=self.config.learning_rate_scheduler,
                    warmup_steps=self.config.learning_rate_warmup_steps,
                    num_cycles=self.config.learning_rate_cycles,
                    min_factor=self.config.learning_rate_min_factor,
                    num_epochs=self.config.epochs,
                    approximate_epoch_length=self.data_loader.get_data_set().approximate_length(),
                    batch_size=self.config.batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    global_step=train_progress.global_step,
                )

            current_epoch_length = self.data_loader.get_data_set().approximate_length()
            step_tqdm = tqdm(
                self.data_loader.get_data_loader(),
                desc="step",
                total=current_epoch_length,
                initial=train_progress.epoch_step,
            )

            gps_instance: TrainGPS | None = getattr(self.model, "deltas", None)
            for batch in step_tqdm:
                if (
                    self.__needs_sample(train_progress)
                    or self.commands.get_and_reset_sample_default_command()
                ):
                    self.__enqueue_sample_during_training(
                        lambda: self.__sample_during_training(
                            train_progress, train_device
                        )
                    )

                if self.__needs_backup(train_progress):
                    self.commands.backup()

                if self.__needs_save(train_progress):
                    self.commands.save()

                sample_commands = self.commands.get_and_reset_sample_custom_commands()
                if sample_commands:

                    def create_sample_commands_fun(sample_commands):
                        def sample_commands_fun():
                            self.__sample_during_training(
                                train_progress, train_device, sample_commands
                            )

                        return sample_commands_fun

                    self.__enqueue_sample_during_training(
                        create_sample_commands_fun(sample_commands)
                    )

                if self.__needs_gc(train_progress):
                    torch_gc()

                if not has_gradient:
                    self.__execute_sample_during_training()
                    transferred_to_temp_device = False

                    if self.commands.get_and_reset_backup_command():
                        self.model.to(self.temp_device)
                        self.backup(train_progress, True, step_tqdm.write)
                        transferred_to_temp_device = True

                    if self.commands.get_and_reset_save_command():
                        self.model.to(self.temp_device)
                        self.save(train_progress, True, step_tqdm.write)
                        transferred_to_temp_device = True

                    if transferred_to_temp_device:
                        self.model_setup.setup_train_device(self.model, self.config)

                # with TorchMemoryRecorder(enabled=False):
                #     model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)

                #     loss = self.model_setup.calculate_loss(
                #         self.model, batch, model_output_data, self.config, train_progress, self.tensorboard
                #     )
                """
                implementação de teste - remover os gradientes fora da mask
                teoricamente isso impede que pesos fora da mask sejam atualizados
                então a rede pode aloprar o quanto quiser ali, não vai mudar nada
                """
                with TorchMemoryRecorder(enabled=False):
                    # Previsão original
                    model_output_data = self.model_setup.predict(
                        self.model,
                        batch,
                        self.config,
                        train_progress,
                    )

                    if self.config.masked_training:
                        # extrai o tensor previsto do dict
                        predicted = model_output_data["predicted"]
                        # zera gradiente fora da máscara
                        predicted.register_hook(lambda g: g * batch["latent_mask"])

                    # Cálculo de loss permanece inalterado
                    loss = self.model_setup.calculate_loss(
                        self.model,
                        batch,
                        model_output_data,
                        self.config,
                        train_progress,
                    )

                    loss = loss / float(self.config.gradient_accumulation_steps)

                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    has_gradient = True
                    if not isinstance(accumulated_loss, torch.Tensor):
                        accumulated_loss = loss.detach()
                    else:
                        accumulated_loss += loss.detach()

                    if self.__is_update_step(train_progress):
                        if (
                            scaler
                            and self.config.optimizer.optimizer.supports_fused_back_pass()
                            and self.config.optimizer.fused_back_pass
                        ):
                            scaler.step_after_unscale_parameter_(self.model.optimizer)
                            scaler.update()
                        elif scaler:
                            scaler.unscale_(self.model.optimizer)
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(
                                    self.parameters, self.config.clip_grad_norm
                                )
                            scaler.step(self.model.optimizer)
                            scaler.update()
                        else:
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(
                                    self.parameters, self.config.clip_grad_norm
                                )

                        self.model.optimizer.step()

                        try:
                            current_stats_list = self.model.optimizer.pop_stats()
                            if current_stats_list: # Only proceed if stats were generated
                                mapped_stats_list = []
                                for stat in current_stats_list:
                                    group_idx = stat["group_idx"] # Acesso direto, pode dar KeyError se não existir
                                    if 0 <= group_idx < len(self.model.param_group_mapping):
                                        name = self.model.param_group_mapping[group_idx]
                                        # AQUI TÁ SUAVE
                                        if name: # Garante que o nome não é vazio ou None
                                            stat_with_name = stat.copy()
                                            stat_with_name["name"] = name
                                            mapped_stats_list.append(stat_with_name)
                                        else: # Opcional: Logar se o nome mapeado for inválido
                                            logFun(f"[Trainer] passou por else: # Opcional: Logar se o nome mapeado for inválido", lvl="warning")
                                            if hasattr(self.config, 'debug') and self.config.debug:
                                                logFun(f"[Trainer] Nome inválido mapeado para group_idx {group_idx}", lvl="debug")
                                    else: # Opcional: Logar se o índice estiver fora do range
                                        logFun(f"[Trainer] passou por else: # Opcional: Logar se o índice estiver fora do range", lvl="warning")
                                        if hasattr(self.config, 'debug') and self.config.debug:
                                            logFun(f"[Trainer] group_idx {group_idx} fora do range do mapeamento (tam: {len(self.model.param_group_mapping)})", lvl="debug")
                                if mapped_stats_list: # Procede apenas se houver stats válidos mapeados
                                    current_deltas = {}
                                    gps_instance: TrainGPS | None = getattr(self.model, "deltas", None) # Re-check instance existence
                                    if gps_instance:
                                        try:
                                            # Get {name: delta_tensor} dict
                                            # AQUI TÁ SUAVE
                                            current_deltas = gps_instance._get_current_module_deltas(self.train_device, self.train_dtype)
                                        except Exception as e_delta:
                                            logFun(f"[Trainer] Error getting current deltas from TrainGPS: {e_delta}", lvl="error")
                                            traceback.print_exc()
                                            # Continue without deltas if error occurs

                                    # Add delta_L2 to each stat dictionary
                                    for s in mapped_stats_list:
                                        module_name = s.get("name")
                                        if module_name in current_deltas:
                                            # Ensure we store a float, not a tensor
                                            # AQUI TÁ SUAVE
                                            s["delta_L2"] = current_deltas[module_name].item()
                                        else:
                                            logFun(f"[Trainer] passou por else: do  if module_name in current_deltas:", lvl="warning")
                                            # Assign a default (e.g., 0.0 or None) if delta is missing for this module
                                            s["delta_L2"] = 0.0
                                            if gps_instance:
                                                logFun(f"[Trainer] Delta_L2 not found for module '{module_name}' in current_deltas from TrainGPS.", lvl="debug")
                                    if self.converge_control:
                                        # AQUI TÁ SUAVE
                                        global_step = self.model.train_progress.global_step
                                        current_epoch = self.model.train_progress.epoch                                        
                                        self.converge_control.set_current_time(current_epoch, global_step)
                                        self.converge_control.ingest_and_process(mapped_stats_list)

                                    for stat in current_stats_list:
                                        group_idx = stat["group_idx"]
                                        if 0 <= group_idx < len(self.model.param_group_mapping):
                                            name = self.model.param_group_mapping[group_idx]
                                            pg = self.model.parameters.by_unique_name(name)
                                            if pg:
                                                for param in pg.parameters:
                                                    # só manda ao ConvergeControl se o grad existir
                                                    if hasattr(param, "grad") and param.grad is not None:
                                                        self.converge_control.update_step_metrics(name, param)
                                                        # logFun(f"[ConvergeControl] ATUALIZOU O '{name}'", lvl="debug")
                                                    # else:
                                                    #     logFun(f"[ConvergeControl] NÃO ATUALIZOU O '{name}'", lvl="debug")
                                    # ➋ snapshot de pesos no fim da época (com base em mapped_stats_list)
                                    weights_dict: Dict[str, torch.Tensor] = {}
                                    for s in mapped_stats_list:
                                        n = s["name"]
                                        pg = self.model.parameters.by_unique_name(n)
                                        if pg and pg.parameters:
                                            # pega o primeiro parâmetro do grupo (ou outro critério seu)
                                            w = pg.parameters[0].detach().clone()
                                            weights_dict[n] = w
                                    self.converge_control.snapshot_epoch_weights(weights_dict)
                                    # 2. Recorder log_step (Se ativo) - Usa a lista mapeada
                                    if self.recorder:
                                        # Pass necessary stats to recorder's log_step
                                        # AQUI TÁ SUAVE
                                        for mapped_stat in mapped_stats_list: # Itera sobre a lista já mapeada
                                            # DataRecorder's log_step expects name, d_hat
                                            self.recorder.log_step(
                                                name=mapped_stat["name"],
                                                d_hat=mapped_stat.get("d_hat", 0.0), # Provide default if missing
                                            )
                        except KeyError as e:
                            logFun(f"[Trainer] KeyError ao acessar stat['group_idx'] ou mapeamento. Chave ausente? Erro: {e}", lvl="error")
                            traceback.print_exc()
                        except IndexError as e:
                            logFun(f"[Trainer] IndexError ao acessar self.model.param_group_mapping. group_idx fora do range? Erro: {e}", lvl="error")
                            traceback.print_exc()
                        except Exception as e:
                            logFun(f"[Trainer@{global_step}] Exception during post-optimizer step processing (ConvergeControl/Recorder): {e}", lvl="error")
                            traceback.print_exc()

                        lr_scheduler.step() # Often done after optimizer step
                        self.model.optimizer.zero_grad(set_to_none=True)
                        has_gradient = False

                        # Report learning rate after potential scheduler step
                        self.model_setup.report_to_tensorboard(
                            self.model, self.config, lr_scheduler
                        )

                        self.tensorboard.add_scalar(
                            "loss/train_step",
                            accumulated_loss.mean().item(),
                            train_progress.global_step,
                        )
                        ema_loss = ema_loss or accumulated_loss.item()
                        ema_loss = (ema_loss * 0.99) + (accumulated_loss.item() * 0.01)
                        step_tqdm.set_postfix(
                            {
                                "loss": accumulated_loss.item(),
                                "smooth loss": ema_loss,
                            }
                        )
                        self.tensorboard.add_scalar(
                            "smooth_loss/train_step",
                            ema_loss,
                            train_progress.global_step,
                        )
                        accumulated_loss = 0.0

                        self.model_setup.after_optimizer_step(
                            self.model, self.config, train_progress
                        )
                        if self.model.ema:
                            update_step = (
                                train_progress.global_step
                                // self.config.gradient_accumulation_steps
                            )
                            self.tensorboard.add_scalar(
                                "ema_decay",
                                self.model.ema.get_current_decay(update_step),
                                train_progress.global_step,
                            )
                            self.model.ema.step(self.parameters, update_step)

                        self.one_step_trained = True

                if self.config.validation:
                    self.__validate(train_progress)

                train_progress.next_step(self.config.batch_size)
                self.callbacks.on_update_train_progress(
                    train_progress, current_epoch_length, self.config.epochs
                )
                
            # Ajusta d_coef POR ÉPOCA usando Δ-L2 actual
            if self.run_number == 2 and self.adaptive_dcoef and gps_instance:
                try:
                    # 1) coleta Δ-L2 (norma L2 atual de cada módulo)
                    # AQUI TÁ SAFE
                    deltas = gps_instance._get_current_module_deltas(self.train_device, self.train_dtype)
                    names  = list(deltas.keys())
                    if names:
                        # AQUI TÁ SAFE        
                        delta_vec = torch.tensor(
                            [deltas[n] for n in names], device=self.train_device
                        )
                        # AQUI TÁ SAFE     
                        scales = self.adaptive_dcoef.scale_vectorized(
                            names, delta_vec, self.train_device, self.train_dtype
                        )
                        updated_count = 0
                        for g in self.model.optimizer.param_groups:
                            # logFun((dir(self.model.optimizer)), lvl="error")
                            # AQUI NÃO TÁ MAIS DANDO MERDA, COLOQUEI UNIQUE NAME NA CRIAÇÃO DO PARAM_GROUPS PRA PODER ACESSAR DAQUI
                            name = g.get("name")
                            if name in scales:
                                base = self.adaptive_dcoef.d_coef_base.get(name, 1.0)
                                g["d_coef"] = (base * scales[name]).item()
                                updated_count += 1
                                if self.adaptive_dcoef.debug:
                                    logFun(f"[AdaptiveDCoef@epoch{train_progress.epoch}] "
                                           f"{name}: base={base:.4g} * scale={scales[name].item():.4g}",
                                            lvl="debug")
                        if updated_count > 0 and (self.adaptive_dcoef.debug or self.adaptive_dcoef.verbose):
                            logFun(f"[AdaptiveDCoef@epoch{train_progress.epoch}] Updated d_coef for {updated_count} parameter groups.", lvl="info")
                except Exception as e:
                    logFun(f"[AdaptiveDCoef@epoch{train_progress.epoch}] Error applying dynamic d_coef adjustment: {e}", lvl="error")
                    traceback.print_exc()
                

                if self.commands.get_stop_command():
                    logFun("[Trainer] Comando STOP recebido durante a época. Encerrando...", lvl="warning")
                    # Limpar handles de gradiente antes de sair se necessário
                    for handle in self.grad_hook_handles:
                        handle.remove()
                    self.grad_hook_handles.clear()
                    return # Sai do método train

            train_progress.next_epoch()
            self.callbacks.on_update_train_progress(
                train_progress, current_epoch_length, self.config.epochs
            )

            if self.commands.get_and_reset_pause_request():
                logFun(f"[Trainer] Requisição de PAUSA recebida. Será executada ao final da Epoch {train_progress.epoch -1}.", lvl="info")
                self.pause_requested_at_epoch_end = True
                self.pause_request_locked = True # Trava a UI
                # Notificar a UI que a requisição foi aceita e o switch está travado
                if hasattr(self.callbacks, 'on_pause_request_accepted'):
                    self.callbacks.on_pause_request_accepted()

            # 2. Executar a pausa se foi agendada
            if self.pause_requested_at_epoch_end and not self.is_paused:
                self.is_paused = True # Marca como pausado
                self.pause_requested_at_epoch_end = False # Limpa a flag de agendamento
                # A trava (pause_request_locked) continua TRUE até o resume

                # Chama a função que move o modelo e entra no loop de espera
                self._handle_pause_logic()

            # Checagem de STOP ao final da época
            if self.commands.get_stop_command():
                logFun("[Trainer] Comando STOP ativo no final da época. Encerrando...", lvl="info")
                break # Sai do loop de épocas
            
            # 1. TrainGPS salva os deltas da epoch
            if gps_instance is not None:
                # Logar deltas do grupo se a opção estiver ativa
                if self.config.train_gps_save_it:
                    try:
                        # Loga para a época que acabou de terminar
                        gps_instance.log_group_deltas(train_progress.epoch - 1)
                    except Exception as e:
                        logFun(f"[TrainGPS] Erro ao logar deltas do grupo na época {train_progress.epoch - 1}: {e}", lvl="error")
                        traceback.print_exc()

                # Logar normas totais para o TensorBoard
                try:
                    current_norm, reference_norm = gps_instance.get_delta_norms()
                    if current_norm is not None:
                        self.tensorboard.add_scalar(
                            "delta_pattern/current_total_delta_norm",
                            current_norm,
                            train_progress.global_step,
                        )
                    if reference_norm is not None:
                        self.tensorboard.add_scalar(
                            "delta_pattern/reference_delta_norm",
                            reference_norm,
                            train_progress.global_step,
                        )
                except Exception as e:
                    logFun(f"[TrainGPS] Erro ao logar normas totais no TensorBoard: {e}", lvl="error")
                    traceback.print_exc()
            self.callbacks.on_update_train_progress(
                train_progress, current_epoch_length, self.config.epochs
            )
            
            # 3. ConvergeControl Decide/Aplica Congelamento (Run >= 2 e Ativo)
            if self.converge_control:
                logFun(f"[ConvergeControl] Δ Checking modules deltas..", lvl="info")
                # Roda a avaliação e obtém o dict {name: should_freeze}
                freeze_decisions = self.converge_control.decide()

                # Aplica em cada módulo que ConvergeControl avaliou
                for name, should_freeze in freeze_decisions.items():
                    # Recupera o NamedParameterGroup pelo nome
                    # AQUI TÁ SUAVE
                    param_group_obj = self.model.parameters.by_unique_name(name)
                    if not param_group_obj:
                        logFun(f"[Trainer] passou por if not param_group_obj:", lvl="warning")
                        continue  # nome inválido ou não existe

                    # Decide se deve estar habilitado (não congelado)
                    should_be_enabled = not should_freeze
                    is_currently_enabled = param_group_obj.is_enabled

                    if is_currently_enabled != should_be_enabled:
                        logFun(f"[Trainer] passou por if is_currently_enabled != should_be_enabled:)", lvl="warning")
                        action = "Unfreeze" if should_be_enabled else "Freeze"
                        if self.converge_control.verbose or self.converge_control.debug:
                            logFun(
                                f"[Trainer ApplyFreeze@{global_step}] "
                                f"{action} module '{name}' based on ConvergeControl decision "
                                f"(Run {self.run_number})",
                                lvl="info"
                            )
                        param_group_obj.set_requires_grad(should_be_enabled)

            if self.commands.get_stop_command():
                return

    def end(self):
        if self.is_paused:
            logFun("[Trainer] Finalizando treinamento enquanto estava pausado. Tentando retomar brevemente para salvar.", lvl="warning")
            # Força a saída da pausa (sem esperar comando) e tenta mover para GPU para salvar
            self.is_paused = False
            self.pause_request_locked = False
            try:
                # Tenta mover de volta pra GPU rapidamente
                self.model_setup.setup_train_device(self.model, self.config)
                torch_gc()
                logFun("[Trainer] Modelo movido para GPU para salvamento final.", lvl="info")
            except Exception as e:
                logFun(f"[Trainer] Falha ao mover modelo para GPU no final (estava pausado): {e}. Salvando do CPU ({self.temp_device}).", lvl="error")
                # O modelo já está no self.temp_device, o save deve funcionar
                pass # Continua para salvar do CPU

        if self.one_step_trained:
            self.model.to(self.temp_device)
            torch_gc()

            if self.config.backup_before_save:
                self.backup(self.model.train_progress) # Backup já usa o modelo no temp_device

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()

            self.callbacks.on_update_status("saving the final model")

            if self.model.ema:
                self.model.ema.copy_ema_to(self.parameters, store_temp=False)
            if (
                os.path.isdir(self.config.output_model_destination)
                and self.config.output_model_format.is_single_file()
            ):
                save_path = os.path.join(
                    self.config.output_model_destination,
                    f"{self.config.save_filename_prefix}{get_string_timestamp()}{self.config.output_model_format.file_extension()}",
                )
            else:
                save_path = self.config.output_model_destination
            print("Saving " + save_path)

            self.model_saver.save(
                model=self.model,
                model_type=self.config.model_type,
                output_model_format=self.config.output_model_format,
                output_model_destination=save_path,
                dtype=self.config.output_dtype.torch_dtype(),
            )
        elif self.model is not None:
            self.model.to(self.temp_device)

        model_filename = os.path.basename(save_path)
        model_name, _ = os.path.splitext(model_filename) # Get model name without extension

        # --- Save Delta Pattern (If train_gps_save_it is True AND instance exists) ---
        gps_instance: TrainGPS | None = getattr(self.model, "deltas", None)
        # A condição agora é apenas checar a flag de salvar e se o módulo foi inicializado
        if getattr(self.config, "train_gps_save_it", False) and gps_instance is not None:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Incluir o número da Run no nome do arquivo para clareza
                output_dir = os.path.join(self.config.workspace_dir, f"training_deltas")
                os.makedirs(output_dir, exist_ok=True)
                delta_filename = f"{model_name}_Deltas_Run{self.run_number}_{timestamp}.json"
                delta_save_path = os.path.join(output_dir, delta_filename)

                logFun(f"[TrainGPS] Salvando deltas (Run {self.run_number}) em: {delta_save_path}", lvl="info")
                # A função save_group_deltas salva o estado atual do delta_log_by_module
                # que foi acumulado durante esta run específica.
                gps_instance.save_group_deltas(delta_save_path)
            except Exception as e:
                logFun(f"[TrainGPS] Erro ao salvar deltas (Run {self.run_number}): {e}", lvl="error")
                traceback.print_exc()

        if self.recorder:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Use a consistent naming scheme/directory if desired
                output_dir = os.path.join(self.config.workspace_dir, "data_recorder") # Or reuse training_deltas dir?
                os.makedirs(output_dir, exist_ok=True)
                # Include Run number for clarity
                profile_filename = f"{model_name}_Profile_Run{self.run_number}_{timestamp}.json.gz"
                profile_save_path = os.path.join(output_dir, profile_filename)

                logFun(f"[DataRecorder] Salvando perfil (Run {self.run_number}) em: {profile_save_path}", lvl="info")

                # DataRecorder's dump method now only takes the path
                # It saves d_hat_final and d_coef_base internally collected.
                self.recorder.dump(profile_save_path)

            except Exception as e:
                logFun(f"[DataRecorder] Erro ao salvar perfil de dinâmica (Run {self.run_number}): {e}", lvl="error")
                traceback.print_exc()

        self.tensorboard.close()

        if self.config.tensorboard:
            super()._stop_tensorboard()

        for handle in self.grad_hook_handles:
            handle.remove()
        torch_gc()
