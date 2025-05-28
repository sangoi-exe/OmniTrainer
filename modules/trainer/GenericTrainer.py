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
from typing import Dict, Optional

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
from requests.exceptions import ConnectionError

from modules.sangoi.ConvergeControl import ConvergeControl
from modules.sangoi.AdaptiveDCoef import AdaptiveDCoef
from modules.sangoi.DataRecorder import DataRecorder
from modules.sangoi.TrainGPS import TrainGPS
# IMPORTANTE: Remova a importação de Progress e Layout se não forem mais usados diretamente
from rich.console import Console as RichConsole, Group # Adicione Group
from rich.text import Text
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.table import Table
# As importações do logFun já estão OK
from modules.sangoi.logFun import logFun, set_logfun_console, init_global_progress, cleanup_global_progress, ProgressContext


def format_time_delta(seconds: float) -> str:
    if seconds < 0 or not isinstance(seconds, (int, float)):
        return "??:??"  # Lida com valores inválidos
    seconds = abs(seconds)  # Garante que é positivo para cálculo
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


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
    is_run2: bool  # Flag to easily check run mode
    _temp_recorder_data: collections.defaultdict  # Temporary storage for recorder data before log_step

    adaptive_dcoef_normal_d_coef: float
    adaptive_dcoef_fine_tune_d_coef: float
    adaptive_dcoef_base_d_coef: float
    adaptive_dcoef_pivot_score: float
    adaptive_dcoef_min_global_target: float
    adaptive_dcoef_max_global_target: float
    adaptive_dcoef_modulation_strength: float
    adaptive_dcoef_initial_boost_d_coef: float
    adaptive_dcoef_convergence_threshold_low: float
    adaptive_dcoef_convergence_threshold_high: float
    data_recorder_run1_path: Optional[str]  # Caminho para o dump da Run 1

    # atributos para pause
    is_paused: bool
    pause_request_locked: bool  # Para travar o switch da UI
    pause_requested_at_epoch_end: bool

    from rich.progress import Progress

    _training_live: Optional[Live] = None # A única instância Live para todo o display
    _global_progress_instance: Optional['Progress'] = None # Referência ao objeto Progress global do logFun

    _current_step_duration_s: float = 0.0
    _epoch_time_elapsed_s: float = 0.0
    _avg_step_time_epoch_s: float = 0.0
    _ema_step_time_s: Optional[float] = None  # Para média móvel exponencial
    _ema_alpha: float = 0.05  # Ajuste para mais ou menos suavização (menor = mais suave)
    _total_training_time_start_s: Optional[float] = None  # Para tempo total de treino

    _epoch_start_time_s: Optional[float] = None  # Para calcular tempo da epoch e ETA
    _num_total_epochs: int = 0
    _steps_per_epoch: int = 0

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super().__init__(config, callbacks, commands)

        tensorboard_log_dir = os.path.join(config.workspace_dir, "tensorboard")
        os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)
        self.tensorboard = TensorBoardManager(log_dir=os.path.join(
            tensorboard_log_dir,
            f"{config.save_filename_prefix}{get_string_timestamp()}",
        ))
        if config.tensorboard:
            super()._start_tensorboard()

        self.model = None
        self.one_step_trained = False
        self.grad_hook_handles = []

        self.is_paused = False
        self.converge_control = None
        self.recorder = None
        self.adaptive_dcoef = None
        self.train_dtype = None
        self.pause_request_locked = False
        self.pause_requested_at_epoch_end = False
        self.train_device = torch.device(getattr(self.config, "train_device"))

        self.adaptive_dcoef_fine_tune_d_coef = getattr(config, "adaptive_dcoef_fine_tune_d_coef", 0.25)
        self.adaptive_dcoef_normal_d_coef = getattr(config, "adaptive_dcoef_normal_d_coef", 1.0)
        self.adaptive_dcoef_base_d_coef = getattr(config, "adaptive_dcoef_base_d_coef", 1.0)
        self.adaptive_dcoef_pivot_score = getattr(config, "adaptive_dcoef_pivot_score", 0.5)
        self.adaptive_dcoef_initial_boost_d_coef = getattr(config, "adaptive_dcoef_initial_boost_d_coef", 2.0)
        self.adaptive_dcoef_modulation_strength = getattr(config, "adaptive_dcoef_modulation_strength", 0.4)
        self.adaptive_dcoef_min_global_target = getattr(config, "adaptive_dcoef_min_global_target", 0.5)
        self.adaptive_dcoef_max_global_target = getattr(config, "adaptive_dcoef_max_global_target", 2.0)
        self.adaptive_dcoef_convergence_threshold_low = getattr(config, "adaptive_dcoef_convergence_threshold_low", 0.33)
        self.adaptive_dcoef_convergence_threshold_high = getattr(config, "adaptive_dcoef_convergence_threshold_high", 0.66)
        self.data_recorder_run1_path = getattr(config, "adpt_dcoef_path", None)

        self.console = RichConsole()
        set_logfun_console(self.console)  # Configurar o console para logFun

        self._num_total_epochs = config.epochs

        # Component initialization flags (from config, with defaults)
        self.dcoef_debug = False
        self.dcoef_verbose = False
        self.recorder_debug = True
        self.recorder_verbose = True
        self.converge_debug = True
        self.converge_verbose = True

        self.run_number = getattr(self.config, "run_number", 1)
        logFun(f"Configurando para Run {self.run_number}.", lvl="LOOP")

        if getattr(self.config, "data_recorder", False):
            # Recorder is active based on its flag. Its behavior (recording) is mainly for Run 1,
            # but it might be optionally used in Run 2 for analysis if needed.
            # No specific check for run_number needed here for activation, just for the 'purpose' log maybe.
            self.recorder = DataRecorder(debug=self.recorder_debug, verbose=self.recorder_verbose)
            purpose = "coleta de dados" if self.run_number == 1 else "gravação opcional de dinâmica"
            logFun(f"[DataRecorder] Inicializado (flag data_recorder=True) para {purpose} na Run {self.run_number}.",
                   lvl="debug")
        else:
            logFun("[DataRecorder] Desativado (flag data_recorder=False).", lvl="debug")

        # 2. AdaptiveDCoef
        self.adaptive_dcoef = None
        if self.run_number == 2 and getattr(self.config, "adpt_dcoef_use_it", False):
            try:
                self.adaptive_dcoef = AdaptiveDCoef(debug=self.dcoef_debug, verbose=self.dcoef_verbose)
                logFun(f"[AdaptiveDCoef] Inicializado com sucesso para Run 2.", lvl="success")
                if self.data_recorder_run1_path:
                    logFun(f"[AdaptiveDCoef] Tentando carregar perfil da Run 1 de: {self.data_recorder_run1_path}",
                           lvl="info")
                    self.adaptive_dcoef.load_prodigy_d_final_run1(self.data_recorder_run1_path)
                else:
                    logFun(
                        "[AdaptiveDCoef] Caminho do perfil da Run 1 (adpt_dcoef_profile_path) não fornecido. 'd_pdgy_final' não será carregado.",
                        lvl="warning")
            except Exception as e:
                logFun(
                    f"[Trainer Aviso] Falha ao inicializar AdaptiveDCoef para Run 2: {e}. d_coef dinâmico será desativado.",
                    lvl="warning")
                traceback.print_exc()
                self.adaptive_dcoef = None
        elif getattr(self.config, "adpt_dcoef_use_it", False) and self.run_number != 2:
            try:
                self.adaptive_dcoef = AdaptiveDCoef(debug=self.dcoef_debug, verbose=self.dcoef_verbose)
                logFun(f"[AdaptiveDCoef] Inicializado com sucesso para Run 1.", lvl="success")
            except Exception as e:
                logFun(
                    f"[Trainer Aviso] Falha ao inicializar AdaptiveDCoef para Run 2: {e}. d_coef dinâmico será desativado.",
                    lvl="warning")
                traceback.print_exc()
        else:
            logFun("[AdaptiveDCoef] Desativado (flag adpt_dcoef_use_it=False).", lvl="debug")


    def _create_training_display_content(self,
                                        current_loss: Optional[float] = None,
                                        current_ema_loss: Optional[float] = None) -> Panel:
        """Cria o conteúdo do painel principal de treinamento com dados atualizados."""
        tp = self.model.train_progress if self.model and hasattr(self.model, 'train_progress') else None

        # Linha 1: Progresso Epoch/Step
        epoch_str = f"Epoch: {tp.epoch + 1}/{self._num_total_epochs}" if tp else f"Epoch: ?/{self._num_total_epochs}"
        step_str = f"Step: {tp.epoch_step + 1}/{self._steps_per_epoch}" if tp and self._steps_per_epoch > 0 else "Step: ?"
        global_step_str = f"Global: {tp.global_step + 1}" if tp else "Global: ?"
        line1 = Text.assemble((epoch_str, "bold cyan"), " | ", (step_str, "bold cyan"), " | ",
                              (global_step_str, "dim cyan"))

        # Linha 2: Losses
        loss_str = f"Loss: {current_loss:.8f}" if current_loss is not None else "Loss: N/A"
        ema_loss_str = f"Smooth: {current_ema_loss:.8f}" if current_ema_loss is not None else "Smooth: N/A"
        line2 = Text.assemble((loss_str, "yellow"), " | ", (ema_loss_str, "bright_yellow"))

        # Linha 3: Temporização
        step_time_disp = f"StepTime: {self._current_step_duration_s:.2f}s"

        effective_avg_step_time = self._avg_step_time_epoch_s
        if self._ema_step_time_s is not None and self._ema_step_time_s > 0:
            effective_avg_step_time = self._ema_step_time_s

        avg_step_disp = f"AvgStep: {effective_avg_step_time:.2f}s/it" if effective_avg_step_time > 0 else "AvgStep: Calc..."
        epoch_elapsed_disp = f"EpochElap: {format_time_delta(self._epoch_time_elapsed_s)}"

        eta_epoch_disp = "ETAEpoch: Calc..."
        if tp and self._steps_per_epoch > 0 and effective_avg_step_time > 0:
            remaining_steps = self._steps_per_epoch - (tp.epoch_step + 1)
            if remaining_steps >= 0:
                eta_s = remaining_steps * effective_avg_step_time
                eta_epoch_disp = f"ETAEpoch: {format_time_delta(eta_s)}"

        eta_total_disp = "ETATotal: Calc..."
        if tp and self._steps_per_epoch > 0 and effective_avg_step_time > 0:
            total_steps = self._num_total_epochs * self._steps_per_epoch
            completed_steps = tp.global_step + 1
            remaining_steps_total = total_steps - completed_steps
            if remaining_steps_total >= 0:
                eta_s_total = remaining_steps_total * effective_avg_step_time
                eta_total_disp = f"ETATotal: {format_time_delta(eta_s_total)}"

        total_time_str = ""
        if self._total_training_time_start_s is not None:
            total_elapsed = time.monotonic() - self._total_training_time_start_s
            total_time_str = f"TotalRun: {format_time_delta(total_elapsed)}"

        line3 = Text.assemble(
            (step_time_disp,      "green"), " | ",
            (avg_step_disp,       "blue"),  " | ",
            (epoch_elapsed_disp,  "magenta"), " | ",
            (eta_epoch_disp,      "magenta"), " | ",
            (eta_total_disp,      "magenta"),
            (" | " + total_time_str if total_time_str else "", "dim white")
        )

        # Linha 4: Status de Componentes (Opcional, se houver espaço e quiser exibir)
        components_status = []
        if self.converge_control:
            # CORREÇÃO AQUI: Usar Text.from_markup() para que as tags sejam interpretadas
            components_status.append(Text.from_markup("CC: [green]Ativo[/green]"))
        if self.adaptive_dcoef:
            # CORREÇÃO AQUI: Usar Text.from_markup()
            components_status.append(Text.from_markup("ADC: [blue]Ativo[/blue]"))
        if self.recorder:
            # CORREÇÃO AQUI: Usar Text.from_markup()
            components_status.append(Text.from_markup("DR: [yellow]Ativo[/yellow]"))
        
        line4 = Text.assemble("Componentes: ", Text(" | ").join(components_status)) if components_status else Text("")

        # Combina as linhas com quebras de linha, centralizadas
        lines_to_join = [line1, line2, line3]
        if components_status: # Adiciona a linha 4 apenas se houver componentes
            lines_to_join.append(line4)

        final_text = Text("\n", justify="center").join(lines_to_join)
        
        return Panel(final_text, title="[bold cyan]OneTrainer - Status do Treinamento[/bold cyan]", border_style="cyan")

    def _get_combined_display_renderable(self, current_loss: Optional[float] = None, current_ema_loss: Optional[float] = None) -> Group:
        """
        Combina o painel de status principal de treinamento e o objeto Progress global
        em um único renderable Group.
        """
        # Garante que o objeto _global_progress_instance esteja inicializado
        if self._global_progress_instance is None:
            self._global_progress_instance = init_global_progress() # Isso agora *apenas cria* o objeto Progress

        status_panel_content = self._create_training_display_content(current_loss, current_ema_loss)

        # Cria um Group contendo o painel de status e a barra de progresso global
        # A ordem aqui define a exibição vertical: status_panel em cima, barra de progresso abaixo
        combined_display = Group(
            status_panel_content,
            self._global_progress_instance # Este é o objeto Progress do logFun
        )
        return combined_display

    def _setup_training_display(self):
        """Configura o sistema de display de treinamento com Live, exibindo painel e progress bars."""
        if self._training_live is None:
            # Obtém o conteúdo combinado inicial para o Live display
            initial_content = self._get_combined_display_renderable()

            self._training_live = Live(
                initial_content,
                console=self.console,
                refresh_per_second=2,
                screen=False,  # Não toma a tela toda
                transient=False  # Não limpa ao parar
            )
            # Inicia o display Live uma vez
            self._training_live.start()
            logFun("Sistema de display Rich Live iniciado", lvl="INFO")

    def _update_training_display(self, current_loss: Optional[float] = None, current_ema_loss: Optional[float] = None):
        """Atualiza o display de treinamento."""
        if self._training_live and self._training_live.is_started:
            # Atualiza a instância Live com o novo conteúdo combinado
            updated_content = self._get_combined_display_renderable(current_loss, current_ema_loss)
            self._training_live.update(updated_content)

    def _stop_training_display(self):
        """Para o display de treinamento."""
        if self._training_live and self._training_live.is_started:
            self._training_live.stop()
            self._training_live = None
            self._global_progress_instance = None # Reseta a referência ao objeto Progress
            logFun("Sistema de display Rich Live finalizado", lvl="INFO")

    def start(self):
        set_logfun_console(self.console)
        # REMOVIDO: init_global_progress() não é mais chamado aqui.
        # Ele será chamado por _get_combined_display_renderable quando o display for configurado.
        # init_global_progress() 
        # REMOVIDO: self._setup_training_display() não é chamado aqui, pois train() o fará.

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
        # self.model.to(self.temp_device) será que dá pra desativar essa bosta?
        self.model.eval()
        torch_gc()

        self.callbacks.on_update_status("creating the data loader/caching")

        self.data_loader = self.create_data_loader(self.model, self.model.train_progress)
        self.model_saver = self.create_model_saver()

        self.model_sampler = self.create_model_sampler(self.model)
        self.previous_sample_time = -1
        self.sample_queue = []

        self.parameters = self.model.parameters.parameters()
        if self.config.validation:
            self.validation_data_loader = self.create_data_loader(self.model,
                                                                  self.model.train_progress,
                                                                  is_validation=True)

        # iniciando o convergecontrol aqui pra poder usar a função de calcular o tamanho do treino
        if getattr(self.config, "convctrl_use_it", False):
            converge_control_instance = ConvergeControl(
                run_number=self.run_number,  # Passa o run_number atual do Trainer
                verbose=self.converge_verbose,
                debug=self.converge_debug,
                data_recorder=self.recorder)

            self.converge_control = converge_control_instance
            logFun(f"[ConvergeControl] Inicializado com sucesso para Run {self.run_number} (usando padrões internos).", lvl="debug")
        else:
            self.converge_control = None
            logFun("[ConvergeControl] Desativado (flag convctrl_use_it=False).", lvl="debug")

    def __save_config_to_workspace(self):
        path = path_util.canonical_join(self.config.workspace_dir, "config")
        os.makedirs(Path(path).absolute(), exist_ok=True)
        path = path_util.canonical_join(path, f"{get_string_timestamp()}.json")
        with open(path, "w") as f:
            json.dump(self.config.to_pack_dict(secrets=False), f, indent=4)

    def __clear_cache(self):
        logFun(f"Limpando diretório de cache {self.config.cache_dir}!", lvl="INFO")
        if os.path.isdir(self.config.cache_dir):
            # ProgressContext já usa o sistema global de progresso
            with ProgressContext("Limpando cache", len(os.listdir(self.config.cache_dir))) as progress: # Estimar total
                files_to_delete = [
                    f for f in os.listdir(self.config.cache_dir)
                    if os.path.isdir(os.path.join(self.config.cache_dir, f)) and
                    (f.startswith("epoch-") or f in ["image", "text"])
                ]
                
                # A iteração para delete deve ser sobre os arquivos_to_delete
                # O total para ProgressContext também deve ser files_to_delete
                if files_to_delete:
                    # Recria o ProgressContext com o total correto para a lista filtrada
                    # Não é ideal ter duas instâncias, mas a primeira é um placeholder.
                    # Poderíamos filtrar antes de criar o ProgressContext.
                    progress.total = len(files_to_delete) # Atualiza o total
                    for filename in files_to_delete:
                        path = os.path.join(self.config.cache_dir, filename)
                        shutil.rmtree(path)
                        progress.update(1)
                else:
                    logFun("Nenhum diretório de cache para limpar.", lvl="INFO")


    def __prune_backups(self, backups_to_keep: int):
        backup_dirpath = os.path.join(self.config.workspace_dir, "backup")
        if os.path.exists(backup_dirpath):
            backup_directories = sorted(
                [
                    dirpath for dirpath in os.listdir(backup_dirpath)
                    if os.path.isdir(os.path.join(backup_dirpath, dirpath))],
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
                        if (self.config.samples_to_tensorboard and sampler_output.file_type == FileType.IMAGE):
                            self.tensorboard.add_image(
                                f"sample{str(i)} - {safe_prompt}",
                                pil_to_tensor(sampler_output.data),  # noqa: B023
                                train_progress.global_step,
                            )
                        self.callbacks.on_sample_default(sampler_output)

                    def on_sample_custom(sampler_output: ModelSamplerOutput):
                        self.callbacks.on_sample_custom(sampler_output)

                    on_sample = (on_sample_custom if is_custom_sample else on_sample_default)
                    on_update_progress = (self.callbacks.on_update_sample_custom_progress
                                          if is_custom_sample else self.callbacks.on_update_sample_default_progress)

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
            current_epoch_length_validation = (self.validation_data_loader.get_data_set().approximate_length())

            if current_epoch_length_validation == 0:
                return

            self.callbacks.on_update_status("calculating validation loss")
            self.model_setup.setup_train_device(self.model, self.config)

            torch_gc()

            accumulated_loss_per_concept = {}
            concept_counts = {}
            mapping_seed_to_label = {}
            mapping_label_to_seed = {}

            # O uso de ProgressContext substitui _rich_progress_footer
            with ProgressContext("Validação", current_epoch_length_validation) as progress:
                for validation_batch in self.validation_data_loader.get_data_loader():
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
                        loss_validation = self.model_setup.calculate_loss(self.model, validation_batch,
                                                                          model_output_data, self.config)

                    # since validation batch size = 1
                    concept_name = validation_batch["concept_name"][0]
                    concept_path = validation_batch["concept_path"][0]
                    concept_seed = validation_batch["concept_seed"].item()
                    loss = loss_validation.item()

                    label = concept_name if concept_name else os.path.basename(concept_path)
                    # check and fix collision to display both graphs in tensorboard
                    if (label in mapping_label_to_seed and mapping_label_to_seed[label] != concept_seed):
                        suffix = 1
                        new_label = f"{label}({suffix})"
                        while (new_label in mapping_label_to_seed and mapping_label_to_seed[new_label] != concept_seed):
                            suffix += 1
                            new_label = f"{label}({suffix})"
                        label = new_label

                    if concept_seed not in mapping_seed_to_label:
                        mapping_seed_to_label[concept_seed] = label
                        mapping_label_to_seed[label] = concept_seed

                    accumulated_loss_per_concept[concept_seed] = (accumulated_loss_per_concept.get(concept_seed, 0) +
                                                                  loss)
                    concept_counts[concept_seed] = concept_counts.get(concept_seed, 0) + 1
                    progress.update(1) # Atualiza a barra de progresso da validação

            for concept_seed, total_loss in accumulated_loss_per_concept.items():
                average_loss = total_loss / concept_counts[concept_seed]

                self.tensorboard.add_scalar(
                    f"loss/validation_step/{mapping_seed_to_label[concept_seed]}",
                    average_loss,
                    train_progress.global_step,
                )

            if len(concept_counts) > 1:
                total_loss = sum(accumulated_loss_per_concept[key] for key in concept_counts)
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
    ):  # // GEMINI-CODE 2024-08-29T11:00:00 - print_cb agora usa logFun
        torch_gc()

        self.callbacks.on_update_status("creating backup")

        backup_name = (f"{get_string_timestamp()}-backup-{train_progress.filename_string()}")
        backup_path = os.path.join(self.config.workspace_dir, "backup", backup_name)

        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

        try:
            if print_msg:
                logFun("Creating Backup " + backup_path, lvl="LOOP") # Uso de logFun

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
    ): # // GEMINI-CODE 2024-08-29T11:00:00 - print_cb agora usa logFun
        torch_gc()

        self.callbacks.on_update_status("saving")

        save_path = os.path.join(
            self.config.workspace_dir,
            "save",
            f"{self.config.save_filename_prefix}{get_string_timestamp()}-save-{train_progress.filename_string()}{self.config.output_model_format.file_extension()}",
        )
        if print_msg:
            logFun("Saving " + save_path, lvl="LOOP") # Uso de logFun

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
        return self.repeating_action_needed("gc", 5, TimeUnit.MINUTE, train_progress, start_at_zero=False)

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
        if (self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass):
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

                            def __grad_hook(tensor: Tensor, param_group=param_group, i=i):
                                if self.__is_update_step(self.model.train_progress):
                                    scaler.unscale_parameter_(tensor, self.model.optimizer)
                                    if self.config.clip_grad_norm is not None:
                                        nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                                    scaler.maybe_opt_step_parameter(tensor, param_group, i, self.model.optimizer)
                                    tensor.grad = None
                        else:

                            def __grad_hook(tensor: Tensor, param_group=param_group, i=i):
                                if self.__is_update_step(self.model.train_progress):
                                    if self.config.clip_grad_norm is not None:
                                        nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                                    self.model.optimizer.step_parameter(tensor, param_group, i)
                                    tensor.grad = None

                        handle = parameter.register_post_accumulate_grad_hook(__grad_hook)
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
        if not self.is_paused:  # Segurança extra
            return

        logFun("Iniciando Pausa...", lvl="LOOP")
        self.callbacks.on_update_status("Pausing... Moving model to CPU")
        try:
            self.model.to(self.temp_device)  # Mover para CPU
            self.model.eval()  # Garantir modo eval
            torch_gc()  # Limpar VRAM
            logFun(f"Modelo movido para {self.temp_device}. VRAM liberada.", lvl="success")
            self.callbacks.on_update_status(f"Paused. Model on {self.temp_device}. Toggle switch to resume.")
            # Notificar UI que a pausa iniciou e o switch pode ser reativado (para desligar)
            if hasattr(self.callbacks, 'on_pause_initiated'):
                self.callbacks.on_pause_initiated()

            # Loop de espera pela retomada
            while self.is_paused:
                if self.commands.get_stop_command():
                    logFun("Comando STOP recebido durante a pausa. Interrompendo.", lvl="warning")
                    self.is_paused = False  # Força a saída do loop de pausa
                    # Mantém o comando de stop ativo para o loop principal
                    break

                if self.commands.get_and_reset_resume_request():
                    logFun("Comando RESUME recebido.", lvl="info")
                    self.is_paused = False  # Sinaliza para sair do loop
                    self.pause_request_locked = False  # Desbloqueia a UI
                    # Notificar UI que o resume começou (switch ainda ativo)
                    if hasattr(self.callbacks, 'on_resume_started'):
                        self.callbacks.on_resume_started()
                    break  # Sai do loop de espera

                time.sleep(0.5)  # Evita busy-waiting, checa a cada 0.5s

            if not self.commands.get_stop_command():  # Só retoma se não for parar
                logFun("Retomando treinamento...", lvl="info")
                self.callbacks.on_update_status("Resuming... Moving model to GPU")
                try:
                    # Recarregar para o dispositivo de treino
                    self.model_setup.setup_train_device(self.model, self.config)
                    torch_gc()  # Limpeza extra
                    logFun(f"Modelo movido de volta para {self.config.train_device}.", lvl="success")
                    self.callbacks.on_update_status("Training resumed.")
                    # Notificar UI que o resume foi concluído
                    if hasattr(self.callbacks, 'on_resume_completed'):
                        self.callbacks.on_resume_completed()

                except Exception as e:
                    logFun(f"Erro ao mover modelo de volta para GPU: {e}", lvl="error")
                    traceback.print_exc()
                    # Tentar continuar mesmo assim? Ou parar? Por segurança, parar.
                    self.commands.stop()
            else:
                logFun("Retomada cancelada devido ao comando STOP.", lvl="warning")

        except Exception as e:
            logFun(f"Erro durante o processo de pausa/retomada: {e}", lvl="error")
            traceback.print_exc()
            self.is_paused = False  # Garante que não fique preso no estado pausado
            self.pause_request_locked = False
            # Considerar parar o treino em caso de erro grave aqui
            self.commands.stop()
            self.callbacks.on_update_status(f"Error during pause/resume: {e}")

    def train(self):
        scheduler_step_counter = 0
        train_device = torch.device(self.config.train_device)
        train_progress = self.model.train_progress

        # Determine target device and dtype from a model parameter if available
        def wrap_scheduler_step(orig_step):
            def wrapped(*args, **kwargs):
                nonlocal scheduler_step_counter
                scheduler_step_counter += 1
                logFun(f"[DEBUG] scheduler.step() chamado {scheduler_step_counter} vezes", lvl="DEBUG")
                logFun(f"[DEBUG] scheduler.last_epoch = {lr_scheduler.last_epoch}", lvl="DEBUG")
                return orig_step(*args, **kwargs)
            return wrapped
        
        # Verificar se é só cache (mantém como está)
        if self.config.only_cache:
            self._handle_cache_only_mode()
            return

        if self.config.only_cache:
            self.callbacks.on_update_status("caching")
            
            with ProgressContext("Caching latents", self.config.epochs - train_progress.epoch) as progress:
                for _epoch in range(train_progress.epoch, self.config.epochs, 1):
                    self.data_loader.get_data_set().start_next_epoch()
                    progress.update(1)
            return

        scaler = (create_grad_scaler() if enable_grad_scaling(self.config.train_dtype, self.parameters) else None)
        self.__apply_fused_back_pass(scaler)

        # False if the model gradients are all None, True otherwise
        # This is used to schedule sampling only when the gradients don't take up any space
        has_gradient = False
        accumulated_loss = 0.0
        ema_loss = 0.0
        lr_scheduler = None
        
        # Inicia o sistema de display principal (que agora gerencia o Live para tudo)
        self._setup_training_display()
        self._total_training_time_start_s = time.monotonic()


        try:
            for _epoch in range(train_progress.epoch, self.config.epochs, 1):

                self._epoch_start_time_s = time.monotonic()
                self._steps_per_epoch = self.data_loader.get_data_set().approximate_length(
                )  # Pega o total de steps para a epoch atual

                if self.is_paused:
                    logFun(f"Treino iniciado em estado PAUSADO (Epoch {train_progress.epoch}). Aguardando resume...", lvl="info")
                    self._handle_pause_logic()
                    if self.commands.get_stop_command():  # Se o stop foi dado durante a pausa inicial
                        logFun("Comando STOP ativo após pausa inicial. Encerrando.", lvl="warning")
                        break  # Sai do loop de épocas

                self.callbacks.on_update_status(f"training")

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

                gps_instance: TrainGPS | None = getattr(self.model, "deltas", None)
                current_epoch_length = self.data_loader.get_data_set().approximate_length()

                for batch_idx, batch in enumerate(self.data_loader.get_data_loader()):
                    step_start_time_s = time.monotonic()
                    if (self.__needs_sample(train_progress) or self.commands.get_and_reset_sample_default_command()):
                        self.__enqueue_sample_during_training(
                            lambda: self.__sample_during_training(train_progress, train_device))

                    if self.__needs_backup(train_progress):
                        self.commands.backup()

                    if self.__needs_save(train_progress):
                        self.commands.save()

                    sample_commands = self.commands.get_and_reset_sample_custom_commands()
                    if sample_commands:

                        def create_sample_commands_fun(sample_commands):

                            def sample_commands_fun():
                                self.__sample_during_training(train_progress, train_device, sample_commands)

                            return sample_commands_fun

                        self.__enqueue_sample_during_training(create_sample_commands_fun(sample_commands))

                    if self.__needs_gc(train_progress):
                        torch_gc()

                    if not has_gradient:
                        self.__execute_sample_during_training()
                        transferred_to_temp_device = False

                        if self.commands.get_and_reset_backup_command():
                            self.model.to(self.temp_device)
                            self.backup(train_progress, True, lambda msg: logFun(msg, lvl="LOOP"))
                            transferred_to_temp_device = True

                        if self.commands.get_and_reset_save_command():
                            self.model.to(self.temp_device)
                            self.save(train_progress, True, lambda msg: logFun(msg, lvl="LOOP"))
                            transferred_to_temp_device = True

                        if transferred_to_temp_device:
                            self.model_setup.setup_train_device(self.model, self.config)

                    with TorchMemoryRecorder(enabled=False):
                        model_output_data = self.model_setup.predict(
                            self.model,
                            batch,
                            self.config,
                            train_progress,
                        )

                        if self.config.masked_training:
                            predicted = model_output_data["predicted"]
                            predicted.register_hook(lambda g: g * batch["latent_mask"])

                        loss = self.model_setup.calculate_loss(
                            self.model,
                            batch,
                            model_output_data,
                            self.config,
                            train_progress,
                        )

                        loss = loss / self.config.gradient_accumulation_steps

                        if scaler:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        has_gradient = True

                        accumulated_loss += loss.item()
                        if self.__is_update_step(train_progress):
                            if (scaler and self.config.optimizer.optimizer.supports_fused_back_pass() and
                                    self.config.optimizer.fused_back_pass):
                                scaler.step_after_unscale_parameter_(self.model.optimizer)
                                scaler.update()
                            elif scaler:
                                scaler.unscale_(self.model.optimizer)
                                if self.config.clip_grad_norm is not None:
                                    nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                                scaler.step(self.model.optimizer)
                                scaler.update()
                            else:
                                if self.config.clip_grad_norm is not None:
                                    nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                                self.model.optimizer.step()

                            try:
                                # FIRST LOOP
                                # --- PASSO 1: Processar estatísticas do otimizador, atualizar CC e gravar dados ---
                                current_stats_list_from_optimizer = self.model.optimizer.pop_stats() # Pop aqui!

                                if self.converge_control: # Atualizar tempo do CC uma vez por step
                                    global_step_cc = self.model.train_progress.global_step
                                    current_epoch_cc = self.model.train_progress.epoch
                                    self.converge_control.set_current_time(current_epoch_cc, global_step_cc)

                                if current_stats_list_from_optimizer and hasattr(self.model, "param_group_mapping"):
                                    for stat_data in current_stats_list_from_optimizer:
                                        group_idx = stat_data.get("group_idx")
                                        # 'name' já vem de stat_data, que deve ser o mesmo de param_group_mapping[group_idx]
                                        name = stat_data.get("name") 
                                        
                                        if name is None: # Sanity check
                                            # Se 'name' não está em stat_data, mas group_idx está, podemos usar o mapeamento
                                            if group_idx is not None and (0 <= group_idx < len(self.model.param_group_mapping)):
                                                name = self.model.param_group_mapping[group_idx]
                                            else:
                                                logFun(f"Skipping stat_data due to missing name and invalid group_idx: {stat_data}", lvl="warning")
                                                continue
                                        
                                        # Obter o d_hat (que é o 'd' do Prodigy que queremos)
                                        d_num_pdgy_raw = stat_data.get("d_num")
                                        d_den_pdgy_raw = stat_data.get("d_den")

                                        # Atualizar ConvergeControl
                                        if self.converge_control and name not in self.converge_control.get_frozen_set():
                                            peft_mod = None
                                            if hasattr(self.model, "unet_lora") and hasattr(self.model.unet_lora, "get_module_for_stats"):
                                                peft_mod = self.model.unet_lora.get_module_for_stats(name)
                                            
                                            if peft_mod:
                                                self.converge_control.update_step_metrics(name, peft_mod)
                                            # else: logFun(f"Módulo PEFT não encontrado para '{name}' para CC update", lvl="LOOP_DEBUG")

                                        # Registrar dados com DataRecorder
                                        if self.recorder:
                                            # if self.converge_control and name in self.converge_control._deques_initialized_for_module:
                                            #     # Coletar métricas do ConvergeControl
                                            #     cc_gd_hist = self.converge_control.gradient_disparity_hist.get(name)
                                            #     cc_gd = cc_gd_hist[-1] if cc_gd_hist else None
                                                
                                            #     cc_snr_hist = self.converge_control.snr_hist.get(name)
                                            #     cc_snr = cc_snr_hist[-1] if cc_snr_hist else None
                                                
                                            #     cc_gns_t_hist = self.converge_control.gns_temporal_hist.get(name)
                                            #     cc_gns_t = cc_gns_t_hist[-1] if cc_gns_t_hist else None
                                                
                                            #     cc_gd_ewma = self.converge_control.gd_ewma.get(name)
                                            #     cc_gd_std_ewma_var = self.converge_control.gd_std_ewma.get(name)

                                            d_num_pdgy = None
                                            if d_num_pdgy_raw is not None:
                                                d_num_pdgy = float(d_num_pdgy_raw.item() if isinstance(d_num_pdgy_raw, torch.Tensor) else d_num_pdgy_raw)
                                                
                                            d_den_pdgy = None
                                            if d_den_pdgy_raw is not None:
                                                d_den_pdgy = float(d_den_pdgy_raw.item() if isinstance(d_den_pdgy_raw, torch.Tensor) else d_den_pdgy_raw)
                                            
                                            self.recorder.log_metrics_step(
                                                name=name,
                                                d_num_pdgy=d_num_pdgy,
                                                d_den_pdgy=d_den_pdgy,
                                                # gd=cc_gd, # Descomentar se precisar
                                                # snr=cc_snr, # Descomentar se precisar
                                                # gns_t=cc_gns_t, # Descomentar se precisar
                                                # gd_ewma=cc_gd_ewma, # Descomentar se precisar
                                                # gd_std_ewma_var=cc_gd_std_ewma_var # Descomentar se precisar
                                            )
                            except Exception as e:
                                logFun(f"Deu merda no first loop: {e}", lvl="error")
                                traceback.print_exc()

                            # Scheduler de learning rate
                            lr_scheduler.step()

                            # Reset de gradientes
                            self.model.optimizer.zero_grad(set_to_none=True)
                            has_gradient = False

                            self._current_step_duration_s = time.monotonic() - step_start_time_s
                            # EMA step time
                            if self._ema_step_time_s is None:
                                self._ema_step_time_s = self._current_step_duration_s
                            else:
                                self._ema_step_time_s = (self._ema_alpha * self._current_step_duration_s +
                                                          (1 - self._ema_alpha) * self._ema_step_time_s)

                            self._epoch_time_elapsed_s = time.monotonic() - self._epoch_start_time_s

                            # Média simples da epoch (se preferir em vez de EMA para ETA da epoch)
                            # epoch_step_times_sum += self._current_step_duration_s
                            # self._avg_step_time_epoch_s = epoch_step_times_sum / (train_progress.epoch_step + 1)
                            # OU usar EMA como avg_step_time_epoch_s se quiser simplificar
                            if self._ema_step_time_s is not None:
                                self._avg_step_time_epoch_s = self._ema_step_time_s  # Usar EMA para o display de AvgStep

                            # Atualiza o display do Rich
                            self._update_training_display(
                                current_loss=accumulated_loss,
                                current_ema_loss=ema_loss
                            )

                            self.model_setup.report_to_tensorboard(self.model, self.config, lr_scheduler)

                            self.tensorboard.add_scalar(
                                "loss/train_step",
                                accumulated_loss,  # Use .mean() se accumulated_loss for um tensor com >1 elemento
                                train_progress.global_step,
                            )
                            ema_loss = ema_loss or accumulated_loss  # Use .item() se accumulated_loss for escalar
                            ema_loss = (ema_loss * 0.99) + (accumulated_loss * 0.01)
                            self.tensorboard.add_scalar(
                                "smooth_loss/train_step",
                                ema_loss,
                                train_progress.global_step,
                            )

                            accumulated_loss = 0.0
                            if self.model.ema:
                                update_step = (train_progress.global_step, self.config.gradient_accumulation_steps)
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
                    self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

                # Fim do loop de batches (fim da época)
                train_progress.next_epoch() # Avança a epoch para a próxima iteração do loop externo
                # Não é necessário chamar on_update_train_progress aqui novamente se já foi chamado no final do loop de batches
                
                # Log de final de epoch no console Rich
                final_epoch_duration = time.monotonic() - self._epoch_start_time_s
                avg_step_final_epoch = self._avg_step_time_epoch_s  # Usa o valor final calculado
                self.console.log(
                    f"[bold green]Epoch {train_progress.epoch} concluída em {format_time_delta(final_epoch_duration)} "
                    f"(Avg step: {avg_step_final_epoch:.3f}s/it)[/bold green]")

                if self.commands.get_and_reset_pause_request():
                    logFun(f"Requisição de PAUSA recebida. Será executada ao final da Epoch {train_progress.epoch -1}.", lvl="info")
                    self.pause_requested_at_epoch_end = True
                    self.pause_request_locked = True  # Trava a UI
                    # Notificar a UI que a requisição foi aceita e o switch está travado
                    if hasattr(self.callbacks, 'on_pause_request_accepted'):
                        self.callbacks.on_pause_request_accepted()

                    # 2. Executar a pausa se foi agendada
                if self.pause_requested_at_epoch_end and not self.is_paused:
                    self.is_paused = True  # Marca como pausado
                    self.pause_requested_at_epoch_end = False  # Limpa a flag de agendamento
                    # A trava (pause_request_locked) continua TRUE até o resume

                    # Chama a função que move o modelo e entra no loop de espera
                    self._handle_pause_logic()

                # Checagem de STOP ao final da época
                if self.commands.get_stop_command():
                    logFun("Comando STOP ativo no final da época. Encerrando...", lvl="info")
                    break  # Sai do loop de épocas

                # 1. TrainGPS salva os deltas da epoch (mantido, independente do CC)
                if gps_instance is not None:
                    if self.config.train_gps_save_it:
                        try:
                            epoch_idx = train_progress.epoch - 1
                            gps_instance.log_group_deltas(epoch_idx)
                        except Exception as e:
                            logFun(f"[TrainGPS] Erro ao logar deltas do grupo na época {train_progress.epoch - 1}: {e}", lvl="error")
                            traceback.print_exc()

                if self.run_number == 2 and self.adaptive_dcoef and gps_instance:
                    current_target_d_coef = self.adaptive_dcoef_normal_d_coef  # Default

                    if self.converge_control:
                        conv_score = self.converge_control.get_global_convergence_score()
                        if conv_score < self.adaptive_dcoef_convergence_threshold_low:
                            current_target_d_coef = self.adaptive_dcoef_initial_boost_d_coef
                            logFun(
                                f"[AdaptiveDCoef] ConvScore ({conv_score:.2f}) < LowThresh ({self.adaptive_dcoef_convergence_threshold_low:.2f}). Target D-Coef: BOOST ({current_target_d_coef})",
                                lvl="info")
                        elif conv_score < self.adaptive_dcoef_convergence_threshold_high:
                            current_target_d_coef = self.adaptive_dcoef_normal_d_coef
                            logFun(
                                f"[AdaptiveDCoef] ConvScore ({conv_score:.2f}) < HighThresh ({self.adaptive_dcoef_convergence_threshold_high:.2f}). Target D-Coef: NORMAL ({current_target_d_coef})",
                                lvl="info")
                        else:
                            current_target_d_coef = self.adaptive_dcoef_fine_tune_d_coef
                            logFun(
                                f"[AdaptiveDCoef] ConvScore ({conv_score:.2f}) >= HighThresh ({self.adaptive_dcoef_convergence_threshold_high:.2f}). Target D-Coef: FINE-TUNE ({current_target_d_coef})",
                                lvl="info")
                    else:
                        logFun(
                            f"[AdaptiveDCoef] ConvergeControl não ativo. Usando target d_coef normal: {current_target_d_coef}",
                            lvl="info")

                    try:
                        # Obter deltas L2 atuais (run 2) do TrainGPS
                        # _get_current_module_deltas retorna Dict[str, torch.Tensor(escalar)]
                        current_deltas_tensor_map = gps_instance._get_current_module_deltas(
                            self.train_device, self.train_dtype)
                        current_deltas_float_map: Dict[str, float] = {
                            name: tensor.item()
                            for name, tensor in current_deltas_tensor_map.items()
                            if tensor is not None}

                        # Obter nomes dos módulos/grupos de parâmetros gerenciados pelo otimizador
                        module_names_in_optimizer = [
                            pg['name'] for pg in self.model.optimizer.param_groups if 'name' in pg]
                        if not module_names_in_optimizer:
                            logFun("[AdaptiveDCoef] Nenhum grupo de parâmetros nomeado encontrado no otimizador.", lvl="warning")

                        # Calcular dcoef individualmente e aplicar se necessário
                        updated_count = 0
                        for group in self.model.optimizer.param_groups:
                            # carrega o perfil d_pdgy da Run 1
                            self.adaptive_dcoef.load_prodigy_d_final_run1(self.data_recorder_run1_path)
                            module_scores_run1 = self.adaptive_dcoef.module_prodigy_d_final_run1 or {}
                            # usa o d_pdgy mais recente capturado pelo DataRecorder
                            current_module_scores = self.recorder.d_pdgy_final  # d_pdgy_final: distância ao ótimo
                            # cálculo do dcoef já com todos os parâmetros
                            dcoefs = self.adaptive_dcoef.calculate_individual_d_coef(
                                module_scores_run1=module_scores_run1,
                                current_module_scores=current_module_scores,
                                gamma=self.adaptive_dcoef.gam,
                                min_scale=self.adaptive_dcoef.min_scale,
                                max_scale=self.adaptive_dcoef.max_scale,
                            )
                            name = group.get("name")
                            if name in dcoefs:
                                old = group["d_coef"]
                                new = dcoefs[name]
                                if old != new:
                                    group["d_coef"] = new
                                    updated_count += 1
                                    logFun(
                                        f"[AdaptiveDCoef] Module '{name}': d_coef alterado para {new:.4f} (era {old})",
                                        lvl="info")
                        if updated_count > 0 and (self.adaptive_dcoef.verbose or self.adaptive_dcoef.debug):
                            logFun(f"[AdaptiveDCoef] d_coef atualizado para {updated_count} grupos de parâmetros.",
                                  lvl="info")

                    except Exception as e:
                        logFun(
                            f"[AdaptiveDCoef@epoch{train_progress.epoch-1}] Erro ao aplicar ajuste dinâmico de d_coef: {e}",
                            lvl="error")
                        traceback.print_exc()

                # 3. ConvergeControl Decide/Aplica Congelamento (Run >= 2 e Ativo)
                if self.converge_control:
                    freeze_decisions = self.converge_control.decide()
                    for name, should_freeze in freeze_decisions.items():
                        param_group_obj = self.model.parameters.by_unique_name(name)
                        if not param_group_obj:
                            continue
                        should_be_enabled = not should_freeze
                        is_currently_enabled = param_group_obj.is_enabled
                        if is_currently_enabled != should_be_enabled:
                            action = "Unfreeze" if should_be_enabled else "Freeze"
                            if self.converge_control.verbose or self.converge_control.debug:
                                logFun(
                                    f"[Trainer ApplyFreeze@{train_progress.global_step}] "
                                    f"{action} module '{name}' based on ConvergeControl decision "
                                    f"(Run {self.run_number})",
                                    lvl="info",
                                    _console=self.console)
                                # TODO: APLICAR CONGELAMENTO NO OPTIMIZER PARAMETER GROUP TAMBÉM SE NECESSÁRIO
                                # A função `set_requires_grad` do `ParameterGroup` em `modules.model.ParameterGroup`
                                # já deve lidar com isso ou é o ponto de controle central.
                                # Se `set_requires_grad` está apenas no ParameterGroup e não afeta o optimizer,
                                # talvez precise de um `optimizer.set_param_group_enabled(name, should_be_enabled)`.
                                # Isso dependerá da implementação do otimizador e do BaseModel.
                            param_group_obj.set_requires_grad(should_be_enabled)

                if self.commands.get_stop_command():
                    return
        finally:
            # === LIMPEZA ===
            self._stop_training_display() # Para o Live display principal
            cleanup_global_progress() # Limpa o objeto Progress global (do logFun)

        # Exibir summary final
        total_training_duration = time.monotonic() - self._total_training_time_start_s if self._total_training_time_start_s else 0
        logFun(f"Treinamento completo! Tempo total: {format_time_delta(total_training_duration)}", lvl="SUCCESS")

    def end(self):
        save_path = os.path.join(
            self.config.workspace_dir,
            "save",
            f"{self.config.save_filename_prefix}{get_string_timestamp()}-save-{self.model.train_progress.filename_string()}{self.config.output_model_format.file_extension()}",
        )        
        if self.is_paused:
            logFun("Finalizando treinamento enquanto estava pausado. Tentando retomar brevemente para salvar.", lvl="warning")
            # Força a saída da pausa (sem esperar comando) e tenta mover para GPU para salvar
            self.is_paused = False
            self.pause_request_locked = False
            try:
                # Tenta mover de volta pra GPU rapidamente
                self.model_setup.setup_train_device(self.model, self.config)
                torch_gc()
                logFun("Modelo movido para GPU para salvamento final.", lvl="info")
            except Exception as e:
                logFun(
                    f"Falha ao mover modelo para GPU no final (estava pausado): {e}. Salvando do CPU ({self.temp_device}).",
                    lvl="error",
                    _console=self.console) # Usar _console aqui
                # O modelo já está no self.temp_device, o save deve funcionar
                pass  # Continua para salvar do CPU

        if self.one_step_trained:
            self.model.to(self.temp_device)
            torch_gc()

            if self.config.backup_before_save:
                self.backup(self.model.train_progress)  # Backup já usa o modelo no temp_device

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()

            self.callbacks.on_update_status("saving the final model")

            if self.model.ema:
                self.model.ema.copy_ema_to(self.parameters, store_temp=False)
            if (os.path.isdir(self.config.output_model_destination) and
                    self.config.output_model_format.is_single_file()):
                save_path = os.path.join(
                    self.config.output_model_destination,
                    f"{self.config.save_filename_prefix}{get_string_timestamp()}{self.config.output_model_format.file_extension()}",
                )
            else:
                save_path = self.config.output_model_destination
            logFun("Saving " + save_path, lvl="LOOP") # Usar logFun aqui

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
        model_name, _ = os.path.splitext(model_filename)  # Get model name without extension

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
                output_dir = os.path.join(self.config.workspace_dir, "data_recorder")  # Or reuse training_deltas dir?
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