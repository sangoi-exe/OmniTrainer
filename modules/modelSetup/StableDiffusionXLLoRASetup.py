import os
import traceback
from typing import Optional
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSetup.BaseStableDiffusionXLSetup import BaseStableDiffusionXLSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.NamedParameterGroup import (
    NamedParameterGroup,
    NamedParameterGroupCollection,
)
from modules.util.optimizer_util import init_model_parameters
from modules.util.torch_util import state_dict_has_prefix
from modules.util.TrainProgress import TrainProgress
from modules.module.LoRAModule import PeftBase
from modules.util.TensorBoardManager import TensorBoardManager

import torch

PRESETS = {
    "attn-mlp": ["attentions"],
    "attn-only": ["attn"],
    "full": [],
}


class StableDiffusionXLLoRASetup(
    BaseStableDiffusionXLSetup,
):
    def __init__(
        self,
        train_device: torch.device,
        temp_device: torch.device,
        debug_mode: bool,
    ):
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
        self,
        model: StableDiffusionXLModel,
        config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        # Grupo para Text Encoder 1 LoRA/DoRA/LoHa
        if config.text_encoder.train and model.text_encoder_1_lora:
            for (
                original_name,
                peft_module,
            ) in model.text_encoder_1_lora.lora_modules.items():
                # Certifique-se de que o módulo PEFT foi inicializado e tem parâmetros
                if peft_module._initialized and list(peft_module.parameters()):
                    # Usar o prefixo do módulo PEFT garante unicidade e reflete a chave do state_dict
                    unique_name = peft_module.prefix.removesuffix(".")
                    parameter_group_collection.add_group(
                        NamedParameterGroup(
                            unique_name=unique_name,
                            # Opcional: Usar nome original para display
                            display_name=f"te1/{original_name}",
                            parameters=peft_module.parameters(),
                            learning_rate=config.text_encoder.learning_rate,
                        )
                    )

        # Grupo para Text Encoder 2 LoRA/DoRA/LoHa
        if config.text_encoder_2.train and model.text_encoder_2_lora:
            for (
                original_name,
                peft_module,
            ) in model.text_encoder_2_lora.lora_modules.items():
                if peft_module._initialized and list(peft_module.parameters()):
                    unique_name = peft_module.prefix.removesuffix(".")
                    parameter_group_collection.add_group(
                        NamedParameterGroup(
                            unique_name=unique_name,
                            display_name=f"te2/{original_name}",
                            parameters=peft_module.parameters(),
                            learning_rate=config.text_encoder_2.learning_rate,
                        )
                    )

        # Grupo para UNet LoRA/DoRA/LoHa
        if config.unet.train and model.unet_lora:
            for original_name, peft_module in model.unet_lora.lora_modules.items():
                if peft_module._initialized and list(peft_module.parameters()):
                    unique_name = peft_module.prefix.removesuffix(".")
                    parameter_group_collection.add_group(
                        NamedParameterGroup(
                            unique_name=unique_name,
                            display_name=f"unet/{original_name}",
                            parameters=peft_module.parameters(),
                            learning_rate=config.unet.learning_rate,
                        )
                    )

        if config.train_any_embedding() or config.train_any_output_embedding():
            if config.text_encoder.train_embedding:
                self._add_embedding_param_groups(
                    model.all_text_encoder_1_embeddings(),
                    parameter_group_collection,
                    config.embedding_learning_rate,
                    "embeddings_1",
                )

            if config.text_encoder_2.train_embedding:
                self._add_embedding_param_groups(
                    model.all_text_encoder_2_embeddings(),
                    parameter_group_collection,
                    config.embedding_learning_rate,
                    "embeddings_2",
                )

        return parameter_group_collection

    def __setup_requires_grad(
        self,
        model: StableDiffusionXLModel,
        config: TrainConfig,
    ):
        self._setup_embeddings_requires_grad(model, config)
        model.text_encoder_1.requires_grad_(False)
        model.text_encoder_2.requires_grad_(False)
        model.unet.requires_grad_(False)
        model.vae.requires_grad_(False)

        if model.text_encoder_1_lora is not None:
            train_text_encoder_1 = config.text_encoder.train and not self.stop_text_encoder_training_elapsed(
                config, model.train_progress
            )
            model.text_encoder_1_lora.requires_grad_(train_text_encoder_1)

        if model.text_encoder_2_lora is not None:
            train_text_encoder_2 = config.text_encoder_2.train and not self.stop_text_encoder_2_training_elapsed(
                config, model.train_progress
            )
            model.text_encoder_2_lora.requires_grad_(train_text_encoder_2)

        if model.unet_lora is not None:
            train_unet = config.unet.train and not self.stop_unet_training_elapsed(config, model.train_progress)
            model.unet_lora.requires_grad_(train_unet)

    def setup_model(
        self, model: StableDiffusionXLModel, config: TrainConfig, tensorboard: Optional[TensorBoardManager] = None
    ):

        model.tensorboard = tensorboard
        print(
            f"[TensorBoardManager] Instância TensorBoard {'atribuída' if model.tensorboard else 'NÃO atribuída'} ao modelo."
        )

        create_te1 = config.text_encoder.train or state_dict_has_prefix(model.lora_state_dict, "lora_te1")
        create_te2 = config.text_encoder_2.train or state_dict_has_prefix(model.lora_state_dict, "lora_te2")

        model.text_encoder_1_lora = LoRAModuleWrapper(model.text_encoder_1, "lora_te1", config) if create_te1 else None

        model.text_encoder_2_lora = LoRAModuleWrapper(model.text_encoder_2, "lora_te2", config) if create_te2 else None

        model.unet_lora = LoRAModuleWrapper(model.unet, "lora_unet", config, config.lora_layers.split(","))

        if model.lora_state_dict:
            if create_te1:
                model.text_encoder_1_lora.load_state_dict(model.lora_state_dict)
            if create_te2:
                model.text_encoder_2_lora.load_state_dict(model.lora_state_dict)

            model.unet_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        if config.text_encoder.train:
            model.text_encoder_1_lora.set_dropout(config.dropout_probability)
        if config.text_encoder_2.train:
            model.text_encoder_2_lora.set_dropout(config.dropout_probability)
        model.unet_lora.set_dropout(config.dropout_probability)

        if create_te1:
            model.text_encoder_1_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_1_lora.hook_to_module()
        if create_te2:
            model.text_encoder_2_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_2_lora.hook_to_module()

        model.unet_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.unet_lora.hook_to_module()

        if config.rescale_noise_scheduler_to_zero_terminal_snr:
            model.rescale_noise_scheduler_to_zero_terminal_snr()
            model.force_v_prediction()

        self._remove_added_embeddings_from_tokenizer(model.tokenizer_1)
        self._remove_added_embeddings_from_tokenizer(model.tokenizer_2)
        self._setup_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)

        parameter_collection = self.create_parameters(model, config)  # Recria ou pega a coleção
        init_model_parameters(model, parameter_collection, self.train_device)

        model.parameters = parameter_collection

        # testar esse mamute do gemini, cheio de verificações e etc
        # START: Modificação solicitada - Refined DeltaPatternRegularizer initialization
        from modules.sangoi.DynamicLossStrength import DeltaPatternRegularizer
        import torch # Ensure torch is imported

        # Garante que deltas é None por padrão
        model.deltas = None

        if config.delta_pattern_use_it or config.delta_pattern_save_it:
            print("[DeltaPattern] Tentando inicializar DeltaPatternRegularizer...")
            try:
                # A coleção de parâmetros AGORA existe e foi atribuída
                param_collection = getattr(model, "parameters", None)
                if param_collection is None or not isinstance(param_collection, NamedParameterGroupCollection):
                     raise ValueError("Coleção de parâmetros (model.parameters) inválida ou não encontrada para DeltaPattern.")
                if not list(param_collection.parameters()):
                    print("[DeltaPattern] Aviso: Coleção de parâmetros está vazia. DeltaPattern não será inicializado.")
                else:
                    # Determina o cache_device (GPU preferencialmente)
                    # Tenta usar o train_device se disponível, senão o device do primeiro parâmetro
                    if hasattr(self, 'train_device'):
                        cache_dev = self.train_device
                    else:
                        first_param = next(iter(param_collection.parameters()), None)
                        if first_param is not None:
                             cache_dev = first_param.device
                        else:
                             # Fallback para CPU se nenhum parâmetro foi encontrado (improvável aqui)
                             print("[DeltaPattern] Aviso: Não foi possível determinar o device dos parâmetros. Usando CPU para cache.")
                             cache_dev = torch.device("cpu")
                    print(f"[DeltaPattern] Usando device '{cache_dev}' para cache interno.")

                    # Atribui a instância ao modelo
                    model.deltas = DeltaPatternRegularizer(
                        model=model, # Passa a instância do modelo completa
                        param_collection=param_collection, # Passa a coleção de parâmetros
                        penalty_metric=getattr(config, "delta_pattern_metric", "cosine"), # Usa config ou default
                        cache_device=cache_dev # Passa o device de cache determinado
                    )
                    print("[DeltaPattern] DeltaPatternRegularizer instanciado.")
                    print(f"[Trainer Setup] DeltaPattern cache_device: {model.deltas.cache_device}")
                    print(f"[Trainer Setup] Trainer train_device: {self.train_device}")
                    if model.deltas.cache_device != self.train_device:
                        print("[Trainer Setup] ALERTA! DeltaPattern cache_device é diferente do train_device!")                    

                    if config.delta_pattern_save_it:
                        print("[DeltaPattern] Capturando pesos iniciais para salvar (Run 1)...")
                        model.deltas.capture_weights() # Usa o cache_device interno

                    if config.delta_pattern_use_it:
                        delta_path = getattr(config, "delta_pattern_path", None)
                        if delta_path and os.path.exists(delta_path):
                            print(f"[DeltaPattern] Carregando padrão de referência de: {delta_path}")
                            # Determina device/dtype de treino para carregar o padrão corretamente
                            train_dtype_torch = config.train_dtype.torch_dtype() # Pega do config
                            train_dev = self.train_device # Usa o train_device da classe setup

                            model.deltas.load_reference_pattern(
                                delta_path,
                                train_device=train_dev,    # Device de treino explícito
                                train_dtype=train_dtype_torch # Dtype de treino explícito
                            )
                            if model.deltas.reference_deltas:  # Checa se carregou
                                print("[DeltaPattern] Capturando pesos iniciais para cálculo de penalidade (Run 2)...")
                                model.deltas.capture_initial_weights_run2() # Usa o cache_device interno
                            else:
                                print(f"[DeltaPattern] Aviso: Falha ao carregar padrão de delta de '{delta_path}'. Penalidade desativada.")
                                config.delta_pattern_use_it = False  # Desativa
                        else:
                            print(f"[DeltaPattern] Aviso: 'delta_pattern_use_it' True, mas caminho '{delta_path}' inválido ou não especificado. Penalidade desativada.")
                            config.delta_pattern_use_it = False  # Desativa

                    print("[DeltaPattern] Configuração do DeltaPatternRegularizer concluída.")

            except Exception as e:
                print(f"[DeltaPattern] Falha CRÍTICA ao inicializar DeltaPatternRegularizer: {e}")
                traceback.print_exc()
                model.deltas = None  # Garante None se falhar
                config.delta_pattern_use_it = False # Desativa a funcionalidade se a inicialização falhar
        else:
            print("[DeltaPattern] DeltaPatternRegularizer não será usado (configurações desativadas).")
            model.deltas = None # Garante None se não for usado
        # END: Modificação solicitada - Refined DeltaPatternRegularizer initialization

        # # Código original comentado para referência
        # model.deltas = DeltaPatternRegularizer(model, model.parameters)
        # # Captura os pesos iniciais se a opção de salvar estiver ativa (Run 1)

        # model.deltas = DeltaPatternRegularizer(model, model.parameters)
        # # Captura os pesos iniciais se a opção de salvar estiver ativa (Run 1)
        # if config.delta_pattern_save_it:
        #     print("[DeltaPattern] Capturando pesos iniciais para logging dos deltas por grupo (Run 1)")
        #     model.deltas.capture_weights()

        # if config.delta_pattern_use_it:
        #     if config.delta_pattern_path and os.path.exists(config.delta_pattern_path):
        #         print(f"[DeltaPattern] Carregando padrão de delta de referência de: {config.delta_pattern_path}")
        #         model.deltas.load_reference_pattern(config.delta_pattern_path)
        #         if model.deltas.reference_deltas:  # Verifica se carregou com sucesso
        #             print("[DeltaPattern] Capturando pesos iniciais para cálculo da penalidade (Run 2).")
        #             model.deltas.capture_initial_weights_run2()
        #         else:
        #             print(
        #                 f"[DeltaPattern] Aviso: Falha ao carregar o padrão de delta de '{config.delta_pattern_path}'. A penalidade será desativada."
        #             )
        #             config.delta_pattern_use_it = False  # Desativa se não conseguiu carregar
        #     else:
        #         print(
        #             f"[DeltaPattern] Aviso: 'delta_pattern_use_it' é True, mas o caminho '{config.delta_pattern_path}' não foi encontrado ou não especificado. A penalidade será desativada."
        #         )
        #         config.delta_pattern_use_it = False  # Desativa se o caminho não existe

    def setup_train_device(
        self,
        model: StableDiffusionXLModel,
        config: TrainConfig,
    ):
        vae_on_train_device = not config.latent_caching
        text_encoder_1_on_train_device = config.train_text_encoder_or_embedding() or not config.latent_caching
        text_encoder_2_on_train_device = config.train_text_encoder_2_or_embedding() or not config.latent_caching

        model.text_encoder_1_to(self.train_device if text_encoder_1_on_train_device else self.temp_device)
        model.text_encoder_2_to(self.train_device if text_encoder_2_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.unet_to(self.train_device)

        if config.text_encoder.train:
            model.text_encoder_1.train()
        else:
            model.text_encoder_1.eval()

        if config.text_encoder_2.train:
            model.text_encoder_2.train()
        else:
            model.text_encoder_2.eval()

        model.vae.eval()

        if config.unet.train:
            model.unet.train()
        else:
            model.unet.eval()

    def after_optimizer_step(
        self,
        model: StableDiffusionXLModel,
        config: TrainConfig,
        train_progress: TrainProgress,
    ):
        if config.preserve_embedding_norm:
            self._normalize_output_embeddings(model.all_text_encoder_1_embeddings())
            self._normalize_output_embeddings(model.all_text_encoder_2_embeddings())
            model.embedding_wrapper_1.normalize_embeddings()
            model.embedding_wrapper_2.normalize_embeddings()
        self.__setup_requires_grad(model, config)
