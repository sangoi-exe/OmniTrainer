import os
import traceback
from typing import Optional

import inspect
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSetup.BaseStableDiffusionXLSetup import BaseStableDiffusionXLSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.sangoi.logFun import logFun
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

def log_tau_requires_grad_status(module: torch.nn.Module, module_name_prefix: str, stage_description: str):
    print(f"\n--- {stage_description} --- Estado de 'requires_grad' para 'log_tau' em '{module_name_prefix}' ---")
    found_any_tau = False
    found_trainable_tau = False
    
    # Se o módulo tiver o nosso ModuleDict customizado
    if hasattr(module, 'registered_learnable_tau_processors'):
        print(f"  Inspecionando '{module_name_prefix}.registered_learnable_tau_processors':")
        for proc_module_name, proc_module_instance in module.registered_learnable_tau_processors.items():
            for param_name, param in proc_module_instance.named_parameters(recurse=False):
                if "log_tau" in param_name:
                    found_any_tau = True
                    full_param_name = f"{module_name_prefix}.registered_learnable_tau_processors.{proc_module_name}.{param_name}"
                    print(f"    {full_param_name}: requires_grad = {param.requires_grad}")
                    if param.requires_grad:
                        found_trainable_tau = True
        if not found_any_tau:
            print(f"    Nenhum 'log_tau' encontrado em '{module_name_prefix}.registered_learnable_tau_processors'.")

    # Tenta encontrar 'log_tau' em outros lugares (caso a estrutura seja diferente)
    # Isso pode ser útil se os processadores foram injetados de outra forma
    # ou se os parâmetros LoRA também tiverem 'log_tau' (improvável, mas para cobrir).
    # else: # Ou inspeciona todos os parâmetros do módulo se o ModuleDict não existir
    #     print(f"  Inspecionando todos os parâmetros de '{module_name_prefix}' (fallback):")
    #     for param_name, param in module.named_parameters():
    #         if "log_tau" in param_name: # Pode gerar muitos falsos positivos se "log_tau" for comum
    #             found_any_tau = True
    #             print(f"    {module_name_prefix}.{param_name}: requires_grad = {param.requires_grad}")
    #             if param.requires_grad:
    #                 found_trainable_tau = True
    
    if not found_any_tau:
        print(f"  Nenhum parâmetro 'log_tau' encontrado em '{module_name_prefix}' durante esta inspeção.")
    elif found_trainable_tau:
        print(f"  Pelo menos um 'log_tau' em '{module_name_prefix}' está TREINÁVEL.")
    else:
        print(f"  Todos os 'log_tau' encontrados em '{module_name_prefix}' estão CONGELADOS.")
    print(f"--- Fim da inspeção para {stage_description} ---")

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
            if hasattr(model.unet, "tau_procs"):                    
                tau_params = [p.log_tau for p in model.unet.tau_procs.values()]               
                if tau_params:                       # deve haver ~140
                    parameter_group_collection.add_group(
                        NamedParameterGroup(
                            unique_name="tau",                     
                            display_name="tau",                     
                            parameters=tau_params,
                            learning_rate=config.unet.learning_rate
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
        # // GEMINI-CODE {timestamp} - INÍCIO DA INSPEÇÃO DE CHAMADA
        # global_step = getattr(model.train_progress, 'step', 'DESCONHECIDO') # Tenta pegar o step atual
        
        # print(f"\n\n--- EXECUTANDO __setup_requires_grad (Step: {global_step}) ---")
        
        # Imprimir o call stack para depuração
        # Limitamos a profundidade para não poluir demais, mas ajuste se necessário
        # O primeiro frame (índice 0) é a própria __setup_requires_grad
        # O segundo frame (índice 1) é quem a chamou diretamente
        # O terceiro frame (índice 2) é quem chamou o chamador, e assim por diante.
        
        # print("  Call Stack (quem está chamando esta função?):")
        # stack = inspect.stack()
        # # Começamos do frame 1 (quem chamou __setup_requires_grad)
        # # e vamos até uns 5 níveis acima, ou menos se a pilha for curta.
        # for i in range(1, min(6, len(stack))): 
        #     frame = stack[i]
        #     # frame[0] é o objeto frame
        #     # frame[1] é o nome do arquivo
        #     # frame[2] é o número da linha
        #     # frame[3] é o nome da função/método
        #     # frame[4] são as linhas de contexto do código (pode ser None)
        #     # frame[5] é o índice da linha atual no contexto (pode ser None)
        #     print(f"    -> Nível {i}: Função '{frame.function}' no arquivo '{frame.filename}', linha {frame.lineno}")
        #     # Se quiser ver o código da linha que chamou (pode ser útil):
        #     # if frame.code_context:
        #     #    print(f"       Contexto: {''.join(frame.code_context).strip()}")
        # if len(stack) <=1:
        #     print("    -> Call stack muito curto, chamada direta de um escopo global ou similar?")
        # print("--- FIM DA INSPEÇÃO DE CHAMADA ---\n")
        # // GEMINI-CODE {timestamp} - Log ANTES de qualquer modificação na UNet
        # if hasattr(model, 'unet'):
        #     log_tau_requires_grad_status(model.unet, "model.unet", "Início de __setup_requires_grad (antes de modificações na UNet)")
        # if hasattr(model, 'unet_lora') and model.unet_lora is not None:
        #      log_tau_requires_grad_status(model.unet_lora, "model.unet_lora", "Início de __setup_requires_grad (antes de modificações na UNet LoRA)")

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
        
        self.__ensure_tau_params_trainable(model) # Passa o objeto model completo

    def __ensure_tau_params_trainable(self, model: StableDiffusionXLModel):
        if hasattr(model, 'unet') and hasattr(model.unet, 'tau_procs'):
            print("\n--- Trainer: Garantindo que parâmetros 'log_tau' são treináveis (pós-setup global) ---")
            made_trainable_count = 0
            total_log_tau_params = 0
            for module_name, module_instance in model.unet.tau_procs.items():
                for param_name, param in module_instance.named_parameters(recurse=False):
                    if "log_tau" in param_name:
                        total_log_tau_params +=1
                        if not param.requires_grad:
                            param.requires_grad_(True)
                            # print(f"  DESCONGELADO (via __ensure_tau_params_trainable): model.unet.tau_procs.{module_name}.{param_name}")
                            made_trainable_count +=1
                        # else:
                        #    print(f"  JÁ TREINÁVEL: model.unet.tau_procs.{module_name}.{param_name}")
            
            if total_log_tau_params > 0:
                if made_trainable_count > 0:
                    print(f"  {made_trainable_count}/{total_log_tau_params} parâmetros 'log_tau' foram explicitamente definidos como treináveis por esta função.")
                else:
                    print(f"  Todos os {total_log_tau_params} parâmetros 'log_tau' encontrados já estavam treináveis ou não foram encontrados para descongelar.")
            else:
                print("  Nenhum parâmetro 'log_tau' encontrado em 'tau_procs' para verificar/descongelar.")

        else:
            print("--- Trainer: Nenhum 'tau_procs' encontrado na UNet para a verificação final do Tau.")

    def setup_model(
        self, model: StableDiffusionXLModel, config: TrainConfig, tensorboard: Optional[TensorBoardManager] = None
    ):

        model.tensorboard = tensorboard
        msg = f"[TensorBoardManager] Instância TensorBoard {'atribuída' if model.tensorboard else 'NÃO atribuída'} ao modelo."
        lvl = "success" if model.tensorboard else "error"
        logFun(msg, lvl=lvl)

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
        from modules.sangoi.TrainGPS import TrainGPS
        # Garante que deltas é None por padrão
        model.deltas = None

        if config.train_gps_use_it or config.train_gps_save_it:
            logFun("Tentando inicializar TrainGPS...", lvl="TRAINGPS")
            try:
                # A coleção de parâmetros AGORA existe e foi atribuída
                param_collection = getattr(model, "parameters", None)
                if param_collection is None or not isinstance(param_collection, NamedParameterGroupCollection):
                      raise ValueError("Coleção de parâmetros (model.parameters) inválida ou não encontrada para DeltaPattern.")
                if not list(param_collection.parameters()):
                    logFun("Aviso: Coleção de parâmetros está vazia. DeltaPattern não será inicializado.", lvl="error")

                if config.train_gps_save_it:
                    # Atribui a instância ao modelo
                    model.deltas = TrainGPS(
                        model=model, # Passa a instância do modelo completa
                        param_collection=param_collection, # Passa a coleção de parâmetros
                        penalty_metric=getattr(config, "delta_pattern_metric", "cosine"), # Usa config ou default
                    )
                
                    logFun("Capturando pesos iniciais para salvar (Run 1)...", lvl="TRAINGPS")
                    model.deltas.capture_weights() # Usa o cache_device interno

                if config.train_gps_use_it:
                    delta_path = getattr(config, "train_gps_path", None)
                    if delta_path and os.path.exists(delta_path):
                        logFun(f"Carregando padrão de referência de: {delta_path}", lvl="TRAINGPS")
                        # Determina device/dtype de treino para carregar o padrão corretamente
                        train_dtype_torch = config.train_dtype.torch_dtype() # Pega do config
                        train_dev = self.train_device # Usa o train_device da classe setup

                        model.deltas.load_reference_pattern(delta_path)
                        if model.deltas.reference_deltas:  # Checa se carregou
                            logFun("Capturando pesos iniciais para cálculo de penalidade (Run 2)...", lvl="TRAINGPS")
                            model.deltas.capture_initial_weights_run2() # Usa o cache_device interno
                        else:
                            logFun(f"Aviso: Falha ao carregar padrão de delta de '{delta_path}'. Penalidade desativada.", lvl="error")
                            config.train_gps_use_it = False  # Desativa
                    else:
                        logFun(f"Aviso: 'train_gps_use_it' True, mas caminho '{delta_path}' inválido ou não especificado. Penalidade desativada.", lvl="warning")
                        config.train_gps_use_it = False  # Desativa

                logFun("Configuração do TrainGPS concluída.", lvl="TRAINGPS")

            except Exception as e:
                logFun(f"Falha CRÍTICA ao inicializar TrainGPS: {e}", lvl="error")
                traceback.print_exc()
                model.deltas = None  # Garante None se falhar
                config.train_gps_use_it = False # Desativa a funcionalidade se a inicialização falhar
        else:
            logFun("TrainGPS não será usado (configurações desativadas).", lvl="warning")
            model.deltas = None # Garante None se não for usado
        # END: Modificação solicitada - Refined TrainGPS initialization

        # # Código original comentado para referência
        # model.deltas = TrainGPS(model, model.parameters)
        # # Captura os pesos iniciais se a opção de salvar estiver ativa (Run 1)

        # model.deltas = TrainGPS(model, model.parameters)
        # # Captura os pesos iniciais se a opção de salvar estiver ativa (Run 1)
        # if config.train_gps_save_it:
        #     print("Capturando pesos iniciais para logging dos deltas por grupo (Run 1)")
        #     model.deltas.capture_weights()

        # if config.train_gps_use_it:
        #     if config.train_gps_path and os.path.exists(config.train_gps_path):
        #         print(f"Carregando padrão de delta de referência de: {config.train_gps_path}")
        #         model.deltas.load_reference_pattern(config.train_gps_path)
        #         if model.deltas.reference_deltas:  # Verifica se carregou com sucesso
        #             print("Capturando pesos iniciais para cálculo da penalidade (Run 2).")
        #             model.deltas.capture_initial_weights_run2()
        #         else:
        #             print(
        #                 f"Aviso: Falha ao carregar o padrão de delta de '{config.train_gps_path}'. A penalidade será desativada."
        #             )
        #             config.train_gps_use_it = False  # Desativa se não conseguiu carregar
        #     else:
        #         print(
        #             f"Aviso: 'train_gps_use_it' é True, mas o caminho '{config.train_gps_path}' não foi encontrado ou não especificado. A penalidade será desativada."
        #         )
        #         config.train_gps_use_it = False  # Desativa se o caminho não existe

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
