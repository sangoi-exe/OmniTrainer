import os
import traceback
import math
import torch
import torch.nn as nn
import torch.nn.functional as F # Importado para referência, usado internamente por AttnProcessor2_0

# Importar os processadores originais para referência e delegação
from diffusers.models.attention_processor import AttnProcessor as OriginalAttnProcessor
from diffusers.models.attention_processor import AttnProcessor2_0 as OriginalAttnProcessor2_0

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelLoader.mixin.SDConfigModelLoaderMixin import SDConfigModelLoaderMixin
from modules.util import create
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


# --- Definições dos Processadores Customizados ---

class LearnableTauProcessor(nn.Module): # MODIFICADO: Herda de nn.Module
    """
    Processador de atenção "vanilla" com tau treinável.
    Herda de nn.Module para que self.log_tau seja um parâmetro treinável reconhecido.
    """
    def __init__(self, init_tau: float = 1.0):
        super().__init__() # ADICIONADO: Chama o construtor de nn.Module
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau)))

    def __call__(
        self,
        attn: nn.Module, # diffusers.models.attention.Attention
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            if encoder_hidden_states is not None and encoder_hidden_states.ndim == 4:
                 encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height*width).transpose(1,2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        effective_scale = attn.scale * self.log_tau.exp() # self.log_tau é usado aqui

        scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[2], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=effective_scale,
        )

        if attention_mask is not None:
            # A camada Attention deve preparar o attention_mask antes de chamar o processador.
            # Ex: attention_mask = attn.prepare_attention_mask(...)
            # Aqui, assumimos que attention_mask já está no formato correto para adição.
            scores = scores + attention_mask

        attention_probs = scores.softmax(dim=-1, dtype=value.dtype if value.dtype is not torch.float16 else torch.float32)
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1 and not isinstance(attn.to_out[1], nn.Identity):
            hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


class LearnableTauAttnProcessor2_0(nn.Module): # MODIFICADO: Herda de nn.Module
    """
    Processador de atenção que tenta manter otimizações (como SDPA de PyTorch 2.0+)
    e introduz um tau treinável. Herda de nn.Module.
    """
    def __init__(self, init_tau: float = 1.0):
        super().__init__() # ADICIONADO: Chama o construtor de nn.Module
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau)))
        # Instanciamos o processador original para delegar a lógica de atenção otimizada.
        self.original_processor_logic = OriginalAttnProcessor2_0()

    def __call__(
        self,
        attn: nn.Module, # diffusers.models.attention.Attention
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs, 
    ) -> torch.Tensor:
        
        original_attn_scale_value = attn.scale 
        new_effective_scale = original_attn_scale_value * self.log_tau.exp() # self.log_tau é usado aqui
        
        try:
            attn.scale = new_effective_scale 
            output = self.original_processor_logic(
                attn, 
                hidden_states, 
                encoder_hidden_states=encoder_hidden_states, 
                attention_mask=attention_mask, 
                **kwargs
            )
        finally:
            attn.scale = original_attn_scale_value 
            
        return output


class StableDiffusionXLModelLoader(
    SDConfigModelLoaderMixin,
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def _default_sd_config_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.STABLE_DIFFUSION_XL_10_BASE:
                return "resources/model_config/stable_diffusion_xl/sd_xl_base.yaml"
            case ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING:
                return "resources/model_config/stable_diffusion_xl/sd_xl_base-inpainting.yaml"
            case _:
                return None

    # // GEMINI-CODE {timestamp} - Modificação em __apply_learnable_tau_processors
    def __apply_learnable_tau_processors(self, unet: UNet2DConditionModel):
        if not hasattr(unet, 'attn_processors'):
            print("Aviso: UNet não possui 'attn_processors'. Tau customizado não aplicado.")
            return

        # Dicionário para os processadores que serão passados para unet.set_attn_processor()
        # Este conterá tanto os seus nn.Modules customizados quanto os processadores originais não modificados.
        final_processors_for_unet_usage = {}
        
        # Dicionário para o nn.ModuleDict, contendo APENAS seus nn.Modules customizados
        # para registro de parâmetros. Os nomes das chaves precisam ser válidos para atributos.
        learnable_tau_modules_for_param_registration = {}
        
        count_applied_tau_procs = 0
        
        print("--- Analisando processadores de atenção originais na UNet ---")
        for name, proc_original_instance in unet.attn_processors.items():
            # print(f"  Original {name}: {type(proc_original_instance)}") # Para depuração detalhada
            init_tau_value = 1.0
            
            # Tenta substituir pelo seu processador customizado
            custom_processor_instance = None
            if isinstance(proc_original_instance, OriginalAttnProcessor2_0):
                custom_processor_instance = LearnableTauAttnProcessor2_0(init_tau=init_tau_value)
                # print(f"    -> Substituindo {name} por LearnableTauAttnProcessor2_0.")
            elif isinstance(proc_original_instance, OriginalAttnProcessor):
                custom_processor_instance = LearnableTauProcessor(init_tau=init_tau_value)
                # print(f"    -> Substituindo {name} por LearnableTauProcessor.")
            
            if custom_processor_instance:
                final_processors_for_unet_usage[name] = custom_processor_instance
                # Nomes para ModuleDict não podem ter '.', então substituímos
                module_dict_key = name.replace('.', '_').replace('-', '_')
                learnable_tau_modules_for_param_registration[module_dict_key] = custom_processor_instance
                count_applied_tau_procs += 1
            else:
                # Mantém o processador original se não for um tipo conhecido ou se não quisermos modificar
                final_processors_for_unet_usage[name] = proc_original_instance
                # print(f"    -> Mantendo {name} como {type(proc_original_instance).__name__} (não modificado).")


        # 1. Atribui os processadores (mistura de customizados e originais) à UNet para uso na lógica de atenção.
        #    O unet.attn_processors continuará sendo um dict Python comum.
        if final_processors_for_unet_usage: # Garante que o dict não está vazio
             unet.set_attn_processor(final_processors_for_unet_usage)
             print(f"Dicionário unet.attn_processors atualizado com {len(final_processors_for_unet_usage)} processadores ({count_applied_tau_procs} customizados com Tau).")
        else:
            print("Nenhum processador final para atribuir a unet.attn_processors.")


        # 2. Se houver processadores customizados (que são nn.Module),
        #    registra-os na UNet através de um nn.ModuleDict para que seus parâmetros sejam encontrados.
        if learnable_tau_modules_for_param_registration:
            # Este atributo é novo e serve APENAS para o PyTorch encontrar os parâmetros.
            # A UNet NÃO usará este atributo para sua lógica de atenção, ela usa unet.attn_processors.
            unet.registered_learnable_tau_processors = nn.ModuleDict(learnable_tau_modules_for_param_registration)
            print(f"nn.ModuleDict 'registered_learnable_tau_processors' com {len(learnable_tau_modules_for_param_registration)} módulos Tau adicionado à UNet para registro de parâmetros.")
            
            # DEBUG: Verificar se os parâmetros agora são visíveis através do ModuleDict
            # print("--- Parâmetros dentro de unet.registered_learnable_tau_processors ---")
            # for param_name, param in unet.registered_learnable_tau_processors.named_parameters():
            #     if "log_tau" in param_name:
            #         print(f"  Encontrado no ModuleDict: {param_name}, Valor: {param.item():.4f}, Requer Grad: {param.requires_grad}")

        elif count_applied_tau_procs > 0:
             print("AVISO: Processadores Tau foram contados, mas o ModuleDict para registro de parâmetros está vazio. Isso não deveria acontecer.")
        else:
            print("Nenhum processador Tau customizado foi aplicado, então nenhum ModuleDict para parâmetros foi criado.")


    def __load_internal(
            self,
            model: StableDiffusionXLModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(model, model_type, weight_dtypes, base_model_name, vae_model_name)
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: StableDiffusionXLModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
    ):
        tokenizer_1 = CLIPTokenizer.from_pretrained(base_model_name, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(base_model_name, subfolder="tokenizer_2")
        noise_scheduler = DDIMScheduler.from_pretrained(base_model_name, subfolder="scheduler")
        noise_scheduler = create.create_noise_scheduler(NoiseScheduler.DDIM, noise_scheduler)
        text_encoder_1 = self._load_transformers_sub_module(CLIPTextModel, weight_dtypes.text_encoder, weight_dtypes.train_dtype, base_model_name, "text_encoder")
        text_encoder_2 = self._load_transformers_sub_module(CLIPTextModelWithProjection, weight_dtypes.text_encoder_2, weight_dtypes.train_dtype, base_model_name, "text_encoder_2")
        
        if vae_model_name:
            vae = self._load_diffusers_sub_module(AutoencoderKL, weight_dtypes.vae, weight_dtypes.fallback_train_dtype, vae_model_name)
        else:
            vae = self._load_diffusers_sub_module(AutoencoderKL, weight_dtypes.vae, weight_dtypes.fallback_train_dtype, base_model_name, "vae")
        
        unet = self._load_diffusers_sub_module(UNet2DConditionModel, weight_dtypes.unet, weight_dtypes.train_dtype, base_model_name, "unet")
        
        self.__apply_learnable_tau_processors(unet)

        model.model_type = model_type
        model.tokenizer_1 = tokenizer_1
        model.tokenizer_2 = tokenizer_2
        model.noise_scheduler = noise_scheduler
        model.text_encoder_1 = text_encoder_1
        model.text_encoder_2 = text_encoder_2
        model.vae = vae
        model.unet = unet

    def __load_ckpt(
            self,
            model: StableDiffusionXLModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
    ):
        pipeline = StableDiffusionXLPipeline.from_single_file(base_model_name, original_config=model.sd_config_filename, safety_checker=None)
        noise_scheduler = create.create_noise_scheduler(NoiseScheduler.DDIM, pipeline.scheduler)
        if vae_model_name:
            pipeline.vae = AutoencoderKL.from_pretrained(vae_model_name, torch_dtype=weight_dtypes.vae.torch_dtype())

        text_encoder_1 = pipeline.text_encoder.to(dtype=weight_dtypes.text_encoder.torch_dtype())
        text_encoder_1.text_model.embeddings.to(dtype=weight_dtypes.text_encoder.torch_dtype(False))
        text_encoder_2 = pipeline.text_encoder_2.to(dtype=weight_dtypes.text_encoder_2.torch_dtype())
        text_encoder_2.text_model.embeddings.to(dtype=weight_dtypes.text_encoder_2.torch_dtype(False))
        vae = pipeline.vae.to(dtype=weight_dtypes.vae.torch_dtype())
        unet = pipeline.unet.to(dtype=weight_dtypes.unet.torch_dtype())

        self.__apply_learnable_tau_processors(unet)

        model.model_type = model_type
        model.tokenizer_1 = pipeline.tokenizer
        model.tokenizer_2 = pipeline.tokenizer_2
        model.noise_scheduler = noise_scheduler
        model.text_encoder_1 = text_encoder_1
        model.text_encoder_2 = text_encoder_2
        model.vae = vae
        model.unet = unet

    def __load_safetensors(
            self,
            model: StableDiffusionXLModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
    ):
        if model_type.has_conditioning_image_input():
            pipeline = StableDiffusionXLInpaintPipeline.from_single_file(base_model_name, original_config=model.sd_config_filename, safety_checker=None, use_safetensors=True)
        else:
            pipeline = StableDiffusionXLPipeline.from_single_file(base_model_name, original_config=model.sd_config_filename, safety_checker=None, use_safetensors=True)
        
        noise_scheduler = create.create_noise_scheduler(NoiseScheduler.DDIM, pipeline.scheduler)
        if vae_model_name:
            vae = self._load_diffusers_sub_module(AutoencoderKL, weight_dtypes.vae, weight_dtypes.fallback_train_dtype, vae_model_name)
        else:
            vae = self._convert_diffusers_sub_module_to_dtype(pipeline.vae, weight_dtypes.vae, weight_dtypes.fallback_train_dtype)
        
        text_encoder_1 = self._convert_transformers_sub_module_to_dtype(pipeline.text_encoder_1, weight_dtypes.text_encoder, weight_dtypes.train_dtype)
        text_encoder_2 = self._convert_transformers_sub_module_to_dtype(pipeline.text_encoder_2, weight_dtypes.text_encoder_2, weight_dtypes.train_dtype)
        unet = self._convert_diffusers_sub_module_to_dtype(pipeline.unet, weight_dtypes.unet, weight_dtypes.train_dtype)

        self.__apply_learnable_tau_processors(unet)

        model.model_type = model_type
        model.tokenizer_1 = pipeline.tokenizer
        model.tokenizer_2 = pipeline.tokenizer_2
        model.noise_scheduler = noise_scheduler
        model.text_encoder_1 = text_encoder_1
        model.text_encoder_2 = text_encoder_2
        model.vae = vae
        model.unet = unet

    def load(
            self,
            model: StableDiffusionXLModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ):
        stacktraces = []
        model.sd_config = self._load_sd_config(model_type, model_names.base_model)
        model.sd_config_filename = self._get_sd_config_name(model_type, model_names.base_model)

        # Ordem de tentativa de carregamento
        load_methods = [
            ("_internal", self.__load_internal),
            ("_diffusers (explícito)", self.__load_diffusers),
            ("_safetensors", self.__load_safetensors),
            ("_ckpt", self.__load_ckpt),
        ]

        for method_name, load_fn in load_methods:
            try:
                print(f"Tentando carregar modelo via {method_name}...")
                load_fn(model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model)
                print(f"Modelo carregado com sucesso via {method_name}.")
                return # Carregamento bem-sucedido, retorna
            except Exception as e:
                stacktraces.append(f"Falha ao carregar via {method_name}:\n{traceback.format_exc()}")
                print(f"Falha ao carregar via {method_name}. Erro: {e}")


        print("\n--- Pilhas de Erro de Todas as Tentativas de Carregamento ---")
        for i, stacktrace_msg in enumerate(stacktraces):
            print(f"\nTentativa {i+1}:")
            print(stacktrace_msg)
        raise Exception("Não foi possível carregar o modelo utilizando nenhum dos métodos disponíveis: " + model_names.base_model)