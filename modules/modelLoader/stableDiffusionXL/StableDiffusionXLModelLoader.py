import os
import traceback
import math
import torch
import torch.nn as nn
import torch.nn.functional as F # Importado para referência, usado internamente por AttnProcessor2_0

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
from diffusers.models.attention_processor import AttnProcessor2_0 as AP2

class LearnableTauAttnProcessor(nn.Module):
    """
    Custom AttnProcessor que replica o fluxo do AttnProcessor2_0, mas
    passa um `scale` explícito ao SDPA contendo o tau treinável.
    """

    def __init__(self, init_tau: float = 1.0):
        super().__init__()
        # tau > 0 via exp; armazenamos no log para estabilidade
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau)))

    def __call__(self,
                 attn: AP2.__class__,  # só p/ type hint
                 hidden_states: torch.Tensor,
                 encoder_hidden_states: torch.Tensor | None = None,
                 attention_mask: torch.Tensor | None = None,
                 temb: torch.Tensor | None = None,
                 **kwargs) -> torch.Tensor:

        # 1) Preparação de Residual + SpatialNorm
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # 2) Flatten 4D → 3D
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch, channel, height*width).transpose(1,2)
        else:
            batch, seq_len, _ = hidden_states.shape

        # 3) Máscara
        processed_attention_mask = None
        if attention_mask is not None:
            processed_attention_mask = attn.prepare_attention_mask(attention_mask, hidden_states.shape[1], batch)
            processed_attention_mask = processed_attention_mask.view(
                batch, attn.heads, -1, processed_attention_mask.shape[-1]
            )

        # 4) GroupNorm (se houver)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1,2)).transpose(1,2)

        # 5) Projeções Q/K/V
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            kv = hidden_states
        elif attn.norm_cross:
            kv = attn.norm_encoder_hidden_states(encoder_hidden_states)
        else:
            kv = encoder_hidden_states
        key   = attn.to_k(kv)
        value = attn.to_v(kv)

        # 6) Rearrange para (batch, heads, seq, head_dim)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch, -1, attn.heads, head_dim).transpose(1,2)
        key   = key.view(batch, -1, attn.heads, head_dim).transpose(1,2)
        value = value.view(batch, -1, attn.heads, head_dim).transpose(1,2)

        # 7) Normas pontuais (opcional)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 8) Calcula o tau e o aplica de forma diferenciável, garantindo device/dtype corretos

        # self.log_tau é o nn.Parameter. Ele já deve estar no device correto (ex: 'cuda:0')
        # e no dtype do modelo (ex: torch.bfloat16) se model.to(device, dtype) funcionou corretamente
        # para todos os nn.Parameter registrados.

        # Calcula tau a partir de log_tau. tau terá o mesmo device e dtype de self.log_tau.
        tau_for_computation = torch.exp(self.log_tau)

        # Agora aplique o tau. Escolha a formulação correta:
        # Opção 1: tau como "temperatura" (maior tau = mais suave, menor tau = mais sharp)
        # Para que a escala efetiva (original_scale / tau_temperatura) seja usada.
        # Como SDPA usa scale_original = 1/sqrt(head_dim), e queremos (Q@K.T * scale_original) / tau_temperatura,
        # isso é equivalente a (Q/sqrt(tau_temperatura) @ K.T/sqrt(tau_temperatura)) * scale_original.
        # Então:
        query_modified = query / torch.sqrt(tau_for_computation)
        key_modified = key / torch.sqrt(tau_for_computation)
        # Se init_tau = 1.0, log_tau = 0, tau_for_computation = 1.0. Nenhuma mudança inicial.
        # Se o modelo aprender log_tau < 0 => tau_for_computation < 1 => sqrt(tau) < 1 => 1/sqrt(tau) > 1
        # Isso AUMENTA a magnitude de Q e K, levando a scores mais "sharps" (focados).
        # Isso parece alinhar com a intenção do Kohya de "aumentar o foco" para compensar sub-variância.

        # Opção 2 (Alternativa): Se você quisesse que tau_for_computation > 1 aumentasse o foco
        # (ou seja, tau_for_computation fosse um multiplicador direto da "força" da atenção)
        # query_modified = query * torch.sqrt(tau_for_computation)
        # key_modified = key * torch.sqrt(tau_for_computation)

        # O print de debug que você tinha é ótimo para verificar o valor de log_tau
        # print("τ-hit, log_tau:", self.log_tau.item(), "tau_for_comp:", tau_for_computation.item(), "query_dtype:", query.dtype, "tau_dtype:", tau_for_computation.dtype)

        # 9) Chama SDPA **sem** o argumento `scale` explícito, para usar a escala padrão interna do SDPA
        #    que será aplicada aos query_modified e key_modified.
        out = F.scaled_dot_product_attention(
            query_modified, # Usando o query modificado
            key_modified,   # Usando o key modificado
            value,
            attn_mask=processed_attention_mask,
            dropout_p=getattr(attn, "dropout", 0.0), # Pega o dropout do módulo Attention original
            is_causal=getattr(attn, "is_causal", False) # Pega is_causal do módulo Attention original
        )

        # 10) Rearranja de volta e aplica saída linear + dropout
        out = out.transpose(1,2).reshape(batch, -1, attn.heads * head_dim)
        out = attn.to_out[0](out)
        if len(attn.to_out) > 1 and not isinstance(attn.to_out[1], nn.Identity):
            out = attn.to_out[1](out)

        # 11) Reconstrói 4D se necessário + residual + rescale
        if input_ndim == 4:
            out = out.transpose(-1,-2).reshape(batch, channel, height, width)
        if attn.residual_connection:
            out = out + residual
        out = out / attn.rescale_output_factor
        return out

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

    def __apply_learnable_tau_processors(self, unet: UNet2DConditionModel):
        if not hasattr(unet, 'attn_processors'):
            print("Aviso: UNet não possui 'attn_processors'. Tau customizado não aplicado.")
            return

        final_processors_for_unet_usage = {}
        tau_procs = {}
        count_applied_tau_procs = 0
        
        print("--- Analisando e substituindo processadores de atenção originais na UNet ---")
        for name, proc_original_instance in unet.attn_processors.items():
            init_tau_value = 1.0 # Ou pegue de uma config se quiser taus iniciais diferentes
            
            if isinstance(proc_original_instance, AP2): # OriginalAttnProcessor2_0 é o AP2
                custom_processor_instance = LearnableTauAttnProcessor(init_tau=init_tau_value) # Sua classe herdada
                final_processors_for_unet_usage[name] = custom_processor_instance
                
                module_dict_key = name.replace('.', '_').replace('-', '_') # Para nome de atributo válido
                tau_procs[module_dict_key] = custom_processor_instance
                count_applied_tau_procs += 1
                # print(f"    -> Substituindo {name} por TauAttnProcessor.")
            else:
                # Mantém o processador original se não for AttnProcessor2_0
                final_processors_for_unet_usage[name] = proc_original_instance
                # print(f"    -> Mantendo {name} como {type(proc_original_instance).__name__}.")

        if count_applied_tau_procs > 0:
            unet.set_attn_processor(final_processors_for_unet_usage)
            unet.tau_procs = nn.ModuleDict(tau_procs)
            print(f"UNet atualizada com {count_applied_tau_procs} instâncias de TauAttnProcessor.")
            
        #     # Debug para verificar parâmetros (opcional, mas útil uma vez)
        #     print("--- Parâmetros Tau registrados na UNet ---")
        #     for param_name, param in unet.tau_procs.named_parameters():
        #         if "log_tau" in param_name:
        #             print(f"  Encontrado: {param_name}, Requer Grad: {param.requires_grad}, Device: {param.device}, Dtype: {param.dtype}")
        # else:
        #     print("Nenhum processador AttnProcessor2_0 encontrado para substituir por TauAttnProcessor.")


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
        
        self.__apply_learnable_tau_processors(unet, weight_dtypes)

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
        unet.to(dtype=weight_dtypes.unet.torch_dtype())

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