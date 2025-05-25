from contextlib import nullcontext
from random import Random
from typing import List, Union

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.model.util.clip_util import encode_clip
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.convert.rescale_noise_scheduler_to_zero_terminal_snr import (
    rescale_noise_scheduler_to_zero_terminal_snr,
)
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType

import torch
import torch.nn.functional as F
from torch import Tensor

from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from modules.sangoi.TrainGPS import TrainGPS

class CacheModule:
    def __init__(self, strategy="full_embeddings", threshold=0.7):
        self.strategy = strategy
        self.threshold = threshold
        self.cache = {}
    
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        self.cache[key] = value

class StableDiffusionXLModelEmbedding:
    def __init__(
            self,
            uuid: str,
            text_encoder_1_vector: Tensor | None,
            text_encoder_2_vector: Tensor | None,
            placeholder: str,
            is_output_embedding: bool,
    ):
        self.text_encoder_1_embedding = BaseModelEmbedding(
            uuid=uuid,
            placeholder=placeholder,
            vector=text_encoder_1_vector,
            is_output_embedding=is_output_embedding,
        )

        self.text_encoder_2_embedding = BaseModelEmbedding(
            uuid=uuid,
            placeholder=placeholder,
            vector=text_encoder_2_vector,
            is_output_embedding=is_output_embedding,
        )


class StableDiffusionXLModel(BaseModel):
    # base model data
    tokenizer_1: CLIPTokenizer | None
    tokenizer_2: CLIPTokenizer | None
    noise_scheduler: DDIMScheduler | None
    text_encoder_1: CLIPTextModel | None
    text_encoder_2: CLIPTextModelWithProjection | None
    vae: AutoencoderKL | None
    unet: UNet2DConditionModel | None

    # autocast context
    autocast_context: torch.autocast | nullcontext
    vae_autocast_context: torch.autocast | nullcontext

    train_dtype: DataType
    vae_train_dtype: DataType

    # persistent embedding training data
    embedding: StableDiffusionXLModelEmbedding | None
    additional_embeddings: list[StableDiffusionXLModelEmbedding] | None
    embedding_wrapper_1: AdditionalEmbeddingWrapper | None
    embedding_wrapper_2: AdditionalEmbeddingWrapper | None

    # persistent lora training data
    text_encoder_1_lora: LoRAModuleWrapper | None
    text_encoder_2_lora: LoRAModuleWrapper | None
    unet_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None
    deltas: TrainGPS | None

    sd_config: dict | None
    sd_config_filename: str | None

    def __init__(
            self,
            model_type: ModelType,
    ):
        super().__init__(model_type=model_type)

        self.tokenizer_1 = None
        self.tokenizer_2 = None
        self.noise_scheduler = None
        self.text_encoder_1 = None
        self.text_encoder_2 = None
        self.vae = None
        self.unet = None

        self.autocast_context = nullcontext()
        self.vae_autocast_context = nullcontext()

        self.train_dtype = DataType.FLOAT_32
        self.vae_train_dtype = DataType.FLOAT_32

        self.embedding = None
        self.additional_embeddings = []
        self.embedding_wrapper_1 = None
        self.embedding_wrapper_2 = None

        self.text_encoder_1_lora = None
        self.text_encoder_2_lora = None
        self.unet_lora = None
        self.lora_state_dict = None
        self.deltas = None

        self.sd_config = None
        self.sd_config_filename = None

        # NOVA: Cache para long prompts - claudio
        self._cached_long_prompts_flag = None
        self._sched_tensors_cached = False

    # Cache a verificação de long prompts
    @property
    def _use_long_prompts(self) -> bool:
        return (hasattr(self, 'enable_long_prompts') and 
                self._cached_long_prompts_flag) or \
              (self.train_config and self.train_config.enable_long_prompts)

    def _process_long_prompts_batch(self, text_list, train_device, batch_size):
        """Processa múltiplos long prompts em batch"""
        all_prompt_outputs_1: List[Tensor] = []
        all_prompt_outputs_2: List[Tensor] = []
        all_pooled_outputs_2: List[Tensor] = []

        max_len_1_tokenizer = self.tokenizer_1.model_max_length
        max_len_2_tokenizer = self.tokenizer_2.model_max_length
        max_chunks = self.train_config.long_prompt_max_chunks

        max_seq_len_1_actual = 0
        max_seq_len_2_actual = 0

        # // Sessão DBG - Depuração de chunking
        if self.train_config.debugoi:
            print(f"// Sessão DBG - encode_text (long_prompts): Processando {len(text_list)} prompts.")
        # // Sessão DBG - Fim da depuração de chunking

        for idx, single_prompt_text in enumerate(text_list):
            # Processa cada prompt individual usando a função otimizada
            prompt_output_1, prompt_output_2, pooled_output = self._process_single_long_prompt(
                single_prompt_text, train_device, 1  # batch_size 1 para cada prompt individual
            )
            
            all_prompt_outputs_1.append(prompt_output_1)
            all_prompt_outputs_2.append(prompt_output_2)
            all_pooled_outputs_2.append(pooled_output)
            
            max_seq_len_1_actual = max(max_seq_len_1_actual, prompt_output_1.shape[1])
            max_seq_len_2_actual = max(max_seq_len_2_actual, prompt_output_2.shape[1])

        # Padding para alinhar todos os comprimentos
        final_text_encoder_1_output_list = []
        for po1 in all_prompt_outputs_1:
            padding_needed = max_seq_len_1_actual - po1.shape[1]
            if padding_needed > 0:
                padded_po1 = torch.nn.functional.pad(po1, (0, 0, 0, padding_needed), mode='constant', value=0)
                final_text_encoder_1_output_list.append(padded_po1)
            else:
                final_text_encoder_1_output_list.append(po1)
        
        final_text_encoder_2_output_list = []
        for po2 in all_prompt_outputs_2:
            padding_needed = max_seq_len_2_actual - po2.shape[1]
            if padding_needed > 0:
                padded_po2 = torch.nn.functional.pad(po2, (0, 0, 0, padding_needed), mode='constant', value=0)
                final_text_encoder_2_output_list.append(padded_po2)
            else:
                final_text_encoder_2_output_list.append(po2)

        # Concatena todos os resultados
        if final_text_encoder_1_output_list:
            final_text_encoder_1_output = torch.cat(final_text_encoder_1_output_list, dim=0)
        else:
            hs1 = self.text_encoder_1.config.hidden_size
            final_text_encoder_1_output = torch.zeros((len(text_list), 0, hs1), 
                                                    device=self.text_encoder_1.device, 
                                                    dtype=self.text_encoder_1.dtype)

        if final_text_encoder_2_output_list:
            final_text_encoder_2_output = torch.cat(final_text_encoder_2_output_list, dim=0)
        else:
            hs2 = self.text_encoder_2.config.hidden_size
            final_text_encoder_2_output = torch.zeros((len(text_list), 0, hs2), 
                                                    device=self.text_encoder_2.device, 
                                                    dtype=self.text_encoder_2.dtype)
        
        if all_pooled_outputs_2:
            final_pooled_text_encoder_2_output = torch.cat(all_pooled_outputs_2, dim=0)
        else:
            ps2 = self.text_encoder_2.config.projection_dim
            final_pooled_text_encoder_2_output = torch.zeros((len(text_list), ps2), 
                                                            device=self.text_encoder_2.device, 
                                                            dtype=self.text_encoder_2.dtype)

        return final_text_encoder_1_output, final_text_encoder_2_output, final_pooled_text_encoder_2_output


    def _cache_long_prompts_setting(self):
        self._cached_long_prompts_flag = (self.train_config and 
                                          self.train_config.enable_long_prompts)

    def _chunk_and_transfer_batch(self, tokenizer, texts_batch, device):
        """Processa batch de textos de uma vez"""
        all_chunks = []
        for text in texts_batch:
            chunks, _ = self._chunk_tokenizer(tokenizer, text, tokenizer.model_max_length)
            all_chunks.extend(chunks)
        
        # Transfer em batch - muito mais eficiente
        if all_chunks:
            stacked_chunks = torch.stack(all_chunks)
            return stacked_chunks.to(device, non_blocking=True)
        return []

    def _efficient_chunk_processing(self, chunks, text_encoder):
        """Pre-aloca tensor para evitar concatenações"""
        if not chunks:
            return torch.zeros((1, 0, text_encoder.config.hidden_size), 
                            device=text_encoder.device, dtype=text_encoder.dtype)
        
        # ✅ VERIFICAÇÃO: Garante que chunks têm dimensionalidade correta
        processed_chunks = []
        for chunk in chunks:
            if chunk.dim() == 1:
                # Se for 1D, adiciona dimensão de batch
                chunk = chunk.unsqueeze(0)
            elif chunk.dim() == 0:
                # Se for escalar, pula
                continue
            processed_chunks.append(chunk)
        
        if not processed_chunks:
            return torch.zeros((1, 0, text_encoder.config.hidden_size), 
                            device=text_encoder.device, dtype=text_encoder.dtype)
        
        # ✅ CORREÇÃO: Usar shape[1] depois da verificação
        total_seq_len = sum(max(0, chunk.shape[1] - 2) for chunk in processed_chunks)  # -2 para BOS/EOS
        batch_size = 1
        hidden_size = text_encoder.config.hidden_size
        
        # Pre-aloca o tensor final
        output_tensor = torch.zeros(
            (batch_size, total_seq_len, hidden_size),
            device=text_encoder.device,
            dtype=text_encoder.dtype
        )
        
        current_pos = 0
        for chunk in processed_chunks:
            try:
                # Mover chunk para o device correto
                chunk = chunk.to(text_encoder.device)
                
                # USA A FUNÇÃO EXISTENTE encode_clip
                chunk_output, _ = encode_clip(
                    text_encoder=text_encoder, 
                    tokens=chunk,  # chunk já tem shape [batch, seq_len]
                    default_layer=-2,
                    layer_skip=0, 
                    add_pooled_output=False,
                    use_attention_mask=False, 
                    add_layer_norm=False,
                )
                
                # Remove BOS/EOS tokens se existirem
                if chunk_output.shape[1] > 2:
                    content = chunk_output[:, 1:-1, :]  # Remove BOS/EOS
                else:
                    content = chunk_output  # Mantém tudo se muito pequeno
                
                seq_len = content.shape[1]
                if seq_len > 0 and current_pos + seq_len <= total_seq_len:
                    output_tensor[:, current_pos:current_pos + seq_len, :] = content
                    current_pos += seq_len
            except Exception as e:
                print(f"⚠️ Erro processando chunk: {e}")
                continue  # Pula chunks problemáticos
        
        return output_tensor


    def _setup_smart_cache(self, config):
        """Cache que funciona com long prompts"""
        if config.enable_long_prompts:
            # Cache apenas os chunks mais comuns
            cache_strategy = "common_chunks"
            cache_threshold = 0.7  # Cache chunks que aparecem em 70%+ dos prompts
        else:
            cache_strategy = "full_embeddings"
        
        return CacheModule(strategy=cache_strategy, threshold=cache_threshold)

    def _encode_negative_prompt_optimized(self, negative_prompt):
        """Método específico para negative prompts sem long prompts"""
        # Para negative prompts, sempre usar tokenização direta (sem long prompts)
        tokens_1 = self.model.tokenizer_1(
            negative_prompt, 
            padding="max_length", 
            max_length=self.model.tokenizer_1.model_max_length,
            truncation=True, 
            return_tensors="pt"
        ).input_ids.to(self.model.text_encoder_1.device)
        
        tokens_2 = self.model.tokenizer_2(
            negative_prompt, 
            padding="max_length",
            max_length=self.model.tokenizer_2.model_max_length,
            truncation=True, 
            return_tensors="pt"
        ).input_ids.to(self.model.text_encoder_2.device)
        
        return self.model.encode_text(
            tokens_1=tokens_1, 
            tokens_2=tokens_2,
            train_device=self.train_device, 
            batch_size=1,
            text=None  # Force non-long-prompt path
        )

    def all_embeddings(self) -> list[StableDiffusionXLModelEmbedding]:
        return self.additional_embeddings \
               + ([self.embedding] if self.embedding is not None else [])

    def all_text_encoder_1_embeddings(self) -> list[BaseModelEmbedding]:
        return [embedding.text_encoder_1_embedding for embedding in self.additional_embeddings] \
               + ([self.embedding.text_encoder_1_embedding] if self.embedding is not None else [])

    def all_text_encoder_2_embeddings(self) -> list[BaseModelEmbedding]:
        return [embedding.text_encoder_2_embedding for embedding in self.additional_embeddings] \
               + ([self.embedding.text_encoder_2_embedding] if self.embedding is not None else [])

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        self.text_encoder_1.to(device=device)
        self.text_encoder_2.to(device=device)

        if self.text_encoder_1_lora is not None:
            self.text_encoder_1_lora.to(device)

        if self.text_encoder_2_lora is not None:
            self.text_encoder_2_lora.to(device)

    def text_encoder_1_to(self, device: torch.device):
        self.text_encoder_1.to(device=device)

        if self.text_encoder_1_lora is not None:
            self.text_encoder_1_lora.to(device)

    def text_encoder_2_to(self, device: torch.device):
        self.text_encoder_2.to(device=device)

        if self.text_encoder_2_lora is not None:
            self.text_encoder_2_lora.to(device)

    def unet_to(self, device: torch.device):
        self.unet.to(device=device)

        if self.unet_lora is not None:
            self.unet_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.unet_to(device)

    def eval(self):
        self.vae.eval()
        self.text_encoder_1.eval()
        self.text_encoder_2.eval()
        self.unet.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        return StableDiffusionXLPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder_1,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer_1,
            tokenizer_2=self.tokenizer_2,
            unet=self.unet,
            scheduler=self.noise_scheduler,
        )

    def force_v_prediction(self):
        self.noise_scheduler.config.prediction_type = 'v_prediction'
        if self.sd_config and 'model' in self.sd_config and 'params' in self.sd_config['model']: # // Sessão III by Gemini - CORREÇÃO: Adicionada verificação de existência de chaves
            self.sd_config['model']['params']['parameterization'] = 'v'
        if self.model_spec: # // Sessão III by Gemini - CORREÇÃO: Adicionada verificação de existência de model_spec
            self.model_spec.prediction_type = 'v'

    def force_epsilon_prediction(self):
        self.noise_scheduler.config.prediction_type = 'epsilon'
        if self.sd_config and 'model' in self.sd_config and 'params' in self.sd_config['model']: # // Sessão III by Gemini - CORREÇÃO: Adicionada verificação de existência de chaves
            self.sd_config['model']['params']['parameterization'] = 'epsilon'
        if self.model_spec: # // Sessão III by Gemini - CORREÇÃO: Adicionada verificação de existência de model_spec
            self.model_spec.prediction_type = 'epsilon'

    def rescale_noise_scheduler_to_zero_terminal_snr(self):
        rescale_noise_scheduler_to_zero_terminal_snr(self.noise_scheduler)

    def add_text_encoder_1_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.all_text_encoder_1_embeddings(), prompt)

    def add_text_encoder_2_embeddings_to_prompt(self, prompt: str) -> str:
        return self._add_embeddings_to_prompt(self.all_text_encoder_2_embeddings(), prompt)

    def encode_text(self, train_device: torch.device, batch_size: int = 1, rand: Random | None = None,
                    text: Union[str, List[str]] = None, tokens_1: Tensor = None, tokens_2: Tensor = None,
                    text_encoder_1_layer_skip: int = 0,
                    text_encoder_2_layer_skip: int = 0,
                    text_encoder_1_output: Tensor = None, 
                    text_encoder_2_output: Tensor = None, 
                    text_encoder_1_dropout_probability: float | None = None,
                    text_encoder_2_dropout_probability: float | None = None,
                    pooled_text_encoder_2_output: Tensor = None, 
            ) -> tuple[Tensor, Tensor, Tensor]:

        # Cache a verificação de long prompts uma vez só
        if not hasattr(self, '_long_prompts_cached'):
            self._cache_long_prompts_setting()
            self._long_prompts_cached = True
        
        if not (self._use_long_prompts and text is not None):
            # === LÓGICA ORIGINAL (sem long prompts) ===
            final_text_encoder_1_output = text_encoder_1_output
            final_text_encoder_2_output = text_encoder_2_output
            final_pooled_text_encoder_2_output = pooled_text_encoder_2_output

            if tokens_1 is None and isinstance(text, str) and text:
                processed_text_1 = self.add_text_encoder_1_embeddings_to_prompt(text)
                tokenizer_output_1 = self.tokenizer_1(
                    processed_text_1, padding='max_length', truncation=True,
                    max_length=self.tokenizer_1.model_max_length, return_tensors="pt",
                )
                tokens_1 = tokenizer_output_1.input_ids.to(self.text_encoder_1.device)
                final_text_encoder_1_output = None 

            if tokens_2 is None and isinstance(text, str) and text:
                processed_text_2 = self.add_text_encoder_2_embeddings_to_prompt(text)
                tokenizer_output_2 = self.tokenizer_2(
                    processed_text_2, padding='max_length', truncation=True,
                    max_length=self.tokenizer_2.model_max_length, return_tensors="pt",
                )
                tokens_2 = tokenizer_output_2.input_ids.to(self.text_encoder_2.device)
                final_text_encoder_2_output = None
                final_pooled_text_encoder_2_output = None
            
            if tokens_1 is not None:
                final_text_encoder_1_output, _ = encode_clip(
                    text_encoder=self.text_encoder_1, tokens=tokens_1, default_layer=-2,
                    layer_skip=text_encoder_1_layer_skip, text_encoder_output=final_text_encoder_1_output,
                    add_pooled_output=False, use_attention_mask=False, add_layer_norm=False,
                )

            if tokens_2 is not None:
                final_text_encoder_2_output, final_pooled_text_encoder_2_output = encode_clip(
                    text_encoder=self.text_encoder_2, tokens=tokens_2, default_layer=-2,
                    layer_skip=text_encoder_2_layer_skip, text_encoder_output=final_text_encoder_2_output,
                    add_pooled_output=True, pooled_text_encoder_output=final_pooled_text_encoder_2_output,
                    use_attention_mask=False, add_layer_norm=False,
                )
        else:
            # === LÓGICA LONG PROMPTS OTIMIZADA ===
            text_list = [text] if isinstance(text, str) else text
            
            if not text_list: 
                current_bs = batch_size 
                hs1 = self.text_encoder_1.config.hidden_size
                hs2 = self.text_encoder_2.config.hidden_size
                ps2 = self.text_encoder_2.config.projection_dim

                final_text_encoder_1_output = torch.zeros((current_bs, 0, hs1), device=self.text_encoder_1.device, dtype=self.text_encoder_1.dtype)
                final_text_encoder_2_output = torch.zeros((current_bs, 0, hs2), device=self.text_encoder_2.device, dtype=self.text_encoder_2.dtype)
                final_pooled_text_encoder_2_output = torch.zeros((current_bs, ps2), device=self.text_encoder_2.device, dtype=self.text_encoder_2.dtype)
            else:
                # PROCESSA APENAS O PRIMEIRO PROMPT (simplificado)
                prompt_text = text_list[0]
                final_text_encoder_1_output, final_text_encoder_2_output, final_pooled_text_encoder_2_output = self._process_single_long_prompt(prompt_text, train_device, batch_size)

        # === DROPOUT (igual ao original) ===
        if final_text_encoder_1_output is not None and final_text_encoder_1_output.numel() > 0:
            current_batch_size = final_text_encoder_1_output.shape[0]
        elif final_text_encoder_2_output is not None and final_text_encoder_2_output.numel() > 0:
            current_batch_size = final_text_encoder_2_output.shape[0]
        else:
            current_batch_size = batch_size

        if text_encoder_1_dropout_probability is not None and final_text_encoder_1_output is not None and final_text_encoder_1_output.numel() > 0:
            dropout_text_encoder_1_mask = (torch.rand(current_batch_size, device=train_device) > text_encoder_1_dropout_probability).float()
            final_text_encoder_1_output = final_text_encoder_1_output * dropout_text_encoder_1_mask.view(-1, 1, 1).to(final_text_encoder_1_output.device)

        if text_encoder_2_dropout_probability is not None:
            dropout_text_encoder_2_mask = (torch.rand(current_batch_size, device=train_device) > text_encoder_2_dropout_probability).float()
            if final_pooled_text_encoder_2_output is not None and final_pooled_text_encoder_2_output.numel() > 0:
                final_pooled_text_encoder_2_output = final_pooled_text_encoder_2_output * dropout_text_encoder_2_mask.view(-1, 1).to(final_pooled_text_encoder_2_output.device)
            if final_text_encoder_2_output is not None and final_text_encoder_2_output.numel() > 0:
                final_text_encoder_2_output = final_text_encoder_2_output * dropout_text_encoder_2_mask.view(-1, 1, 1).to(final_text_encoder_2_output.device)

        return final_text_encoder_1_output, final_text_encoder_2_output, final_pooled_text_encoder_2_output

    def _process_single_long_prompt(self, prompt_text, train_device, batch_size):
        """Processa um único long prompt com otimizações"""
        try:
            processed_text_1 = self.add_text_encoder_1_embeddings_to_prompt(prompt_text or "")
            processed_text_2 = self.add_text_encoder_2_embeddings_to_prompt(prompt_text or "")

            token_chunks_1, _ = self._chunk_tokenizer(self.tokenizer_1, processed_text_1, 
                                                      self.tokenizer_1.model_max_length)
            token_chunks_2, _ = self._chunk_tokenizer(self.tokenizer_2, processed_text_2, 
                                                      self.tokenizer_2.model_max_length)

            # ✅ VERIFICAÇÃO: Limita chunks se configurado
            max_chunks = getattr(self.train_config, 'long_prompt_max_chunks', 10)
            token_chunks_1 = token_chunks_1[:max_chunks] if token_chunks_1 else []
            token_chunks_2 = token_chunks_2[:max_chunks] if token_chunks_2 else []

            # ✅ VERIFICAÇÃO: Se não há chunks, retorna tensors vazios
            if not token_chunks_1 and not token_chunks_2:
                hs1 = self.text_encoder_1.config.hidden_size
                hs2 = self.text_encoder_2.config.hidden_size
                ps2 = self.text_encoder_2.config.projection_dim
                
                prompt_output_1 = torch.zeros((1, 0, hs1), device=self.text_encoder_1.device, dtype=self.text_encoder_1.dtype)
                prompt_output_2 = torch.zeros((1, 0, hs2), device=self.text_encoder_2.device, dtype=self.text_encoder_2.dtype)
                pooled_output = torch.zeros((1, ps2), device=self.text_encoder_2.device, dtype=self.text_encoder_2.dtype)
                
                return prompt_output_1, prompt_output_2, pooled_output

            # OTIMIZAÇÃO: Usa a função eficiente para cada encoder
            prompt_output_1 = self._efficient_chunk_processing(token_chunks_1, self.text_encoder_1)
            prompt_output_2 = self._efficient_chunk_processing(token_chunks_2, self.text_encoder_2)
            
            # Pooled output do primeiro chunk do encoder 2
            if token_chunks_2:
                try:
                    first_chunk = token_chunks_2[0].to(self.text_encoder_2.device)
                    _, pooled_output = encode_clip(
                        text_encoder=self.text_encoder_2, tokens=first_chunk, default_layer=-2,
                        add_pooled_output=True, use_attention_mask=False, add_layer_norm=False,
                    )
                except Exception as e:
                    print(f"⚠️ Erro no pooled output: {e}")
                    ps2 = self.text_encoder_2.config.projection_dim
                    pooled_output = torch.zeros((1, ps2), device=self.text_encoder_2.device, 
                                              dtype=self.text_encoder_2.dtype)
            else:
                ps2 = self.text_encoder_2.config.projection_dim
                pooled_output = torch.zeros((1, ps2), device=self.text_encoder_2.device, 
                                          dtype=self.text_encoder_2.dtype)

            return prompt_output_1, prompt_output_2, pooled_output
            
        except Exception as e:
            print(f"🚨 ERRO em _process_single_long_prompt: {e}")
            # Fallback para tensors vazios
            hs1 = self.text_encoder_1.config.hidden_size
            hs2 = self.text_encoder_2.config.hidden_size
            ps2 = self.text_encoder_2.config.projection_dim
            
            return (
                torch.zeros((1, 0, hs1), device=self.text_encoder_1.device, dtype=self.text_encoder_1.dtype),
                torch.zeros((1, 0, hs2), device=self.text_encoder_2.device, dtype=self.text_encoder_2.dtype),
                torch.zeros((1, ps2), device=self.text_encoder_2.device, dtype=self.text_encoder_2.dtype)
            )


    def combine_text_encoder_output(self, te1_out, te2_out, pooled_out):
        if te1_out is None or te2_out is None:
            # Criar tensors de fallback com dimensões consistentes
            batch_size = pooled_out.shape[0] if pooled_out is not None else 1
            device = pooled_out.device if pooled_out is not None else self.train_device
            
            if te1_out is None:
                te1_out = torch.zeros(
                    (batch_size, 0, self.text_encoder_1.config.hidden_size),
                    device=device, dtype=self.train_dtype.torch_dtype()
                )
            
            if te2_out is None:
                te2_out = torch.zeros(
                    (batch_size, 0, self.text_encoder_2.config.hidden_size),
                    device=device, dtype=self.train_dtype.torch_dtype()
                )
        
        # Alinhar comprimentos de sequência
        max_seq_len = max(te1_out.shape[1], te2_out.shape[1])
        
        if te1_out.shape[1] < max_seq_len:
            padding = max_seq_len - te1_out.shape[1]
            te1_out = F.pad(te1_out, (0, 0, 0, padding))
        
        if te2_out.shape[1] < max_seq_len:
            padding = max_seq_len - te2_out.shape[1]
            te2_out = F.pad(te2_out, (0, 0, 0, padding))
        
        # Concatenação simples
        combined = torch.cat([te1_out, te2_out], dim=-1)
        return combined, pooled_out

    
    def _chunk_tokenizer(self, tokenizer, text: str, max_length: int):
        if not text or not text.strip():
            return [], []
        
        # Garantir que sempre temos espaço para BOS/EOS
        content_max_length = max(max_length - 2, 1)
        
        # Tokenizar sem special tokens primeiro
        all_input_ids = tokenizer(text, add_special_tokens=False).input_ids
        
        if not all_input_ids:
            return [], []
        
        chunks = []
        for i in range(0, len(all_input_ids), content_max_length):
            chunk_ids = all_input_ids[i:i + content_max_length]
            
            # Adicionar BOS/EOS e padding de forma consistente
            input_ids = [tokenizer.bos_token_id] + chunk_ids + [tokenizer.eos_token_id]
            
            # Padding até max_length
            while len(input_ids) < max_length:
                input_ids.append(tokenizer.pad_token_id or tokenizer.eos_token_id)
            
            # ✅ MUDANÇA: Criar tensor 2D com unsqueeze(0) para [1, seq_len]
            chunk_tensor = torch.tensor(input_ids[:max_length], dtype=torch.long).unsqueeze(0)
            chunks.append(chunk_tensor)
        
        return chunks, []
