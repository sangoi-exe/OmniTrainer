from contextlib import nullcontext
from random import Random
from typing import List, Union # // Sessão II by Gemini - CORREÇÃO: Adicionada importação Union

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
from torch import Tensor

from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from modules.sangoi.TrainGPS import TrainGPS


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
        super().__init__(
            model_type=model_type,
        )

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
        self.embedding_wrapper_2 = None # // Sessão III by Gemini - CORREÇÃO: Corrigido para embedding_wrapper_2

        self.text_encoder_1_lora = None
        self.text_encoder_2_lora = None
        self.unet_lora = None
        self.lora_state_dict = None
        self.deltas = None

        self.sd_config = None
        self.sd_config_filename = None

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

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1, # Este é o batch_size do dataloader
            rand: Random | None = None,
            text: Union[str, List[str]] = None, # // Sessão II by Gemini - CORREÇÃO: Permitir List[str] para text
            tokens_1: Tensor = None, 
            tokens_2: Tensor = None, 
            text_encoder_1_layer_skip: int = 0,
            text_encoder_2_layer_skip: int = 0,
            text_encoder_1_output: Tensor = None, 
            text_encoder_2_output: Tensor = None, 
            text_encoder_1_dropout_probability: float | None = None,
            text_encoder_2_dropout_probability: float | None = None,
            pooled_text_encoder_2_output: Tensor = None, 
    ) -> tuple[Tensor, Tensor, Tensor]:
        # // Sessão II by Gemini - CORREÇÃO: Refatoração completa da lógica de encode_text para lidar com batches de long prompts
        if not (self.train_config and self.train_config.enable_long_prompts and text is not None):
            # --- Original Logic Start (sem long prompts ou sem texto fornecido para long prompts) ---
            final_text_encoder_1_output = text_encoder_1_output # // Sessão III by Gemini - CORREÇÃO: Inicializar com valor de entrada ou None
            final_text_encoder_2_output = text_encoder_2_output # // Sessão III by Gemini - CORREÇÃO: Inicializar com valor de entrada ou None
            final_pooled_text_encoder_2_output = pooled_text_encoder_2_output # // Sessão III by Gemini - CORREÇÃO: Inicializar com valor de entrada ou None

            if tokens_1 is None and isinstance(text, str) and text:
                processed_text_1 = self.add_text_encoder_1_embeddings_to_prompt(text)
                tokenizer_output_1 = self.tokenizer_1(
                    processed_text_1, padding='max_length', truncation=True,
                    max_length=self.tokenizer_1.model_max_length, return_tensors="pt",
                )
                tokens_1 = tokenizer_output_1.input_ids.to(self.text_encoder_1.device)
                # // Sessão III by Gemini - CORREÇÃO: Se tokens_1 foi gerado, text_encoder_1_output também deve ser gerado, não usado da entrada
                final_text_encoder_1_output = None 

            if tokens_2 is None and isinstance(text, str) and text:
                processed_text_2 = self.add_text_encoder_2_embeddings_to_prompt(text)
                tokenizer_output_2 = self.tokenizer_2(
                    processed_text_2, padding='max_length', truncation=True,
                    max_length=self.tokenizer_2.model_max_length, return_tensors="pt",
                )
                tokens_2 = tokenizer_output_2.input_ids.to(self.text_encoder_2.device)
                # // Sessão III by Gemini - CORREÇÃO: Se tokens_2 foi gerado, text_encoder_2_output e pooled também devem ser gerados
                final_text_encoder_2_output = None
                final_pooled_text_encoder_2_output = None
            
            # // Sessão III by Gemini - CORREÇÃO: Assegurar que, se os tokens foram fornecidos ou gerados, os encoders são chamados.
            # Se text_encoder_X_output foi fornecido E os tokens correspondentes NÃO foram gerados/fornecidos,
            # então usamos os text_encoder_X_output diretamente (já encodados).
            # Caso contrário, se os tokens existem, encodamos.

            if tokens_1 is not None: # Se temos tokens_1 (fornecidos ou gerados)
                final_text_encoder_1_output, _ = encode_clip(
                    text_encoder=self.text_encoder_1, tokens=tokens_1, default_layer=-2,
                    layer_skip=text_encoder_1_layer_skip, text_encoder_output=final_text_encoder_1_output, # Passar o output (pode ser None)
                    add_pooled_output=False, use_attention_mask=False, add_layer_norm=False,
                )
            # Se tokens_1 é None E final_text_encoder_1_output (da entrada) não é None, usamos o da entrada.
            # Se ambos são None, final_text_encoder_1_output permanece None (ou tensor vazio se apropriado).

            if tokens_2 is not None: # Se temos tokens_2 (fornecidos ou gerados)
                final_text_encoder_2_output, final_pooled_text_encoder_2_output = encode_clip(
                    text_encoder=self.text_encoder_2, tokens=tokens_2, default_layer=-2,
                    layer_skip=text_encoder_2_layer_skip, text_encoder_output=final_text_encoder_2_output, # Passar o output (pode ser None)
                    add_pooled_output=True, pooled_text_encoder_output=final_pooled_text_encoder_2_output, # Passar o pooled (pode ser None)
                    use_attention_mask=False, add_layer_norm=False,
                )
            # Se tokens_2 é None E final_text_encoder_2_output (da entrada) não é None, usamos o da entrada.
            # Se ambos são None, final_text_encoder_2_output e final_pooled_text_encoder_2_output permanecem None.

            # --- Original Logic End ---
        else:
            # --- Long Prompt Logic Start (Refatorado para Batch) ---
            text_list = [text] if isinstance(text, str) else text
            
            # // Sessão III by Gemini - CORREÇÃO: Lidar com text_list sendo None ou vazio
            if not text_list: 
                # Se text_list é None ou vazio, usar o batch_size do dataloader para criar tensores vazios.
                # Isso garante que o resto do pipeline não quebre se não houver prompts.
                current_bs = batch_size 
                hs1 = self.text_encoder_1.config.hidden_size
                hs2 = self.text_encoder_2.config.hidden_size
                ps2 = self.text_encoder_2.config.projection_dim

                final_text_encoder_1_output = torch.zeros((current_bs, 0, hs1), device=self.text_encoder_1.device, dtype=self.text_encoder_1.dtype)
                final_text_encoder_2_output = torch.zeros((current_bs, 0, hs2), device=self.text_encoder_2.device, dtype=self.text_encoder_2.dtype)
                final_pooled_text_encoder_2_output = torch.zeros((current_bs, ps2), device=self.text_encoder_2.device, dtype=self.text_encoder_2.dtype)
            else:
                all_prompt_outputs_1: List[Tensor] = []
                all_prompt_outputs_2: List[Tensor] = []
                all_pooled_outputs_2: List[Tensor] = []

                max_len_1_tokenizer = self.tokenizer_1.model_max_length
                max_len_2_tokenizer = self.tokenizer_2.model_max_length
                max_chunks = self.train_config.long_prompt_max_chunks

                max_seq_len_1_actual = 0
                max_seq_len_2_actual = 0

                for single_prompt_text in text_list:
                    # Se single_prompt_text for None ou string vazia, _chunk_tokenizer deve lidar com isso
                    # retornando chunks vazios, e a lógica subsequente criará tensores (1, 0, H).
                    processed_text_1 = self.add_text_encoder_1_embeddings_to_prompt(single_prompt_text or "") # // Sessão III by Gemini - CORREÇÃO: Passar "" se None
                    processed_text_2 = self.add_text_encoder_2_embeddings_to_prompt(single_prompt_text or "") # // Sessão III by Gemini - CORREÇÃO: Passar "" se None

                    token_chunks_1, _ = self._chunk_tokenizer(self.tokenizer_1, processed_text_1, max_len_1_tokenizer)
                    token_chunks_2, _ = self._chunk_tokenizer(self.tokenizer_2, processed_text_2, max_len_2_tokenizer)

                    token_chunks_1 = token_chunks_1[:max_chunks]
                    token_chunks_2 = token_chunks_2[:max_chunks]
                    
                    num_chunks_1 = len(token_chunks_1)
                    num_chunks_2 = len(token_chunks_2)

                    chunk_embeddings_1: List[Tensor] = []
                    current_pooled_output_2_for_prompt: Tensor = None 

                    for i in range(num_chunks_1):
                        tokens_1_chunk = token_chunks_1[i].unsqueeze(0).to(self.text_encoder_1.device)
                        chunk_output_1, _ = encode_clip(
                            text_encoder=self.text_encoder_1, tokens=tokens_1_chunk, default_layer=-2,
                            layer_skip=text_encoder_1_layer_skip, add_pooled_output=False,
                            use_attention_mask=False, add_layer_norm=False,
                        )
                        chunk_output_1 = self._apply_output_embeddings(
                            self.all_text_encoder_1_embeddings(), self.tokenizer_1,
                            tokens_1_chunk, chunk_output_1,
                        )
                        chunk_embeddings_1.append(chunk_output_1[:, 1:-1, :]) 

                    if chunk_embeddings_1:
                        prompt_output_1 = torch.cat(chunk_embeddings_1, dim=1)
                    else:
                        hs1 = self.text_encoder_1.config.hidden_size
                        prompt_output_1 = torch.zeros((1, 0, hs1), device=self.text_encoder_1.device, dtype=self.text_encoder_1.dtype)
                    
                    all_prompt_outputs_1.append(prompt_output_1)
                    max_seq_len_1_actual = max(max_seq_len_1_actual, prompt_output_1.shape[1])

                    chunk_embeddings_2: List[Tensor] = []
                    for i in range(num_chunks_2):
                        tokens_2_chunk = token_chunks_2[i].unsqueeze(0).to(self.text_encoder_2.device)
                        chunk_output_2, pooled_out_2_chunk = encode_clip(
                            text_encoder=self.text_encoder_2, tokens=tokens_2_chunk, default_layer=-2,
                            layer_skip=text_encoder_2_layer_skip, add_pooled_output=True,
                            use_attention_mask=False, add_layer_norm=False,
                        )
                        chunk_output_2 = self._apply_output_embeddings(
                            self.all_text_encoder_2_embeddings(), self.tokenizer_2,
                            tokens_2_chunk, chunk_output_2,
                        )
                        chunk_embeddings_2.append(chunk_output_2[:, 1:-1, :]) 
                        if i == 0: 
                            current_pooled_output_2_for_prompt = pooled_out_2_chunk
                    
                    if chunk_embeddings_2:
                        prompt_output_2 = torch.cat(chunk_embeddings_2, dim=1)
                    else:
                        hs2 = self.text_encoder_2.config.hidden_size
                        prompt_output_2 = torch.zeros((1, 0, hs2), device=self.text_encoder_2.device, dtype=self.text_encoder_2.dtype)

                    if current_pooled_output_2_for_prompt is None: 
                        ps2 = self.text_encoder_2.config.projection_dim
                        current_pooled_output_2_for_prompt = torch.zeros((1, ps2), device=self.text_encoder_2.device, dtype=self.text_encoder_2.dtype)

                    all_prompt_outputs_2.append(prompt_output_2)
                    all_pooled_outputs_2.append(current_pooled_output_2_for_prompt)
                    max_seq_len_2_actual = max(max_seq_len_2_actual, prompt_output_2.shape[1])

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

                if final_text_encoder_1_output_list: # Verifica se a lista não está vazia
                    final_text_encoder_1_output = torch.cat(final_text_encoder_1_output_list, dim=0)
                else: # Se text_list era uma lista de N elementos, mas todos resultaram em prompts vazios
                    hs1 = self.text_encoder_1.config.hidden_size
                    final_text_encoder_1_output = torch.zeros((len(text_list), 0, hs1), device=self.text_encoder_1.device, dtype=self.text_encoder_1.dtype)

                if final_text_encoder_2_output_list:
                    final_text_encoder_2_output = torch.cat(final_text_encoder_2_output_list, dim=0)
                else:
                    hs2 = self.text_encoder_2.config.hidden_size
                    final_text_encoder_2_output = torch.zeros((len(text_list), 0, hs2), device=self.text_encoder_2.device, dtype=self.text_encoder_2.dtype)
                
                if all_pooled_outputs_2:
                    final_pooled_text_encoder_2_output = torch.cat(all_pooled_outputs_2, dim=0)
                else:
                    ps2 = self.text_encoder_2.config.projection_dim
                    final_pooled_text_encoder_2_output = torch.zeros((len(text_list), ps2), device=self.text_encoder_2.device, dtype=self.text_encoder_2.dtype)
            # --- End Long Prompt Logic (Refactored) ---

        # --- Common Logic (Dropout) ---
        # // Sessão II by Gemini - CORREÇÃO: O batch_size para dropout agora é o batch_size real dos tensores finais.
        # // Sessão III by Gemini - CORREÇÃO: Garantir que current_batch_size seja derivado corretamente.
        # Se final_text_encoder_1_output existir e tiver elementos, usar seu batch_size.
        # Senão, se final_text_encoder_2_output existir e tiver elementos, usar o dele.
        # Senão, se final_pooled_text_encoder_2_output existir e tiver elementos, usar o dele.
        # Como fallback, usar o batch_size original do dataloader.
        if final_text_encoder_1_output is not None and final_text_encoder_1_output.numel() > 0:
            current_batch_size = final_text_encoder_1_output.shape[0]
        elif final_text_encoder_2_output is not None and final_text_encoder_2_output.numel() > 0:
            current_batch_size = final_text_encoder_2_output.shape[0]
        elif final_pooled_text_encoder_2_output is not None and final_pooled_text_encoder_2_output.numel() > 0:
            current_batch_size = final_pooled_text_encoder_2_output.shape[0]
        else: # Todos os tensores de saída são None ou vazios. Usa o batch_size do input.
            current_batch_size = batch_size
            # Se current_batch_size for 0 e o dropout for aplicado, pode dar erro.
            # A condição .numel() > 0 antes do dropout deve prevenir isso.

        if text_encoder_1_dropout_probability is not None and final_text_encoder_1_output is not None and final_text_encoder_1_output.numel() > 0:
            # // Sessão III by Gemini - CORREÇÃO: Usar torch.rand para gerar máscara
            dropout_text_encoder_1_mask = (torch.rand(current_batch_size, device=train_device) > text_encoder_1_dropout_probability).float()
            final_text_encoder_1_output = final_text_encoder_1_output * dropout_text_encoder_1_mask.view(-1, 1, 1).to(final_text_encoder_1_output.device)

        if text_encoder_2_dropout_probability is not None: # Aplicar dropout em pooled e hidden states separadamente
            # // Sessão III by Gemini - CORREÇÃO: Usar torch.rand para gerar máscara
            dropout_text_encoder_2_mask = (torch.rand(current_batch_size, device=train_device) > text_encoder_2_dropout_probability).float()
            if final_pooled_text_encoder_2_output is not None and final_pooled_text_encoder_2_output.numel() > 0:
                 final_pooled_text_encoder_2_output = final_pooled_text_encoder_2_output * dropout_text_encoder_2_mask.view(-1, 1).to(final_pooled_text_encoder_2_output.device)
            if final_text_encoder_2_output is not None and final_text_encoder_2_output.numel() > 0:
                final_text_encoder_2_output = final_text_encoder_2_output * dropout_text_encoder_2_mask.view(-1, 1, 1).to(final_text_encoder_2_output.device)

        if final_text_encoder_1_output is not None:
            final_text_encoder_1_output = final_text_encoder_1_output.to(train_device)
        if final_text_encoder_2_output is not None:
            final_text_encoder_2_output = final_text_encoder_2_output.to(train_device)
        if final_pooled_text_encoder_2_output is not None:
            final_pooled_text_encoder_2_output = final_pooled_text_encoder_2_output.to(train_device)

        return final_text_encoder_1_output, final_text_encoder_2_output, final_pooled_text_encoder_2_output

    def combine_text_encoder_output(
            self,
            text_encoder_1_output: Tensor,
            text_encoder_2_output: Tensor,
            pooled_text_encoder_2_output: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # // Sessão II by Gemini - CORREÇÃO: Simplificar e garantir consistência de device/dtype
        # Assume-se que text_encoder_1_output e text_encoder_2_output já estão padronizados
        # para o mesmo comprimento de sequência se vierem da lógica de long_prompts refatorada.

        # // Sessão III by Gemini - CORREÇÃO: Lidar com possíveis tensores None
        if pooled_text_encoder_2_output is None:
            # Se pooled_output é None, não podemos determinar target_device/dtype seguramente.
            # Isso indica um problema anterior. Retornar None ou tensores vazios?
            # Para robustez, tentaremos usar o device/dtype de um dos outros tensores se existirem.
            if text_encoder_1_output is not None:
                target_device = text_encoder_1_output.device
                target_dtype = text_encoder_1_output.dtype
            elif text_encoder_2_output is not None:
                target_device = text_encoder_2_output.device
                target_dtype = text_encoder_2_output.dtype
            else: # Todos são None, não há o que fazer.
                # // Sessão III by Gemini - CORREÇÃO: Retornar None para todos se tudo for None
                return None, None 
            # Se pooled for None, criamos um tensor zerado para ele com o batch_size dos outros, se possível
            # e uma dimensão de pooled padrão (ex: 1280 para SDXL)
            # No entanto, o chamador de combine_text_encoder_output (o método predict) espera um pooled_output.
            # O erro deve ser tratado antes. Aqui, se pooled_text_encoder_2_output é None, retornamos ele como está.
            # O código abaixo irá falhar se pooled_text_encoder_2_output for None.
            # A lógica em encode_text deve garantir que pooled_text_encoder_2_output nunca seja None.
            # Mesmo que seja um tensor de zeros (Batch, PoolDim).
            # A asserção abaixo ajuda a pegar isso se acontecer.
            assert pooled_text_encoder_2_output is not None, "pooled_text_encoder_2_output não pode ser None em combine_text_encoder_output"


        target_device = pooled_text_encoder_2_output.device
        target_dtype = pooled_text_encoder_2_output.dtype 

        # // Sessão III by Gemini - CORREÇÃO: Garantir que tensores não sejam None antes de .to()
        te1_output_c = text_encoder_1_output.to(device=target_device, dtype=target_dtype) if text_encoder_1_output is not None else None
        te2_output_c = text_encoder_2_output.to(device=target_device, dtype=target_dtype) if text_encoder_2_output is not None else None
        pooled_output_c = pooled_text_encoder_2_output # Já está no device/dtype correto ou é None

        # // Sessão III by Gemini - CORREÇÃO: Lidar com te1_output_c ou te2_output_c sendo None
        if te1_output_c is None or te2_output_c is None:
            # Se um dos hidden_states for None, a concatenação não é possível.
            # Isso pode acontecer se um encoder não foi treinado/usado e retornou None.
            # Dependendo da arquitetura do UNet, pode ser necessário um tensor de zeros.
            # Por agora, se um for None, retornamos o outro (ou None se ambos forem None)
            # e o pooled_output. Isso pode causar problemas no UNet.
            # Uma abordagem mais robusta seria preencher com zeros da dimensão do outro.
            print(f"[WARN] Um dos text encoder outputs é None em combine_text_encoder_output. TE1: {te1_output_c is not None}, TE2: {te2_output_c is not None}")
            if te1_output_c is not None and te2_output_c is None:
                # Se apenas te2 é None, talvez o unet só precise de te1. Ou preencher te2 com zeros.
                # Para SDXL, ambos são geralmente necessários.
                # Criar um tensor de zeros para te2_output_c com as dimensões de te1_output_c (exceto a última)
                # e a dimensão do hidden_state do encoder2
                hs2 = self.text_encoder_2.config.hidden_size 
                te2_output_c = torch.zeros((te1_output_c.shape[0], te1_output_c.shape[1], hs2), device=target_device, dtype=target_dtype)

            elif te2_output_c is not None and te1_output_c is None:
                hs1 = self.text_encoder_1.config.hidden_size
                te1_output_c = torch.zeros((te2_output_c.shape[0], te2_output_c.shape[1], hs1), device=target_device, dtype=target_dtype)
            
            elif te1_output_c is None and te2_output_c is None:
                 # Se ambos são None, mas pooled_output_c existe, o UNet pode só precisar do pooled.
                 # Mas para SDXL, hidden states são normalmente concatenados.
                 # Retornar um tensor (Batch, 0, CombinedHidden)
                 # No entanto, o predict() espera um text_encoder_output que não seja None.
                 # Isso indica um problema mais fundamental se ambos forem None.
                 # A lógica em encode_text deve gerar tensores de (Batch, 0, Hidden) se não houver tokens/chunks.
                 # Assumindo que encode_text não retorna None para os hidden states.
                 # Se eles são (Batch, 0, Hidden), a concatenação resultará em (Batch, 0, CombinedHidden).
                 pass # A lógica abaixo tratará shape[0]==0 ou shape[1]==0

        # Prossiga com a lógica de concatenação, agora que te1_output_c e te2_output_c são tensores (podem ser de seq_len 0)
        if (te1_output_c is None or te1_output_c.shape[0] == 0 or te1_output_c.shape[1] == 0) and \
           (te2_output_c is None or te2_output_c.shape[0] == 0 or te2_output_c.shape[1] == 0):
            # Ambos estão vazios (ou um é None e o outro vazio)
            # Determinar o batch_size a partir do pooled_output_c se possível, senão 0.
            # Determinar combined_hidden_size
            current_bs = pooled_output_c.shape[0] if pooled_output_c is not None else 0
            hs1 = self.text_encoder_1.config.hidden_size if te1_output_c is not None and te1_output_c.shape[-1]>0 else (self.text_encoder_1.config.hidden_size if hasattr(self.text_encoder_1, 'config') else 0)
            hs2 = self.text_encoder_2.config.hidden_size if te2_output_c is not None and te2_output_c.shape[-1]>0 else (self.text_encoder_2.config.hidden_size if hasattr(self.text_encoder_2, 'config') else 0)
            combined_hidden_size = hs1 + hs2
            if combined_hidden_size == 0 and (hs1 > 0 or hs2 > 0) : # Evitar combined_hidden_size=0 se um dos encoders tiver hidden_size
                combined_hidden_size = max(hs1, hs2) # Não ideal, mas melhor que 0 se um for 0
            
            text_encoder_output = torch.zeros((current_bs, 0, combined_hidden_size), device=target_device, dtype=target_dtype)

        elif te1_output_c is not None and te2_output_c is not None and te1_output_c.shape[1] != te2_output_c.shape[1]:
            print(f"[WARN] Mismatched sequence lengths in combine_text_encoder_output: TE1={te1_output_c.shape[1]}, TE2={te2_output_c.shape[1]}. Concatenating with shortest length.")
            min_seq_len = min(te1_output_c.shape[1], te2_output_c.shape[1])
            # Se min_seq_len for 0, um dos tensores tem seq_len 0.
            if min_seq_len == 0 : # Um ou ambos têm seq_len 0. Concatenar resultará em (Batch, 0, CombinedHidden)
                 combined_hidden_size = te1_output_c.shape[-1] + te2_output_c.shape[-1]
                 text_encoder_output = torch.zeros((te1_output_c.shape[0], 0, combined_hidden_size), device=target_device, dtype=target_dtype)
            else: 
                 text_encoder_output = torch.cat(
                     [te1_output_c[:, :min_seq_len, :],
                      te2_output_c[:, :min_seq_len, :]],
                     dim=-1
                 )
        elif te1_output_c is not None and te2_output_c is not None: # Comprimentos de sequência iguais e não nulos (ou ambos zero)
            text_encoder_output = torch.cat([te1_output_c, te2_output_c], dim=-1)
        else:
            # // Sessão III by Gemini - CORREÇÃO: Caso de fallback se um for None e o outro não (após tentativa de preenchimento com zeros)
            # Isso não deveria ser alcançado se a lógica de preenchimento acima funcionar.
            # Mas como segurança:
            print(f"[ERROR] Inconsistência nos text encoder outputs para concatenação.")
            current_bs = pooled_output_c.shape[0] if pooled_output_c is not None else 0
            hs1 = self.text_encoder_1.config.hidden_size if hasattr(self.text_encoder_1, 'config') else 768 # Default
            hs2 = self.text_encoder_2.config.hidden_size if hasattr(self.text_encoder_2, 'config') else 1280 # Default
            combined_hidden_size = hs1 + hs2
            text_encoder_output = torch.zeros((current_bs, 0, combined_hidden_size), device=target_device, dtype=target_dtype)


        return text_encoder_output, pooled_output_c
    
    def _chunk_tokenizer(self, tokenizer, text: str, max_length: int) -> tuple[List[Tensor], List[Tensor]]: # // Sessão III by Gemini - CORREÇÃO: Adicionada anotação de tipo para text
        """Helper to tokenize and chunk text."""
        # // Sessão III by Gemini - CORREÇÃO: Lidar com text sendo None ou vazio no início
        if text is None or not text.strip():
            return [], []

        all_input_ids = tokenizer(text, add_special_tokens=False).input_ids

        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        # // Sessão III by Gemini - CORREÇÃO: Garantir que bos e eos são inteiros
        if not isinstance(bos, int) or not isinstance(eos, int):
            raise ValueError(f"BOS/EOS token IDs must be integers. Got BOS: {bos}, EOS: {eos}")

        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos 

        content_max_length = max_length - 2 
        if content_max_length <=0: # // Sessão III by Gemini - CORREÇÃO: Evitar content_max_length negativo/zero se max_length for muito pequeno
            # Se max_length é 2, content_max_length é 0. Se 1, é -1.
            # Se content_max_length é 0, o loop range(0, len(all_input_ids), 0) é infinito.
            # Se for <=0, não podemos ter conteúdo, apenas BOS/EOS, o que não faz sentido para chunking de prompt.
            # Retornar um único chunk com BOS, EOS e padding se necessário, ou chunks vazios.
            # Para simplificar, se não há espaço para conteúdo, retornamos chunks vazios.
            # Ou, um chunk apenas com BOS/EOS se o prompt for vazio, mas all_input_ids seria vazio.
            print(f"[WARN] Tokenizer max_length {max_length} é muito pequeno para chunking, resultando em content_max_length <= 0. Retornando chunks vazios.")
            return [], []


        chunks = []
        attention_masks = []

        for i in range(0, len(all_input_ids), content_max_length):
            chunk_ids = all_input_ids[i:i + content_max_length]
            
            input_ids = [bos] + chunk_ids + [eos]
            mask = [1] * len(input_ids)

            padding_len = max_length - len(input_ids)
            if padding_len > 0:
                input_ids = input_ids + ([pad] * padding_len)
                mask = mask + ([0] * padding_len)
            elif padding_len < 0: # // Sessão III by Gemini - CORREÇÃO: Truncar se exceder max_length devido a BOS/EOS e chunk_ids
                # Isso não deveria acontecer se content_max_length for calculado corretamente.
                # Mas como segurança.
                input_ids = input_ids[:max_length]
                mask = mask[:max_length]


            chunks.append(torch.tensor(input_ids, dtype=torch.long))
            attention_masks.append(torch.tensor(mask, dtype=torch.long))

        return chunks, attention_masks