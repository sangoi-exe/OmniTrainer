from contextlib import nullcontext
from random import Random
from typing import List

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

from modules.util.loss.DynamicLossStrength import DeltaPatternRegularizer


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
    deltas: DeltaPatternRegularizer | None

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
        self.embedding_wrapper_1 = None

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
        self.sd_config['model']['params']['parameterization'] = 'v'
        self.model_spec.prediction_type = 'v'

    def force_epsilon_prediction(self):
        self.noise_scheduler.config.prediction_type = 'epsilon'
        self.sd_config['model']['params']['parameterization'] = 'epsilon'
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
            batch_size: int = 1, # Batch size is typically 1 here during encoding
            rand: Random | None = None,
            text: str = None,
            tokens_1: Tensor = None, # Will be ignored if enable_long_prompts=True and text is provided
            tokens_2: Tensor = None, # Will be ignored if enable_long_prompts=True and text is provided
            text_encoder_1_layer_skip: int = 0,
            text_encoder_2_layer_skip: int = 0,
            text_encoder_1_output: Tensor = None, # Likely None if enable_long_prompts=True
            text_encoder_2_output: Tensor = None, # Likely None if enable_long_prompts=True
            text_encoder_1_dropout_probability: float | None = None,
            text_encoder_2_dropout_probability: float | None = None,
            pooled_text_encoder_2_output: Tensor = None, # Likely None if enable_long_prompts=True
    ) -> tuple[Tensor, Tensor, Tensor]:
        # If long prompts are not enabled, use the original logic
        if not (self.train_config and self.train_config.enable_long_prompts and text is not None):
            # --- Original Logic Start ---
            if tokens_1 is None and text is not None:
                # Apply embedding placeholders before tokenization
                processed_text_1 = self.add_text_encoder_1_embeddings_to_prompt(text)
                tokenizer_output_1 = self.tokenizer_1(
                    processed_text_1,
                    padding='max_length',
                    truncation=True,
                    max_length=self.tokenizer_1.model_max_length,
                    return_tensors="pt",
                )
                tokens_1 = tokenizer_output_1.input_ids.to(self.text_encoder_1.device)

            if tokens_2 is None and text is not None:
                 # Apply embedding placeholders before tokenization
                processed_text_2 = self.add_text_encoder_2_embeddings_to_prompt(text)
                tokenizer_output_2 = self.tokenizer_2(
                    processed_text_2,
                    padding='max_length',
                    truncation=True,
                    max_length=self.tokenizer_2.model_max_length,
                    return_tensors="pt",
                )
                tokens_2 = tokenizer_output_2.input_ids.to(self.text_encoder_2.device)

            # Encode using original method
            text_encoder_1_output, _ = encode_clip(
                text_encoder=self.text_encoder_1,
                tokens=tokens_1,
                default_layer=-2,
                layer_skip=text_encoder_1_layer_skip,
                text_encoder_output=text_encoder_1_output,
                add_pooled_output=False,
                use_attention_mask=False, # Original didn't use mask here
                add_layer_norm=False,
            )

            text_encoder_2_output, pooled_text_encoder_2_output = encode_clip(
                text_encoder=self.text_encoder_2,
                tokens=tokens_2,
                default_layer=-2,
                layer_skip=text_encoder_2_layer_skip,
                text_encoder_output=text_encoder_2_output,
                add_pooled_output=True,
                pooled_text_encoder_output=pooled_text_encoder_2_output,
                use_attention_mask=False, # Original didn't use mask here
                add_layer_norm=False,
            )
            # --- Original Logic End ---

        else:
            # --- Long Prompt Logic Start ---
            max_chunks = self.train_config.long_prompt_max_chunks
            max_len_1 = self.tokenizer_1.model_max_length
            max_len_2 = self.tokenizer_2.model_max_length

            # Apply embedding placeholders to the full text first
            processed_text_1 = self.add_text_encoder_1_embeddings_to_prompt(text)
            processed_text_2 = self.add_text_encoder_2_embeddings_to_prompt(text)

            # Chunk and tokenize
            token_chunks_1, mask_chunks_1 = self._chunk_tokenizer(self.tokenizer_1, processed_text_1, max_len_1)
            token_chunks_2, mask_chunks_2 = self._chunk_tokenizer(self.tokenizer_2, processed_text_2, max_len_2)

            # Limit chunks
            token_chunks_1 = token_chunks_1[:max_chunks]
            mask_chunks_1 = mask_chunks_1[:max_chunks]
            token_chunks_2 = token_chunks_2[:max_chunks]
            mask_chunks_2 = mask_chunks_2[:max_chunks]
            
            num_chunks = len(token_chunks_1) # Assume both tokenizers produce same number of chunks

            # Store embeddings per chunk
            chunk_embeddings_1: List[Tensor] = []
            chunk_embeddings_2: List[Tensor] = []
            pooled_text_encoder_2_output = None # Get from the first chunk

            # Move encoders to the correct device for processing
            # Note: This assumes encode_text is called when encoders are already on train_device
            # If called during sampling/caching, ensure correct device placement
            text_encoder_1_device = self.text_encoder_1.device
            text_encoder_2_device = self.text_encoder_2.device

            for i in range(num_chunks):
                tokens_1_chunk = token_chunks_1[i].unsqueeze(0).to(text_encoder_1_device) # Add batch dim
                # mask_1_chunk = mask_chunks_1[i].unsqueeze(0).to(text_encoder_1_device) # Add batch dim if needed by encode_clip

                tokens_2_chunk = token_chunks_2[i].unsqueeze(0).to(text_encoder_2_device) # Add batch dim
                # mask_2_chunk = mask_chunks_2[i].unsqueeze(0).to(text_encoder_2_device) # Add batch dim if needed by encode_clip

                # Encode chunk 1
                chunk_output_1, _ = encode_clip(
                    text_encoder=self.text_encoder_1,
                    tokens=tokens_1_chunk,
                    default_layer=-2,
                    layer_skip=text_encoder_1_layer_skip,
                    add_pooled_output=False,
                    use_attention_mask=False, # Set to True if mask is used
                    # attention_mask=mask_1_chunk,
                    add_layer_norm=False,
                )

                 # Encode chunk 2
                chunk_output_2, pooled_output_2_chunk = encode_clip(
                    text_encoder=self.text_encoder_2,
                    tokens=tokens_2_chunk,
                    default_layer=-2,
                    layer_skip=text_encoder_2_layer_skip,
                    add_pooled_output=True,
                    use_attention_mask=False, # Set to True if mask is used
                    # attention_mask=mask_2_chunk,
                    add_layer_norm=False,
                )
                
                # --- Apply output embeddings *per chunk* before concatenation ---
                # Note: This assumes _apply_output_embeddings works with single-item batches
                # and uses the chunk's tokens for indexing.
                chunk_output_1 = self._apply_output_embeddings(
                    self.all_text_encoder_1_embeddings(),
                    self.tokenizer_1,
                    tokens_1_chunk, # Use the chunk tokens
                    chunk_output_1,
                )
                chunk_output_2 = self._apply_output_embeddings(
                    self.all_text_encoder_2_embeddings(),
                    self.tokenizer_2,
                    tokens_2_chunk, # Use the chunk tokens
                    chunk_output_2,
                )
                # --- End Output Embedding Application ---


                # Store results (remove BOS/EOS embeddings before storing)
                # Assumes sequence length is axis 1 after batch dim 0
                chunk_embeddings_1.append(chunk_output_1[:, 1:-1, :]) # Exclude BOS/EOS
                chunk_embeddings_2.append(chunk_output_2[:, 1:-1, :]) # Exclude BOS/EOS

                if i == 0:
                    pooled_text_encoder_2_output = pooled_output_2_chunk # Store pooled from first chunk

            # Concatenate chunk embeddings along the sequence length dimension
            if chunk_embeddings_1:
                text_encoder_1_output = torch.cat(chunk_embeddings_1, dim=1)
            else:
                 # Handle case with no valid chunks (e.g., empty prompt)
                 # Create a zero tensor with expected shape or handle error
                 # Assuming hidden size can be inferred from text_encoder_1
                 hidden_size_1 = self.text_encoder_1.config.hidden_size
                 text_encoder_1_output = torch.zeros((batch_size, 0, hidden_size_1), device=text_encoder_1_device, dtype=self.text_encoder_1.dtype)


            if chunk_embeddings_2:
                text_encoder_2_output = torch.cat(chunk_embeddings_2, dim=1)
            else:
                # Handle case with no valid chunks
                hidden_size_2 = self.text_encoder_2.config.hidden_size
                text_encoder_2_output = torch.zeros((batch_size, 0, hidden_size_2), device=text_encoder_2_device, dtype=self.text_encoder_2.dtype)
                # Ensure pooled output is also zero/None if no chunks
                if pooled_text_encoder_2_output is None:
                     pooled_size_2 = self.text_encoder_2.config.projection_dim
                     pooled_text_encoder_2_output = torch.zeros((batch_size, pooled_size_2), device=text_encoder_2_device, dtype=self.text_encoder_2.dtype)

            # --- End Long Prompt Logic ---

        # --- Common Logic (Dropout, Final Application of Output Embeddings if not done per chunk) ---
        # Apply output embeddings *after* concatenation if not done per chunk
        # Note: Applying per-chunk (as implemented above) is generally easier
        # text_encoder_1_output = self._apply_output_embeddings(
        #     self.all_text_encoder_1_embeddings(), self.tokenizer_1, concatenated_tokens_1, text_encoder_1_output
        # )
        # text_encoder_2_output = self._apply_output_embeddings(
        #     self.all_text_encoder_2_embeddings(), self.tokenizer_2, concatenated_tokens_2, text_encoder_2_output
        # )

        # Apply dropout
        if text_encoder_1_dropout_probability is not None and text_encoder_1_output.numel() > 0:
            dropout_text_encoder_1_mask = (torch.tensor(
                [rand.random() > text_encoder_1_dropout_probability for _ in range(batch_size)], # Uses batch_size here
                device=train_device)).float() # Ensure mask is on train_device
            # Adjust mask shape if needed for broadcasting (usually [batch, 1, 1])
            text_encoder_1_output = text_encoder_1_output * dropout_text_encoder_1_mask.view(-1, 1, 1).to(text_encoder_1_output.device)


        if text_encoder_2_dropout_probability is not None and text_encoder_2_output.numel() > 0:
            dropout_text_encoder_2_mask = (torch.tensor(
                [rand.random() > text_encoder_2_dropout_probability for _ in range(batch_size)], # Uses batch_size here
                device=train_device)).float() # Ensure mask is on train_device
            # Adjust mask shape
            if pooled_text_encoder_2_output is not None and pooled_text_encoder_2_output.numel() > 0:
                 pooled_text_encoder_2_output = pooled_text_encoder_2_output * dropout_text_encoder_2_mask.view(-1, 1).to(pooled_text_encoder_2_output.device) # Mask shape (batch, 1)
            text_encoder_2_output = text_encoder_2_output * dropout_text_encoder_2_mask.view(-1, 1, 1).to(text_encoder_2_output.device) # Mask shape (batch, 1, 1)


        return text_encoder_1_output, text_encoder_2_output, pooled_text_encoder_2_output
    # END OneTrainer Long Prompt Mod

    def combine_text_encoder_output(
            self,
            text_encoder_1_output: Tensor,
            text_encoder_2_output: Tensor,
            pooled_text_encoder_2_output: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # START OneTrainer Long Prompt Mod
        # Ensure dimensions match for concatenation, even if one is empty due to chunking failure
        target_device = pooled_text_encoder_2_output.device # Use pooled output device as target
        target_dtype = pooled_text_encoder_2_output.dtype # Use pooled output dtype as target

        if text_encoder_1_output.shape[1] != text_encoder_2_output.shape[1]:
            # This indicates an issue, likely one encoder failed or produced different sequence length
            # For robustness, try to pad or truncate, but ideally lengths should match after chunking/concatenation
            # Or, handle based on which one might be empty if processing failed
            print(f"[WARN] Mismatched sequence lengths in combine_text_encoder_output: TE1={text_encoder_1_output.shape[1]}, TE2={text_encoder_2_output.shape[1]}. This might cause errors.")
            # Simple fallback: use the shorter sequence length (may lose info)
            min_seq_len = min(text_encoder_1_output.shape[1], text_encoder_2_output.shape[1])
            if min_seq_len == 0: # If one is completely empty, return zeros for combined (or raise error)
                 combined_hidden_size = text_encoder_1_output.shape[-1] + text_encoder_2_output.shape[-1]
                 text_encoder_output = torch.zeros((text_encoder_1_output.shape[0], 0, combined_hidden_size), device=target_device, dtype=target_dtype)
            else:
                 text_encoder_output = torch.cat(
                     [text_encoder_1_output[:, :min_seq_len, :].to(target_device, target_dtype),
                      text_encoder_2_output[:, :min_seq_len, :].to(target_device, target_dtype)],
                     dim=-1
                 )

        elif text_encoder_1_output.shape[1] == 0 and text_encoder_2_output.shape[1] == 0:
             # Handle case where both are empty
             combined_hidden_size = text_encoder_1_output.shape[-1] + text_encoder_2_output.shape[-1]
             text_encoder_output = torch.zeros((text_encoder_1_output.shape[0], 0, combined_hidden_size), device=target_device, dtype=target_dtype)
        else:
             # Original concatenation
            text_encoder_output = torch.cat(
                [text_encoder_1_output.to(target_device, target_dtype),
                 text_encoder_2_output.to(target_device, target_dtype)],
                dim=-1
            )
        # END OneTrainer Long Prompt Mod

        return text_encoder_output, pooled_text_encoder_2_output.to(target_device, target_dtype) # Ensure pooled is also on correct device/dtype
    
    def _chunk_tokenizer(self, tokenizer, text, max_length):
        """Helper to tokenize and chunk text."""
        # Tokenize full text without truncation first to get all ids
        # We handle special tokens manually per chunk later.
        all_input_ids = tokenizer(text, add_special_tokens=False).input_ids

        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos # Use EOS for padding if PAD is None

        # Max length for actual content tokens per chunk
        content_max_length = max_length - 2 # Account for BOS and EOS

        chunks = []
        attention_masks = []

        for i in range(0, len(all_input_ids), content_max_length):
            chunk_ids = all_input_ids[i:i + content_max_length]
            
            # Create chunk with special tokens
            input_ids = [bos] + chunk_ids + [eos]
            mask = [1] * len(input_ids)

            # Pad chunk if necessary
            padding_len = max_length - len(input_ids)
            if padding_len > 0:
                input_ids = input_ids + ([pad] * padding_len)
                mask = mask + ([0] * padding_len)

            chunks.append(torch.tensor(input_ids, dtype=torch.long))
            attention_masks.append(torch.tensor(mask, dtype=torch.long))

        return chunks, attention_masks
