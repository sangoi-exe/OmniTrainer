from abc import ABCMeta
from random import Random

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel, StableDiffusionXLModelEmbedding
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupDiffusionMixin import ModelSetupDiffusionMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.util.checkpointing_util import (
  enable_checkpointing_for_basic_transformer_blocks,
  enable_checkpointing_for_clip_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.conv_util import apply_circular_padding_to_conv2d
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.TrainProgress import TrainProgress
from modules.util.TensorBoardManager import TensorBoardManager

import torch
import torch.nn.functional as F
from torch import Tensor

from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod


class BaseStableDiffusionXLSetup(
  BaseModelSetup,
  ModelSetupDiffusionLossMixin,
  ModelSetupDebugMixin,
  ModelSetupNoiseMixin,
  ModelSetupDiffusionMixin,
  ModelSetupEmbeddingMixin,
  metaclass=ABCMeta
):

  def setup_optimizations(
      self,
      model: StableDiffusionXLModel,
      config: TrainConfig,
  ):
    model.unet_offload_conductor = None # Inicializa como None
    # model.text_encoder_1_offload_conductor = None # Não usado para CLIP, mas boa prática zerar
    # model.text_encoder_2_offload_conductor = None # Não usado para CLIP, mas boa prática zerar
    
    if config.gradient_checkpointing.enabled():
      # Limpa/Inicializa os possíveis condutores
      model.unet_offload_conductor = None
      model.text_encoder_1_offload_conductor = None # Não usado para CLIP, mas seguro zerar
      model.text_encoder_2_offload_conductor = None # Não usado para CLIP, mas seguro zerar

    # Aplica checkpointing se não estiver OFF
    if config.gradient_checkpointing != GradientCheckpointingMethod.OFF:
      # Chama as funções modificadas. Elas decidirão internamente se aplicam
      # checkpointing baseado em config.gradient_checkpointing_layers se o modo for ON.
      # Passamos offload_enabled=True para permitir que a função crie o conductor se o modo for CPU_OFFLOADED.
      unet_conductor_or_none = enable_checkpointing_for_basic_transformer_blocks(model.unet, config, offload_enabled=True)
      enable_checkpointing_for_clip_encoder_layers(model.text_encoder_1, config)
      enable_checkpointing_for_clip_encoder_layers(model.text_encoder_2, config)

      # Atribui o conductor retornado APENAS se o modo for CPU_OFFLOADED
      if config.gradient_checkpointing == GradientCheckpointingMethod.CPU_OFFLOADED:
        model.unet_offload_conductor = unet_conductor_or_none

    if config.force_circular_padding:
      apply_circular_padding_to_conv2d(model.vae)
      apply_circular_padding_to_conv2d(model.unet)
      if model.unet_lora is not None:
        apply_circular_padding_to_conv2d(model.unet_lora)

    model.autocast_context, model.train_dtype = create_autocast_context(self.train_device, config.train_dtype, [
      config.weight_dtypes().unet,
      config.weight_dtypes().text_encoder,
      config.weight_dtypes().text_encoder_2,
      config.weight_dtypes().vae,
      config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
      config.weight_dtypes().embedding if config.train_any_embedding() else None,
    ], config.enable_autocast_cache)

    model.vae_autocast_context, model.vae_train_dtype = disable_fp16_autocast_context(
      self.train_device,
      config.train_dtype,
      config.fallback_train_dtype,
      [
        config.weight_dtypes().vae,
      ],
      config.enable_autocast_cache,
    )

    quantize_layers(model.text_encoder_1, self.train_device, model.train_dtype)
    quantize_layers(model.text_encoder_2, self.train_device, model.train_dtype)
    quantize_layers(model.vae, self.train_device, model.vae_train_dtype)
    quantize_layers(model.unet, self.train_device, model.train_dtype)
    
  def _setup_embeddings(
      self,
      model: StableDiffusionXLModel,
      config: TrainConfig,
  ):
    additional_embeddings = []
    for embedding_config in config.all_embedding_configs():
      embedding_state = model.embedding_state_dicts.get(embedding_config.uuid, None)
      if embedding_state is None:
        embedding_state_1 = self._create_new_embedding(
          embedding_config,
          model.tokenizer_1,
          model.text_encoder_1,
          lambda text: model.encode_text(
            text=text,
            train_device=self.temp_device,
          )[0][0][1:],
        )

        embedding_state_2 = self._create_new_embedding(
          embedding_config,
          model.tokenizer_2,
          model.text_encoder_2,
          lambda text: model.encode_text(
            text=text,
            train_device=self.temp_device,
          )[1][0][1:],
        )
      else:
        embedding_state_1 = embedding_state.get("clip_l_out", embedding_state.get("clip_l", None))
        embedding_state_2 = embedding_state.get("clip_g_out", embedding_state.get("clip_g", None))

      embedding_state_1 = embedding_state_1.to(
        dtype=model.text_encoder_1.get_input_embeddings().weight.dtype,
        device=self.train_device,
      ).detach()

      embedding_state_2 = embedding_state_2.to(
        dtype=model.text_encoder_2.get_input_embeddings().weight.dtype,
        device=self.train_device,
      ).detach()

      embedding = StableDiffusionXLModelEmbedding(
        embedding_config.uuid,
        embedding_state_1,
        embedding_state_2,
        embedding_config.placeholder,
        embedding_config.is_output_embedding,
      )
      if embedding_config.uuid == config.embedding.uuid:
        model.embedding = embedding
      else:
        additional_embeddings.append(embedding)

    model.additional_embeddings = additional_embeddings

    self._add_embeddings_to_tokenizer(model.tokenizer_1, model.all_text_encoder_1_embeddings())
    self._add_embeddings_to_tokenizer(model.tokenizer_2, model.all_text_encoder_2_embeddings())

  def _setup_embedding_wrapper(
      self,
      model: StableDiffusionXLModel,
      config: TrainConfig,
  ):
    model.embedding_wrapper_1 = AdditionalEmbeddingWrapper(
      tokenizer=model.tokenizer_1,
      orig_module=model.text_encoder_1.text_model.embeddings.token_embedding,
      embeddings=model.all_text_encoder_1_embeddings(),
    )
    model.embedding_wrapper_2 = AdditionalEmbeddingWrapper(
      tokenizer=model.tokenizer_2,
      orig_module=model.text_encoder_2.text_model.embeddings.token_embedding,
      embeddings=model.all_text_encoder_2_embeddings(),
    )

    model.embedding_wrapper_1.hook_to_module()
    model.embedding_wrapper_2.hook_to_module()

  def _setup_embeddings_requires_grad(
      self,
      model: StableDiffusionXLModel,
      config: TrainConfig,
  ):
    for embedding, embedding_config in zip(model.all_text_encoder_1_embeddings(),
                         config.all_embedding_configs(), strict=True):
      train_embedding_1 = \
        embedding_config.train \
        and config.text_encoder.train_embedding \
        and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
      embedding.requires_grad_(train_embedding_1)

    for embedding, embedding_config in zip(model.all_text_encoder_2_embeddings(),
                         config.all_embedding_configs(), strict=True):
      train_embedding_2 = \
        embedding_config.train \
        and config.text_encoder_2.train_embedding \
        and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
      embedding.requires_grad_(train_embedding_2)

  def predict(
      self,
      model: StableDiffusionXLModel,
      batch: dict,
      config: TrainConfig,
      train_progress: TrainProgress,
      *,
      deterministic: bool = False,
  ) -> dict:
    with model.autocast_context:
      if deterministic:
          batch_seed = 0  # reprodutibilidade total
      # flag idiota, essa merda AINDA É deterministic mesmo false, porque usa o global step
      # mudei pra ser realmente aleatória, pra poder funcionar o OHEM lá no predict 
      else:
          # gera seed imprevisível via Torch (GPU-safe) a cada invocação
          batch_seed = torch.randint(
              low=1, high=2**31 - 1, size=(1,),
              device=config.train_device
          ).item()

      generator = torch.Generator(device=config.train_device).manual_seed(batch_seed)
      rand      = Random(batch_seed)

      vae_scaling_factor = model.vae.config['scaling_factor']

      # // Sessão DBG - Início da depuração do prompt
      if config.debugoi:
          print(f"// Sessão DBG - predict: Tipo de batch['prompt']: {type(batch['prompt'])}")
          if isinstance(batch['prompt'], list):
              print(f"// Sessão DBG - predict: batch['prompt'] (primeiros 3 itens): {batch['prompt'][:3]}")
          else:
              print(f"// Sessão DBG - predict: batch['prompt'] (inteiro): {batch['prompt']}")
            # // Sessão DBG - Fim da depuração do prompt			

      text_encoder_output, pooled_text_encoder_2_output = model.combine_text_encoder_output(*model.encode_text(
        train_device=self.train_device,
        batch_size=batch['latent_image'].shape[0],
        rand=rand,
        tokens_1=batch['tokens_1'],
        tokens_2=batch['tokens_2'],
        text_encoder_1_layer_skip=config.text_encoder_layer_skip,
        text_encoder_2_layer_skip=config.text_encoder_2_layer_skip,

        # só busco hidden_states do batch se estivermos usando o pipeline normal (não long prompts)
        text_encoder_1_output=(
            batch['text_encoder_1_hidden_state']
            if not config.train_text_encoder_or_embedding() and not config.enable_long_prompts
            else None
        ),
        text_encoder_2_output=(
            batch['text_encoder_2_hidden_state']
            if not config.train_text_encoder_2_or_embedding() and not config.enable_long_prompts
            else None
        ),
        pooled_text_encoder_2_output=(
            batch['text_encoder_2_pooled_state']
            if not config.train_text_encoder_2_or_embedding() and not config.enable_long_prompts
            else None
        ),
        # e, para long prompts, passe o texto bruto em vez de tokens
        text=(
            batch['prompt']
            if config.enable_long_prompts
            else None
        ),

        text_encoder_1_dropout_probability=config.text_encoder.dropout_probability,
        text_encoder_2_dropout_probability=config.text_encoder_2.dropout_probability,
      ))

      # // Sessão DBG - Depuração após encode_text e combine_text_encoder_output
      if config.debugoi:
          print(f"// Sessão DBG - predict: Após encode_text/combine_text_encoder_output:")
          if text_encoder_output is not None:
              print(f"// Sessão DBG - predict: text_encoder_output.shape: {text_encoder_output.shape}, .device: {text_encoder_output.device}")
              print(f"// Sessão DBG - predict: text_encoder_output.dtype: {text_encoder_output.dtype}")
              print(f"// Sessão DBG - predict: text_encoder_output.mean(): {text_encoder_output.mean().item():.4f}, .std(): {text_encoder_output.std().item():.4f}")
          else:
              print(f"// Sessão DBG - predict: text_encoder_output is None.")
          
          if pooled_text_encoder_2_output is not None:
              print(f"// Sessão DBG - predict: pooled_text_encoder_2_output.shape: {pooled_text_encoder_2_output.shape}, .device: {pooled_text_encoder_2_output.device}")
              print(f"// Sessão DBG - predict: pooled_text_encoder_2_output.dtype: {pooled_text_encoder_2_output.dtype}")
              print(f"// Sessão DBG - predict: pooled_text_encoder_2_output.mean(): {pooled_text_encoder_2_output.mean().item():.4f}, .std(): {pooled_text_encoder_2_output.std().item():.4f}")
          else:
              print(f"// Sessão DBG - predict: pooled_text_encoder_2_output is None.")

          # Inspecionar inputs para o UNet
          print(f"// Sessão DBG - predict: Inputs para model.unet:")
          print(f"// Sessão DBG - predict: UNet encoder_hidden_states.shape: {text_encoder_output.shape if text_encoder_output is not None else 'None'}")
          if text_encoder_output is not None:
              print(f"// Sessão DBG - predict: UNet encoder_hidden_states.device: {text_encoder_output.device}")
          print(f"// Sessão DBG - predict: UNet added_cond_kwargs['text_embeds'].shape: {pooled_text_encoder_2_output.shape if pooled_text_encoder_2_output is not None else 'None'}")
          if pooled_text_encoder_2_output is not None:
              print(f"// Sessão DBG - predict: UNet added_cond_kwargs['text_embeds'].device: {pooled_text_encoder_2_output.device}")

      # // Sessão DBG - Fim da depuração

      latent_image = batch['latent_image'].to(device=self.train_device, memory_format=torch.channels_last)
      scaled_latent_image = latent_image * vae_scaling_factor

      scaled_latent_conditioning_image = None
      if config.model_type.has_conditioning_image_input():
        batch['latent_conditioning_image'] = batch['latent_conditioning_image'].to(device=self.train_device, memory_format=torch.channels_last)
        scaled_latent_conditioning_image = batch['latent_conditioning_image'] * vae_scaling_factor

      latent_noise = self._create_noise(scaled_latent_image, config, generator)

      timestep = self._get_timestep_discrete(
        model.noise_scheduler.config['num_train_timesteps'],
        deterministic,
        generator,
        scaled_latent_image.shape[0],
        config,
        train_progress,
      )

      scaled_noisy_latent_image = self._add_noise_discrete(
        scaled_latent_image,
        latent_noise,
        timestep,
        model.noise_scheduler.betas,
      )

      # original size of the image
      original_height = batch['original_resolution'][0]
      original_width = batch['original_resolution'][1]
      crops_coords_top = batch['crop_offset'][0]
      crops_coords_left = batch['crop_offset'][1]
      target_height = batch['crop_resolution'][0]
      target_width = batch['crop_resolution'][1]

      add_time_ids = torch.stack([
        original_height,
        original_width,
        crops_coords_top,
        crops_coords_left,
        target_height,
        target_width
      ], dim=1)

      add_time_ids = add_time_ids.to(
        dtype=scaled_noisy_latent_image.dtype,
        device=scaled_noisy_latent_image.device,
      )

      # <<< CHATGPT ADD – põe embeddings/hidden-states no mesmo CUDA do UNet >>>
      text_encoder_output = text_encoder_output.to(
        device=self.train_device,
        dtype=model.train_dtype.torch_dtype(),
        non_blocking=True,
      )
      if pooled_text_encoder_2_output is not None:
        pooled_text_encoder_2_output = pooled_text_encoder_2_output.to(
          device=self.train_device,
          dtype=model.train_dtype.torch_dtype(),
          non_blocking=True,
        )

      if config.model_type.has_mask_input() and config.model_type.has_conditioning_image_input():
        latent_input = torch.concat(
          [
            scaled_noisy_latent_image,
            batch['latent_mask'].to(self.train_device, non_blocking=True),
            scaled_latent_conditioning_image,
          ],
          dim=1,
        )
      else:
        latent_input = scaled_noisy_latent_image

      added_cond_kwargs = {
        "text_embeds": pooled_text_encoder_2_output,
        "time_ids": add_time_ids,
      }
      predicted_latent_noise = model.unet(
        sample=latent_input.to(
          device=self.train_device,
          dtype=model.train_dtype.torch_dtype(),
        ),
        timestep=timestep,
        encoder_hidden_states=text_encoder_output,
        added_cond_kwargs=added_cond_kwargs,
      ).sample

      model_output_data = {}

      if model.noise_scheduler.config.prediction_type == 'epsilon':
        model_output_data = {
          'loss_type': 'target',
          'timestep': timestep,
          'predicted': predicted_latent_noise,
          'target': latent_noise,
        }
      elif model.noise_scheduler.config.prediction_type == 'v_prediction':
        target_velocity = model.noise_scheduler.get_velocity(scaled_latent_image, latent_noise, timestep)
        model_output_data = {
          'loss_type': 'target',
          'timestep': timestep,
          'predicted': predicted_latent_noise,
          'target': target_velocity,
        }
      if config.debug_mode or config.save_predictions:
        with torch.no_grad():
          self._save_text(
            self._decode_tokens(batch['tokens_1'], model.tokenizer_1),
            config.debug_dir + "/training_batches",
            "7-prompt",
            train_progress.global_step,
          )

          # noise
          self._save_image(
            self._project_latent_to_image_sdxl(latent_noise),
            config.debug_dir + "/training_batches",
            "1-noise",
            train_progress.global_step,
            True
          )

          # predicted noise
          self._save_image(
            self._project_latent_to_image_sdxl(predicted_latent_noise),
            config.debug_dir + "/training_batches",
            "2-predicted_noise",
            train_progress.global_step,
            True
          )

          # noisy image
          self._save_image(
            self._project_latent_to_image_sdxl(scaled_noisy_latent_image),
            config.debug_dir + "/training_batches",
            "3-noisy_image",
            train_progress.global_step,
            True
          )

          # predicted image
          alphas_cumprod = model.noise_scheduler.alphas_cumprod.to(config.train_device)
          sqrt_alpha_prod = alphas_cumprod[timestep] ** 0.5
          sqrt_alpha_prod = sqrt_alpha_prod.flatten().reshape(-1, 1, 1, 1)

          sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timestep]) ** 0.5
          sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten().reshape(-1, 1, 1, 1)
          
          scaled_predicted_latent_image = \
            (scaled_noisy_latent_image - predicted_latent_noise * sqrt_one_minus_alpha_prod) \
            / sqrt_alpha_prod

          self._save_image(
            self._project_latent_to_image_sdxl(scaled_predicted_latent_image),
            config.debug_dir + "/training_batches",
            "4-predicted_image",
            model.train_progress.global_step,
            True
          )

          # image
          self._save_image(
            self._project_latent_to_image_sdxl(scaled_latent_image),
            config.debug_dir + "/training_batches",
            "5-image",
            model.train_progress.global_step,
            True
          )
      else:
          # predicted image
          alphas_cumprod = model.noise_scheduler.alphas_cumprod.to(config.train_device)
          sqrt_alpha_prod = alphas_cumprod[timestep] ** 0.5
          sqrt_alpha_prod = sqrt_alpha_prod.flatten().reshape(-1, 1, 1, 1)

          sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timestep]) ** 0.5
          sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten().reshape(-1, 1, 1, 1)
          
          scaled_predicted_latent_image = \
            (scaled_noisy_latent_image - predicted_latent_noise * sqrt_one_minus_alpha_prod) \
            / sqrt_alpha_prod

      if config.full_vae_mf:
          model_output_data["target_image_rgb"]    = self._decode_latent_sdxl(model, scaled_latent_image.detach())
          model_output_data["predicted_image_rgb"] = self._decode_latent_sdxl(model, scaled_predicted_latent_image.detach())
      else:
          model_output_data["target_image_latent"] = scaled_latent_image.detach()
          model_output_data["predicted_image_latent"] = scaled_predicted_latent_image.detach()              

      model_output_data['prediction_type'] = model.noise_scheduler.config.prediction_type

    return model_output_data

  def _decode_latent_sdxl(self, model, latents: torch.Tensor):
      model.vae_to(self.train_device)
      scaling = getattr(model.vae.config, "scaling_factor", 0.18215)
      latents = latents / scaling
      # mantemos o mesmo dtype do treino (bf16)
      target_dtype = latents.dtype
      latents = latents.to(device=model.vae.device, dtype=target_dtype)
      with torch.autocast("cuda", dtype=target_dtype):   # API nova          
          decoded = model.vae.decode(latents, return_dict=False)[0]
      return decoded

  def calculate_loss(
      self,
      model: StableDiffusionXLModel,
      batch: dict,
      data: dict,
      config: TrainConfig,
      progress: TrainProgress,
  ) -> Tensor:
    return self._diffusion_losses(
      batch=batch,
      data=data,
      model=model,
      config=config,
      progress=progress,
      train_device=self.train_device,
      betas=model.noise_scheduler.betas.to(device=self.train_device),
    ).mean()
