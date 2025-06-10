from abc import ABCMeta
from random import Random
from typing import Optional

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
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn.functional as F
from torch import Tensor

from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod

def apply_channels_last_to_lora_conv(lora_module):
    """
    Percorre um módulo LoRA e aplica `channels_last` a qualquer camada Conv2d encontrada.
    """
    for module in lora_module.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.to(memory_format=torch.channels_last)

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
      if config.gradient_checkpointing.enabled():
          model.unet.enable_gradient_checkpointing()
          enable_checkpointing_for_basic_transformer_blocks(model.unet, config, offload_enabled=False)
          enable_checkpointing_for_clip_encoder_layers(model.text_encoder_1, config)
          enable_checkpointing_for_clip_encoder_layers(model.text_encoder_2, config)

      if config.force_circular_padding:
          apply_circular_padding_to_conv2d(model.vae)
          apply_circular_padding_to_conv2d(model.unet)
          if model.unet_lora is not None:
              apply_circular_padding_to_conv2d(model.unet_lora)

      from modules.sangoi.logFun import logFun
      logFun("Otimização 'channels_last' ativada. Convertendo modelos...", lvl="INFO")
      try:
          # Converte os componentes principais que usam convoluções 4D
          if hasattr(model, 'unet'):
              model.unet.to(memory_format=torch.channels_last)
          if hasattr(model, 'vae'):
              model.vae.to(memory_format=torch.channels_last)
          
          # Se você treina um LoRA, a camada de convolução dele também precisa ser convertida
          if hasattr(model, 'unet_lora'):
              # LoRA pode não ter sido criado ainda, então verificamos se não é None
              if model.unet_lora is not None:
                  apply_channels_last_to_lora_conv(model.unet_lora)

          logFun("Modelos convertidos para 'channels_last' com sucesso.", lvl="SUCCESS")
      except Exception as e:
          logFun(f"Falha ao converter modelos para 'channels_last': {e}", lvl="ERROR")

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
                  model,
                  embedding_config,
                  model.tokenizer_1,
                  model.text_encoder_1,
                  lambda text: model.encode_text(
                      text=text,
                      train_device=self.temp_device,
                  )[0][0][1:],
              )

              embedding_state_2 = self._create_new_embedding(
                  model,
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
      tensorboard: Optional[SummaryWriter] = None,
      *,
      deterministic: bool = False,
  ) -> dict:
    tensorboard = model.tensorboard
    with model.autocast_context:
      if deterministic:
          batch_seed = 0  # reprodutibilidade total
      # flag idiota, essa merda AINDA É deterministic mesmo false, porque usa o global step
      else:
          # gera seed imprevisível via Torch (GPU-safe) a cada invocação
          batch_seed = torch.randint(
              low=1, high=2**31 - 1, size=(1,),
              device=config.train_device
          ).item()

      generator = torch.Generator(device=config.train_device).manual_seed(batch_seed)
      rand      = Random(batch_seed)

      vae_scaling_factor = model.vae.config['scaling_factor']

      text_encoder_output, pooled_text_encoder_2_output = model.combine_text_encoder_output(*model.encode_text(
          train_device=self.train_device,
          batch_size=batch['latent_image'].shape[0],
          rand=rand,
          tokens_1=batch['tokens_1'],
          tokens_2=batch['tokens_2'],
          text_encoder_1_layer_skip=config.text_encoder_layer_skip,
          text_encoder_2_layer_skip=config.text_encoder_2_layer_skip,
          text_encoder_1_output=batch[
              'text_encoder_1_hidden_state'] if not config.train_text_encoder_or_embedding() else None,
          text_encoder_2_output=batch[
              'text_encoder_2_hidden_state'] if not config.train_text_encoder_2_or_embedding() else None,
          pooled_text_encoder_2_output=batch[
              'text_encoder_2_pooled_state'] if not config.train_text_encoder_2_or_embedding() else None,
          text_encoder_1_dropout_probability=config.text_encoder.dropout_probability,
          text_encoder_2_dropout_probability=config.text_encoder_2.dropout_probability,
      ))

      latent_image = batch['latent_image']
      scaled_latent_image = latent_image * vae_scaling_factor

      scaled_latent_conditioning_image = None
      if config.model_type.has_conditioning_image_input():
        batch['latent_conditioning_image'] = batch['latent_conditioning_image']
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

      if config.model_type.has_mask_input() and config.model_type.has_conditioning_image_input():
          latent_input = torch.concat(
              [scaled_noisy_latent_image, batch['latent_mask'], scaled_latent_conditioning_image], 1
          )
      else:
          latent_input = scaled_noisy_latent_image

      added_cond_kwargs = {"text_embeds": pooled_text_encoder_2_output, "time_ids": add_time_ids}
      predicted_latent_noise = model.unet(
          sample=latent_input.to(dtype=model.train_dtype.torch_dtype()),
          timestep=timestep,
          encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
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
      
      tensorboard.add_scalar(
          f"SangoiW/Timestep",
          timestep,
          train_progress.global_step,
      )

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

  def update_sampler_priorities(self, timesteps: torch.Tensor, batch_loss: float, config: TrainConfig):
      """
      Chama o método de atualização de prioridades do noiseMixin.
      """
      # Como esta classe herda de ModelSetupNoiseMixin, self.update_priorities está disponível.
      self.update_priorities(timesteps, batch_loss, config)
  
