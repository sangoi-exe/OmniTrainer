import math
from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.lumina.lumina_util import LUMINA_NUM_TRAIN_TIMESTEPS
from modules.model.LuminaModel import LuminaModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseLuminaSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta,
):
    LAYER_PRESETS = {
        "attn-mlp": ["attention", "feed_forward"],
        "attn-only": ["attention"],
        "blocks": ["layers", "context_refiner", "noise_refiner"],
        "full": [],
    }

    def setup_optimizations(self, model: LuminaModel, config: TrainConfig):
        if config.gradient_checkpointing.enabled():
            if config.gradient_checkpointing.offload():
                raise ValueError("Lumina gradient checkpointing CPU offload is not supported")
            model.transformer.enable_gradient_checkpointing(cpu_offload=False)

        model.autocast_context, model.train_dtype = create_autocast_context(
            self.train_device,
            config.train_dtype,
            [
                config.weight_dtypes().transformer,
                config.weight_dtypes().text_encoder,
                config.weight_dtypes().vae,
                config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            ],
            config.enable_autocast_cache,
        )

        model.text_encoder_autocast_context, model.text_encoder_train_dtype = disable_fp16_autocast_context(
            self.train_device,
            config.train_dtype,
            config.fallback_train_dtype,
            [
                config.weight_dtypes().text_encoder,
                config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            ],
            config.enable_autocast_cache,
        )

        model.vae_autocast_context, model.vae_train_dtype = disable_fp16_autocast_context(
            self.train_device,
            config.train_dtype,
            config.fallback_train_dtype,
            [config.weight_dtypes().vae],
            config.enable_autocast_cache,
        )

        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.vae_train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)
        self._set_attention_mechanism(model.transformer, config.attention_mechanism)

    def __sample_lumina_sigmas(
        self,
        latent_image: Tensor,
        config: TrainConfig,
        generator: torch.Generator,
        deterministic: bool,
        validation_index=None,
        validation_count=None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = latent_image.shape[0]
        device = latent_image.device

        if deterministic:
            timestep = self._get_timestep_discrete(
                LUMINA_NUM_TRAIN_TIMESTEPS,
                True,
                generator,
                batch_size,
                config,
                shift=config.timestep_shift,
                validation_index=validation_index,
                validation_count=validation_count,
            ).to(device=device, dtype=torch.float32)
            sigmas = (timestep / LUMINA_NUM_TRAIN_TIMESTEPS).clamp(1e-7, 1.0)
            return timestep, sigmas.view(-1, 1, 1, 1)

        if config.timestep_distribution == TimestepDistribution.LOGIT_NORMAL:
            sigmas = torch.normal(
                mean=config.noising_bias,
                std=config.noising_weight + 1.0,
                size=(batch_size,),
                generator=generator,
                device=device,
            ).sigmoid()
            shift = config.timestep_shift
            sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
        elif config.timestep_distribution == TimestepDistribution.NEXTDIT_SHIFT:
            _, _, height, width = latent_image.shape
            image_seq_len = (height // 2) * (width // 2)
            mu = 0.5 + (1.15 - 0.5) * (image_seq_len - 256) / (4096 - 256)
            sigmas = torch.rand(batch_size, generator=generator, device=device).clamp_min(1e-7)
            exp_mu = math.exp(mu)
            sigmas = exp_mu / (exp_mu + (1.0 / sigmas - 1.0))
        elif config.timestep_distribution == TimestepDistribution.UNIFORM:
            sigmas = torch.rand(batch_size, generator=generator, device=device)
        elif config.timestep_distribution == TimestepDistribution.SIGMOID:
            sigmas = torch.randn(batch_size, generator=generator, device=device)
            sigmas = (sigmas * (config.noising_weight + 1.0) + config.noising_bias).sigmoid()
        else:
            timestep = self._get_timestep_discrete(
                LUMINA_NUM_TRAIN_TIMESTEPS,
                False,
                generator,
                batch_size,
                config,
                shift=config.timestep_shift,
            ).to(device=device, dtype=torch.float32)
            sigmas = (timestep / LUMINA_NUM_TRAIN_TIMESTEPS).clamp(1e-7, 1.0)
            return timestep, sigmas.view(-1, 1, 1, 1)

        sigmas = sigmas.clamp(1e-7, 1.0)
        timestep = sigmas * LUMINA_NUM_TRAIN_TIMESTEPS
        return timestep, sigmas.view(-1, 1, 1, 1)

    def predict(
        self,
        model: LuminaModel,
        batch: dict,
        config: TrainConfig,
        train_progress: TrainProgress,
        *,
        deterministic: bool = False,
    ) -> dict:
        if config.force_epsilon_prediction or config.force_v_prediction:
            raise ValueError("Lumina uses raw flow prediction; force_epsilon_prediction and force_v_prediction are unsupported")

        with model.autocast_context:
            if deterministic:
                batch_seed = int(batch.get("__validation_noise_seed__", 0))
            else:
                batch_seed = train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            text_encoder_output, text_encoder_attention_mask = model.encode_text(
                train_device=self.train_device,
                batch_size=batch["latent_image"].shape[0],
                rand=rand,
                tokens=batch["tokens"],
                attention_mask=batch["tokens_mask"],
                text_encoder_output=batch["text_encoder_hidden_state"]
                if not config.train_text_encoder_or_embedding()
                else None,
                text_encoder_dropout_probability=config.text_encoder.dropout_probability if not deterministic else None,
            )

            latent_image = batch["latent_image"]
            latent_noise = self._create_noise(latent_image, config, generator)
            timestep, sigmas = self.__sample_lumina_sigmas(
                latent_image,
                config,
                generator,
                deterministic,
                batch.get("__validation_timestep_index__"),
                batch.get("__validation_timestep_count__"),
            )
            noisy_latent_image = (1.0 - sigmas) * latent_image + sigmas * latent_noise
            model_timestep = 1.0 - timestep / LUMINA_NUM_TRAIN_TIMESTEPS

            predicted_flow = model.transformer(
                x=noisy_latent_image.to(dtype=config.train_dtype.torch_dtype()),
                t=model_timestep.to(dtype=config.train_dtype.torch_dtype()),
                cap_feats=text_encoder_output.to(dtype=config.train_dtype.torch_dtype()),
                cap_mask=text_encoder_attention_mask.to(dtype=torch.int32),
            )

            flow = latent_image - latent_noise
            model_output_data = {
                "loss_type": "target",
                "timestep": timestep.long().clamp(0, LUMINA_NUM_TRAIN_TIMESTEPS - 1),
                "predicted": predicted_flow,
                "target": flow,
            }

            if self.debug_mode:
                with torch.no_grad():
                    predicted_latent_image = noisy_latent_image + predicted_flow * sigmas
                    self._save_tokens("7-prompt", batch["tokens"], model.tokenizer, config, train_progress)
                    self._save_latent("1-noise", latent_noise, config, train_progress)
                    self._save_latent("2-noisy_image", noisy_latent_image, config, train_progress)
                    self._save_latent("3-predicted_flow", predicted_flow, config, train_progress)
                    self._save_latent("4-flow", flow, config, train_progress)
                    self._save_latent("5-predicted_image", predicted_latent_image, config, train_progress)
                    self._save_latent("6-image", latent_image, config, train_progress)

        return model_output_data

    def calculate_loss(self, model: LuminaModel, batch: dict, data: dict, config: TrainConfig) -> Tensor:
        return self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=getattr(model.noise_scheduler, "sigmas", None),
        ).mean()

    def prepare_text_caching(self, model: LuminaModel, config: TrainConfig):
        model.to(self.temp_device)

        if not config.train_text_encoder_or_embedding():
            model.text_encoder_to(self.train_device)

        model.eval()
        torch_gc()
