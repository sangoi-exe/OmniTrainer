import copy
from collections.abc import Callable

from modules.model.lumina.lumina_util import LUMINA_NUM_TRAIN_TIMESTEPS
from modules.model.LuminaModel import LuminaModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.torch_util import torch_gc

import torch

from PIL import Image
from tqdm import tqdm


class LuminaSampler(BaseModelSampler):
    def __init__(self, train_device: torch.device, temp_device: torch.device, model: LuminaModel, model_type: ModelType):
        super().__init__(train_device, temp_device)
        self.model = model
        self.model_type = model_type

    @staticmethod
    def __tensor_to_pil(image: torch.Tensor) -> Image.Image:
        image = image.detach().float().cpu().clamp(-1, 1)
        image = ((image + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
        image = image.permute(1, 2, 0).numpy()
        return Image.fromarray(image)

    @torch.no_grad()
    def __sample_base(
        self,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        seed: int,
        random_seed: bool,
        diffusion_steps: int,
        cfg_scale: float,
        text_encoder_sequence_length: int | None,
        system_prompt: str,
        on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            scheduler = copy.deepcopy(self.model.noise_scheduler)
            scheduler.set_timesteps(diffusion_steps, device=self.train_device)
            timesteps = scheduler.timesteps

            self.model.text_encoder_to(self.train_device)
            prompt_embedding, prompt_attention_mask = self.model.encode_text(
                text=prompt,
                train_device=self.train_device,
                max_length=text_encoder_sequence_length or 256,
                system_prompt=system_prompt,
            )
            negative_prompt_embedding, negative_prompt_attention_mask = self.model.encode_text(
                text=negative_prompt,
                train_device=self.train_device,
                max_length=text_encoder_sequence_length or 256,
                system_prompt=system_prompt,
                is_negative=True,
            )
            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            latent_image = torch.randn(
                size=(1, self.model.transformer.in_channels, height // 8, width // 8),
                generator=generator,
                device=self.train_device,
                dtype=torch.float32,
            )

            self.model.transformer_to(self.train_device)
            for step_index, timestep in enumerate(tqdm(timesteps, desc="sampling")):
                current_timestep = 1 - timestep / LUMINA_NUM_TRAIN_TIMESTEPS
                current_timestep = current_timestep * torch.ones(latent_image.shape[0], device=latent_image.device)

                predicted_positive = self.model.transformer(
                    x=latent_image.to(dtype=self.model.train_dtype.torch_dtype()),
                    t=current_timestep.to(dtype=self.model.train_dtype.torch_dtype()),
                    cap_feats=prompt_embedding.to(dtype=self.model.train_dtype.torch_dtype()),
                    cap_mask=prompt_attention_mask.to(dtype=torch.int32),
                )

                if current_timestep[0] < 0.25:
                    predicted_negative = self.model.transformer(
                        x=latent_image.to(dtype=self.model.train_dtype.torch_dtype()),
                        t=current_timestep.to(dtype=self.model.train_dtype.torch_dtype()),
                        cap_feats=negative_prompt_embedding.to(dtype=self.model.train_dtype.torch_dtype()),
                        cap_mask=negative_prompt_attention_mask.to(dtype=torch.int32),
                    )
                    predicted = predicted_negative + cfg_scale * (predicted_positive - predicted_negative)
                else:
                    predicted = predicted_positive

                latent_image = scheduler.step(-predicted, timestep, latent_image, return_dict=False)[0]
                on_update_progress(step_index + 1, len(timesteps))

            self.model.transformer_to(self.temp_device)
            torch_gc()

            self.model.vae_to(self.train_device)
            image = self.model.vae.decode_tensor(latent_image.to(dtype=self.model.vae.dtype))[0]
            self.model.vae_to(self.temp_device)
            torch_gc()

            return ModelSamplerOutput(file_type=FileType.IMAGE, data=self.__tensor_to_pil(image))

    def sample(
        self,
        sample_config: SampleConfig,
        destination: str,
        image_format: ImageFormat | None = None,
        video_format: VideoFormat | None = None,
        audio_format: AudioFormat | None = None,
        on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
        on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        sampler_output = self.__sample_base(
            prompt=sample_config.prompt,
            negative_prompt=sample_config.negative_prompt,
            height=self.quantize_resolution(sample_config.height, 16),
            width=self.quantize_resolution(sample_config.width, 16),
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            text_encoder_sequence_length=self.model.train_config.text_encoder_sequence_length
            if self.model.train_config is not None
            else 256,
            system_prompt=self.model.train_config.lumina_system_prompt if self.model.train_config is not None else "",
            on_update_progress=on_update_progress,
        )
        self.save_sampler_output(sampler_output, destination, image_format, video_format, audio_format)
        on_sample(sampler_output)


factory.register(BaseModelSampler, LuminaSampler, ModelType.LUMINA_2)
