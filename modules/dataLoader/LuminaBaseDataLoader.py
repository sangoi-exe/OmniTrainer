import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.model.BaseModel import BaseModel
from modules.model.LuminaModel import LuminaModel
from modules.modelSetup.BaseLuminaSetup import BaseLuminaSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.EncodeGemmaText import EncodeGemmaText
from mgds.pipelineModules.MapData import MapData
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModules.Tokenize import Tokenize


class LuminaBaseDataLoader(BaseDataLoader, DataLoaderText2ImageMixin):
    def _preparation_modules(self, config: TrainConfig, model: LuminaModel):
        max_token_length = config.text_encoder_sequence_length or 256

        rescale_image = RescaleImageChannels(
            image_in_name="image",
            image_out_name="image",
            in_range_min=0,
            in_range_max=1,
            out_range_min=-1,
            out_range_max=1,
        )
        encode_image = MapData(in_name="image", out_name="latent_image", map_fn=model.encode_image)
        downscale_mask = ScaleImage(in_name="mask", out_name="latent_mask", factor=0.125)
        add_system_prompt = MapData(
            in_name="prompt",
            out_name="prompt",
            map_fn=lambda prompt: model.add_system_prompt(config.lumina_system_prompt, prompt),
        )
        tokenize_prompt = Tokenize(
            in_name="prompt",
            tokens_out_name="tokens",
            mask_out_name="tokens_mask",
            tokenizer=model.tokenizer,
            max_token_length=max_token_length,
        )
        encode_prompt = EncodeGemmaText(
            tokens_in_name="tokens",
            tokens_attention_mask_in_name="tokens_mask",
            hidden_state_out_name="text_encoder_hidden_state",
            add_layer_norm=False,
            text_encoder=model.text_encoder,
            hidden_state_output_index=-2,
            autocast_contexts=[model.autocast_context, model.text_encoder_autocast_context],
            dtype=model.text_encoder_train_dtype.torch_dtype(),
        )

        modules = [rescale_image, encode_image]
        if config.masked_training:
            modules.append(downscale_mask)

        modules += [add_system_prompt, tokenize_prompt]
        if not config.train_text_encoder_or_embedding():
            modules.append(encode_prompt)

        return modules

    def _cache_modules(self, config: TrainConfig, model: LuminaModel, model_setup: BaseLuminaSetup):
        image_split_names = ["latent_image"]
        if config.masked_training:
            image_split_names.append("latent_mask")

        image_aggregate_names = ["crop_resolution", "image_path"]
        text_split_names = ["tokens", "tokens_mask", "text_encoder_hidden_state"]
        sort_names = text_split_names + image_aggregate_names + image_split_names + ["prompt", "concept"]

        return self._cache_modules_from_names(
            model,
            model_setup,
            image_split_names=image_split_names,
            image_aggregate_names=image_aggregate_names,
            text_split_names=text_split_names,
            sort_names=sort_names,
            config=config,
            text_caching=not config.train_text_encoder_or_embedding(),
        )

    def _output_modules(self, config: TrainConfig, model: LuminaModel, model_setup: BaseLuminaSetup):
        output_names = [
            "image_path",
            "latent_image",
            "prompt",
            "tokens",
            "tokens_mask",
        ]

        if config.masked_training:
            output_names.append("latent_mask")

        if not config.train_text_encoder_or_embedding():
            output_names.append("text_encoder_hidden_state")

        return self._output_modules_from_out_names(
            model,
            model_setup,
            output_names=output_names,
            config=config,
            use_conditioning_image=False,
            vae=model.vae,
            autocast_context=[model.autocast_context, model.vae_autocast_context],
            train_dtype=model.vae_train_dtype,
        )

    def _debug_modules(self, config: TrainConfig, model: LuminaModel):
        debug_dir = os.path.join(config.debug_dir, "dataloader")

        def before_save_fun():
            model.vae_to(self.train_device)

        decode_image = MapData(in_name="latent_image", out_name="decoded_image", map_fn=model.decode_image)
        upscale_mask = ScaleImage(in_name="latent_mask", out_name="decoded_mask", factor=8)
        decode_prompt = DecodeTokens(in_name="tokens", out_name="decoded_prompt", tokenizer=model.tokenizer)
        save_image = SaveImage(
            image_in_name="decoded_image",
            original_path_in_name="image_path",
            path=debug_dir,
            in_range_min=-1,
            in_range_max=1,
            before_save_fun=before_save_fun,
        )
        save_mask = SaveImage(
            image_in_name="decoded_mask",
            original_path_in_name="image_path",
            path=debug_dir,
            in_range_min=0,
            in_range_max=1,
            before_save_fun=before_save_fun,
        )
        save_prompt = SaveText(
            text_in_name="decoded_prompt",
            original_path_in_name="image_path",
            path=debug_dir,
            before_save_fun=before_save_fun,
        )

        modules = [decode_image, save_image]
        if config.masked_training:
            modules += [upscale_mask, save_mask]
        modules += [decode_prompt, save_prompt]
        return modules

    def _create_dataset(
        self,
        config: TrainConfig,
        model: BaseModel,
        model_setup: BaseModelSetup,
        train_progress: TrainProgress,
        is_validation: bool = False,
    ):
        return DataLoaderText2ImageMixin._create_dataset(
            self,
            config,
            model,
            model_setup,
            train_progress,
            is_validation,
            aspect_bucketing_quantization=16,
        )


factory.register(BaseDataLoader, LuminaBaseDataLoader, ModelType.LUMINA_2)
