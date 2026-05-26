import traceback

from modules.model.lumina.lumina_util import (
    GEMMA_TOKENIZER_ID,
    load_lumina_autoencoder,
    load_lumina_gemma2,
    load_lumina_transformer,
)
from modules.model.LuminaModel import LuminaModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer


class LuminaModelLoader(HFModelLoaderMixin):
    def __init__(self):
        super().__init__()

    def __load_safetensors(
        self,
        model: LuminaModel,
        model_type: ModelType,
        weight_dtypes: ModelWeightDtypes,
        model_names: ModelNames,
        _quantization: QuantizationConfig,
    ):
        if not model_names.base_model:
            raise ValueError("Lumina base_model_name must point to the NextDiT safetensors checkpoint")
        if not model_names.text_encoder_model:
            raise ValueError("Lumina text_encoder.model_name must point to the Gemma2 safetensors checkpoint")
        if not model_names.vae_model:
            raise ValueError("Lumina vae.model_name must point to the AE safetensors checkpoint")

        tokenizer = AutoTokenizer.from_pretrained(GEMMA_TOKENIZER_ID)
        tokenizer.padding_side = "right"

        text_encoder = load_lumina_gemma2(
            model_names.text_encoder_model,
            weight_dtypes.text_encoder.torch_dtype(),
            "cpu",
        )
        vae = load_lumina_autoencoder(
            model_names.vae_model,
            weight_dtypes.vae.torch_dtype(),
            "cpu",
        )
        transformer = load_lumina_transformer(
            model_names.base_model,
            weight_dtypes.transformer.torch_dtype(),
            "cpu",
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=6.0)

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer
        model.noise_scheduler = noise_scheduler

    def load(
        self,
        model: LuminaModel,
        model_type: ModelType,
        model_names: ModelNames,
        weight_dtypes: ModelWeightDtypes,
        quantization: QuantizationConfig,
    ) -> LuminaModel | None:
        stacktraces = []

        try:
            self.__load_safetensors(model, model_type, weight_dtypes, model_names, quantization)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load Lumina model: " + model_names.base_model)
