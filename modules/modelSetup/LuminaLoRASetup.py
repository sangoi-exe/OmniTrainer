from modules.model.LuminaModel import LuminaModel
from modules.modelSetup.BaseLuminaSetup import BaseLuminaSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.torch_util import state_dict_has_prefix
from modules.util.TrainProgress import TrainProgress

import torch

LUMINA_TRANSFORMER_LORA_TARGET_FILTERS = ["context_refiner", "noise_refiner", "layers", "final_layer"]
LUMINA_TEXT_ENCODER_LORA_TARGET_FILTERS = ["self_attn", "mlp"]
LUMINA_TRANSFORMER_LORA_TARGET_CLASSES = ["JointTransformerBlock", "FinalLayer"]
LUMINA_TEXT_ENCODER_LORA_TARGET_CLASSES = [
    "Gemma2Attention",
    "Gemma2FlashAttention2",
    "Gemma2SdpaAttention",
    "Gemma2MLP",
]


class LuminaLoRASetup(BaseLuminaSetup):
    def __init__(self, train_device: torch.device, temp_device: torch.device, debug_mode: bool):
        super().__init__(train_device=train_device, temp_device=temp_device, debug_mode=debug_mode)

    def create_parameters(self, model: LuminaModel, config: TrainConfig) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()
        self._create_model_part_parameters(
            parameter_group_collection, "text_encoder_lora", model.text_encoder_lora, config.text_encoder
        )
        self._create_model_part_parameters(
            parameter_group_collection, "transformer_lora", model.transformer_lora, config.transformer
        )
        return parameter_group_collection

    def __setup_requires_grad(self, model: LuminaModel, config: TrainConfig):
        model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)

        self._setup_model_part_requires_grad(
            "text_encoder_lora", model.text_encoder_lora, config.text_encoder, model.train_progress
        )
        self._setup_model_part_requires_grad(
            "transformer_lora", model.transformer_lora, config.transformer, model.train_progress
        )

    def setup_model(self, model: LuminaModel, config: TrainConfig):
        if config.peft_type != PeftType.LORA:
            raise ValueError("Lumina currently supports only LoRA PEFT")
        if config.train_any_embedding() or config.train_any_output_embedding():
            raise ValueError("Lumina embedding training is not supported")

        create_te = config.text_encoder.train or state_dict_has_prefix(model.lora_state_dict, "lora_te")
        model.text_encoder_lora = (
            LoRAModuleWrapper(model.text_encoder, "lora_te", config, LUMINA_TEXT_ENCODER_LORA_TARGET_FILTERS)
            if create_te
            else None
        )
        model.transformer_lora = LoRAModuleWrapper(
            model.transformer,
            "lora_transformer",
            config,
            LUMINA_TRANSFORMER_LORA_TARGET_FILTERS,
        )

        if model.lora_state_dict:
            if model.text_encoder_lora is not None:
                model.text_encoder_lora.load_state_dict(model.lora_state_dict)
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        if model.text_encoder_lora is not None:
            model.text_encoder_lora.set_dropout(config.dropout_probability)
            model.text_encoder_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_lora.hook_to_module()

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)
        self._export_lora_key_manifest(model, config)

    def setup_train_device(self, model: LuminaModel, config: TrainConfig):
        vae_on_train_device = self.debug_mode or not config.latent_caching
        text_encoder_on_train_device = config.text_encoder.train or not config.latent_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        if config.text_encoder.train:
            model.text_encoder.train()
        else:
            model.text_encoder.eval()

        model.vae.eval()

        if config.transformer.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(self, model: LuminaModel, config: TrainConfig, train_progress: TrainProgress):
        self.__setup_requires_grad(model, config)


factory.register(BaseModelSetup, LuminaLoRASetup, ModelType.LUMINA_2, TrainingMethod.LORA)
