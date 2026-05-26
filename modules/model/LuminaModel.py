from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel
from modules.model.lumina.flux_autoencoder import LuminaAutoEncoder
from modules.model.lumina.lumina_models import NextDiT
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

from transformers import Gemma2Model, PreTrainedTokenizer


class LuminaModel(BaseModel):
    tokenizer: PreTrainedTokenizer | None
    text_encoder: Gemma2Model | None
    vae: LuminaAutoEncoder | None
    transformer: NextDiT | None

    text_encoder_autocast_context: torch.autocast | nullcontext
    vae_autocast_context: torch.autocast | nullcontext

    text_encoder_train_dtype: DataType
    vae_train_dtype: DataType

    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    text_encoder_lora: LoRAModuleWrapper | None
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(self, model_type: ModelType):
        super().__init__(model_type=model_type)

        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None

        self.text_encoder_autocast_context = nullcontext()
        self.vae_autocast_context = nullcontext()

        self.text_encoder_train_dtype = DataType.FLOAT_32
        self.vae_train_dtype = DataType.FLOAT_32

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None

        self.text_encoder_lora = None
        self.transformer_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [adapter for adapter in [self.text_encoder_lora, self.transformer_lora] if adapter is not None]

    @staticmethod
    def add_system_prompt(system_prompt: str | None, prompt: str, *, is_negative: bool = False) -> str:
        if is_negative or not system_prompt:
            return prompt
        return f"{system_prompt} <Prompt Start> {prompt}"

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        if (
            self.text_encoder_offload_conductor is not None
            and self.text_encoder_offload_conductor.layer_offload_activated()
        ):
            self.text_encoder_offload_conductor.to(device)
        else:
            self.text_encoder.to(device=device)

        if self.text_encoder_lora is not None:
            self.text_encoder_lora.to(device)

    def transformer_to(self, device: torch.device):
        if (
            self.transformer_offload_conductor is not None
            and self.transformer_offload_conductor.layer_offload_activated()
        ):
            self.transformer_offload_conductor.to(device)
        else:
            self.transformer.to(device=device)

        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)

    def eval(self):
        self.vae.eval()
        self.text_encoder.eval()
        self.transformer.eval()

    def encode_text(
        self,
        train_device: torch.device,
        batch_size: int = 1,
        rand: Random | None = None,
        text: str | None = None,
        tokens: Tensor | None = None,
        attention_mask: Tensor | None = None,
        max_length: int | None = None,
        system_prompt: str | None = None,
        is_negative: bool = False,
        text_encoder_dropout_probability: float | None = None,
        text_encoder_output: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if tokens is None and text is not None:
            tokenizer_output = self.tokenizer(
                self.add_system_prompt(system_prompt, text, is_negative=is_negative),
                max_length=256 if max_length is None else max_length,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                pad_to_multiple_of=8,
            )
            tokens = tokenizer_output.input_ids.to(self.text_encoder.device)
            attention_mask = tokenizer_output.attention_mask.to(self.text_encoder.device)

        if text_encoder_output is None:
            with self.text_encoder_autocast_context:
                output = self.text_encoder(
                    input_ids=tokens.to(self.text_encoder.device),
                    attention_mask=attention_mask.to(self.text_encoder.device),
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
            text_encoder_output = output.hidden_states[-2]

        if text_encoder_dropout_probability is not None:
            dropout_text_encoder_mask = torch.tensor(
                [rand.random() > text_encoder_dropout_probability for _ in range(batch_size)], device=train_device
            ).float()
            attention_mask = attention_mask * dropout_text_encoder_mask[:, None]
            text_encoder_output = text_encoder_output * dropout_text_encoder_mask[:, None, None]

        return text_encoder_output, attention_mask

    def encode_image(self, image: Tensor) -> Tensor:
        image = image.to(device=self.vae.device, dtype=self.vae_train_dtype.torch_dtype())
        with self.vae_autocast_context:
            return self.vae.encode_tensor(image.unsqueeze(0)).squeeze(dim=0)

    def decode_image(self, latent_image: Tensor) -> Tensor:
        latent_image = latent_image.to(device=self.vae.device, dtype=self.vae_train_dtype.torch_dtype())
        with self.vae_autocast_context:
            return self.vae.decode_tensor(latent_image.unsqueeze(0)).clamp(-1, 1).squeeze(dim=0)
