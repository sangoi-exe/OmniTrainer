from modules.model.BaseModel import BaseModel
from modules.model.HunyuanVideoModel import HunyuanVideoModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.convert.lora.convert_hunyuan_video_lora import convert_hunyuan_video_lora_key_sets
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.ModelNames import ModelNames


class HunyuanVideoLoRALoader(LoRALoaderMixin):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        return convert_hunyuan_video_lora_key_sets()

    def _preprocess_state_dict(self, state_dict: dict) -> dict:
        if not any(key.startswith("transformer.") for key in state_dict):
            return state_dict

        processed_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("transformer."):
                key = key.replace(".guidance_in.in_layer.", ".guidance_in.mlp.0.")
                key = key.replace(".guidance_in.out_layer.", ".guidance_in.mlp.2.")
                key = key.replace(".time_in.in_layer.", ".time_in.mlp.0.")
                key = key.replace(".time_in.out_layer.", ".time_in.mlp.2.")
                key = key.replace(".txt_in.c_embedder.in_layer.", ".txt_in.c_embedder.linear_1.")
                key = key.replace(".txt_in.c_embedder.out_layer.", ".txt_in.c_embedder.linear_2.")
                key = key.replace(".txt_in.t_embedder.in_layer.", ".txt_in.t_embedder.linear_1.")
                key = key.replace(".txt_in.t_embedder.out_layer.", ".txt_in.t_embedder.linear_2.")
                key = key.replace(".mlp.fc1.", ".mlp.0.")
                key = key.replace(".mlp.fc2.", ".mlp.2.")
                key = key.replace(".fc1.", ".fc0.")
                processed_state_dict["lora_transformer." + key[len("transformer.") :]] = value
            elif key.startswith("lora_te1_"):
                processed_state_dict["lora_te2_" + key[len("lora_te1_") :]] = value
            elif key.startswith("lora_llama_"):
                processed_state_dict["lora_te1_" + key[len("lora_llama_") :]] = value
            else:
                processed_state_dict[key] = value

        return processed_state_dict

    def load(
        self,
        model: HunyuanVideoModel,
        model_names: ModelNames,
    ):
        return self._load(model, model_names)
