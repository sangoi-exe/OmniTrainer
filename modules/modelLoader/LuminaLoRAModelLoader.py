from modules.model.LuminaModel import LuminaModel
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.lumina.LuminaLoRALoader import LuminaLoRALoader
from modules.modelLoader.lumina.LuminaModelLoader import LuminaModelLoader
from modules.util.enum.ModelType import ModelType

LuminaLoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.LUMINA_2: "resources/sd_model_spec/lumina_2-lora.json"},
    model_class=LuminaModel,
    model_loader_class=LuminaModelLoader,
    embedding_loader_class=None,
    lora_loader_class=LuminaLoRALoader,
)
