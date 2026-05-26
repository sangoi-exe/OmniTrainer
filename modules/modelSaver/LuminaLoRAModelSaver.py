from modules.model.LuminaModel import LuminaModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.lumina.LuminaLoRASaver import LuminaLoRASaver
from modules.util.enum.ModelType import ModelType

LuminaLoRAModelSaver = make_lora_model_saver(
    ModelType.LUMINA_2,
    model_class=LuminaModel,
    lora_saver_class=LuminaLoRASaver,
    embedding_saver_class=None,
)
