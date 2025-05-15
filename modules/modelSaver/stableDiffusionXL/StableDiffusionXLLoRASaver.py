import os.path
from pathlib import Path

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.sangoi.logFun import logFun
from modules.util.enum.ModelFormat import ModelFormat

import torch
from torch import Tensor

from safetensors.torch import save_file


class StableDiffusionXLLoRASaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def __get_state_dict(
            self,
            model: StableDiffusionXLModel,
    ) -> dict[str, Tensor]:
        logFun("[LoRA Saver] Iniciando __get_state_dict...", lvl="DEBUG_SAVER") # Log de início
        state_dict = {}

        # 1. Text Encoders (sem tau)
        if model.text_encoder_1_lora is not None:
            te1_lora_sd = model.text_encoder_1_lora.state_dict()
            logFun(f"[LoRA Saver] Coletado {len(te1_lora_sd)} params de text_encoder_1_lora.", lvl="DEBUG_SAVER")
            state_dict.update(te1_lora_sd)
        else:
            logFun("[LoRA Saver] model.text_encoder_1_lora é None.", lvl="DEBUG_SAVER")

        if model.text_encoder_2_lora is not None:
            te2_lora_sd = model.text_encoder_2_lora.state_dict()
            logFun(f"[LoRA Saver] Coletado {len(te2_lora_sd)} params de text_encoder_2_lora.", lvl="DEBUG_SAVER")
            state_dict.update(te2_lora_sd)
        else:
            logFun("[LoRA Saver] model.text_encoder_2_lora é None.", lvl="DEBUG_SAVER")
        
        # 2. UNet LoRA (pesos A e B)
        if model.unet_lora is not None:
            unet_lora_sd = model.unet_lora.state_dict()
            logFun(f"[LoRA Saver] Coletado {len(unet_lora_sd)} params de unet_lora (pesos A/B).", lvl="DEBUG_SAVER")
            state_dict.update(unet_lora_sd)
        else:
            logFun("[LoRA Saver] model.unet_lora é None.", lvl="DEBUG_SAVER")
        
        # 3. Parâmetros 'log_tau' da UNet principal
        logFun("[LoRA Saver] Verificando model.unet.tau_procs...", lvl="DEBUG_SAVER")
        if hasattr(model.unet, 'tau_procs') and model.unet.tau_procs:
            logFun(f"[LoRA Saver] Encontrado model.unet.tau_procs com {len(model.unet.tau_procs)} processadores.", lvl="DEBUG_SAVER")
            taus_collected_count = 0
            for proc_key, proc_module in model.unet.tau_procs.items():
                state_dict_key_for_tau = f"unet.tau_procs.{proc_key}.log_tau"
                state_dict[state_dict_key_for_tau] = proc_module.log_tau.data 
                # logFun(f"[LoRA Saver]   Coletado tau para '{proc_key}' como '{state_dict_key_for_tau}'. Valor: {proc_module.log_tau.data.item():.4f}", lvl="DEBUG_SAVER")
                taus_collected_count += 1
            logFun(f"[LoRA Saver] Total de {taus_collected_count} parâmetros 'log_tau' coletados da UNet.", lvl="DEBUG_SAVER")
        else:
            logFun("[LoRA Saver] Atributo 'model.unet.tau_procs' não encontrado ou está vazio.", lvl="DEBUG_SAVER")
        
        # 4. model.lora_state_dict geral
        if model.lora_state_dict is not None:
            logFun(f"[LoRA Saver] Coletando {len(model.lora_state_dict)} params de model.lora_state_dict.", lvl="DEBUG_SAVER")
            state_dict.update(model.lora_state_dict)
        else:
            logFun("[LoRA Saver] model.lora_state_dict é None.", lvl="DEBUG_SAVER")

        # 5. Embeddings adicionais
        if model.additional_embeddings and model.train_config.bundle_additional_embeddings:
            logFun(f"[LoRA Saver] Coletando {len(model.additional_embeddings)} embeddings adicionais...", lvl="DEBUG_SAVER")
            emb_count = 0
            for embedding in model.additional_embeddings:
                placeholder = embedding.text_encoder_1_embedding.placeholder
                if embedding.text_encoder_1_embedding.vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.clip_l"] = embedding.text_encoder_1_embedding.vector
                    emb_count+=1
                if embedding.text_encoder_2_embedding.vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.clip_g"] = embedding.text_encoder_2_embedding.vector
                    emb_count+=1
            logFun(f"[LoRA Saver] Coletado {emb_count} tensores de embeddings adicionais.", lvl="DEBUG_SAVER")
        else:
            logFun("[LoRA Saver] Sem embeddings adicionais para coletar ou bundle_additional_embeddings=False.", lvl="DEBUG_SAVER")

        logFun(f"[LoRA Saver] __get_state_dict finalizado. Total de {len(state_dict)} chaves no state_dict.", lvl="DEBUG_SAVER")
        return state_dict

    def __save_ckpt(
            self,
            model: StableDiffusionXLModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = self.__get_state_dict(model)
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        torch.save(save_state_dict, destination)

    def __save_safetensors(
            self,
            model: StableDiffusionXLModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = self.__get_state_dict(model)
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def __save_internal(
            self,
            model: StableDiffusionXLModel,
            destination: str,
    ):
        os.makedirs(destination, exist_ok=True)

        self.__save_safetensors(model, os.path.join(destination, "lora", "lora.safetensors"), None)

    def save(
            self,
            model: StableDiffusionXLModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                raise NotImplementedError
            case ModelFormat.CKPT:
                self.__save_ckpt(model, output_model_destination, dtype)
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
