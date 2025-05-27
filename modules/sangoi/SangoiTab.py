# Sessão V by user - Reorganização da SangoiTab
from pathlib import Path

import customtkinter as ctk

from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class SangoiTab:
    """
    A aba para configurações personalizadas do usuário (Sangoi),
    incluindo Delta Pattern e configurações de Camadas.
    """

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state

        self.refresh_ui()

    def refresh_ui(self):
        """
        Cria e atualiza os elementos da interface do usuário para a aba Sangoi.
        """
        if hasattr(self, 'main_frame') and self.main_frame:
            self.main_frame.destroy()

        self.main_frame = ctk.CTkScrollableFrame(self.master, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True)

        # Configuração do grid para o frame principal da aba:
        # 6 colunas para acomodar até 3 "slots" de (Label + Widget) por linha.
        # Slot 1: col 0 (Label), col 1 (Widget)
        # Slot 2: col 2 (Label), col 3 (Widget)
        # Slot 3: col 4 (Label), col 5 (Widget)
        self.main_frame.grid_columnconfigure(0, weight=0) 
        self.main_frame.grid_columnconfigure(1, weight=1) 
        self.main_frame.grid_columnconfigure(2, weight=0) 
        self.main_frame.grid_columnconfigure(3, weight=1) 
        self.main_frame.grid_columnconfigure(4, weight=0) 
        self.main_frame.grid_columnconfigure(5, weight=1)

        row_index = 0

        # --- Row 1: Generate Lora Layers List & Use Long Prompts ---
        # Slot 1
        components.label(self.main_frame, row_index, 0, "Generate Lora Layers List",
                         tooltip="Generate a txt with lora layers that is going to be trained")
        components.switch(self.main_frame, row_index, 1, self.ui_state, "gen_lora_keys")
        
        # Slot 2
        components.label(self.main_frame, row_index, 2, "Use Long Prompts",
                         tooltip="Process more than 77 tokens")
        components.switch(self.main_frame, row_index, 3, self.ui_state, "enable_long_prompts")

        # Slot 3 (cols 4, 5) fica em branco
        components.label(self.main_frame, row_index, 4, "Sangoi Debug",
                         tooltip="Debug pros meus badalhos")
        components.switch(self.main_frame, row_index, 5, self.ui_state, "debugoi")
        row_index += 1

        # Slot 3 (cols 4, 5) fica em branco
        components.label(self.main_frame, row_index, 0, "Full VAE",
                         tooltip="Use full vae on sangoi loss fun, WILL FALLBACK TO SYSTEM RAM (6gb for VAE)")
        components.switch(self.main_frame, row_index, 1, self.ui_state, "full_vae_mf")
        row_index += 1
        
        # --- Row 2: Use Delta Pattern & Save Delta Pattern ---
        # Slot 1
        components.label(self.main_frame, row_index, 0, "Use Delta Pattern",
                         tooltip="Apply a pre-defined delta pattern during training.")
        components.switch(self.main_frame, row_index, 1, self.ui_state, "train_gps_use_it")

        # Slot 2
        components.label(self.main_frame, row_index, 2, "Save Delta Pattern",
                         tooltip="Save the calculated delta pattern after training.")
        components.switch(self.main_frame, row_index, 3, self.ui_state, "train_gps_save_it")
        # Slot 3 (cols 4, 5) fica em branco
        row_index += 1

        # --- Row 3: Delta Pattern Weight & Delta Pattern Path ---
        # Slot 1
        components.label(self.main_frame, row_index, 0, "Delta Pattern Weight",
                         tooltip="Weight multiplier for the delta pattern application.")
        components.entry(self.main_frame, row_index, 1, self.ui_state, "train_gps_weight")

        # Slot 2 (para Delta Pattern Path, o entry e seu botão ocuparão cols 3 e 4)
        components.label(self.main_frame, row_index, 2, "Delta Pattern Path",
                         tooltip="Path to the delta pattern file (.pt, .safetensors). Leave empty if not using.")
        # components.file_entry coloca seu CTkEntry em (row, col=3) e o botão em (row, col=3+1=4)
        components.file_entry(self.main_frame, row_index, 3, self.ui_state, "train_gps_path")
        # Slot 3 (col 5) fica em branco
        row_index += 1

        # --- Row 4: Layer Blacklist ---
        components.label(self.main_frame, row_index, 0, "Layer Blacklist",
                         tooltip="Comma-separated list of layers to exclude from training (blacklist).")
        blacklist_entry = components.entry(
            self.main_frame, row_index, 1, self.ui_state, "lora_layers_blacklist",
            tooltip="Comma-separated list of layers to exclude from training. Example: 'down_blocks.0,mid_block'"
        )
        # Entry ocupa as colunas 1, 2, 3, 4, 5
        blacklist_entry.grid(row=row_index, column=1, columnspan=5, sticky="ew") 
        row_index += 1

        # --- Row 5: Grad Ckpt Layers ---
        components.label(self.main_frame, row_index, 0, "Grad Ckpt Layers",
                         tooltip="Comma-separated list of layer patterns to apply gradient checkpointing to. Leave empty for default behavior (usually ON for specific blocks).")
        gradient_ckpt_entry = components.entry(
            self.main_frame, row_index, 1, self.ui_state, "gradient_checkpointing_layers",
            tooltip="Define specific layers for gradient checkpointing. Example: 'mid_block,up_blocks.1'"
        )
        # Entry ocupa as colunas 1, 2, 3, 4, 5
        gradient_ckpt_entry.grid(row=row_index, column=1, columnspan=5, sticky="ew")
        row_index += 1

        # Seção DCoef Pattern foi removida conforme solicitado.