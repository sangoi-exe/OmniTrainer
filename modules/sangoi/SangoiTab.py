from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.PathIOType import PathIOType
from modules.util.enum.TrainGPSPenaltyMetric import TrainGPSPenaltyMetric
from modules.util.ui import components
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class SangoiTab:
    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state

        self.refresh_ui()

    def refresh_ui(self):
        if hasattr(self, "main_frame") and self.main_frame:
            self.main_frame.destroy()

        self.main_frame = ctk.CTkScrollableFrame(self.master, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True)

        self.main_frame.grid_columnconfigure(0, weight=0)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(2, weight=0)
        self.main_frame.grid_columnconfigure(3, weight=1)
        self.main_frame.grid_columnconfigure(4, weight=0)
        self.main_frame.grid_columnconfigure(5, weight=1)

        row = 0

        components.label(
            self.main_frame,
            row,
            0,
            "LoRA Key Export Path",
            tooltip="Optional JSON manifest path for LoRA keys. Relative paths are written under the workspace directory.",
        )
        components.path_entry(
            self.main_frame,
            row,
            1,
            self.ui_state,
            "lora_key_export_path",
            mode="file",
            io_type=PathIOType.OUTPUT,
        )

        components.label(self.main_frame, row, 2, "Data Recorder", tooltip="Record training statistics.")
        components.switch(self.main_frame, row, 3, self.ui_state, "data_recorder")
        row += 1

        components.label(
            self.main_frame, row, 0, "Use TrainGPS", tooltip="Apply a saved TrainGPS delta pattern during training."
        )
        components.switch(self.main_frame, row, 1, self.ui_state, "train_gps_use_it")

        components.label(
            self.main_frame, row, 2, "Save TrainGPS", tooltip="Save TrainGPS delta information after training."
        )
        components.switch(self.main_frame, row, 3, self.ui_state, "train_gps_save_it")
        row += 1

        components.label(
            self.main_frame, row, 0, "TrainGPS Weight", tooltip="Multiplier for the TrainGPS delta penalty."
        )
        components.entry(self.main_frame, row, 1, self.ui_state, "train_gps_weight")

        components.label(self.main_frame, row, 2, "TrainGPS Path", tooltip="Path to a TrainGPS delta file.")
        components.path_entry(self.main_frame, row, 3, self.ui_state, "train_gps_path", mode="file")
        row += 1

        components.label(
            self.main_frame,
            row,
            0,
            "TrainGPS Metric",
            tooltip="Penalty metric for TrainGPS. COSINE requires at least two common reference groups.",
        )
        components.options(
            self.main_frame,
            row,
            1,
            [str(x) for x in list(TrainGPSPenaltyMetric)],
            self.ui_state,
            "train_gps_penalty_metric",
        )
        row += 1

        components.label(
            self.main_frame,
            row,
            0,
            "Layer Blacklist",
            tooltip="Comma-separated list of LoRA layer filters to exclude from training.",
        )
        blacklist_entry = components.entry(
            self.main_frame,
            row,
            1,
            self.ui_state,
            "lora_layers_blacklist",
            tooltip="Comma-separated list of layer filters to exclude from training. Example: down_blocks.0,mid_block",
        )
        blacklist_entry.grid(row=row, column=1, columnspan=5, sticky="ew")
        row += 1

        components.label(
            self.main_frame,
            row,
            0,
            "Grad Ckpt Layers",
            tooltip="Comma-separated layer filters for gradient checkpointing.",
        )
        gradient_checkpointing_entry = components.entry(
            self.main_frame,
            row,
            1,
            self.ui_state,
            "gradient_checkpointing_layers",
            tooltip="Comma-separated layer filters to checkpoint. Empty checkpoints every eligible layer.",
        )
        gradient_checkpointing_entry.grid(row=row, column=1, columnspan=5, sticky="ew")
