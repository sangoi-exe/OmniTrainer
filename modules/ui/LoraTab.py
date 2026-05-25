from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import PeftType
from modules.util.ui import components
from modules.util.ui.UIState import UIState
from modules.util.ui.validation_helpers import check_range

import customtkinter as ctk


class LoraTab:
    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__()

        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state

        self.scroll_frame = None
        self.options_frame = None

        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_frame:
            self.scroll_frame.destroy()
        self.scroll_frame = ctk.CTkFrame(self.master, fg_color="transparent")
        self.scroll_frame.grid(row=0, column=0, sticky="nsew")

        self.scroll_frame.grid_columnconfigure(0, weight=0)
        self.scroll_frame.grid_columnconfigure(1, weight=1)
        self.scroll_frame.grid_columnconfigure(2, weight=2)

        components.label(self.scroll_frame, 0, 0, "Type", tooltip="The type of low-parameter finetuning method.")
        # This will instantly call self.setup_lora.
        components.options_kv(
            self.scroll_frame,
            0,
            1,
            [
                ("LoRA", PeftType.LORA),
                ("LoHa", PeftType.LOHA),
                ("OFT v2", PeftType.OFT_2),
                ("LoKr", PeftType.LOKR),
            ],
            self.ui_state,
            "peft_type",
            command=self.setup_lora,
        )

    def setup_lora(self, peft_type: PeftType):
        if peft_type == PeftType.LOHA:
            name = "LoHa"
        elif peft_type == PeftType.OFT_2:
            name = "OFT v2"
        elif peft_type == PeftType.LOKR:
            name = "LoKr"
        else:
            name = "LoRA"

        if self.options_frame:
            self.options_frame.destroy()
        self.options_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        self.options_frame.grid(row=1, column=0, columnspan=3, sticky="nsew")
        master = self.options_frame

        master.grid_columnconfigure(0, weight=0, uniform="a")
        master.grid_columnconfigure(1, weight=1, uniform="a")
        master.grid_columnconfigure(2, minsize=50, uniform="a")
        master.grid_columnconfigure(3, weight=0, uniform="a")
        master.grid_columnconfigure(4, weight=1, uniform="a")

        # lora model name
        components.label(
            master,
            0,
            0,
            f"{name} base model",
            tooltip=f"The base {name} to train on. Leave empty to create a new {name}",
        )
        entry = components.path_entry(
            master, 0, 1, self.ui_state, "lora_model_name", mode="file", path_modifier=components.json_path_modifier
        )
        entry.grid(row=0, column=1, columnspan=4)

        # LoRA decomposition
        if peft_type == PeftType.LORA:
            components.label(master, 1, 3, "Decompose Weights (DoRA)", tooltip="Decompose LoRA Weights (aka, DoRA).")
            components.switch(master, 1, 4, self.ui_state, "lora_decompose")

            components.label(
                master,
                2,
                3,
                "Use Norm Epsilon (DoRA Only)",
                tooltip="Add an epsilon to the norm divison calculation in DoRA. Can aid in training stability, and also acts as regularization.",
            )
            components.switch(master, 2, 4, self.ui_state, "lora_decompose_norm_epsilon")
            components.label(
                master,
                3,
                3,
                "Apply on output axis (DoRA Only)",
                tooltip="Apply the weight decomposition on the output axis instead of the input axis.",
            )
            components.switch(master, 3, 4, self.ui_state, "lora_decompose_output_axis")

        # LoRA and LoHA shared settings
        if peft_type == PeftType.LORA or peft_type == PeftType.LOHA:
            # rank
            components.label(
                master, 1, 0, f"{name} rank", tooltip=f"The rank parameter used when creating a new {name}"
            )
            components.entry(
                master,
                1,
                1,
                self.ui_state,
                "lora_rank",
                required=True,
                extra_validate=check_range(lower=1, message="Rank must be at least 1"),
            )

            # alpha
            components.label(
                master, 2, 0, f"{name} alpha", tooltip=f"The alpha parameter used when creating a new {name}"
            )
            components.entry(master, 2, 1, self.ui_state, "lora_alpha", required=True)

            # Dropout Percentage
            components.label(
                master,
                3,
                0,
                "Dropout Probability",
                tooltip="Dropout probability. This percentage of model nodes will be randomly ignored at each training step. Helps with overfitting. 0 disables, 1 maximum.",
            )
            components.entry(master, 3, 1, self.ui_state, "dropout_probability")

            # weight dtype
            components.label(
                master,
                4,
                0,
                f"{name} Weight Data Type",
                tooltip=f"The {name} weight data type used for training. This can reduce memory consumption, but reduces precision",
            )
            components.options_kv(
                master,
                4,
                1,
                [
                    ("float32", DataType.FLOAT_32),
                    ("bfloat16", DataType.BFLOAT_16),
                ],
                self.ui_state,
                "lora_weight_dtype",
            )

            # For use with additional embeddings.
            components.label(
                master,
                5,
                0,
                "Bundle Embeddings",
                tooltip=f"Bundles any additional embeddings into the {name} output file, rather than as separate files",
            )
            components.switch(master, 5, 1, self.ui_state, "bundle_additional_embeddings")

            if peft_type == PeftType.LORA:
                components.label(
                    master,
                    6,
                    0,
                    "Text Encoder Scale",
                    tooltip="Scale applied to loaded text encoder LoRA weights. 1.0 keeps weights unchanged.",
                )
                components.entry(master, 6, 1, self.ui_state, "lora_te_scale", required=True)

                components.label(
                    master,
                    7,
                    0,
                    "UNet Scale",
                    tooltip="Scale applied to loaded UNet LoRA weights. 1.0 keeps weights unchanged.",
                )
                components.entry(master, 7, 1, self.ui_state, "lora_unet_scale", required=True)

        # OFTv2
        elif peft_type == PeftType.OFT_2:
            # Block Size
            components.label(
                master, 1, 0, f"{name} Block Size", tooltip=f"The block size parameter used when creating a new {name}"
            )
            components.entry(master, 1, 1, self.ui_state, "oft_block_size", required=True)

            # Block Share
            components.label(
                master,
                1,
                3,
                "Block Share",
                tooltip="Share the OFT parameters between blocks. A single rotation matrix is shared across all blocks within a layer, drastically cutting the number of trainable parameters and yielding very compact adapter files, potentially improving generalization but at the cost of significant expressiveness, which can lead to underfitting on more complex or diverse tasks.",
            )
            components.switch(master, 1, 4, self.ui_state, "oft_block_share")

            components.label(
                master,
                2,
                3,
                "Constrained OFT",
                tooltip="Constrains learned rotations near the identity matrix for smaller, more stable updates.",
            )
            components.switch(master, 2, 4, self.ui_state, "oft_coft")

            components.label(
                master,
                3,
                3,
                "COFT Epsilon",
                tooltip="Constraint strength for Constrained OFT.",
            )
            components.entry(master, 3, 4, self.ui_state, "coft_eps", required=True)

            components.label(
                master,
                4,
                3,
                "Scaled OFT",
                tooltip="Scales OFT weights so block-size changes do not strongly change the effective learning rate.",
            )
            components.switch(master, 4, 4, self.ui_state, "scaled_oft")

            components.label(
                master,
                5,
                3,
                "DoRA OFT",
                tooltip="Learns an OFT magnitude vector in addition to the rotation.",
            )
            components.switch(master, 5, 4, self.ui_state, "dora_oft")

            # Dropout Percentage
            components.label(
                master,
                2,
                0,
                "Dropout Probability",
                tooltip="Dropout probability. This percentage of the rotated adapter nodes that will be randomly restored to the base model initial statue. Helps with overfitting. 0 disables, 1 maximum.",
            )
            components.entry(master, 2, 1, self.ui_state, "dropout_probability")

            # OFT weight dtype
            components.label(
                master,
                3,
                0,
                f"{name} Weight Data Type",
                tooltip=f"The {name} weight data type used for training. This can reduce memory consumption, but reduces precision",
            )
            components.options_kv(
                master,
                3,
                1,
                [
                    ("float32", DataType.FLOAT_32),
                    ("bfloat16", DataType.BFLOAT_16),
                ],
                self.ui_state,
                "lora_weight_dtype",
            )

            # For use with additional embeddings.
            components.label(
                master,
                4,
                0,
                "Bundle Embeddings",
                tooltip=f"Bundles any additional embeddings into the {name} output file, rather than as separate files",
            )
            components.switch(master, 4, 1, self.ui_state, "bundle_additional_embeddings")

        elif peft_type == PeftType.LOKR:
            components.label(
                master, 1, 0, f"{name} dimension", tooltip="Dimension used for the secondary decomposition."
            )
            components.entry(
                master,
                1,
                1,
                self.ui_state,
                "lokr_dim",
                required=True,
                extra_validate=check_range(lower=1, message="LoKr dimension must be at least 1"),
            )

            components.label(
                master,
                2,
                0,
                "Decomposition Factor",
                tooltip="-1 uses automatic factorization. Positive values force that factor when divisible.",
            )
            components.entry(master, 2, 1, self.ui_state, "lokr_decompose_factor", required=True)

            components.label(master, 3, 0, f"{name} alpha", tooltip=f"The alpha parameter used when creating a new {name}")
            components.entry(master, 3, 1, self.ui_state, "lora_alpha", required=True)

            components.label(
                master,
                4,
                0,
                "Dropout Probability",
                tooltip="Dropout probability. This percentage of adapter nodes will be randomly ignored at each training step.",
            )
            components.entry(master, 4, 1, self.ui_state, "dropout_probability")

            components.label(
                master,
                5,
                0,
                f"{name} Weight Data Type",
                tooltip=f"The {name} weight data type used for training.",
            )
            components.options_kv(
                master,
                5,
                1,
                [
                    ("float32", DataType.FLOAT_32),
                    ("bfloat16", DataType.BFLOAT_16),
                ],
                self.ui_state,
                "lora_weight_dtype",
            )

            components.label(
                master,
                6,
                0,
                "Kronecker-Vec Trick",
                tooltip="Uses a faster Linear forward path without materializing the full Kronecker product.",
            )
            components.switch(master, 6, 1, self.ui_state, "lokr_vec_trick")

            components.label(
                master,
                7,
                0,
                "Bundle Embeddings",
                tooltip=f"Bundles any additional embeddings into the {name} output file, rather than as separate files",
            )
            components.switch(master, 7, 1, self.ui_state, "bundle_additional_embeddings")

            components.label(
                master,
                1,
                3,
                "Decompose Both Matrices",
                tooltip="Rank-decomposes both Kronecker product matrices.",
            )
            components.switch(master, 1, 4, self.ui_state, "lokr_decompose_both")

            components.label(
                master,
                2,
                3,
                "Use Tucker Decomposition",
                tooltip="Uses Tucker decomposition for convolutional LoKr layers.",
            )
            components.switch(master, 2, 4, self.ui_state, "lokr_use_tucker")

            components.label(
                master,
                3,
                3,
                "Force Full Matrix",
                tooltip="Forces the second Kronecker matrix to use full matrix mode.",
            )
            components.switch(master, 3, 4, self.ui_state, "lokr_full_matrix")

            components.label(
                master,
                4,
                3,
                "Decompose Weights",
                tooltip="Applies DoRA-style weight decomposition to LoKr.",
            )
            components.switch(master, 4, 4, self.ui_state, "lokr_weight_decompose")

            components.label(
                master,
                5,
                3,
                "DoRA Output Axis",
                tooltip="Applies LoKr DoRA decomposition on the output axis instead of the input axis.",
            )
            components.switch(master, 5, 4, self.ui_state, "lokr_dora_on_output")
