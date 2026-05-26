# Lumina 2 Training

Lumina 2 is available as a native model family with LoRA training.

## Required Model Paths

Select `Lumina 2` as the model type and set these paths in the model tab:

- Base model: Lumina NextDiT `.safetensors` checkpoint.
- Text Encoder: Gemma2 `.safetensors` checkpoint.
- Lumina AE: Lumina autoencoder `.safetensors` checkpoint.

The Lumina loader requires these three separate checkpoints. It does not infer missing Gemma2 or autoencoder weights
from the base model path. Directories, Hugging Face repository IDs, GGUF files, and non-`.safetensors` extensions are
not accepted for Lumina.

## Training Settings

The Lumina UI exposes LoRA as the training method. Use the text encoder sequence length field for Gemma2 token length;
the Lumina model-type default is `256`.

Lumina uses flow-matching training with raw flow prediction. The Lumina defaults set:

- Timestep distribution: `NEXTDIT_SHIFT`
- Timestep shift: `6.0`
- Noising weight: `0.0`
- Noising bias: `0.0`

The model tab also exposes `Lumina System Prompt`. When set, this text is prepended to positive training prompts before
the `<Prompt Start>` marker. Negative prompts are left unchanged.

## Caching

Latent caching encodes images with the Lumina autoencoder and uses a bucket quantization of `16`. Text caching stores
Gemma2 token IDs, token masks, and hidden states from the second-to-last hidden-state layer.
