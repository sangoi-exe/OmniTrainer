# OneTrainerSangoi Instructions

Scope: `/home/lucas/work/OneTrainerSangoi` unless a deeper `AGENTS.md` overrides it.

## Purpose

This checkout is the Sangoi-modified OneTrainer workspace. Prefer preserving the local SDXL training behavior while integrating upstream OneTrainer changes cleanly.

## Local Toolchain Law

Repository validation commands must use this checkout's local uv-managed Python toolchain.

- Python commands must run through local uv: `./.uv/bin/uv run --python .venv/bin/python --no-sync ...`.
- Tool-only validation that does not import project dependencies may use repo-local uv tools: `./.uv/bin/uv tool run --python 3.13 --from <tool> ...`. Tool installs must stay under `.uv/tools` and `.uv/tool-bin`.
- Do not use system/global `python`, `pip`, or `uv` for repository tests or validation.
- If `./.uv/bin/uv` or `.venv/bin/python` is missing, stop and report the missing local toolchain path unless the user explicitly requests non-local execution for that command.
- `.venv` is the fixed local Python environment path for this checkout; do not use `OT_PYTHON_VENV` or alternate venv paths for validation.
- Keep `PYTHONPATH="$PWD"` for direct Python module checks from the repository root.
- For local WSL validation, set `OT_PLATFORM_REQUIREMENTS=requirements-default.txt` before any command that sources `lib.include.sh` or refreshes dependencies.

## WSL Validation

This WSL checkout's torch environment is CPU-only for local validation.

- Keep lightweight smoke checks CPU-only.
- Do not run CUDA training, CUDA compile checks, or GPU-dependent validation from this checkout unless the user confirms a GPU-capable environment.
- A CUDA-unavailable result in this WSL checkout is an environment limit, not proof that the repository change is broken.
- Do not let WSL `nvidia-smi` detection switch local validation to CUDA requirements unless the user explicitly asks for that.

## Key Paths

- Bootstrap owner: `lib.include.sh`, `install.sh`, `run-cmd.sh`, `start-ui.sh`, `update.sh`, `.dockerignore`.
- Dependency owners: `requirements.txt`, `requirements-global.txt`, `requirements-cuda.txt`, `requirements-rocm.txt`, `requirements-default.txt`, `pyproject.toml`, `.python-version`.
- Sangoi training mods: `modules/sangoi/`, `modules/modelSetup/StableDiffusionXLLoRASetup.py`, `modules/modelSetup/mixin/`, `modules/trainer/GenericTrainer.py`.
- Tk UI config surfaces: `modules/ui/`, `modules/sangoi/SangoiTab.py`, `modules/util/config/TrainConfig.py`.
- Task context: `.sangoi/`.

## Durable Notes

- Do not reintroduce compatibility aliases or stale config fields for removed Sangoi options.
- Keep documentation and examples describing current behavior only.
- `training_presets/*` may be allowed to return to upstream behavior for the current master-update cleanup work.

Last reviewed: 2026-05-24.
