# OneTrainer Launch Scripts

## Mac And Linux Systems

### Scripts

- `install.sh`: installs repo-local `uv`, installs managed Python, creates `.venv`, and installs requirements.
- `update.sh`: pulls Git updates and refreshes the uv-managed Python environment.
- `start-ui.sh`: launches the main OneTrainer interface through the uv-managed environment.
- `run-cmd.sh`: executes a script from `scripts/` through the uv-managed environment.

### Environment Variables

- `OT_UV_VERSION`: uv version installed into `.uv/bin`. Defaults to `0.9.17`.
- `OT_PYTHON_VERSION`: managed Python version installed by uv. Defaults to `3.13`.
- `OT_LAZY_UPDATES`: if `true`, `update.sh` refreshes dependencies only when the repository commit changed since the previous dependency update. Defaults to `false`.
- `OT_CUDA_LOWMEM_MODE`: if `true`, sets aggressive PyTorch CUDA allocator settings for low-memory GPUs. Defaults to `false`.
- `OT_PLATFORM_REQUIREMENTS`: selects the platform requirements file. Defaults to `detect`, which picks NVIDIA, AMD/ROCm, or CPU requirements. Valid explicit values are `requirements-cuda.txt`, `requirements-rocm.txt`, and `requirements-default.txt`.
- `OT_SCRIPT_DEBUG`: if `true`, enables additional script debug logging. Defaults to `false`.

The old Conda/host-Python environment variables are no longer supported. If `OT_CONDA_CMD`, `OT_CONDA_ENV`, `OT_PYTHON_CMD`, `OT_PYTHON_VENV`, `OT_PREFER_VENV`, or `OT_CONDA_USE_PYTHON_VERSION` is set, the launch scripts fail immediately instead of silently ignoring it.

### Examples

```bash
env OT_CUDA_LOWMEM_MODE=true OT_PLATFORM_REQUIREMENTS=requirements-cuda.txt ./start-ui.sh
```

```bash
env OT_PYTHON_VERSION=3.13 OT_UV_VERSION=0.9.17 ./install.sh
```

### Runtime Layout

- `.uv/bin/uv`: repo-local uv executable.
- `.uv/python/`: uv-managed Python installs.
- `.uv/cache/`: repo-local uv cache unless `UV_CACHE_DIR` is already set.
- `.venv/`: fixed uv-created virtual environment used by all Unix launch scripts.

If the environment is corrupt or was created with the wrong Python version, delete `.venv` and run `./install.sh` again. Do not manually delete `.uv` while an install/update process is running.

### Running Custom Script Commands

Always use `run-cmd.sh` for OneTrainer CLI tasks. It validates the script name, prepares the uv-managed runtime, and executes the target script.

```bash
./run-cmd.sh train --config-path path/to/config.json
./run-cmd.sh train -h
```

The names of valid scripts are the Python files in `scripts/` without the `.py` suffix.

### Automation

When automating OneTrainer from your own shell scripts, call `run-cmd.sh` for CLI tasks and `start-ui.sh` for the UI. Use `set -e` in automation scripts so failures stop the workflow immediately.
