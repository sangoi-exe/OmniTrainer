# Selected Upstream PR Port Validation Artifact

Date: 2026-05-25
Branch: `master-update`
HEAD: `57d0a21f8e2b831d9e1158e9ce27bdca8bef97f3`

## Command

```bash
set -euo pipefail
export OT_PLATFORM_REQUIREMENTS=requirements-default.txt
export PYTHONPATH="$PWD"
export TQDM_DISABLE=1
./.uv/bin/uv run --python .venv/bin/python --no-sync python -m compileall -q modules scripts
./.uv/bin/uv run --python .venv/bin/python --no-sync python scripts/validate_selected_upstream_pr_ports.py \
    --check resume-actions \
    --check timed-actions \
    --check accumulator \
    --check accumulator-mid-step \
    --check tensorboard \
    --check patience \
    --check component-stop \
    --check validation-timesteps \
    --check strict-config \
    --check timestep-distributions \
    --check flow-timestep-rejection \
    --check immiscible \
    --check perturbations \
    --check flux2-edit \
    --check flux2-dropout-block \
    --check hidream-tokenizer \
    --check hunyuan-comfy \
    --check adapters \
    --check prefetch-cache \
    --check prefetch-device-boundary \
    --check attention-backend \
    --check sangoi-invariants \
    --check lora-manifest \
    --check source-guards
git diff --exit-code HEAD -- requirements.txt requirements-default.txt requirements-cuda.txt requirements-rocm.txt requirements-global.txt requirements-dev.txt pyproject.toml .python-version
git diff --exit-code HEAD -- lib.include.sh install.sh run-cmd.sh start-ui.sh update.sh .dockerignore scripts/train_remote.py modules/cloud resources/docker
unexpected_guarded_untracked="$(git ls-files --others --exclude-standard -- requirements.txt requirements-default.txt requirements-cuda.txt requirements-rocm.txt requirements-global.txt requirements-dev.txt pyproject.toml .python-version .gitmodules lib.include.sh install.sh run-cmd.sh start-ui.sh update.sh .dockerignore modules/ipex_to_cuda modules/cloud resources/docker || true)"
if [ -n "$unexpected_guarded_untracked" ]; then
    printf 'unexpected guarded untracked files:\n%s\n' "$unexpected_guarded_untracked" >&2
    exit 1
fi
git diff --check
git diff HEAD --check
echo VALIDATION_STATUS=0
```

## Output

```text
No module named 'triton', continuing without triton
ok resume-actions
ok timed-actions
ok accumulator
ok accumulator-mid-step
ok tensorboard
ok patience
ok component-stop
ok validation-timesteps
ok strict-config
ok timestep-distributions
ok flow-timestep-rejection
ok immiscible
ok perturbations
ok flux2-edit
ok flux2-dropout-block
ok hidream-tokenizer
ok hunyuan-comfy
ok adapters
ok prefetch-cache
ok prefetch-device-boundary
ok attention-backend
ok sangoi-invariants
ok lora-manifest
ok source-guards
VALIDATION_STATUS=0
```
