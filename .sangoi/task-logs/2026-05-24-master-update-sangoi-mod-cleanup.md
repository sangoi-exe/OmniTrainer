# Master Update Sangoi Mod Cleanup Task Log

Date: 2026-05-24
Branch: `master-update`
Plan: `.sangoi/plans/2026-05-24-master-update-sangoi-mod-cleanup.md`

## Objective

Clean up handmade Sangoi modifications after the upstream OneTrainer merge while preserving supported SDXL training behavior and replacing stale bootstrap/tooling surfaces with repo-local uv-managed Python.

## Completed Work

- Added root `AGENTS.md` with local uv, fixed `.venv`, WSL CPU-only validation, and key owner-path instructions.
- Reworked POSIX bootstrap around repo-local `.uv/bin/uv`, uv-managed Python 3.13, fixed `.venv`, `.python-version`, local uv tool dirs, and fail-loud removed Conda/host-Python variables.
- Added `.dockerignore` for current-source Docker contexts and updated Dockerfiles/docs to build from the current checkout instead of cloning upstream.
- Updated cloud bootstrap fan-in so `LinuxCloud` checks `.uv/.venv`, requires explicit non-upstream install commands, and rejects upstream `Nerogar/OneTrainer` remotes.
- Cleaned Sangoi config/UI/runtime surfaces: removed stale config/debug fields, added canonical aspect bucketing override, TrainGPS metric enum, DataRecorder schema, LoRA key export path, and deterministic LoRA manifest export.
- Repaired training contracts: parabolic scheduler wiring, plain Prodigy fused-back-pass fail-loud behavior, per-sample priority loss payloads, flow-matching rejection for Sangoi loss mode/weight, and TrainGPS fail-loud schema/metric/group/save behavior.
- Removed stale orphan owners: `requirements-sangoi.txt`, `.style.yapf`, `modules/sangoi/Logger.py`, `modules/util/TensorBoardManager.py`, `modules/util/loss/dynamic_loss_strength.py`, and `scripts/loadTime.py`.
- Converted remote Git requirements to PEP 508 direct URLs so `uv pip` can install them without editable remote requirements.
- Ran pre-commit; hooks reformatted existing files repo-wide under the current Ruff/pre-commit policy.

## Validation

Passed:

```bash
OT_PLATFORM_REQUIREMENTS=requirements-default.txt ./install.sh
awk 'found && /^```$/ { exit } found { print } /^```bash$/ { found=1 }' .sangoi/plans/2026-05-24-master-update-sangoi-mod-cleanup.md > /tmp/onetrainer-sangoi-validation.sh
bash /tmp/onetrainer-sangoi-validation-wrapper.sh
grep -n 'VALIDATION_EXIT_STATUS=0\|VALIDATION_STATUS=0' /tmp/onetrainer-sangoi-validation-with-status.log
```

Validation artifact: `/tmp/onetrainer-sangoi-validation-with-status.log`.
Validation result: `VALIDATION_EXIT_STATUS=0` and `VALIDATION_STATUS=0`.

Covered by the validation script:

- local uv/Python metadata and `.venv` guards
- pre-commit config and `pre-commit run --all-files`
- shell syntax/source checks
- requirements owner checks and CPU torch backend checks
- Docker/cloud/static source fan-in checks
- full `compileall` over `modules` and `scripts`
- dependency import smoke for `rich` and `pytorch_msssim`
- scheduler, config ownership, aspect bucketing, Prodigy, tensor utility, GenerateLosses, LoRA export, priority sampling, Sangoi flow rejection, TrainGPS, DataRecorder, and UI callback ownership smokes
- final `git diff --check`, `git diff HEAD --check`, conflict-marker scan, and stale-reference scans

## Not Run

- Docker image build.
- Cloud remote install/update/run.
- CUDA/GPU training validation from this WSL checkout.
- Rendered Tk UI inspection or screenshot capture.

## Notes

- The WSL validation path used `requirements-default.txt` and installed `torch==2.9.1+cpu` / `torchvision==0.24.1+cpu` via `uv pip --torch-backend cpu`.
- UI changes are source/compile validated but rendered inspection remains `UNVERIFIED`.
