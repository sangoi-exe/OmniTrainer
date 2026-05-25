# Sangoi Changelog

## 2026-05-25 - Selected upstream PR ports

- Ported selected stale upstream training/runtime improvements manually into `master-update` while preserving Sangoi LoRA rules, TrainGPS/DataRecorder behavior, local uv bootstrap, and WSL CPU validation constraints.
- Added canonical validation timestep controls, deterministic validation noise seeding, validation metrics for TensorBoard/patience, TensorBoard run resume metadata, validation patience restore, prefetching, target-resolution cache identity, and mid-accumulation resume state.
- Added configurable attention backend selection, immiscible noise oversampling, BETA/SPEED timestep distributions, and SDXL CEP/CIOP controls.
- Added LoKr, Scaled OFT, DoRA-OFT, LoRA load scaling controls, Flux 2 edit-conditioning support, HiDream tokenizer fixes, and HunyuanVideo LoRA ComfyUI conversion support.
- Added `scripts/validate_selected_upstream_pr_ports.py` to verify the new ownership contracts, guarded-file constraints, strict config behavior, Sangoi invariants, and conversion hooks.
- Validation performed: targeted `py_compile`, runtime CPU smokes, formal selected-PR validator, `git diff --check`, and `git diff HEAD --check`. Full training runs, GPU validation, Docker/cloud execution, and rendered Tk inspection were not run.

## 2026-05-24 - Sangoi mod cleanup and uv-managed bootstrap

- Cleaned the handmade Sangoi mod surface on `master-update` by removing stale config/UI/debug fields and unsupported orphan modules while preserving supported SDXL training behavior.
- Added repo-local uv-managed POSIX bootstrap with `.uv/bin/uv`, managed Python 3.13, fixed `.venv`, `.python-version`, local uv tool dirs, and WSL CPU validation via `requirements-default.txt`.
- Added root `AGENTS.md` and `.dockerignore`; updated Docker/cloud fan-in so current-source builds and remote installs do not silently clone or update upstream OneTrainer.
- Repaired training contracts for parabolic LR scheduling, Prodigy fused-back-pass rejection, per-sample priority timestep losses, flow-matching Sangoi loss rejection, TrainGPS schema/metric/group handling, DataRecorder schema, and deterministic LoRA key export.
- Replaced editable remote Git requirements with PEP 508 direct URLs for uv, and moved CPU torch wheel selection into `uv pip --torch-backend cpu`.
- Validation performed: local uv bootstrap passed, formal plan validation script passed with pre-commit, compileall, dependency imports, source-contract scans, and targeted Python smokes. Docker builds, cloud runs, CUDA validation, and rendered Tk UI inspection were not run.

## 2026-05-24 - SotA Sangoi loss and masked-gradient port

- Ported the requested SotA training-impact items from `origin/SotA04022025+Mods` into `master-update`: dynamic Sangoi loss behavior and masked-training prediction-gradient suppression.
- Added `LossMode.ORIGINAL`/`LossMode.SANGOI` config ownership, defaulting to `ORIGINAL` so current loss behavior remains the default.
- Added `LossTracker` and scalar-scheduled `DynamicLossControl` for dynamic MSE/MAE/log-cosh weighting without broad config/progress object reads.
- Replaced `LossWeight.SANGOI` with the SotA SNR/MAPE/progress weighting path for diffusion losses, requiring `model.train_progress` and preserving TensorBoard as optional.
- Preserved current additive Huber, Sangoi Huber, Sangoi Charbonnier, VB, TrainGPS, and priority-sampling behavior while integrating the new base-loss mode.
- Added a graph-local masked-training gradient hook that zeros prediction gradients outside the latent mask and fails loud for LoRA masked prior preservation when that outside-mask training would be contradicted.
- Added the loss-mode selector to the training UI and plumbed model progress into PixArtAlpha, Sana, and Wuerstchen diffusion loss callers.
- Validation performed: `git diff --check`, targeted `compileall`, AST/source contract checks, diffusion caller matrix, hook-order checks, and scoped forbidden-residue scans passed. Runtime tensor/import smokes were blocked because the current shell Python cannot import `torch`.
- Final `Senior Code Reviewer` gate completed with `READY`.

## 2026-05-24 - Upstream master merge resolution

- Resolved the `master-update` merge against the upstream OneTrainer tree while preserving current Sangoi training contracts.
- Kept upstream structural changes, new model families, current LoRA conversion utilities, UI validation changes, and `training_presets/*` matching upstream `main`.
- Restored Sangoi-owned training features on the upstream structure: custom Sangoi Huber/Charbonnier weights, `LossWeight.SANGOI`, priority timestep sampling, LoRA blacklist/rank/alpha rules, selective gradient checkpointing, TrainGPS, DataRecorder, Prodigy stats, SDXL prediction/debug hooks, pause/resume commands, and Sangoi UI bindings.
- Repaired TrainGPS penalty application so current parameter deltas keep their autograd path while captured initial/reference values stay detached.
- Removed stale merge residue for old local contracts such as `delta_pattern_metric`, bare `charbonnier_strength`, and obsolete `components.file_entry` calls.
- Added `pytorch-msssim` to the active global requirements path for Sangoi SSIM loss/debug metrics.
- Validation performed: no unmerged paths, no conflict markers, no targeted stale-contract residues, staged Python syntax compilation passed, TrainGPS focused syntax/residue checks passed, Sangoi fail-loud paths were recompiled, and `git diff --cached --check` passed. Runtime config smoke was blocked because the current shell Python cannot import `torch`.
