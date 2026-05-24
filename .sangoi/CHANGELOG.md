# Sangoi Changelog

## 2026-05-24 - Upstream master merge resolution

- Resolved the `master-update` merge against the upstream OneTrainer tree while preserving current Sangoi training contracts.
- Kept upstream structural changes, new model families, current LoRA conversion utilities, UI validation changes, and `training_presets/*` matching upstream `main`.
- Restored Sangoi-owned training features on the upstream structure: custom Sangoi Huber/Charbonnier weights, `LossWeight.SANGOI`, priority timestep sampling, LoRA blacklist/rank/alpha rules, selective gradient checkpointing, TrainGPS, DataRecorder, Prodigy stats, SDXL prediction/debug hooks, pause/resume commands, and Sangoi UI bindings.
- Repaired TrainGPS penalty application so current parameter deltas keep their autograd path while captured initial/reference values stay detached.
- Removed stale merge residue for old local contracts such as `delta_pattern_metric`, bare `charbonnier_strength`, and obsolete `components.file_entry` calls.
- Added `pytorch-msssim` to the active global requirements path for Sangoi SSIM loss/debug metrics.
- Validation performed: no unmerged paths, no conflict markers, no targeted stale-contract residues, staged Python syntax compilation passed, TrainGPS focused syntax/residue checks passed, Sangoi fail-loud paths were recompiled, and `git diff --cached --check` passed. Runtime config smoke was blocked because the current shell Python cannot import `torch`.
