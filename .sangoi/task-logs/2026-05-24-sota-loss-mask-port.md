# SotA Loss And Masked-Gradient Port Task Log

Date: 2026-05-24
Branch: `master-update`
Mode: Doctrine mode
Plan: `.sangoi/plans/2026-05-24-sota-loss-mask-port.md`

## Objective

Port only the requested direct training-impact items from `origin/SotA04022025+Mods` into `master-update`: the SotA Sangoi loss behavior and the masked-training prediction-gradient hook. Keep current master-update structure and preserve existing Sangoi/upstream features outside that scope.

## Baselines

- Target branch baseline: `master-update` at `f3f9f264eca07f36817653562cf1f61ebc1a380d`.
- Source branch reference: `origin/SotA04022025+Mods` at `a350f7d074c2252f11c43fba0229f7e20f982a0b`.
- Merge base: `9a5ccbc450b447869cbafc7bf53bc33358a7f4c2`.

## Gate History

- `Senior Plan Auditor`: first pass returned `REPLAN`; the plan was repaired to fail loud for masked LoRA plus positive masked prior preservation.
- `Senior Plan Auditor`: second pass returned `REPLAN`; the plan was repaired to cover every diffusion caller, remove hidden state, and tighten validation/scans.
- `Senior Plan Auditor`: third pass returned `APPROVE_WITH_FIXES`; local plan fixes were applied before implementation.
- `Senior Code Reviewer`: completed final gate with `READY`.

## Implementation Summary

- Added `modules/util/enum/LossMode.py` with canonical `ORIGINAL` and `SANGOI` values.
- Added `loss_mode_fn`, `loss_tracker_window`, `loss_tracker_use_mad`, `dls_use_ema`, `dls_ema_decay`, and `dls_outlier_threshold` to `modules/util/config/TrainConfig.py` with direct defaults.
- Added `modules/sangoi/DynamicLossControl.py` with batch-safe scalar tracking, read-only `sample_count`, and scalar `progress_fraction` scheduling.
- Reworked `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py` so `LossMode.ORIGINAL` preserves current base loss behavior and `LossMode.SANGOI` dynamically weights MSE/MAE/log-cosh while leaving Huber, Sangoi Huber, Sangoi Charbonnier, VB, and TrainGPS additive.
- Replaced the current MAE-relative `LossWeight.SANGOI` path with SNR/MAPE/progress weighting for diffusion losses and fail-loud progress requirements.
- Made flow matching fail loud for `LossMode.SANGOI` until flow callers plumb the required progress/model state.
- Plumbed `model=model` into PixArtAlpha, Sana, and Wuerstchen diffusion loss calls. Stable Diffusion and SDXL were verified already compliant.
- Added graph-local masked prediction-gradient suppression helpers to `modules/trainer/GenericTrainer.py` and registered the hook before loss/backward.
- Added a trainer guard that raises `ValueError` before prediction/loss when LoRA masked training combines the hook with positive masked prior preservation.
- Added a `Loss Mode` selector to `modules/ui/TrainingTab.py` and preserved the current `loss_weight_strength` `1..20` gamma validation.

## Changed Files

- `.sangoi/plans/2026-05-24-sota-loss-mask-port.md`
- `.sangoi/CHANGELOG.md`
- `.sangoi/task-logs/2026-05-24-sota-loss-mask-port.md`
- `modules/modelSetup/BasePixArtAlphaSetup.py`
- `modules/modelSetup/BaseSanaSetup.py`
- `modules/modelSetup/BaseWuerstchenSetup.py`
- `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py`
- `modules/sangoi/DynamicLossControl.py`
- `modules/trainer/GenericTrainer.py`
- `modules/ui/TrainingTab.py`
- `modules/util/config/TrainConfig.py`
- `modules/util/enum/LossMode.py`

## Validation

Completed successfully:

```bash
git diff --check
python -m compileall modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py modules/modelSetup/BasePixArtAlphaSetup.py modules/modelSetup/BaseSanaSetup.py modules/modelSetup/BaseStableDiffusionSetup.py modules/modelSetup/BaseStableDiffusionXLSetup.py modules/modelSetup/BaseWuerstchenSetup.py modules/trainer/GenericTrainer.py modules/sangoi/DynamicLossControl.py modules/util/enum/LossMode.py modules/util/config/TrainConfig.py modules/ui/TrainingTab.py
```

Static/AST validation also passed for:

- unique new `TrainConfig.default_values()` keys;
- all diffusion `_diffusion_losses(...)` callers passing `model=model`;
- scalar-only `DynamicLossControl.adjust_weights(...)` API;
- masked-gradient hook order before `calculate_loss(...)`;
- no graph-local prediction hook handles stored in `grad_hook_handles`;
- loss mixin anchors for `LossMode`, additive losses, TrainGPS, SANGOI weighting, and flow fail-loud behavior;
- `TrainingTab.py` importing `LossMode`, binding `loss_mode_fn`, and preserving gamma range validation;
- scoped forbidden-pattern scans over touched runtime/config/UI files.

Blocked validation:

```bash
python - <<'PY'
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LossMode import LossMode
PY
```

The runtime import/tensor smokes could not run in the current shell because `/home/lucas/.pyenv/shims/python` resolves to Python 3.12.10 without `torch` installed, causing `ModuleNotFoundError: No module named 'torch'` during `TrainConfig` import through `DataType`.

## Final Review

`Senior Code Reviewer` returned `READY`.

Reviewer residual risk noted as non-blocking:

- no actual training run was performed;
- runtime tensor/import smokes requiring local `torch` were not reproduced in this shell;
- UI was source-validated but not rendered.
