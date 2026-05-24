# SotA Loss And Masked-Gradient Port Plan

Difficulty: hard
Status: approved with fixes applied
Date: 2026-05-24
Branch: `master-update`

## Summary

Port the direct training-impact pieces requested from `origin/SotA04022025+Mods` into the current `master-update` branch: the SotA Sangoi loss behavior and the masked-training gradient hook. Keep `master-update` as the structural baseline so upstream training, prior-preservation, TrainGPS, DataRecorder, priority timestep sampling, UI/config, and optimizer integrations remain intact.

## User Intent

The user discovered that `SotA04022025+Mods` was the better SDXL training branch and asked to bring over items 2 and 3 from the comparison: the SotA Sangoi loss behavior and the masked-training gradient hook. The user did not ask to port AlignProp, SDXL DataLoader changes, LoRA override syntax, or optimizer behavior in this pass.

## Current Evidence

- Current branch: `master-update` at `f3f9f264eca07f36817653562cf1f61ebc1a380d`.
- Source branch ref: `origin/SotA04022025+Mods` at `a350f7d074c2252f11c43fba0229f7e20f982a0b`.
- Merge base: `9a5ccbc450b447869cbafc7bf53bc33358a7f4c2`.
- The working tree is clean before this plan.
- `master-update` currently has a simpler `LossWeight.SANGOI` MAE-relative weighting in `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py`.
- `origin/SotA04022025+Mods` has `LossMode`, `LossTracker`, `DynamicLossControl`, and SNR/MAPE/progress-based `LossWeight.SANGOI` behavior.
- `origin/SotA04022025+Mods` has a masked-training hook in `modules/trainer/GenericTrainer.py` that preserves full forward context and zeros gradient outside the latent mask.

## Scope

In scope:

- Add the SotA `LossMode` config surface needed to select original versus dynamic Sangoi loss composition.
- Add a current, robust `modules/sangoi/DynamicLossControl.py` owner for `LossTracker` and `DynamicLossControl`.
- Integrate dynamic MSE/MAE/log-cosh weighting into the existing master-update loss mixin without dropping current Huber, Sangoi Huber, Sangoi Charbonnier, VB, TrainGPS, or priority sampling behavior.
- Replace the current `LossWeight.SANGOI` MAE-relative weighting with the SotA SNR/MAPE/progress weighting.
- Add the masked-training prediction-gradient hook to the current master-update trainer loop.
- Add the `LossMode` selector to the existing training loss UI, matching the SotA UI surface.
- Update doctrine bookkeeping after final review.

Out of scope:

- AlignProp.
- SDXL DataLoader prompt/cache behavior.
- SotA string-based LoRA override syntax.
- Prodigy/optimizer behavior changes.
- Reverting `training_presets/*` or changing presets.
- Adding compatibility aliases for old field names.
- Adding fallback adapters, dual-read config paths, or silent remapping.

## Locked Contract Decisions

- The canonical branch to modify is `master-update`; the SotA branch is a known-good comparison anchor, not the target branch.
- The canonical new enum is `modules/util/enum/LossMode.py` with exactly `ORIGINAL` and `SANGOI` values.
- The canonical config field is `loss_mode_fn: LossMode`, defaulting to `LossMode.ORIGINAL` so existing master-update configs keep current loss behavior unless the user selects the Sangoi mode.
- The dynamic loss config fields are `loss_tracker_window`, `loss_tracker_use_mad`, `dls_use_ema`, `dls_ema_decay`, and `dls_outlier_threshold` under `TrainConfig`; each must have one direct canonical owner and direct consumers.
- The dynamic loss mode only controls the MSE/MAE/log-cosh base loss trio. Current master-update additive loss terms remain additive: upstream Huber, Sangoi Huber, Sangoi Charbonnier, and VB loss.
- In `LossMode.ORIGINAL`, the MSE/MAE/log-cosh behavior must remain equivalent to current master-update behavior.
- In `LossMode.SANGOI`, MSE/MAE/log-cosh component losses are computed for tracker input and dynamically weighted, then multiplied by the configured strengths.
- `LossWeight.SANGOI` uses SotA SNR/MAPE/progress weighting for diffusion losses only. It requires `model.train_progress` and fails loud if selected without progress.
- `LossWeight.SANGOI` keeps the existing current/default strength domain used by the training UI: `loss_weight_strength` is a gamma-like value validated as `1..20` and defaults to `5.0`. The SotA reward formula maps the computed reward into `[gamma, 1]`; with this domain the reward can exceed `1.0`, matching the current config/UI contract rather than introducing a second fractional strength contract.
- TensorBoard logging is optional; absence of `model.tensorboard` must not block training. Absence of required progress for selected SANGOI loss weighting must block training.
- The masked-gradient hook is active only when `config.masked_training` is true and a training backward pass will use `model_output_data["predicted"]`. It must not alter the forward prediction tensor values.
- The masked-gradient hook uses the latent mask as the gradient multiplier, broadcasting channels to the prediction tensor. Shape mismatches after broadcasting are runtime errors, not silently repaired.
- `masked_training` with positive `masked_prior_preservation_weight` is an invalid combination for this hook in LoRA training. Current prior preservation intentionally trains outside the mask, while the requested hook zeros prediction gradients outside the mask. The implementation must fail loud for that combination instead of computing a misleading prior-preservation loss whose outside-mask gradient is suppressed.
- SANGOI dynamic masked base components preserve the existing `masked_losses_with_prior` semantics when `prior_target` is present, but the trainer-level invalid-combination guard prevents positive masked prior-preservation from silently colliding with the gradient hook.
- Dynamic `LossMode.SANGOI` mutates `LossTracker`/EMA state only on gradient-enabled training paths where `predicted.requires_grad` is true. Validation and loss-generation paths may read existing tracker/controller state to compute the loss but must not update tracker samples or EMA weights.
- Dynamic `LossMode.SANGOI` is supported for diffusion losses only in this pass. Flow-matching losses must fail loud if `config.loss_mode_fn == LossMode.SANGOI`, because current flow callers do not plumb model/progress into `_flow_matching_losses`.

## Feature Owner Matrix

| Feature / Contract | Canonical Owner | Live Consumers / Producers | Validation |
| --- | --- | --- | --- |
| `LossMode` enum | `modules/util/enum/LossMode.py` | `TrainConfig.py`, `TrainingTab.py`, `ModelSetupDiffusionLossMixin.py` | Import smoke and enum string asserts |
| `loss_mode_fn` and dynamic loss config fields | `modules/util/config/TrainConfig.py` | loss mixin and UI state serialization | default config smoke; duplicate key scan |
| Dynamic loss tracker/control | `modules/sangoi/DynamicLossControl.py` | `ModelSetupDiffusionLossMixin.py` | import and synthetic dynamic-weight smoke |
| Dynamic MSE/MAE/log-cosh composition | `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py` | diffusion setups using `_diffusion_losses` | synthetic `_diffusion_losses` smoke where dependencies allow; compileall otherwise |
| SotA SANGOI loss weighting | `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py` | diffusion model setup loss calls | direct helper path exercised by synthetic smoke or source-level assertions |
| Masked-training gradient hook | `modules/trainer/GenericTrainer.py` | current trainer training loop | synthetic tensor hook test or source-level hook assertion plus compileall |
| Loss mode UI selector | `modules/ui/TrainingTab.py` | training UI state | compileall and import smoke if UI deps are available |

## Alternatives Considered

1. Cherry-pick the SotA commits wholesale. Rejected because the target branch already integrated upstream refactors and master-update local features; wholesale cherry-pick would resurrect stale DataLoader/LoRA/optimizer surfaces outside the requested scope.
2. Replace `ModelSetupDiffusionLossMixin.py` with the SotA file. Rejected because that would drop master-update Huber, Sangoi Huber/Charbonnier, TrainGPS, flow-matching, and prior-preservation integration.
3. Keep master-update loss code and only add the masked-gradient hook. Rejected because the user explicitly requested item 2, and the static comparison showed item 2 as a likely quality driver.
4. Add a parallel Sangoi loss wrapper layer. Rejected because it would duplicate loss ownership and create avoidable contract soup around the shared loss mixin.
5. Integrate SotA behavior inside the current owners. Chosen because it preserves the current upstream/master-update structure while restoring the two requested quality-affecting behaviors at their canonical runtime seams.
6. Expose every dynamic-loss tuning field in the UI. Rejected for this pass because the SotA UI only exposed `loss_mode_fn`; dynamic tuning fields are persisted config fields and can be edited in config JSON without expanding the UI surface beyond the requested feature.

## Planned Touchpoints

Runtime/config/UI files:

- `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py`
- `modules/modelSetup/BasePixArtAlphaSetup.py`
- `modules/modelSetup/BaseSanaSetup.py`
- `modules/modelSetup/BaseStableDiffusionSetup.py` (validation-only; already passes `model=model`)
- `modules/modelSetup/BaseStableDiffusionXLSetup.py` (validation-only; already passes `model=model`)
- `modules/modelSetup/BaseWuerstchenSetup.py`
- `modules/trainer/GenericTrainer.py`
- `modules/sangoi/DynamicLossControl.py`
- `modules/util/enum/LossMode.py`
- `modules/util/config/TrainConfig.py`
- `modules/ui/TrainingTab.py`

Doctrine artifacts:

- `.sangoi/plans/2026-05-24-sota-loss-mask-port.md`
- `.sangoi/CHANGELOG.md`
- `.sangoi/task-logs/2026-05-24-sota-loss-mask-port.md`

## Implementation Steps

1. Gate this plan before implementation.
   - Submit this plan to `Senior Plan Auditor` using a complete plan-gate brief.
   - Do not edit runtime/config/UI files until the gate returns `APPROVE` or `APPROVE_WITH_FIXES` and any required local plan fixes are applied.
   - Done when the gate verdict is recorded in the task context and the plan status is updated if the auditor requires fixes.

2. Add config and enum ownership.
   - Create `modules/util/enum/LossMode.py` with `ORIGINAL` and `SANGOI`.
   - Import `LossMode` in `modules/util/config/TrainConfig.py`.
   - Add type annotations for `loss_tracker_window`, `loss_tracker_use_mad`, `dls_use_ema`, `dls_ema_decay`, `dls_outlier_threshold`, and `loss_mode_fn`.
   - Add defaults next to the existing Sangoi and training loss fields: tracker window `100`, MAD disabled, EMA disabled, EMA decay `0.9`, outlier threshold `3.0`, and `loss_mode_fn = LossMode.ORIGINAL`.
   - Done when `TrainConfig.default_values()` owns each new key exactly once.

3. Add dynamic loss control owner.
   - Add `modules/sangoi/DynamicLossControl.py` based on the SotA branch but repair batch-size handling by storing detached mean scalar values instead of calling `.item()` on potentially batched tensors.
   - Keep the SotA schedule defaults: `mae` `0.6 -> 0.0`, `mse` `0.2 -> 0.6`, and `log_cosh` `0.2 -> 0.4`.
   - Make constructor parameters match the canonical config fields.
   - `DynamicLossControl.adjust_weights` must accept scalar scheduling input such as `progress_fraction`; it must not read broad `config` or `progress` objects.
   - Expose a read-only `LossTracker.sample_count` property for validation and debugging; it reports the number of tracked MSE samples and does not mutate tracker state.
   - Done when the module imports without requiring runtime training state.

4. Integrate dynamic loss composition in the current loss mixin.
   - Import `LossMode`, `LossTracker`, and `DynamicLossControl` in `ModelSetupDiffusionLossMixin.py`.
   - Initialize lazy dynamic-loss state in the mixin without changing public setup signatures.
   - Add a helper that syncs `LossTracker`/`DynamicLossControl` from current config when relevant config values change.
   - The sync helper may cache only the tracker, controller, and an exact canonical-field tuple. Do not mirror flat config fields, add aliases, use `getattr` for canonical keys, or bridge old field names.
   - Preserve current masked and unmasked loss behavior for `LossMode.ORIGINAL`.
   - For `LossMode.SANGOI`, compute MSE/MAE/log-cosh per-sample tensors, update the tracker, compute dynamic weights, and apply them to the configured base strengths.
   - For `LossMode.SANGOI` masked losses, use `masked_losses_with_prior` for MSE/MAE/log-cosh component tensors so existing `prior_target` semantics remain intact when present. Positive masked prior-preservation is still rejected by the trainer-level guard when the gradient hook would make it contradictory.
   - Pass an explicit state-mutation flag into the dynamic weighting path; mutate tracker/EMA only when gradients are enabled and the prediction tensor requires grad.
   - Preserve current additional Huber, Sangoi Huber, Sangoi Charbonnier, VB, TrainGPS, and flow-matching behavior.
   - In `_flow_matching_losses`, fail loud when `config.loss_mode_fn == LossMode.SANGOI`; keep `LossMode.ORIGINAL` flow behavior unchanged.
   - The loss mixin must not read `self.config`, `self.progress`, `self.tensorboard`, or `model.train_config`. Any required `train_progress`, `total_epochs`, `config`, or TensorBoard writer must be passed explicitly through the current call.
   - Done when no loss path reads missing config fields through `getattr` and original mode remains the default.

5. Replace `LossWeight.SANGOI` weighting with the SotA weighting.
   - Plumb `model=model` into every current diffusion `_diffusion_losses(...)` caller: SD, SDXL, PixArtAlpha, Sana, and Wuerstchen.
   - SD and SDXL are verification-only for this plan because they already pass `model=model`; PixArtAlpha, Sana, and Wuerstchen require edits.
   - Replace the current MAE-relative `__sangoi_loss_weighting` with a helper that receives timesteps, prediction, target, explicit `train_progress`, explicit `total_epochs`, optional TensorBoard writer, device, and gamma.
   - Compute SNR, per-sample clamped MAPE, epoch-progress interpolation, low-SNR-first/high-SNR-first scenario weighting, and the SotA reward formula.
   - Log the same scalar families through the existing optional TensorBoard writer when present.
   - Fail loud if `LossWeight.SANGOI` is selected without a model carrying `train_progress`.
   - Done when diffusion `LossWeight.SANGOI` no longer uses the MAE-relative weighting.

6. Add the masked-training gradient hook to the current trainer loop.
   - Add static helper methods in `GenericTrainer.py` to prepare/broadcast the latent mask and to register a hook that multiplies prediction gradients by the mask.
   - Add a fail-loud trainer guard before prediction/loss work: if `self.config.masked_training`, `self.config.masked_prior_preservation_weight > 0`, and `self.config.training_method == TrainingMethod.LORA`, raise `ValueError` explaining that the SotA masked-gradient hook conflicts with masked prior preservation outside-mask gradients.
   - In the current training loop, after `model_output_data` is finalized and before `calculate_loss(...)`, register the hook when `self.config.masked_training` is true.
   - Register the hook only when `model_output_data["predicted"].requires_grad` is true.
   - Treat the prediction hook as graph-local. Do not store its handle in `self.grad_hook_handles`, because it is tied to one prediction tensor, not a long-lived parameter hook.
   - `prepare_mask_for_prediction` may broadcast only a single mask channel across prediction channels. It must reject batch-size mismatches, spatial mismatches, and non-singleton channel mismatches with `ValueError`.
   - Keep prior-preservation flow intact: when the prior model branch runs, apply the hook only to the current trainable `model_output_data["predicted"]` after prior target replacement.
   - Do not run the hook in validation or sampling paths.
   - Done when the hook is registered before backward and does not alter forward prediction values.

7. Add the UI selector.
   - Import `LossMode` in `modules/ui/TrainingTab.py`.
   - Add a `Loss Mode` option row at the top of the loss frame, then shift existing row bookkeeping without removing existing Huber/Sangoi Huber/Charbonnier/VB/loss-weight controls.
   - Done when UI state binds the canonical `loss_mode_fn` field.

8. Validate locally.
   - Run `git diff --check`.
   - Run `python -m compileall` on every runtime/config/UI touchpoint, including the three diffusion caller files now being plumbed.
   - Run a config smoke asserting `TrainConfig.default_values().loss_mode_fn == LossMode.ORIGINAL` and all dynamic fields exist with expected defaults.
   - Run a source-level duplicate-key check for the newly added `TrainConfig` keys.
   - Run an AST caller matrix check proving every `_diffusion_losses(...)` caller outside the mixin passes `model=model`.
   - Run a `TrainConfig.from_dict` roundtrip smoke for all canonical dynamic fields and `LossMode.SANGOI`.
   - Run a synthetic masked-gradient hook check against `GenericTrainer` static helpers, including forward equality, channel-only broadcast, and batch/spatial/channel mismatch failures.
   - Run source-order validation proving the masked-gradient hook is registered before `calculate_loss(...)` in the training loop and that no prediction hook handle is appended to `self.grad_hook_handles`.
   - Run an `ORIGINAL`-mode numeric equivalence check over small synthetic unmasked and masked loss tensors for MSE/MAE/log-cosh to prove default behavior remains unchanged.
   - Run SANGOI dynamic smokes that exercise batch-size-2 tracker mutation on a gradient-enabled path and no mutation under `torch.no_grad()`.
   - Run a SANGOI formula/progress smoke for `LossWeight.SANGOI`, including fail-loud behavior when progress is unavailable, TensorBoard-absent success when progress is available, Wuerstchen `alphas_cumprod_fun` support, and `epochs == 1`.
   - Run source validation proving masked LoRA with positive masked prior preservation raises `ValueError` before prediction/loss work.
   - Run source or synthetic validation proving SANGOI leaves additive loss terms and TrainGPS penalty in place.
   - Run a flow fail-loud smoke proving `_flow_matching_losses` rejects `LossMode.SANGOI`.
   - Run source validation proving `TrainingTab.py` imports `LossMode`, binds `loss_mode_fn`, and preserves the existing `loss_weight_strength` `1..20` validation for SANGOI.
   - Run scoped forbidden-pattern scans on only touched/new runtime/config/UI files for `getattr(config`, `hasattr(config`, `== "SANGOI"`, `== 'SANGOI'`, `self.config`, `self.progress`, `self.tensorboard`, `model.train_config`, `dynamic_loss_strengthing`, `lora_module_overrides`, `enable_token_grad_analyzer`, `dora_paper_calc`, `dyloco_params`, stale misspellings such as `charbonier`, and alternate old field names.
   - Run `rg -n "loss_mode_fn|LossMode|DynamicLossControl|stop_grad_outside_mask|prepare_mask_for_prediction" modules .sangoi` to confirm references are current.

9. Final review and bookkeeping.
   - Assemble a code-review bundle and run the mandatory `Senior Code Reviewer` gate on the substantive change set.
   - Apply exactly any approved local non-substantive fixes if the verdict is `APPROVE_WITH_FIXES`.
   - Update `.sangoi/CHANGELOG.md` and create `.sangoi/task-logs/2026-05-24-sota-loss-mask-port.md` after final review.
   - Done when the reviewer gate is complete, validation evidence is recorded, and doctrine artifacts describe current behavior only.

## Validation Commands

```bash
set -euo pipefail

TOUCHPOINTS=(
  modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py
  modules/modelSetup/BasePixArtAlphaSetup.py
  modules/modelSetup/BaseSanaSetup.py
  modules/modelSetup/BaseStableDiffusionSetup.py
  modules/modelSetup/BaseStableDiffusionXLSetup.py
  modules/modelSetup/BaseWuerstchenSetup.py
  modules/trainer/GenericTrainer.py
  modules/sangoi/DynamicLossControl.py
  modules/util/enum/LossMode.py
  modules/util/config/TrainConfig.py
  modules/ui/TrainingTab.py
)

git diff --check
python -m compileall "${TOUCHPOINTS[@]}"
python - <<'PY'
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LossMode import LossMode
config = TrainConfig.default_values()
assert config.loss_mode_fn == LossMode.ORIGINAL
assert config.loss_tracker_window == 100
assert config.loss_tracker_use_mad is False
assert config.dls_use_ema is False
assert config.dls_ema_decay == 0.9
assert config.dls_outlier_threshold == 3.0

roundtrip = TrainConfig.default_values().from_dict({
    **config.to_dict(),
    "loss_mode_fn": "SANGOI",
    "loss_tracker_window": 7,
    "loss_tracker_use_mad": True,
    "dls_use_ema": True,
    "dls_ema_decay": 0.75,
    "dls_outlier_threshold": 2.5,
})
assert roundtrip.loss_mode_fn == LossMode.SANGOI
assert roundtrip.loss_tracker_window == 7
assert roundtrip.loss_tracker_use_mad is True
assert roundtrip.dls_use_ema is True
assert roundtrip.dls_ema_decay == 0.75
assert roundtrip.dls_outlier_threshold == 2.5
print("config smoke ok")
PY
python - <<'PY'
import ast
from collections import Counter
from pathlib import Path
keys = {
    "loss_mode_fn",
    "loss_tracker_window",
    "loss_tracker_use_mad",
    "dls_use_ema",
    "dls_ema_decay",
    "dls_outlier_threshold",
}
path = Path("modules/util/config/TrainConfig.py")
tree = ast.parse(path.read_text(), filename=str(path))
found = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "append":
        if node.args and isinstance(node.args[0], ast.Tuple) and node.args[0].elts:
            first = node.args[0].elts[0]
            if isinstance(first, ast.Constant) and first.value in keys:
                found.append(first.value)
counts = Counter(found)
missing = keys - set(counts)
duplicates = {key: count for key, count in counts.items() if count != 1}
if missing or duplicates:
    raise SystemExit(f"missing={sorted(missing)} duplicates={duplicates}")
print("TrainConfig key ownership ok")
PY
python - <<'PY'
import ast
from pathlib import Path

missing = []
for path in Path("modules/modelSetup").glob("Base*Setup.py"):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_diffusion_losses":
            if not any(keyword.arg == "model" for keyword in node.keywords):
                missing.append(f"{path}:{node.lineno}")
if missing:
    raise SystemExit("diffusion caller missing model kw:\\n" + "\\n".join(missing))
print("diffusion caller model plumbing ok")
PY
python - <<'PY'
import ast
from pathlib import Path
import torch
from modules.trainer.GenericTrainer import GenericTrainer

predicted = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2).clone().requires_grad_(True)
before = predicted.detach().clone()
mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
prepared = GenericTrainer.prepare_mask_for_prediction(mask, predicted)
GenericTrainer.stop_grad_outside_mask(predicted, prepared)
assert torch.equal(predicted.detach(), before)
predicted.sum().backward()
assert torch.equal(predicted.grad, mask.expand_as(predicted))
try:
    GenericTrainer.prepare_mask_for_prediction(torch.ones(1, 1, 3, 3), predicted)
except ValueError:
    pass
else:
    raise SystemExit("spatial mismatch did not fail")
for bad_mask in (torch.ones(2, 1, 2, 2), torch.ones(1, 2, 2, 2)):
    try:
        GenericTrainer.prepare_mask_for_prediction(bad_mask, predicted)
    except ValueError:
        pass
    else:
        raise SystemExit("batch/channel mismatch did not fail")

source = Path("modules/trainer/GenericTrainer.py").read_text()
hook_index = source.index("self.stop_grad_outside_mask(")
loss_index = source.index("loss = self.model_setup.calculate_loss(")
assert hook_index < loss_index
tree = ast.parse(source, filename="modules/trainer/GenericTrainer.py")
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "stop_grad_outside_mask":
        helper_source = ast.get_source_segment(source, node)
        assert "grad_hook_handles" not in helper_source
        break
else:
    raise SystemExit("stop_grad_outside_mask helper missing")
print("masked-gradient hook smoke ok")
PY
python - <<'PY'
import torch
import torch.nn.functional as F
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LossMode import LossMode
from modules.util.enum.LossWeight import LossWeight

class Probe(ModelSetupDiffusionLossMixin):
    pass

config = TrainConfig.default_values()
config.loss_mode_fn = LossMode.ORIGINAL
config.mse_strength = 0.25
config.mae_strength = 0.5
config.log_cosh_strength = 0.75
config.huber_strength = 0.0
config.sangoi_huber_strength = 0.0
config.sangoi_charbonnier_strength = 0.0
config.vb_loss_strength = 0.0
config.loss_weight_fn = LossWeight.CONSTANT
config.batch_size = 2
config.gradient_accumulation_steps = 1
predicted = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]], [[[2.0, 3.0], [4.0, 5.0]]]])
target = torch.ones_like(predicted)
batch = {"loss_weight": torch.ones(2)}
data = {"loss_type": "target", "predicted": predicted, "target": target}
actual = Probe()._diffusion_losses(batch, data, config, torch.device("cpu"))
mean_dim = tuple(range(1, predicted.ndim))
diff = predicted - target
log_cosh = diff + F.softplus(-2.0 * diff) - torch.log(torch.full_like(diff, 2.0))
expected = (
    F.mse_loss(predicted, target, reduction="none").mean(mean_dim) * config.mse_strength
    + F.l1_loss(predicted, target, reduction="none").mean(mean_dim) * config.mae_strength
    + log_cosh.mean(mean_dim) * config.log_cosh_strength
)
assert torch.allclose(actual, expected)

config.masked_training = True
config.unmasked_weight = 0.25
config.normalize_masked_area_loss = False
config.masked_prior_preservation_weight = 0.0
mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]], [[[0.0, 1.0], [1.0, 0.0]]]])
batch = {"loss_weight": torch.ones(2), "latent_mask": mask}
actual = Probe()._diffusion_losses(batch, data, config, torch.device("cpu"))
weight = torch.clamp(mask, config.unmasked_weight, 1)
expected = (
    (F.mse_loss(predicted, target, reduction="none") * weight).mean(mean_dim) * config.mse_strength
    + (F.l1_loss(predicted, target, reduction="none") * weight).mean(mean_dim) * config.mae_strength
    + (log_cosh * weight).mean(mean_dim) * config.log_cosh_strength
)
assert torch.allclose(actual, expected)
print("ORIGINAL loss equivalence smoke ok")
PY
python - <<'PY'
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LossMode import LossMode
from modules.util.enum.LossWeight import LossWeight

class Probe(ModelSetupDiffusionLossMixin):
    pass

config = TrainConfig.default_values()
config.loss_mode_fn = LossMode.SANGOI
config.masked_training = True
config.batch_size = 2
config.mse_strength = 1.0
config.mae_strength = 1.0
config.log_cosh_strength = 1.0
config.huber_strength = 0.0
config.sangoi_huber_strength = 0.0
config.sangoi_charbonnier_strength = 0.0
config.vb_loss_strength = 0.0
config.loss_weight_fn = LossWeight.CONSTANT
predicted = torch.ones(2, 1, 2, 2, requires_grad=True)
target = torch.zeros_like(predicted)
batch = {"loss_weight": torch.ones(2), "latent_mask": torch.ones(2, 1, 2, 2)}
data = {"loss_type": "target", "predicted": predicted, "target": target, "prior_target": target + 0.5}
probe = Probe()
probe._diffusion_losses(batch, data, config, torch.device("cpu"))
assert probe._ModelSetupDiffusionLossMixin__loss_tracker.sample_count == 1

probe = Probe()
with torch.no_grad():
    probe._diffusion_losses(batch, data, config, torch.device("cpu"))
assert probe._ModelSetupDiffusionLossMixin__loss_tracker.sample_count == 0
try:
    probe._flow_matching_losses(batch, data, config, torch.device("cpu"))
except ValueError:
    pass
else:
    raise SystemExit("flow SANGOI did not fail")

config = TrainConfig.default_values()
config.loss_mode_fn = LossMode.ORIGINAL
config.loss_weight_fn = LossWeight.SANGOI
config.loss_weight_strength = 5.0
config.mse_strength = 1.0
config.mae_strength = 0.0
config.log_cosh_strength = 0.0
config.huber_strength = 0.0
config.sangoi_huber_strength = 0.0
config.sangoi_charbonnier_strength = 0.0
config.vb_loss_strength = 0.0
config.batch_size = 2
config.epochs = 3
predicted = torch.tensor([[[[1.0, 2.0]]], [[[2.0, 3.0]]]])
target = torch.ones_like(predicted)
batch = {"loss_weight": torch.ones(2)}
data = {"loss_type": "target", "predicted": predicted, "target": target, "timestep": torch.tensor([0, 1])}
betas = torch.tensor([0.1, 0.2], dtype=torch.float32)
try:
    Probe()._diffusion_losses(batch, data, config, torch.device("cpu"), betas=betas)
except (RuntimeError, ValueError):
    pass
else:
    raise SystemExit("SANGOI loss weight did not require progress")
model = SimpleNamespace(train_progress=SimpleNamespace(epoch=1, global_step=10), tensorboard=None)
actual = Probe()._diffusion_losses(batch, data, config, torch.device("cpu"), model=model, betas=betas)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
snr = alphas_cumprod / (1 - alphas_cumprod)
mape = torch.abs((target - predicted) / (target + 1e-8)).clamp(0, 1).mean(dim=(1, 2, 3))
alpha = model.train_progress.epoch / float(config.epochs - 1)
scenario = (1 - alpha) * torch.log(1 + 1 / (snr[data["timestep"]] + 1e-8)) + alpha * torch.log(snr[data["timestep"]] + 1)
reward = config.loss_weight_strength + (1 - config.loss_weight_strength) * torch.exp(-(1 - mape) * scenario).clamp(0, 1)
expected = F.mse_loss(predicted, target, reduction="none").mean(dim=(1, 2, 3)) * reward
assert torch.allclose(actual, expected)

config.epochs = 1
actual = Probe()._diffusion_losses(batch, data, config, torch.device("cpu"), model=model, betas=betas)
assert torch.isfinite(actual).all()

actual = Probe()._diffusion_losses(
    batch,
    data,
    config,
    torch.device("cpu"),
    model=model,
    alphas_cumprod_fun=lambda timesteps, _unused: torch.tensor([0.9, 0.72])[timesteps],
)
assert torch.isfinite(actual).all()
print("SANGOI validation/flow smoke ok")
PY
python - <<'PY'
import ast
from pathlib import Path

trainer_source = Path("modules/trainer/GenericTrainer.py").read_text()
guard_index = trainer_source.index("masked_prior_preservation_weight")
predict_index = trainer_source.index("model_output_data = self.model_setup.predict")
loss_index = trainer_source.index("loss = self.model_setup.calculate_loss(")
assert guard_index < predict_index < loss_index
assert "ValueError" in trainer_source[guard_index:predict_index]

loss_source = Path("modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py").read_text()
assert "config.huber_strength" in loss_source
assert "config.sangoi_huber_strength" in loss_source
assert "config.sangoi_charbonnier_strength" in loss_source
assert "config.vb_loss_strength" in loss_source
assert "train_gps.compute_penalty" in loss_source

ui_source = Path("modules/ui/TrainingTab.py").read_text()
assert "from modules.util.enum.LossMode import LossMode" in ui_source
assert '"loss_mode_fn"' in ui_source
assert "check_range(lower=1, upper=20" in ui_source
print("source contract smoke ok")
PY
rg -n "loss_mode_fn|LossMode|DynamicLossControl|stop_grad_outside_mask|prepare_mask_for_prediction" modules .sangoi
! rg -n 'getattr\\(config|hasattr\\(config|== "SANGOI"|== '\\''SANGOI'\\''|dynamic_loss_strengthing|lora_module_overrides|enable_token_grad_analyzer|dora_paper_calc|dyloco_params|charbonier|loss_mode\\s*=' "${TOUCHPOINTS[@]}"
! rg -n 'self\\.config|self\\.progress|self\\.tensorboard|model\\.train_config' modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py modules/sangoi/DynamicLossControl.py
```

## Acceptance Criteria

- `LossMode.ORIGINAL` remains the default and preserves current master-update loss behavior.
- `LossMode.SANGOI` activates dynamic MSE/MAE/log-cosh weighting without disabling current additive Huber/Sangoi Huber/Sangoi Charbonnier/VB paths.
- `LossWeight.SANGOI` uses the SotA SNR/MAPE/progress weighting instead of the current MAE-relative weighting.
- Existing TrainGPS penalty remains applied after diffusion loss weighting.
- Masked training registers a prediction-gradient hook before backward and leaves forward prediction values unchanged.
- Masked LoRA training with positive masked prior preservation fails loud instead of silently suppressing outside-mask prior-preservation gradients.
- Flow-matching training with `LossMode.SANGOI` fails loud until flow callers plumb the required progress/model state.
- No compatibility aliases, dual-read fields, fallback config lookups, or stale SotA-only names are introduced.
- Required validation passes or any environmental blocker is explicitly documented.
- Mandatory Plan Auditor and Code Reviewer gates are complete before final handoff.
