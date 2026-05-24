# Master Update Merge Resolution Task Log

Date: 2026-05-24
Branch: `master-update`
Mode: Doctrine mode
Plan: `.sangoi/plans/2026-05-24-master-update-merge-resolution.md`

## Objective

Resolve the in-progress merge from the upstream-synced `main` branch into `master-update`, preserving current Sangoi modifications while adopting upstream structural changes. Restore upstream `training_presets/*` files as explicitly allowed by the user.

## Gate History

- `Senior Plan Auditor`: completed before implementation with `APPROVE_WITH_FIXES`.
- Applied plan fixes before conflict resolution: narrowed LoRA parsing validation, included `|||||||` marker checks, locked UIState list ownership, added config mirror/residue checks, and required a staged surface ledger for final review.
- `Senior Code Reviewer`: first completed final-gate pass returned `NOT_READY`; the three findings are fixed below and a fresh final gate remains pending.

## Merge Resolution Summary

- Resolved every unmerged path; `git diff --name-only --diff-filter=U` is empty.
- Accepted upstream `training_presets/*` contents and verified the staged preset tree matches `main`.
- Used upstream files as structural baselines where upstream had deep refactors, then reapplied current Sangoi behavior at canonical owners.
- Preserved Sangoi config fields in `modules/util/config/TrainConfig.py` and removed dead local config residue from the merge surface.
- Preserved upstream `huber_strength`/`huber_delta` as upstream Huber fields and kept Sangoi custom loss weights under `sangoi_huber_strength` and `sangoi_charbonnier_strength`.
- Preserved `LossWeight.SANGOI` for diffusion loss weighting while leaving flow matching unsupported.
- Restored `PRIORITY_SAMPLING` through `TimestepDistribution`, `ModelSetupNoiseMixin`, and `GenericTrainer` without string enum comparison or scalar `loss.item()` priority updates.
- Preserved LoRA blacklist plus rank/alpha rules with dict-list config ownership in `LoRAModule.py`; removed string literal parsing and generic UIState list-string handling for rule lists.
- Restored selective gradient checkpointing through `gradient_checkpointing_layers` config ownership, Sangoi UI binding, UIState comma-text conversion, and `checkpointing_util.py` layer-name filtering.
- Kept current upstream LoRA saver/conversion ownership under `modules.util.convert.lora.*` with legacy safetensors support.
- Reapplied TrainGPS/DataRecorder/Prodigy/TensorBoard hooks in the upstream trainer lifecycle.
- Fixed `TrainGPS` to iterate `NamedParameterGroupCollection` through `unique_name_mapping` and `by_unique_name`, capture every parameter in each group, preserve epoch/group reference shape, keep penalty tensors connected to autograd, and load saved delta patterns back into the shape consumed by the penalty path.
- Updated Sangoi/UI bindings to use current upstream `components.path_entry` and removed stale `components.file_entry` calls.

## Key Files Resolved Or Repaired

- `modules/trainer/GenericTrainer.py`
- `modules/sangoi/TrainGPS.py`
- `modules/sangoi/SangoiTab.py`
- `modules/util/config/TrainConfig.py`
- `modules/util/checkpointing_util.py`
- `modules/util/config/BaseConfig.py`
- `modules/util/enum/LossWeight.py`
- `modules/util/enum/ModelFormat.py`
- `modules/util/enum/Optimizer.py`
- `modules/util/enum/TimestepDistribution.py`
- `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py`
- `modules/modelSetup/mixin/ModelSetupNoiseMixin.py`
- `modules/modelSetup/BaseStableDiffusionSetup.py`
- `modules/modelSetup/BaseStableDiffusionXLSetup.py`
- `modules/modelSetup/StableDiffusionXLLoRASetup.py`
- `modules/module/LoRAModule.py`
- `modules/util/loss/masked_loss.py`
- `modules/util/create.py`
- `modules/util/bf16_stochastic_rounding.py`
- `modules/util/optimizer_util.py`
- `modules/util/ui/UIState.py`
- `modules/ui/TrainUI.py`
- `modules/ui/TrainingTab.py`
- `modules/ui/SampleFrame.py`
- `modules/util/commands/TrainCommands.py`
- `requirements-global.txt`
- `training_presets/*`

## Validation

Commands completed successfully:

```bash
git diff --name-only --diff-filter=U
rg -uu --glob '!.git/**' --glob '!.venv*/**' --glob '!venv*/**' --glob '!__pycache__/**' -n '^(<<<<<<<|\|\|\|\|\|\||=======|>>>>>>>)' .
rg -n '(^|[^A-Za-z0-9_])(delta_pattern_metric|charbonnier_strength|file_entry)([^A-Za-z0-9_]|$)' modules/sangoi modules/modelSetup modules/module modules/ui modules/util modules/trainer
rg -n 'getattr\(config, "(debugoi|lora_layers_blacklist|gradient_checkpointing_layers|lora_modules_rank_rules|lora_modules_alpha_rules|gen_lora_keys|full_vae_mf|save_predictions|bucket_ratio|sangoi_huber_strength|sangoi_charbonnier_strength|priority_temperature|priority_lr|priority_beta|priority_radius|train_gps_save_it|train_gps_use_it|train_gps_weight|train_gps_path|data_recorder|run_number)"' modules
git diff --cached --name-only --diff-filter=ACMR -z -- '*.py' | xargs -0 -r python -m py_compile
git diff --cached --check
```

Blocked validation:

```bash
python - <<'PY'
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LossWeight import LossWeight
PY
```

The runtime smoke could not run in the current shell because `/home/lucas/.pyenv/versions/3.12.10/bin/python` cannot import `torch` (`ModuleNotFoundError: No module named 'torch'`). No local virtual environment was visible from the workspace root during validation.

## Remaining Gate

Run `Senior Code Reviewer` against the staged change set before final handoff.

## Post-Review-Prep Correction

Before a final reviewer verdict was issued, the lead stopped the stale review lane after finding a substantive `TrainGPS` issue in the staged surface.

Correction applied:

- `modules/sangoi/TrainGPS.py` now preserves the saved reference shape as `epoch -> group -> scalar delta norm` instead of flattening group names and losing the epoch key.
- TrainGPS now captures all parameters in each `NamedParameterGroupCollection` group, computes group delta norms across every parameter in the group, and compares current group norms against the matching epoch/group reference values.
- Invalid TrainGPS reference pattern files now re-raise after logging, so `setup_for_use()` cannot silently initialize with an empty reference pattern.

Additional validation after correction:

```bash
python -m py_compile modules/sangoi/TrainGPS.py
rg -n 'param_collection\.groups|key\.split\("/"|startswith\(reference_key_prefix\)|save_group_deltas|log_group_deltas' modules/sangoi/TrainGPS.py modules/trainer/GenericTrainer.py modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py
```

## Post-Review-Prep Fail-Loud Correction

A second stale review lane was stopped before verdict after the lead found fail-silent TrainGPS paths.

Correction applied:

- `modules/modelSetup/StableDiffusionXLLoRASetup.py` now raises directly when TrainGPS is enabled but cannot initialize instead of logging and disabling `model.train_gps`.
- `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py` now raises when `train_gps_use_it` is enabled but no TrainGPS instance was initialized, and no longer suppresses TrainGPS penalty computation errors.
- `modules/sangoi/TrainGPS.py` now raises on incompatible reference group sets, empty reference patterns, and failed delta-log saves.
- `modules/sangoi/DataRecorder.py` now re-raises dump failures after logging.

Additional validation after correction:

```bash
python -m py_compile modules/sangoi/TrainGPS.py modules/sangoi/DataRecorder.py modules/modelSetup/StableDiffusionXLLoRASetup.py modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py
rg -n 'Initialization failed|Failed to calculate/apply penalty|Erro em compute_penalty|param_collection\.groups|save_group_deltas|log_group_deltas' modules/sangoi/TrainGPS.py modules/sangoi/DataRecorder.py modules/modelSetup/StableDiffusionXLLoRASetup.py modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py modules/trainer/GenericTrainer.py
```

## Senior Code Reviewer NOT_READY Fixes

The first completed final-gate review returned `NOT_READY` with three findings. Corrections applied:

- Restored the eight upstream preset paths that were missing from the staged index and removed an accidental leading-space ` training_presets/` directory created during the restore. `git diff --cached --name-status main -- training_presets` now reports no differences.
- Reconnected TrainGPS penalty computation to live trainable parameters by using detached tensors only for initial/reference captures and logging, while `compute_penalty()` builds current deltas from non-detached parameters.
- Restored the local selective gradient-checkpointing feature: `TrainConfig.gradient_checkpointing_layers`, explicit `UIState` comma-text handling, a `SangoiTab` entry, and `checkpointing_util.py` filtering by named module-list layer paths.

Validation after the fixes:

```bash
git diff --name-only --diff-filter=U
git diff --name-only
git ls-files --others --exclude-standard
git diff --cached --name-status main -- training_presets
rg -uu --glob '!.git/**' --glob '!.venv*/**' --glob '!venv*/**' --glob '!__pycache__/**' -n '^(<<<<<<<|\|\|\|\|\|\||=======|>>>>>>>)' .
rg -n '(^|[^A-Za-z0-9_])(delta_pattern_metric|charbonnier_strength|file_entry)([^A-Za-z0-9_]|$)' modules/sangoi modules/modelSetup modules/module modules/ui modules/util modules/trainer
rg -n 'getattr\(config, "(debugoi|lora_layers_blacklist|gradient_checkpointing_layers|lora_modules_rank_rules|lora_modules_alpha_rules|gen_lora_keys|full_vae_mf|save_predictions|bucket_ratio|sangoi_huber_strength|sangoi_charbonnier_strength|priority_temperature|priority_lr|priority_beta|priority_radius|train_gps_save_it|train_gps_use_it|train_gps_weight|train_gps_path|data_recorder|run_number)"' modules
rg -n 'Initialization failed|Failed to calculate/apply penalty|Erro em compute_penalty|param_collection\.groups|key\.split\("/"|startswith\(reference_key_prefix\)|save_group_deltas|log_group_deltas' modules/sangoi/TrainGPS.py modules/sangoi/DataRecorder.py modules/modelSetup/StableDiffusionXLLoRASetup.py modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py modules/trainer/GenericTrainer.py
rg -n 'gradient_checkpointing_layers|should_checkpoint_layer|_checkpoint_layer_patterns' modules/util/config/TrainConfig.py modules/util/ui/UIState.py modules/sangoi/SangoiTab.py modules/util/checkpointing_util.py
python -m py_compile modules/sangoi/TrainGPS.py modules/sangoi/DataRecorder.py modules/modelSetup/StableDiffusionXLLoRASetup.py modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py modules/util/config/TrainConfig.py modules/util/ui/UIState.py modules/sangoi/SangoiTab.py modules/util/checkpointing_util.py
git diff --cached --name-only --diff-filter=ACMR -z -- '*.py' | xargs -0 -r python -m py_compile
git diff --cached --check
```

The source-level AST check also passed for duplicate `TrainConfig.default_values()` keys, required Sangoi config defaults, and forbidden `getattr(config, "...")` access to required Sangoi keys.

## Final Code Review

`Senior Code Reviewer` completed with `APPROVE_WITH_FIXES`.

Required fix applied:

- Removed the overwritten duplicate `'name': group.display_name` key from `modules/util/NamedParameterGroup.py` while preserving the surviving runtime key `'name': group.unique_name`.

Verification after the required fix:

```bash
python - <<'PY'
import ast
from pathlib import Path
path = Path('modules/util/NamedParameterGroup.py')
tree = ast.parse(path.read_text(), filename=str(path))
hits = []
for node in ast.walk(tree):
    if isinstance(node, ast.Dict):
        seen = {}
        for key_node in node.keys:
            if isinstance(key_node, ast.Constant):
                key = key_node.value
                if key in seen:
                    hits.append(f'{path}:{key_node.lineno}: duplicate literal dict key {key!r}; first at line {seen[key]}')
                else:
                    seen[key] = key_node.lineno
if hits:
    raise SystemExit('\n'.join(hits))
print('duplicate literal dict-key scan passed')
PY
git diff --cached --check
python -m py_compile modules/util/NamedParameterGroup.py
```
