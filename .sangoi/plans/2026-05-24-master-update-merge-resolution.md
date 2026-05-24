# Master Update Merge Resolution Plan

Difficulty: complex
Status: implemented, final code review approved with fixes applied
Date: 2026-05-24
Branch: `master-update`

## Summary

Resolve the in-progress merge of local `main` into `master-update`, where local `main` is synced to `upstream/master` because `https://github.com/Nerogar/OneTrainer` currently has no `main` ref. Preserve Sangoi-local training behavior and integration points while adopting the upstream tree shape, new model families, dependency updates, and refactors from upstream. Restore upstream `training_presets/*` files where the merge currently has delete/modify conflicts.

## User Intent

The user wants the outdated OneTrainer workspace updated from upstream while keeping local modifications. The merge must not blindly keep the old local version when upstream changed a subsystem deeply. Local modifications must be integrated into the current upstream contracts and structure. `training_presets/*` conflicts may be resolved by accepting upstream.

## Current Evidence

- Current branch: `master-update`.
- Merge state: `HEAD` is local `master` at `47320350`; `MERGE_HEAD` is local `main` at `26987343`.
- Local `main` tracks `upstream/master` because `upstream/main` does not exist.
- Merge base: `71d4646b6782a5bf14bc308291a1a5dacfa2bfda`.
- Divergence from merge base: `master` has 145 local-only commits; `main` has 351 upstream-only commits.
- In-progress merge has 50 unmerged paths: 29 `UU`, 1 `AA`, 20 `DU`.
- Working tree also contains hundreds of automatically merged upstream additions and modifications; do not treat the current working tree as already validated.

## Scope

In scope:

- Resolve all current merge conflicts.
- Preserve local Sangoi behavior by intent, not necessarily by old line placement.
- Accept upstream versions of conflicted `training_presets/*` files.
- Keep local added Sangoi support files unless validation proves an exact local feature is obsolete or broken against upstream contracts.
- Update `.sangoi/CHANGELOG.md` and `.sangoi/task-logs/` after substantive review.

Out of scope:

- Rewriting the training architecture beyond what conflict resolution requires.
- Adding compatibility aliases, dual-read fields, fallback adapters, or old-name sanitizers.
- Preserving deleted local `training_presets/*` deletions.
- Pulling or rebasing from remotes beyond the already fetched upstream state.
- Solving unrelated upstream bugs or unrelated local technical debt.

## Resolution Principles

1. Upstream structure wins when upstream performed broad refactors, new model-family additions, or shared API changes.
2. Local feature intent wins when the local side adds Sangoi-specific behavior, metrics, logging, optimizer behavior, loss weighting, TrainGPS behavior, TensorBoard/DataRecorder behavior, or SDXL-specific training behavior that is still referenced by live local code.
3. The final code must expose one canonical contract for every config field, enum, module API, and UI binding. Do not add compatibility wrappers or stale aliases.
4. Use three-way evidence for each non-trivial conflict: `:1:` for base, `:2:` for local master, and `:3:` for upstream main.
5. Stage each resolved block only after checking that imports and references still point to live canonical names.

## Locked Contract Decisions

- This plan must pass the `Senior Plan Auditor` gate before conflict implementation starts.
- Upstream `huber_strength` and `huber_delta` remain the upstream Huber loss contract only.
- Sangoi custom Huber and Charbonnier weights use canonical local names `sangoi_huber_strength` and `sangoi_charbonnier_strength`; old local uses of `huber_strength` and `charbonnier_strength` must be renamed at every live producer and consumer in the merge resolution.
- `PRIORITY_SAMPLING` is preserved with required canonical `TrainConfig` fields: `priority_temperature: float = 1.0`, `priority_lr: float = 1.0`, `priority_beta: float = 0.9`, and `priority_radius: int = 0`.
- `delta_pattern_metric` is stale residue. Do not preserve it through `getattr`, aliases, or config fields. `TrainGPS` keeps its existing constructor default unless a current canonical field is introduced in this plan; this plan does not introduce one.
- LoRA layer rules have one canonical persisted shape: `lora_modules_rank_rules` is `list[dict[str, int]]`, and `lora_modules_alpha_rules` is `list[dict[str, int | float]]`. Each list item maps a layer-name pattern string to the rule value. `LoRAModule.py` is the parser/validator owner and consumes only this dict shape; do not keep string-literal parsing, dual-shape parsers, or `getattr(..., [])` fallbacks for canonical config keys.
- `lora_layers_blacklist` remains `list[str]` and may be edited through the Sangoi UI as comma-separated text via `UIState`; this is a UI representation for the one canonical list shape, not a second runtime contract. Generic list-string conversion is forbidden for `lora_modules_rank_rules` and `lora_modules_alpha_rules`.
- `gradient_checkpointing_layers` remains a canonical Sangoi `list[str]` selector. It is edited through the Sangoi UI as comma-separated text, consumed by `modules/util/checkpointing_util.py`, and an empty list means every eligible upstream checkpointing layer is checkpointed.
- Required local config fields must be accessed directly after `TrainConfig` ownership is restored. Do not mask missing canonical local fields with `getattr(..., default)`, and do not mirror same-concept local config values into `self.<key>` fields such as `self.run_number`.
- `pytorch-msssim` must be reachable through the active `lib.include.sh` install path. The chosen dependency placement is `requirements-global.txt`; `requirements.txt` and `requirements-sangoi.txt` are not sufficient by themselves because the active installer uses `requirements-global.txt` plus platform requirements.
- Dead local config residue is not preserved. Current local keys `alpha_sangoi`, `sangoi_use_huber`, `sangoi_schedule`, `sangoi_huber_factor`, `sangoi_use_loss_schedule`, `sangoi_enable_tf`, `adpt_dcoef_use_it`, `adpt_dcoef_path`, and `convctrl_use_it` have no live non-config consumer in the inspected local branch and must be removed unless implementation finds a live non-config consumer before editing `TrainConfig.py`.

## Feature Owner Matrix

| Feature / Contract | Canonical Owner | Live Consumers / Producers | Validation |
| --- | --- | --- | --- |
| Sangoi config fields | `modules/util/config/TrainConfig.py` | `modules/sangoi/*`, `modules/modelSetup/*`, `modules/module/LoRAModule.py`, `modules/trainer/GenericTrainer.py`, `modules/ui/*`, `modules/util/ui/UIState.py` | Config uniqueness smoke; no required-key `getattr(config, ...)` fallback |
| `LossWeight.SANGOI` | `modules/util/enum/LossWeight.py` | `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py` | `LossWeight.SANGOI.value == "SANGOI"` and `not LossWeight.SANGOI.supports_flow_matching()` |
| Priority timestep sampling | `modules/util/enum/TimestepDistribution.py` and `modules/modelSetup/mixin/ModelSetupNoiseMixin.py` | `modules/trainer/GenericTrainer.py`, `modules/ui/TimestepDistributionWindow.py`, `modules/util/config/TrainConfig.py` | Priority enum/config asserts and direct field access |
| Sangoi custom Huber/Charbonnier weighting | `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py` | `modules/ui/TrainingTab.py`, `modules/util/config/TrainConfig.py` | Uses `sangoi_huber_strength` and `sangoi_charbonnier_strength`; upstream `huber_strength` remains upstream-only |
| LoRA rank/alpha rules and blacklist | `modules/module/LoRAModule.py` | `modules/util/config/TrainConfig.py`, `modules/util/ui/UIState.py`, `modules/sangoi/SangoiTab.py` | Dict-list rule shape only; no `ast.literal_eval`; blacklist remains `list[str]` |
| Selective gradient checkpointing | `modules/util/checkpointing_util.py` | `modules/util/config/TrainConfig.py`, `modules/util/ui/UIState.py`, `modules/sangoi/SangoiTab.py` | `gradient_checkpointing_layers` remains `list[str]`; empty list keeps upstream all-eligible behavior |
| TrainGPS | `modules/sangoi/TrainGPS.py` | SDXL setup/trainer files and `modules/util/config/TrainConfig.py` | No `delta_pattern_metric`; existing constructor default metric only |
| DataRecorder/TensorBoard local metrics | `modules/sangoi/DataRecorder.py` and `modules/util/TensorBoardManager.py` | `modules/trainer/GenericTrainer.py`, loss/setup code | Imports succeed or missing optional runtime dependency is documented |
| Plain Prodigy local stats/stochastic patch | `modules/util/optimizer/prodigy_extensions.py` and `modules/util/create.py` | `modules/trainer/GenericTrainer.py`, `modules/util/enum/Optimizer.py`, `modules/util/optimizer_util.py` | `Optimizer.PRODIGY.supports_fused_back_pass()` and `hasattr(optimizer, "pop_stats")` guard for stat collection |
| SSIM dependency | `requirements-global.txt` | `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py`, `modules/trainer/GenericTrainer.py` | `grep -n '^pytorch-msssim' requirements-global.txt` |
| Config loading and UI list state | `modules/util/config/BaseConfig.py` and `modules/util/ui/UIState.py` | `TrainConfig` and UI tab bindings | No new fallback behavior for local required config keys; no string/dual-shape support for LoRA rule lists |

## Alternatives Considered

1. Accept all `ours`. Rejected because it would discard 351 upstream commits, new model families, new optimizer/config contracts, and dependency changes.
2. Accept all `theirs`. Rejected because it would drop Sangoi-local modules, TrainGPS/DataRecorder behavior, custom loss weighting, local optimizer instrumentation, and UI/config bindings.
3. Abort the merge, reset onto upstream, and cherry-pick 145 local commits. Rejected for this task because it would replay a large patch stack through the same refactor surface while losing the already exposed `zdiff3` conflict context and rerere preimages.
4. Resolve conflict markers hunk-by-hunk inside the current working tree as the primary method. Rejected because hunk context alone is insufficient after deep upstream refactors; use working-tree markers only as navigation.
5. Use upstream as the structural baseline and transplant local feature intent with three-way inspection. Chosen because it preserves current upstream contracts while retaining local behavior in the smallest auditable way.
6. Create new abstraction layers to isolate Sangoi behavior before resolving. Rejected because conflict resolution must repair existing owners first and new wrappers would create avoidable ownership and compatibility residue.

## Planned Touchpoints

Conflict paths to resolve:

- `modules/dataLoader/StableDiffusionXLBaseDataLoader.py`
- `modules/model/BaseModel.py`
- `modules/modelLoader/stableDiffusionXL/StableDiffusionXLModelLoader.py`
- `modules/modelSampler/StableDiffusionXLSampler.py`
- `modules/modelSaver/mixin/LoRASaverMixin.py`
- `modules/modelSaver/stableDiffusionXL/StableDiffusionXLLoRASaver.py`
- `modules/modelSetup/BaseModelSetup.py`
- `modules/modelSetup/BaseStableDiffusionSetup.py`
- `modules/modelSetup/BaseStableDiffusionXLSetup.py`
- `modules/modelSetup/StableDiffusionXLLoRASetup.py`
- `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py`
- `modules/modelSetup/mixin/ModelSetupNoiseMixin.py`
- `modules/module/LoRAModule.py`
- `modules/trainer/GenericTrainer.py`
- `modules/ui/LoraTab.py`
- `modules/ui/TopBar.py`
- `modules/ui/TrainUI.py`
- `modules/ui/TrainingTab.py`
- `modules/util/bf16_stochastic_rounding.py`
- `modules/util/checkpointing_util.py`
- `modules/util/commands/TrainCommands.py`
- `modules/util/config/TrainConfig.py`
- `modules/util/create.py`
- `modules/util/enum/ModelFormat.py`
- `modules/util/enum/Optimizer.py`
- `modules/util/enum/TimestepDistribution.py`
- `modules/util/loss/masked_loss.py`
- `modules/util/optimizer_util.py`
- `modules/util/ui/UIState.py`
- `requirements-global.txt`
- `training_presets/*` files currently in `DU` conflict state

Adjacent auto-merged invariant paths to inspect and update if required by resolved contracts:

- `requirements.txt`
- `requirements-sangoi.txt`
- `lib.include.sh`
- `modules/model/StableDiffusionXLModel.py`
- `modules/util/config/BaseConfig.py`
- `modules/util/enum/LossWeight.py`
- `modules/ui/TimestepDistributionWindow.py`
- `modules/ui/SampleFrame.py`
- `modules/util/ui/components.py`

Local added paths expected to remain unless proven dead during validation:

- `modules/sangoi/DataRecorder.py`
- `modules/sangoi/Logger.py`
- `modules/sangoi/SangoiTab.py`
- `modules/sangoi/TrainGPS.py`
- `modules/sangoi/logFun.py`
- `modules/util/TensorBoardManager.py`
- `modules/util/optimizer/prodigy_extensions.py`
- `requirements-sangoi.txt`
- `scripts/loadTime.py`

Doctrine artifacts:

- `.sangoi/plans/2026-05-24-master-update-merge-resolution.md`
- `.sangoi/CHANGELOG.md`
- `.sangoi/task-logs/2026-05-24-master-update-merge-resolution.md`

## Implementation Steps

1. Preserve merge anchors before editing.
   - Record `git rev-parse HEAD`, `git rev-parse MERGE_HEAD`, `git merge-base HEAD MERGE_HEAD`, and `git diff --name-only --diff-filter=U`.
   - Confirm `git config --get rerere.enabled` remains `true` and `git config --get merge.conflictstyle` remains `zdiff3`.
   - Do not edit conflict files until the current plan version has an `APPROVE` or `APPROVE_WITH_FIXES` result from `Senior Plan Auditor`.

2. Resolve `training_presets/*` conflicts by accepting upstream.
   - For every unmerged path under `training_presets/`, use `git checkout --theirs -- <path>` followed by `git add <path>`.
   - Done when no path appears in `git diff --name-only --diff-filter=U -- training_presets`.

3. Resolve config, enum, and dependency contracts first.
   - `TrainConfig.py`: start from upstream shape and add local Sangoi fields and defaults in one canonical location. Preserve or create these canonical local keys because they have a live non-config consumer or a live owner named in this plan: `debugoi`, `lora_layers_blacklist`, `gradient_checkpointing_layers`, `lora_modules_rank_rules`, `lora_modules_alpha_rules`, `gen_lora_keys`, `full_vae_mf`, `save_predictions`, `bucket_ratio`, `sangoi_huber_strength`, `sangoi_charbonnier_strength`, `priority_temperature`, `priority_lr`, `priority_beta`, `priority_radius`, `train_gps_save_it`, `train_gps_use_it`, `train_gps_weight`, `train_gps_path`, `data_recorder`, and `run_number`. Remove dead local config residue: `alpha_sangoi`, `sangoi_use_huber`, `sangoi_schedule`, `sangoi_huber_factor`, `sangoi_use_loss_schedule`, `sangoi_enable_tf`, `adpt_dcoef_use_it`, `adpt_dcoef_path`, and `convctrl_use_it`, unless a live non-config consumer is found before editing. Keep upstream `huber_strength` and `huber_delta` as upstream-owned fields.
   - `BaseConfig.py`: inspect the upstream config loader while resolving `TrainConfig.py`; do not add new fallback, dual-shape, alias, or list-string compatibility for Sangoi config keys. Keep upstream generic loader behavior only where unchanged by this merge resolution.
   - `LossWeight.py` is auto-merged but must be verified: keep upstream `supports_flow_matching()` and add local `SANGOI` only if the current merged file lacks it; `SANGOI` must not report flow-matching support.
   - `TimestepDistribution.py`: keep upstream `INVERTED_PARABOLA` and local `PRIORITY_SAMPLING`.
   - `ModelFormat.py`: keep upstream `LEGACY_SAFETENSORS` and `COMFY_LORA`; avoid duplicate local additions.
   - `Optimizer.py`: keep upstream optimizer members and keep local Prodigy fused-back-pass eligibility for `Optimizer.PRODIGY` because the chosen behavior preserves the local plain-Prodigy `patch_prodigy` path. Also keep upstream ProdigyPlus/Adv fused-back-pass entries.
   - `requirements-global.txt`, `requirements.txt`, `requirements-sangoi.txt`, and `lib.include.sh`: keep upstream version bumps, put the exact line `pytorch-msssim # Sangoi SSIM loss/debug metrics` in `requirements-global.txt` so the active installer reaches it, and avoid restoring removed obsolete global dependencies unless imported by resolved code. This keeps the prior local unpinned requirement deliberately unpinned because the local requirement was unpinned before the merge.

4. Resolve core training math and optimizer behavior.
   - `ModelSetupDiffusionLossMixin.py`: start from upstream current implementation and reapply local `sangoi_huber_loss`, `sangoi_masked_loss` integration, `LossWeight.SANGOI`, SSIM helpers, and TensorBoard/DataRecorder metrics only where tensor shapes and upstream flow-matching logic remain valid. Local custom Huber and Charbonnier code must read `sangoi_huber_strength` and `sangoi_charbonnier_strength`; upstream `huber_strength` and `huber_delta` remain upstream Huber only.
   - `masked_loss.py`: preserve upstream behavior and add local `sangoi_masked_loss` as a canonical helper only if it is still called by resolved loss code.
   - `ModelSetupNoiseMixin.py`: preserve upstream distributions and add `PRIORITY_SAMPLING` support without replacing upstream timestep handling. The priority implementation must access `config.priority_temperature`, `config.priority_lr`, `config.priority_beta`, and `config.priority_radius` directly after adding them to `TrainConfig`.
   - `bf16_stochastic_rounding.py`: keep upstream seeding and stochastic rounding behavior, then reapply local buffered helpers if local optimizer extensions still call them.
   - `create.py`: preserve upstream optimizer creation, new optimizer families, and new config parameters; apply local `patch_prodigy` to plain Prodigy after upstream construction using `optimizer_config.stochastic_rounding`; keep upstream `slice_p` and ProdigyPlus/Adv handling.
   - `optimizer_util.py`: keep upstream optimizer default dictionaries and merge local Prodigy defaults only where they correspond to live optimizer config fields.

5. Resolve SDXL-specific training pipeline conflicts.
   - For `StableDiffusionXLBaseDataLoader.py`, `StableDiffusionXLModelLoader.py`, `StableDiffusionXLSampler.py`, `StableDiffusionXLLoRASaver.py`, `BaseStableDiffusionXLSetup.py`, and `StableDiffusionXLLoRASetup.py`, use upstream files as the structural baseline and transplant local SDXL-specific behavior: TrainGPS hooks, long-prompt handling, bucket/save-prediction behavior, LoRA layer blacklist/rank/alpha rules, legacy safetensors behavior, and local debug metrics.
   - For `modules/modelSaver/mixin/LoRASaverMixin.py`, resolve the add/add conflict as one canonical upstream-owned mixin: keep upstream import ownership from `modules.util.convert.lora.convert_lora_util`, keep upstream OMI/legacy conversion behavior, keep `LEGACY_SAFETENSORS` handling, and do not preserve the older external `omi_model_standards` import path.
   - Verify that any local import from `modules.sangoi.*` is still used by code in the resolved file. Remove unused imports rather than leaving stale hooks.

6. Resolve shared model/setup/trainer/checkpoint command conflicts.
   - `BaseModel.py` and `BaseModelSetup.py`: preserve upstream model lifecycle fields and add local TrainGPS/DataRecorder or debug fields only where owned by those classes.
   - `BaseStableDiffusionSetup.py`: preserve upstream base behavior and add local behavior only where it still applies to all SD-style models.
   - `LoRAModule.py`: preserve upstream module refactors and integrate local rank/alpha rule logic, blacklist logic, and logging without duplicating ownership. The only accepted rule shape is the canonical dict-list shape from `TrainConfig`; `lora_modules_alpha_rules` accepts `int | float` values and rejects strings; remove `ast.literal_eval` string parsing and any fallback `getattr` access for required rule fields.
   - `GenericTrainer.py`: preserve upstream trainer flow, multi-GPU additions, abort behavior, sampling/caching changes, and stochastic rounding seed logic; reapply local DataRecorder, TrainGPS, TensorBoard, Prodigy stats extraction, SSIM debug, priority-sampling priority updates, and save-profile behavior at current upstream lifecycle points. Prodigy stat collection must call the canonical local `pop_stats` API only when the optimizer exposes it; non-Progidy optimizers must not get a compatibility fake. Priority sampling must compare `self.config.timestep_distribution` to `TimestepDistribution.PRIORITY_SAMPLING`, must call the update hook after upstream `calculate_loss(...)` produces the pre-division loss and before or around gradient normalization/backward, and must pass a tensor-compatible detached loss such as `loss.detach()` rather than `loss.item()`.
   - `checkpointing_util.py` and `TrainCommands.py`: preserve upstream checkpoint command behavior, reapply selective `gradient_checkpointing_layers` filtering at the checkpointing owner, then reapply local continue/save command intent without stale command aliases.

7. Resolve UI and UI-state conflicts after runtime contracts are fixed.
   - `TrainUI.py`: keep upstream tab creation structure and add the Sangoi tab as one canonical tab if all referenced `UIState` keys exist.
   - `SangoiTab.py`: keep local file, but update UI helper calls if upstream component signatures changed. In particular, upstream renamed `components.file_entry` to `components.path_entry`; no stale `file_entry` call may remain.
   - `LoraTab.py`, `TrainingTab.py`, `TopBar.py`, `TimestepDistributionWindow.py`, `SampleFrame.py`, and `UIState.py`: keep upstream validation/state changes and add local bindings only for resolved config keys. `SangoiTab.py` exposes `gradient_checkpointing_layers` and binds the canonical `TrainConfig` field. `UIState.py` may keep comma-text conversion for explicit `list[str]` UI fields such as `lora_layers_blacklist` and `gradient_checkpointing_layers`, but must not generically stringify `lora_modules_rank_rules` or `lora_modules_alpha_rules`.

8. Stage and inspect incrementally.
   - After each block, run `git diff --name-only --diff-filter=U` and `git status --short`.
   - Stage resolved files with `git add` only after checking for stale imports, duplicate enum/config entries, and broken references.
   - Let rerere record each completed resolution.

9. Validate before committing the merge.
   - Required:
     - `git diff --name-only --diff-filter=U`
     - `rg -uu --glob '!.git/**' --glob '!.venv*/**' --glob '!venv*/**' -n '^(<<<<<<<|\\|\\|\\|\\|\\|\\|\\||=======|>>>>>>>)' .`
     - `source lib.include.sh && prepare_runtime_environment && run_python_in_active_env -m compileall modules scripts`
     - Run an AST no-fallback check for required local config keys:
       ```bash
       source lib.include.sh && prepare_runtime_environment && run_python_in_active_env - <<'PY'
       import ast
       from pathlib import Path

       required = {
           "debugoi",
           "lora_layers_blacklist",
           "gradient_checkpointing_layers",
           "lora_modules_rank_rules",
           "lora_modules_alpha_rules",
           "gen_lora_keys",
           "full_vae_mf",
           "save_predictions",
           "bucket_ratio",
           "sangoi_huber_strength",
           "sangoi_charbonnier_strength",
           "priority_temperature",
           "priority_lr",
           "priority_beta",
           "priority_radius",
           "train_gps_save_it",
           "train_gps_use_it",
           "train_gps_weight",
           "train_gps_path",
           "data_recorder",
           "run_number",
       }
       hits = []
       for path in Path("modules").rglob("*.py"):
           tree = ast.parse(path.read_text(), filename=str(path))
           for node in ast.walk(tree):
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
                   if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value in required:
                       hits.append(f"{path}:{node.lineno}:{node.args[1].value}")
       if hits:
           raise SystemExit("\\n".join(hits))
       PY
       ```
     - Run a source-level duplicate check for `TrainConfig.default_values()` keys:
       ```bash
       source lib.include.sh && prepare_runtime_environment && run_python_in_active_env - <<'PY'
       import ast
       from collections import Counter
       from pathlib import Path

       path = Path("modules/util/config/TrainConfig.py")
       tree = ast.parse(path.read_text(), filename=str(path))
       keys = []
       for class_node in [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainConfig"]:
           for fn in [n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == "default_values"]:
               for node in ast.walk(fn):
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "append":
                       if node.args and isinstance(node.args[0], ast.Tuple) and node.args[0].elts:
                           first = node.args[0].elts[0]
                           if isinstance(first, ast.Constant) and isinstance(first.value, str):
                               keys.append(first.value)
       dupes = sorted(k for k, count in Counter(keys).items() if count > 1)
       if dupes:
           raise SystemExit("duplicate TrainConfig keys: " + ", ".join(dupes))
       PY
       ```
     - `! rg -n '(^|[^A-Za-z0-9_])(delta_pattern_metric|charbonnier_strength|file_entry)([^A-Za-z0-9_]|$)' modules/sangoi modules/modelSetup modules/module modules/ui modules/util`
     - `! rg -n 'ast\\.literal_eval|lora_modules_(rank|alpha)_rules.*str|isinstance\\([^\\n]+, str\\)' modules/module/LoRAModule.py`
     - `! rg -n 'alpha_sangoi|sangoi_use_huber|sangoi_schedule|sangoi_huber_factor|sangoi_use_loss_schedule|sangoi_enable_tf|adpt_dcoef_use_it|adpt_dcoef_path|convctrl_use_it' modules`
     - `! rg -n 'timestep_distribution\\s*==\\s*[\"'\"']PRIORITY_SAMPLING[\"'\"']|update_sampler_priorities\\([^\\n]*loss\\.item\\(' modules/trainer/GenericTrainer.py`
     - `! rg -n 'lora_modules_(rank|alpha)_rules' modules/util/ui/UIState.py`
     - `! rg -n 'self\\.(run_number|debugoi|data_recorder|gen_lora_keys|bucket_ratio|train_gps_(save_it|use_it|weight|path))\\s*=' modules/trainer/GenericTrainer.py`
     - Run a source guard check for Prodigy stats:
       ```bash
       source lib.include.sh && prepare_runtime_environment && run_python_in_active_env - <<'PY'
       from pathlib import Path

       path = Path("modules/trainer/GenericTrainer.py")
       lines = path.read_text().splitlines()
       violations = []
       for index, line in enumerate(lines):
           if ".pop_stats(" in line:
               window = "\n".join(lines[max(0, index - 10): index + 1])
               if "hasattr" not in window or "pop_stats" not in window:
                   violations.append(f"{path}:{index + 1}: unguarded pop_stats call")
       if violations:
           raise SystemExit("\n".join(violations))
       PY
       ```
     - `rg -n '(^|[^A-Za-z0-9_])huber_strength([^A-Za-z0-9_]|$)' modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py modules/ui/TrainingTab.py modules/util/config/TrainConfig.py`
     - Inspect the `huber_strength` matches above and verify they are upstream Huber-only, not Sangoi custom-loss code.
   - Targeted import/config checks:
     - Core dependency/config smoke:
       ```bash
       source lib.include.sh && prepare_runtime_environment && run_python_in_active_env - <<'PY'
       from modules.util.config.TrainConfig import TrainConfig
       from modules.util.enum.LossWeight import LossWeight
       from modules.util.enum.ModelFormat import ModelFormat
       from modules.util.enum.Optimizer import Optimizer
       from modules.util.enum.TimestepDistribution import TimestepDistribution
       import modules.util.create
       import modules.util.loss.masked_loss
       import modules.util.optimizer.prodigy_extensions
       import modules.sangoi.SangoiTab

       assert LossWeight.SANGOI.value == "SANGOI"
       assert not LossWeight.SANGOI.supports_flow_matching()
       assert TimestepDistribution.PRIORITY_SAMPLING.value == "PRIORITY_SAMPLING"
       assert ModelFormat.LEGACY_SAFETENSORS.file_extension() == ".safetensors"
       assert Optimizer.PRODIGY.supports_fused_back_pass()
       TrainConfig.default_values()
       PY
       ```
     - Sangoi config uniqueness smoke:
       ```bash
       source lib.include.sh && prepare_runtime_environment && run_python_in_active_env - <<'PY'
       from modules.util.config.TrainConfig import TrainConfig

       expected = {
           "debugoi",
           "lora_layers_blacklist",
           "gradient_checkpointing_layers",
           "lora_modules_rank_rules",
           "lora_modules_alpha_rules",
           "gen_lora_keys",
           "full_vae_mf",
           "save_predictions",
           "bucket_ratio",
           "sangoi_huber_strength",
           "sangoi_charbonnier_strength",
           "priority_temperature",
           "priority_lr",
           "priority_beta",
           "priority_radius",
           "train_gps_save_it",
           "train_gps_use_it",
           "train_gps_weight",
           "train_gps_path",
           "data_recorder",
           "run_number",
       }
       config = TrainConfig.default_values()
       missing = expected.difference(config.types)
       assert not missing, sorted(missing)
       for key in expected:
           assert list(config.types).count(key) == 1, key
       assert "huber_strength" in config.types
       assert "huber_delta" in config.types
       PY
       ```
     - Local-added module smoke:
       ```bash
       source lib.include.sh && prepare_runtime_environment && run_python_in_active_env - <<'PY'
       import modules.sangoi.DataRecorder
       import modules.sangoi.Logger
       import modules.sangoi.TrainGPS
       import modules.sangoi.logFun
       import modules.util.TensorBoardManager
       import modules.util.optimizer.prodigy_extensions
       import modules.sangoi.SangoiTab
       PY
       source lib.include.sh && prepare_runtime_environment && run_python_in_active_env -m py_compile scripts/loadTime.py
       ```
     - UI helper smoke/static check:
       ```bash
       ! rg -n 'file_entry' modules/sangoi modules/ui
       source lib.include.sh && prepare_runtime_environment && run_python_in_active_env - <<'PY'
       import modules.util.ui.components
       import modules.sangoi.SangoiTab
       PY
       ```
     - Dependency reachability check:
       ```bash
       grep -n '^pytorch-msssim' requirements-global.txt
       grep -n 'requirements-global.txt' lib.include.sh
       ```
   - If optional runtime dependencies are missing, record the exact missing import and still complete syntax/conflict/static validation.

10. Review and finish.
    - Inspect `git diff --cached --stat` and targeted diffs for resolved conflict files.
    - Assemble a `Senior Code Reviewer` bundle that includes the plan, pre/post unmerged path list, change-set anchors from `git diff --cached --name-status` and `git diff --cached --stat`, a staged surface ledger classifying every staged path as conflict-resolved, accepted-upstream preset, retained local-added, auto-merged upstream-only, or doctrine/bookkeeping, focused diffs for every resolved conflict group, feature owner matrix closure, validation command output, known risks, environment limitations, explicitly out-of-scope surfaces, and confirmation that the review bundle matches the current staged artifacts.
    - Run `Senior Code Reviewer` on the substantive merge resolution before handoff.
    - Apply only reviewer-enumerated local non-substantive fixes if the verdict is `APPROVE_WITH_FIXES`; rerun review if any fix changes runtime behavior, contracts, expected invariants, user-facing technical claims, validation truth, or the substantive reviewed artifact surface.
    - Create or update `.sangoi/CHANGELOG.md` and `.sangoi/task-logs/2026-05-24-master-update-merge-resolution.md` after the substantive reviewer result. Bookkeeping-only memorialization of the reviewer result does not require a fresh review.
    - Commit the merge only after validation and review are clean enough for handoff.

## Validation Acceptance Criteria

- `git diff --name-only --diff-filter=U` prints nothing.
- `rg -uu --glob '!.git/**' --glob '!.venv*/**' --glob '!venv*/**' -n '^(<<<<<<<|\\|\\|\\|\\|\\|\\|\\||=======|>>>>>>>)' .` prints nothing.
- `source lib.include.sh && prepare_runtime_environment && run_python_in_active_env -m compileall modules scripts` succeeds, or fails only on an explicitly documented pre-existing environment/dependency limitation that is not caused by the merge resolution.
- Resolved `TrainConfig` contains upstream current fields plus local Sangoi fields exactly once.
- Resolved enums contain upstream current members plus local members that are still referenced.
- `requirements-global.txt` keeps upstream dependency updates and contains `pytorch-msssim` for local Sangoi SSIM imports.
- `training_presets/*` conflicts are resolved to upstream versions.
- Local Sangoi modules remain importable or any non-importability is explained by a missing optional UI/runtime dependency, not by broken local paths.
- Anti-residue checks find no `delta_pattern_metric`, no bare local `charbonnier_strength`, no stale local custom-loss use of bare `huber_strength`, no `components.file_entry` calls, no dead local config residue, no LoRA-rule `ast.literal_eval` or string parsing, no generic UIState string handling for LoRA rule lists, no same-concept local config mirrors, no stale priority string/loss-scalar update, no unguarded Prodigy `pop_stats`, no duplicate `TrainConfig.default_values()` keys, and no `getattr(config, ...)` fallback for required local config keys.

## Stop Conditions

- A local feature cannot be mapped onto the upstream contract without inventing a new owner, wrapper, compatibility alias, or fallback adapter.
- A required local config key is referenced by live code but cannot be represented in upstream `TrainConfig` without conflicting with an upstream field.
- `PRIORITY_SAMPLING` cannot be integrated with the canonical priority config fields and direct accesses.
- LoRA rule consumers require string parsing or dual-shape compatibility after canonical dict-list ownership is restored.
- Validation reveals a syntax or import break in a resolved conflict file that cannot be fixed inside the existing owner.
- A dependency required by resolved local code was removed upstream and no current requirements path can carry it without contradicting upstream constraints.
- A conflict resolution would require dropping Sangoi-local behavior outside the user-approved `training_presets/*` exception.

## Locked Assumptions

- The user-approved `training_presets/*` exception means accept upstream versions for the current delete/modify preset conflicts.
- The user wants local modifications preserved as behavior, not necessarily as old code layout.
- No compatibility preservation is requested.
- `upstream/master` is the correct upstream source because `upstream/main` does not exist.
- The merge should remain on `master-update` and should not rewrite `master` or remote branches during this task.
