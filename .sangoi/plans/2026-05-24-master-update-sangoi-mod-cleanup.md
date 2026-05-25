# Master Update Sangoi Mod Cleanup Plan

Difficulty: complex
Status: implementation in progress; user-added uv-managed Python/bootstrap instruction scope is being folded into the approved cleanup before validation
Date: 2026-05-24
Branch: `master-update`
Baseline diff: `upstream/master..master-update`
Patrol report: `.sangoi/reports/2026-05-24-master-update-diff-patrol.md`

## Summary

Clean up the handmade Sangoi modifications on `master-update` that were flagged by the 25-receipt diff-scoped Patrol. Preserve intended SDXL training-quality behavior, but remove or rename stale fields, broken advertised capabilities, unused debug surfaces, and hidden side effects. The cleanup is executed as staged doctrine-mode repair by canonical owner, not as another patch stack.

## User Intent

The user wants the local Sangoi modifications cleaned up after the upstream merge because several training parameters were previously hardcoded or hand-edited before Codex CLI was available. The cleanup must keep useful local training behavior, especially SDXL-oriented behavior, while integrating it into current upstream contracts instead of leaving fragile patches, hidden config fields, or stale UI promises.

## Current Evidence

- Current branch is `master-update` at `bfe2ff5a5c2d9aa41cab285696fa896103d1df48`.
- Upstream structural baseline is `upstream/master` at `2698734305620513dd97b3956973f45b10c43be0`.
- The Patrol diff scope was `upstream/master..master-update`.
- Patrol completed 5 usable rounds for each role: Recon Scout, Bug Hunter, Contract Soup Hunter, Standards Auditor, and Overengineering Auditor.
- Patrol merged 15 hotspot clusters, `MH-01` through `MH-15`.
- First Senior Plan Auditor gate returned `REPLAN` with `Repair mode: Local patch`; the first revision incorporated that local patch set.
- Second Senior Plan Auditor gate returned `REPLAN` with `Repair mode: Local patch`; this revision incorporates that second local patch set before runtime/config/UI implementation starts.
- Third Senior Plan Auditor gate returned `APPROVE_WITH_FIXES` with two local plan fixes: narrow the removed-name scan so legitimate scheduler `current_step` parameters are not forbidden globally, and add a TrainGPS missing-reference-group smoke. Those fixes are applied in this plan.
- The previous broad `/.sangoi` ignore made the plan untracked. This revision repairs `.gitignore` before runtime implementation: `.sangoi/plans` and `.sangoi/task-logs` are trackable, while generated `.sangoi/reports/` and `.sangoi/.tools/` remain ignored.
- Shell entrypoints `install.sh`, `run-cmd.sh`, `start-ui.sh`, and `update.sh` are now staged as executable `100755` to match upstream entrypoint mode.
- `scripts/train_ui.py` now calls `script_imports()` so ZLUDA behavior matches upstream by default.
- `requirements.txt` no longer includes `-r requirements-sangoi.txt`; active Sangoi runtime dependencies now live in `requirements-global.txt`.
- After implementation began, the user explicitly deferred tests until the fix batch and uv-managed Python bootstrap are complete.
- The requested bootstrap reference is `/home/lucas/work/stable-diffusion-webui-codex`: it uses repo-local `.uv/bin/uv`, uv-managed Python, `.venv`, and validation through local uv instead of system Python.
- This WSL checkout's local torch validation environment is CPU-only. Local validation must not treat CUDA unavailability as proof of repository breakage.
- A new root `AGENTS.md` is required for this workspace because launch, dependency, UI, and training-mod owners are now spread across multiple high-risk files.
- The uv addendum `Senior Plan Auditor` gate returned `REPLAN` with `Repair mode: Local patch`: lock `.venv` ownership, split bootstrap from validation, tighten TrainGPS fail-loud behavior, make Docker use current source, and strengthen validation/fan-in.
- The follow-up uv addendum `Senior Plan Auditor` gate returned `REPLAN` with `Repair mode: Local patch`: remove internal `OT_PYTHON_VENV` ownership, make `lib.include.sh` safe to re-source, add LinuxCloud/Docker docs fan-in, and strengthen validation for local uv, `.uv` scans, priority sampling, and Sangoi flow rejection.
- The latest uv addendum `Senior Plan Auditor` gate returned `REPLAN` with `Repair mode: Local patch`: lock TrainGPS empty-save failure, fix validation ordering, align `uv tool run` with `AGENTS.md`, make cloud upstream install fail loud, declare DataRecorder nullability, and add Docker fan-in validation.
- The final uv addendum `Senior Plan Auditor` gate returned `APPROVE_WITH_FIXES`: add cloud source-identity preflight, `.dockerignore`, current-source docs, NVIDIA UI CUDA requirements, and matching validation. Those local fixes are applied before bootstrap.

## Scope

In scope:

- Fix all 15 Patrol hotspot clusters, either by repairing the behavior or removing the broken/stale surface completely.
- Preserve intended SDXL training behavior: Sangoi loss modes/weights, masked-gradient behavior, LoRA rank/alpha rules, LoRA blacklist, TrainGPS when it is supported, DataRecorder when it records valid step-aligned data, and Prodigy normal-step statistics.
- Expose useful configuration through canonical `TrainConfig` fields and UI bindings when the feature remains supported.
- Remove config fields, UI controls, modules, scripts, requirements files, and docs references whose only live behavior is stale, unused, or misleading.
- Repair packaging/tooling surfaces that can hide future changes or break launch behavior.
- Replace POSIX Conda/host-Python bootstrap with repo-local uv-managed Python: `.uv/bin/uv`, `.uv/python`, `.venv`, `.python-version`, and `uv pip` installation against the existing requirements owners.
- Add root `AGENTS.md` with durable local toolchain, WSL CPU validation, and owner-map instructions copied/adapted from the webui workspace where they fit OneTrainer.
- Update `.sangoi/CHANGELOG.md` and create a task log before final code review so substantive doctrine claims are included in the review bundle. Post-review bookkeeping is limited to outcome receipt and consistency confirmation.

Out of scope:

- Preserving old config keys for compatibility. No aliases, dual-read paths, migration adapters, old-name sanitizers, or fallback wrappers.
- Migrating user-local JSON configs unless the user separately asks for one-time migration.
- Reimplementing plain Prodigy fused-back-pass unless a separate user request makes that exact capability a goal.
- Rebuilding the whole training architecture or changing upstream model-family behavior unrelated to the Patrol hotspots.
- Changing `training_presets/*` contents beyond ignore-policy repair; the user already allowed presets to return to upstream.
- Adding a new debug image/prediction export product surface in this pass.
- Converting Windows `.bat` launchers to uv-managed Python in this pass unless the user explicitly expands the scope to Windows runtime parity. Current uv-managed bootstrap scope is the active WSL/Linux path plus Docker surfaces that call the POSIX scripts.

## Locked Contract Decisions

1. `master-update` is the only implementation branch for this cleanup.
2. `upstream/master` remains the structural baseline. `origin/SotA04022025+Mods` remains a comparison anchor only for training-quality behavior already ported into `master-update`, not a wholesale source branch.
3. Breaking propagation is the default. Removed or renamed fields must be removed from `TrainConfig`, UI bindings, runtime consumers, docs, tests, examples, prompts, and validation scans in the same change.
4. Plain `Optimizer.PRODIGY` will no longer advertise `supports_fused_back_pass()` in this cleanup. Keep normal Prodigy behavior and local statistics extraction; fail loud if the user enables fused-back-pass with plain Prodigy. A correct `step_parameter` implementation is deferred unless the user explicitly asks for Prodigy fused-back-pass.
5. `LearningRateScheduler.PARABOLIC` remains supported and must be wired in `create_lr_scheduler()` to `lr_lambda_parabolic`. It must not silently fall through to constant scheduling.
6. `bucket_ratio` is removed. The new canonical config field is `aspect_bucketing_quantization_override: int`, default `0`. `0` means use the model-specific quantization passed by the setup call. Positive values override that quantization. Negative values raise `ValueError` at data-loader construction. Every live producer and consumer uses only the new field.
7. `modules/util/torch_util.py` owns exactly one `get_tensor_data`, one `add_dummy_grad_fn_`, and one `has_grad_fn`. Recursive calls inside `get_tensor_data` call `get_tensor_data`, never a stale `get_tensors` name.
8. `BaseModelSetup.calculate_loss(...)` continues to return a scalar tensor. Per-sample loss for priority timestep sampling is a separate canonical payload key: `model_output_data["loss_per_sample"]`, produced by diffusion loss code before scalar reduction and consumed only by `ModelSetupNoiseMixin.update_priorities()` through the trainer.
9. `TimestepDistribution.PRIORITY_SAMPLING` remains supported only where per-sample diffusion loss is available. It must fail loud if `loss_per_sample` is missing, scalar, wrong-length for the batch timesteps, NaN/Inf, or negative. It must not expand a scalar batch loss across all timesteps.
10. `LossMode.SANGOI` and `LossWeight.SANGOI` remain diffusion-only. Flow-matching paths must reject them with clear errors, and UI/config validation must prevent or surface unsupported selections instead of allowing a later obscure crash.
11. TrainGPS remains supported only for model types where `ModelType.is_stable_diffusion_xl()` is true plus `TrainingMethod.LORA` in this pass. Other model types or training methods fail loud when `train_gps_use_it` or `train_gps_save_it` is enabled.
12. Add canonical `TrainGPSPenaltyMetric` enum with `MSE` and `COSINE`. The new `TrainConfig.train_gps_penalty_metric` default is `TrainGPSPenaltyMetric.MSE`. `COSINE` is accepted only when at least two common reference groups exist; one-group cosine fails loud because it is inert for positive norm scalars.
13. TrainGPS JSON has exactly this current schema and rejects every other shape, including the old `epoch_* -> group -> value` dictionary:

```json
{
  "schema_version": 1,
  "kind": "sangoi_train_gps_delta_log",
  "metric": "MSE",
  "records": [
    {"epoch": 1, "group": "text_encoder_lora", "delta_norm": 0.123}
  ]
}
```

14. DataRecorder output JSON has exactly this current schema and removes `set_current_step` / `current_step` hidden state:

```json
{
  "schema_version": 1,
  "kind": "sangoi_data_recorder",
  "records": [
    {
      "step": 42,
      "group": "unet_lora",
      "d_num_pdgy": 0.1,
      "d_den_pdgy": null,
      "dlr_pdgy": 0.0003,
      "grad_norm": 1.5
    }
  ]
}
```

Each metric value is either a finite JSON number or `null` when the producer has no value for that metric at that step.

15. `gen_lora_keys` is removed. The new canonical config field is `lora_key_export_path: str`, default `""`. Empty disables export. A non-empty path writes one deterministic JSON manifest after LoRA modules are constructed. Relative paths resolve under `config.workspace_dir`; output is atomically written; keys are fully prefixed; an existing file is overwritten only when the exact configured path names it. The old `generate_keys_by_block_file` symbol is removed entirely. `modules/module/LoRAModule.py` owns `build_lora_key_manifest(wrappers: Iterable[LoRAModuleWrapper]) -> dict` and `export_lora_key_manifest(export_path: str | Path, wrappers: Iterable[LoRAModuleWrapper], workspace_dir: str | Path) -> Path`; LoRA setup code calls the exporter after all model adapters are assigned. Manifest shape is exactly `{"schema_version": 1, "kind": "sangoi_lora_key_manifest", "keys_by_block": {"BASE": ["prefixed.key"]}}`, with deterministic sorted block and key order.
16. `save_predictions` and `full_vae_mf` are removed from config, UI, and SDXL setup code because current code creates ephemeral payload keys with no consumer or persistence owner. If image/latent export is wanted later, it must be a separate explicit artifact-export feature.
17. `debugoi` is removed unless implementation finds a live non-config runtime consumer before editing. Current evidence shows only config/UI references.
18. `modules/sangoi/logFun.py` remains the Sangoi logging/progress owner. `rich` is added to the active install path in `requirements-global.txt`. Unused `modules/sangoi/Logger.py` is deleted if the implementation-time import scan still shows no live importer.
19. `pytorch-msssim` has one dependency owner: `requirements-global.txt`. `requirements.txt` keeps `-r requirements-global.txt` and must no longer include `-r requirements-sangoi.txt`. `requirements-sangoi.txt` is deleted unless implementation finds a live installer path that consumes it and justifies keeping it as a real owner.
20. `.style.yapf` is deleted. Ruff is the Python lint/format owner through `pyproject.toml` and `.pre-commit-config.yaml`; the pre-commit config uses `ruff-check` and `ruff-format`, not the old single legacy `ruff` hook.
21. `modules/util/loss/dynamic_loss_strength.py`, `scripts/loadTime.py`, unused `TensorBoardManager.py`, and unused `NamedParameterGroup.is_enabled` / `set_requires_grad()` are removed when import scans confirm they have no live runtime consumer.
22. UI lifecycle callbacks have one owner per event. Workspace-dir TensorBoard restart has one owner: the `UIState` variable trace. The duplicate direct `path_entry(..., command=...)` restart callback and its stale `_on_workspace_dir_change` method are deleted. Pause/resume requests are owned by `TrainCommands`; trainer state transitions are owned by `GenericTrainer`; UI state changes are scheduled back onto Tk via `TrainUI.after(0, ...)` callback wrappers only.
23. Launch behavior returns to the upstream ZLUDA contract: `scripts/train_ui.py` calls `script_imports()` unless the user later asks for a supported explicit disable switch.
24. Shell entrypoint executable modes are restored to upstream: `install.sh`, `run-cmd.sh`, `start-ui.sh`, and `update.sh` are `100755`.
25. `.gitignore` must not ignore the entire `.sangoi` directory. Keep `.sangoi/CHANGELOG.md`, `.sangoi/plans/`, and `.sangoi/task-logs/` trackable. Ignore generated `.sangoi/reports/` and `.sangoi/.tools/`. Remove root `/training_presets` ignore and let `training_presets/.gitignore` own user-preset ignoring while preserving built-in `#*.json` presets.
26. POSIX launch scripts use repo-local uv-managed Python. `OT_UV_VERSION` defaults to `0.9.17`, `OT_PYTHON_VERSION` defaults to `3.13`, `.python-version` is `3.13`, uv installs under `.uv/bin` and `.uv/python`, and `.venv` is the only supported POSIX virtual-environment path. Old Conda/host-Python environment variables, including `OT_PYTHON_VENV`, fail loud instead of being ignored.
27. Bootstrap commands may provision `.uv` and `.venv`: `OT_PLATFORM_REQUIREMENTS=requirements-default.txt ./install.sh` for this WSL checkout, or `./update.sh` after the plan fixes. Repository validation starts only after `./.uv/bin/uv` and `.venv/bin/python` exist; if either is missing during validation, stop and report the missing local toolchain path.
28. Validation commands for this workspace use the local uv runner: `./.uv/bin/uv run --python .venv/bin/python --no-sync ...`. Direct system/global `python`, `pip`, or `uv` is not an acceptable validation path unless the user explicitly requests it for that exact command. `uv tool run` is allowed only through the repo-local uv executable, must pin `--python 3.13`, and must store tool environments under `.uv/tools` / `.uv/tool-bin`.
29. Local WSL validation is CPU-only. Lightweight smokes stay CPU-only, validation sets `OT_PLATFORM_REQUIREMENTS=requirements-default.txt`, POSIX bootstrap installs that platform requirements file through `uv pip --torch-backend cpu`, and CUDA training/GPU validation is not run from this checkout unless the user confirms a GPU-capable environment.
30. TrainGPS reference JSON metric is authoritative and must match `TrainConfig.train_gps_penalty_metric`. A missing reference epoch for an active TrainGPS use run fails loud instead of producing a zero penalty. `GenericTrainer` performs a common TrainGPS support preflight before model-family-specific setup. Any requested TrainGPS save with no initialized TrainGPS object, or with no logged epoch delta records, fails loud.
31. Dockerfiles changed in this cleanup build from the current Docker build context/source tree, not from a fresh upstream clone. RunPod/Vast build comments must instruct building from the repository root with `-f resources/docker/<file>`. `.dockerignore` excludes local environments, git metadata, generated `.sangoi` outputs, and workspace/model/user-data directories from Docker build contexts. Docker work is limited to existing Dockerfiles that already use POSIX launch scripts; no new Docker parity or build guarantee is introduced.
32. `modules/cloud/LinuxCloud.py` participates in POSIX bootstrap fan-in because it installs/updates/runs OneTrainer remotely through the same shell scripts. It checks remote `.uv/bin/uv` and `.venv/bin/python`, not a stale `venv` directory. `CloudConfig.install_cmd` defaults to empty for this checkout, and cloud auto-install fails loud unless the user provides an explicit install command that does not point at upstream `Nerogar/OneTrainer`. Existing remote git checkouts also fail loud when their remotes point at upstream `Nerogar/OneTrainer`.

## Feature Owner Matrix

| Feature / Contract | Canonical Owner | Live Consumers / Producers | Validation |
| --- | --- | --- | --- |
| Parabolic LR scheduler | `modules/util/create.py`, `modules/util/lr_scheduler_util.py`, `modules/util/enum/LearningRateScheduler.py` | training optimizer/scheduler setup and UI enum selection | helper and factory scheduler smoke |
| Prodigy normal-step stats | `modules/util/optimizer/prodigy_extensions.py`, `modules/util/create.py`, `modules/trainer/GenericTrainer.py` | plain Prodigy optimizer creation and recorder logging | Prodigy no longer reports fused support; stats API smoke when dependency exists |
| Tensor utility recursion/offload helpers | `modules/util/torch_util.py` | checkpointing, offload, optimizer device movement | AST duplicate/undefined-name check and nested tensor smoke |
| Aspect bucket quantization override | `modules/util/config/TrainConfig.py`, `modules/dataLoader/mixin/DataLoaderText2ImageMixin.py`, `modules/ui/TrainUI.py` | data-loader aspect bucketing | default uses setup quantization; positive override wins; negative rejects |
| Priority timestep sampling | `modules/modelSetup/mixin/ModelSetupNoiseMixin.py`, diffusion loss producers, `modules/trainer/GenericTrainer.py` | training loop with `TimestepDistribution.PRIORITY_SAMPLING` | no scalar expansion; missing/bad vector rejection; accepted vector update smoke |
| Sangoi loss support boundary | `modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py`, `modules/ui/TrainingTab.py`, loss enums | diffusion setups and UI | flow rejection smoke and UI/source support gate check |
| GenerateLosses offline loss calculation | `modules/module/GenerateLossesModel.py`, `scripts/calculate_loss.py` | offline loss JSON generation | call-signature AST check and progress update smoke |
| TrainGPS | `modules/sangoi/TrainGPS.py`, `modules/modelSetup/StableDiffusionXLLoRASetup.py`, `modules/trainer/GenericTrainer.py`, `modules/sangoi/SangoiTab.py` | SDXL LoRA setup, trainer finalization, Sangoi UI | support gate, schema, metric, and one-group cosine fail-loud smokes |
| DataRecorder | `modules/sangoi/DataRecorder.py`, `modules/trainer/GenericTrainer.py`, Prodigy stats extraction | gzip JSON recorder output | schema smoke with explicit step/group records |
| LoRA key export | `modules/module/LoRAModule.py`, `modules/util/config/TrainConfig.py`, `modules/sangoi/SangoiTab.py` | LoRA module setup and Sangoi UI | deterministic manifest write smoke and old-key scan |
| Removed debug/prediction surfaces | `TrainConfig`, `TrainUI`, `BaseStableDiffusionXLSetup`, orphan modules/scripts | none after cleanup | forbidden-reference scan |
| Rich logging/progress | `modules/sangoi/logFun.py`, `requirements-global.txt`, `requirements.txt` | Sangoi logging and GenerateLosses progress; POSIX/Windows/Docker/manual installs | import smoke and dependency-owner scan |
| Launch/tooling policy | `scripts/train_ui.py`, shell scripts, `.gitignore`, `.editorconfig`, `pyproject.toml`, `.pre-commit-config.yaml` | runtime entrypoints and repository hygiene | mode check, config syntax/source checks, pre-commit validation |
| uv-managed POSIX bootstrap | `lib.include.sh`, `install.sh`, `run-cmd.sh`, `start-ui.sh`, `update.sh`, `modules/cloud/LinuxCloud.py`, `pyproject.toml`, `.python-version`, `.gitignore`, `.dockerignore`, `LAUNCH-SCRIPTS.md`, Dockerfiles, `docs/DockerImage.md` | WSL/Linux launch, cloud remote launch, CLI, update, Docker installs, local validation | local uv bootstrap/version smoke after fix batch; no system Python validation |
| Local workspace instructions | `AGENTS.md` | future Codex/tooling sessions in this checkout | source review plus toolchain/stale-instruction scan |

## Alternatives Considered

1. Keep all local knobs and only patch crashes. Rejected because Patrol found hidden fields, side effects, and unused surfaces that would keep breaking as upstream evolves.
2. Remove all Sangoi additions and return to upstream. Rejected because the user explicitly wants to preserve training-improving local SDXL behavior.
3. Implement every advertised capability, including plain Prodigy fused-back-pass and debug prediction export. Rejected because those are large new features; the current advertised surfaces are broken and not proven necessary for training quality. Removing broken advertisement is the safer exact cleanup.
4. Add compatibility aliases for old config fields such as `bucket_ratio` and `gen_lora_keys`. Rejected by workspace policy and because aliases would preserve the current contract soup.
5. Create wrapper modules around current code to isolate Sangoi behavior. Rejected because the existing canonical owners are clear; wrappers would add another ownership layer.
6. Stage the cleanup by contract owner and validation boundary. Chosen because it repairs runtime bugs first, then fixes config/UI ownership, then deletes stale surfaces with scans proving no live references remain.

## Hotspot Coverage Map

| Patrol hotspot | Cleanup owner in this plan | Resolution |
| --- | --- | --- |
| `MH-01` PARABOLIC exposed but not wired | Scheduler batch | Wire `PARABOLIC` in scheduler factory and validate helper/factory behavior |
| `MH-02` bucket quantization owner conflict | Config/data-loader batch | Remove `bucket_ratio`; add explicit override field with model default semantics |
| `MH-03` Prodigy fused/stat incomplete | Optimizer batch | Remove plain Prodigy fused support; keep normal stats patch |
| `MH-04` torch_util helper drift | Runtime utility batch | Collapse duplicate helpers and fix undefined recursion |
| `MH-05` TrainGPS unsafe | TrainGPS batch | Gate support to SDXL LoRA, add metric enum/schema, fix math/API |
| `MH-06` save_predictions/full_vae_mf no persistence | Removal batch | Remove unsupported config/UI/runtime payload keys |
| `MH-07` Sangoi loss/masking/priority owner clarity | Loss/sampler batch | Keep diffusion-only Sangoi loss, enforce support gates, use per-sample priority loss |
| `MH-08` GenerateLosses/Rich drift | Offline/progress batch | Fix call signature/progress owner; declare `rich` in dependency owners |
| `MH-09` LoRA key export lossy/side-effectful | LoRA batch | Replace bool side effect with explicit export path and deterministic manifest |
| `MH-10` startup/launch regressions | Launch batch | Restore ZLUDA default and shell executable bits; repair preset ignore policy |
| `MH-11` dependency/tooling split | Tooling batch | Single dependency owners, Ruff/pre-commit cleanup, remove YAPF owner |
| `MH-12` orphan/debug surfaces | Removal batch | Delete confirmed-unused debug modules/scripts/fields |
| `MH-13` UI callback ownership spread | UI lifecycle batch | Collapse workspace-dir restart to one trace owner and prove pause/resume command, trainer, and Tk UI callback sequencing |
| `MH-14` DataRecorder telemetry schema | Recorder batch | Versioned record schema with explicit step/group |
| `MH-15` repo policy/tooling drift | Tooling batch | Repair `.gitignore`, `.editorconfig`, script modes, and lint residues |

## Implementation Steps

1. Gate this revised plan before validation and final review.
   - Original runtime/config/UI implementation was gated by the third `Senior Plan Auditor` verdict before the user added uv-managed bootstrap scope.
   - Submit this revised plan to `Senior Plan Auditor` with a complete plan-gate brief as a recovery/addendum gate for the expanded uv-managed bootstrap and local instruction scope.
   - Do not run validation or final code review until this revised gate returns `APPROVE` or `APPROVE_WITH_FIXES` and required plan fixes are applied.
   - Done when the revised gate verdict is recorded and this plan status is updated.

2. Confirm repository-policy and doctrine-artifact trackability before runtime edits.
   - Keep the already-applied `.gitignore` repair: no broad `/.sangoi`, no root `/training_presets`, generated `.sangoi/reports/` and `.sangoi/.tools/` ignored, `.refs` ignored.
   - Confirm `.sangoi/plans/2026-05-24-master-update-sangoi-mod-cleanup.md` and future `.sangoi/task-logs/2026-05-24-master-update-sangoi-mod-cleanup.md` are not ignored.
   - Done when `git check-ignore` proves plan/task-log paths are trackable and generated report/tool paths remain ignored.

3. Freeze current evidence and create implementation branch state.
   - Record `git branch --show-current`, `git rev-parse HEAD upstream/master`, `git status --short --ignored . .sangoi`, and `git diff --name-status upstream/master..master-update`.
   - Record current import scans for planned deletion candidates.
   - Done when evidence is stored in the task log draft or command transcript for later handoff.

4. Fix runtime bugs that can break execution before broader renames.
   - Import `lr_lambda_parabolic` in `modules/util/create.py` and add the `LearningRateScheduler.PARABOLIC` match case.
   - Collapse `modules/util/torch_util.py` duplicate helper definitions and replace stale `get_tensors` recursion with `get_tensor_data`.
   - Fix `GenerateLossesModel.start()` so `calculate_loss` is called with the current canonical signature, uses `self.model.train_progress`, updates progress inside the batch loop, and does not keep an unused `tqdm` iterator when the Rich progress owner is used.
   - Remove `Optimizer.PRODIGY` from `supports_fused_back_pass()` and ensure plain Prodigy with `fused_back_pass=True` fails at optimizer creation or UI validation with a clear unsupported-combination error.
   - Done when compile and source checks pass for these owners.

5. Repair aspect bucketing and config/UI ownership.
   - Rename `TrainConfig.bucket_ratio` to `aspect_bucketing_quantization_override` everywhere.
   - In `DataLoaderText2ImageMixin._aspect_bucketing_in`, reject negative override values, use `_aspect_bucketing_quantization` when override is `0`, and use the configured override when it is positive.
   - Update `TrainUI` to expose the new field with a label that states `0` uses the model default.
   - Remove all `bucket_ratio` references from code, UI, and current-behavior docs. This plan may keep removed-name inventories as execution targets.
   - Done when default config keeps upstream/model-specific quantization, a positive override is validated, and a negative override raises.

6. Repair Sangoi loss and priority sampling contracts.
   - Keep `LossMode.SANGOI`, `LossWeight.SANGOI`, and the masked-gradient hook only on supported diffusion paths.
   - Add fail-loud support checks for flow-matching or unsupported model setups that select Sangoi loss mode/weight or priority sampling without per-sample loss.
   - Make diffusion loss code place unreduced per-sample losses in `model_output_data["loss_per_sample"]` before scalar reduction when priority sampling is active.
   - Update `GenericTrainer` to pass `model_output_data["loss_per_sample"]` to `ModelSetupNoiseMixin.update_priorities()`; never pass the reduced scalar loss for priority updates.
   - Update `ModelSetupNoiseMixin.update_priorities()` to reject missing, scalar, wrong-length, non-finite, or negative per-sample priority losses.
   - Done when priority sampling rejects bad inputs and changes different timestep priorities for different accepted per-sample losses in a synthetic smoke.

7. Repair TrainGPS into a supported SDXL LoRA feature.
   - Add `TrainGPSPenaltyMetric` enum and `TrainConfig.train_gps_penalty_metric` with default `MSE`.
   - Bind the metric in `SangoiTab` near existing TrainGPS controls.
   - Gate TrainGPS setup so only SDXL LoRA can enable save/use flags.
   - Replace deprecated `torch.norm` calls with `torch.linalg.vector_norm`.
   - Change default penalty math to MSE and make one-group cosine fail loud.
   - Replace unversioned TrainGPS JSON with the exact schema from locked decision 13.
   - Update save and load paths to validate `schema_version`, `kind`, `metric`, records, group names, epochs, and numeric finite `delta_norm` strictly.
   - Done when synthetic TrainGPS tests cover MSE, two-group cosine, one-group cosine failure, missing group failure, empty-save failure, current-schema acceptance, and old-schema rejection.

8. Repair DataRecorder telemetry.
   - Replace nested metric-list storage with the exact record-row schema from locked decision 14.
   - Remove `set_current_step` and `current_step`.
   - Update trainer/Prodigy stats extraction to pass `step=train_progress.global_step` and group names explicitly.
   - Remove hidden model-attribute telemetry carriers and stale mirrors.
   - Done when a gzip JSON smoke validates schema version, kind, record shape, finite numeric metric values, accepted `null` metric values, and rejected non-finite metric values.

9. Repair LoRA key export and LoRA rule side effects.
   - Remove `gen_lora_keys` from `TrainConfig`, `SangoiTab`, `LoRAModule`, and validation surfaces.
   - Add `lora_key_export_path: str = ""` to `TrainConfig` and UI.
   - Export after `self.lora_modules` is fully constructed only when the configured path is non-empty.
   - Resolve relative export paths under `config.workspace_dir`, write JSON atomically, include fully prefixed module keys, and avoid timestamp-based implicit file naming.
   - Add `build_lora_key_manifest(...)` and `export_lora_key_manifest(...)` in `modules/module/LoRAModule.py`; call the exporter from LoRA setup after all model adapters are assigned, not from each wrapper constructor in a way that can overwrite partial manifests.
   - Remove `generate_keys_by_block_file` entirely.
   - Keep existing dict-list `lora_modules_rank_rules` and `lora_modules_alpha_rules` as the only rule shape.
   - Done when old `gen_lora_keys` and `generate_keys_by_block_file` references are gone and an export smoke proves deterministic fully-prefixed output.

10. Remove unsupported debug/prediction surfaces.
    - Delete `save_predictions` and `full_vae_mf` config fields, UI switches, and SDXL setup branches that only populate unsaved payload keys.
    - Delete `debugoi` config/UI if implementation-time scan still shows no live runtime consumer.
    - Delete `modules/util/loss/dynamic_loss_strength.py`, `scripts/loadTime.py`, `modules/sangoi/Logger.py`, `requirements-sangoi.txt`, `.style.yapf`, and `modules/util/TensorBoardManager.py` if import scans still show no live consumer.
    - Remove unused `NamedParameterGroup.is_enabled` and `set_requires_grad()` if the implementation-time scan still shows no live consumer.
    - Done when forbidden-reference scans show no stale names and compileall passes.

11. Repair dependency owners, launch, and repo policy.
    - Keep `modules/sangoi/logFun.py` as the single Sangoi Rich logging/progress owner.
    - Add `rich` to `requirements-global.txt` if any `rich` import remains.
    - Keep exactly one `pytorch-msssim` dependency line in `requirements-global.txt`.
    - Either pin `rich` and `pytorch-msssim` to implementation-tested versions or add a current unpinned-exception rationale beside the dependency lines.
    - Remove `-r requirements-sangoi.txt` from `requirements.txt` before deleting `requirements-sangoi.txt`.
    - Update any install/docs/Docker references that describe Sangoi-specific requirements so they describe only the current dependency owners.
    - Restore `scripts/train_ui.py` to call `script_imports()` so ZLUDA behavior matches upstream unless disabled by an explicit future feature.
    - Restore executable mode `100755` on `install.sh`, `run-cmd.sh`, `start-ui.sh`, and `update.sh`.
    - Add `.python-version` with `3.13`.
    - Update `pyproject.toml` so the project declares `requires-python = ">=3.10,<3.14"` and uv uses managed Python.
    - Rewrite POSIX launch bootstrap so `install.sh`, `update.sh`, `start-ui.sh`, and `run-cmd.sh` all source `lib.include.sh`, install repo-local uv into `.uv/bin`, install managed Python under `.uv/python`, create the fixed `.venv`, and run scripts through `uv run --python .venv/bin/python --no-sync`.
    - Keep the existing requirements files as dependency owners; install them through `uv pip --python .venv/bin/python`.
    - Reject removed Conda/host-Python environment variables loudly in `lib.include.sh`, including user-provided `OT_PYTHON_VENV`; do not use `OT_PYTHON_VENV` as an internal owner after rejecting it.
    - Make `update.sh` source `lib.include.sh` and run removed-env preflight before `git pull`, then re-source the newest `lib.include.sh` after the pull.
    - Update Dockerfiles that call the POSIX scripts so required bootstrap tools are installed, uv cache cleanup uses the repo-local uv, NVIDIA UI installs CUDA requirements explicitly, and RunPod/Vast Dockerfiles build from the current source tree instead of cloning upstream.
    - Add `.dockerignore` so current-source Docker contexts exclude local environments, git metadata, generated `.sangoi` outputs, and workspace/model/user-data directories.
    - Update `modules/cloud/LinuxCloud.py` so remote install/update checks for `.uv/bin/uv` plus `.venv/bin/python` and does not branch on a stale `venv` directory.
    - Make `modules/util/config/CloudConfig.py` default `install_cmd` empty and keep the cloud tab exposing it as a required explicit command when automatic install is enabled. `LinuxCloud` must fail loud for blank commands, upstream `Nerogar/OneTrainer` clone commands, and existing remote git checkouts whose remotes point at upstream OneTrainer instead of silently installing/updating upstream OneTrainer.
    - Update `docs/DockerImage.md` so it points at the real `resources/docker/NVIDIA-UI.Dockerfile` path and states that the old NVIDIA UI image is an example, not validated training infrastructure.
    - Update `README.md`, `LAUNCH-SCRIPTS.md`, and `docs/CliTraining.md` so Linux/Mac/WSL documentation describes the uv-managed flow only.
    - Add root `AGENTS.md` with local uv validation rules, WSL CPU torch note, and current key owner paths.
    - Repair `.editorconfig` so `root = true` is first and duplicate wildcard sections are removed.
    - Replace the pre-commit legacy `ruff` hook with explicit `ruff-check` and `ruff-format` hooks that match the active `pyproject.toml` Ruff sections.
    - Done when dependency scans, mode checks, and repo policy scans pass.

12. Repair UI lifecycle callback ownership.
    - Keep the `workspace_dir` `UIState` trace as the only automatic restart owner for always-on TensorBoard.
    - Delete the duplicate direct `path_entry(..., command=...)` restart callback.
    - Keep cleanup of the trace id on close.
    - Prove pause/resume sequencing remains split across `TrainCommands` request methods, `GenericTrainer` state transitions, `TrainCallbacks`, and `TrainUI.after(0, ...)` UI handlers.
    - Done when validation shows exactly one restart path for workspace-dir changes and pause/resume callbacks never update Tk widgets directly from trainer threads.

13. Run full validation.
    - Run every command in `## Validation Commands`.
    - This step is intentionally deferred until the user-approved uv bootstrap fix batch is complete.
    - Before entering repository validation, bootstrap the local environment once with `OT_PLATFORM_REQUIREMENTS=requirements-default.txt ./install.sh`; validation itself must stop if the local uv or `.venv` toolchain is missing.
    - If the active runtime environment cannot be prepared because a dependency install fails, stop and report the fatal bootstrap failure instead of substituting partial validation.
    - If pre-commit or Ruff auto-fixes files, rerun `git diff --check`, compileall, and the stale-reference scans after the auto-fix.
    - Done when required validations pass or a precise blocker is recorded.

14. Doctrine bookkeeping, final review, and outcome receipt.
    - Update `.sangoi/CHANGELOG.md` and create `.sangoi/task-logs/2026-05-24-master-update-sangoi-mod-cleanup.md` before assembling the final review bundle.
    - Freeze the review surface with `git status --porcelain=v1 --untracked-files=all`, `git diff HEAD --check`, required-artifact `git ls-files --error-unmatch` checks after staging, and exact cumulative diff anchors for the review bundle.
    - Assemble a code-review bundle that includes runtime/config/UI/bootstrap changes plus `.sangoi/CHANGELOG.md` and the task log, then run the mandatory `Senior Code Reviewer` gate.
    - Apply exactly any reviewer-approved local non-substantive fixes if the verdict is `APPROVE_WITH_FIXES`.
    - After final review, add only an outcome receipt or reviewer-verdict note if needed, then run a consistency check that no post-review note changed runtime behavior, contracts, expected invariants, validation truth, or substantive technical claims.
    - Done when the reviewer gate is complete and doctrine artifacts describe current behavior only.

## Validation Commands

```bash
set -euo pipefail

# Current-maintainer surfaces only. Historical `.sangoi` evidence is excluded from stale-name scans.
current_roots=()
for path in \
    modules scripts tests docs examples resources prompts AGENTS.md README.md LAUNCH-SCRIPTS.md \
    pyproject.toml .python-version .pre-commit-config.yaml .gitignore .dockerignore requirements.txt \
    requirements-global.txt requirements-dev.txt install.sh run-cmd.sh start-ui.sh update.sh \
    install.bat update.bat lib.include.sh; do
    if [ -e "$path" ]; then
        current_roots+=("$path")
    fi
done

run_removed_name_scans() {
    rg -n 'bucket_ratio|gen_lora_keys|generate_keys_by_block_file|full_vae_mf|save_predictions|debugoi|dynamic_loss_strength|requirements-sangoi|TensorBoardManager' "${current_roots[@]}" && exit 1 || true
    rg -n 'modules/sangoi/Logger\.py|from modules\.sangoi\.Logger|import modules\.sangoi\.Logger' "${current_roots[@]}" && exit 1 || true
    rg -n 'set_current_step|self\.current_step|\.set_current_step\(' modules/sangoi/DataRecorder.py modules/trainer/GenericTrainer.py && exit 1 || true
}

run_current_source_doc_scans() {
    source_doc_roots=()
    for path in README.md docs examples prompts AGENTS.md LAUNCH-SCRIPTS.md; do
        if [ -e "$path" ]; then
            source_doc_roots+=("$path")
        fi
    done
    if [ "${#source_doc_roots[@]}" -gt 0 ]; then
        rg -n 'git clone (https://github\.com/Nerogar/OneTrainer(\.git)?|git@github\.com:Nerogar/OneTrainer(\.git)?)' "${source_doc_roots[@]}" && exit 1 || true
    fi
}

# Local bootstrap must be complete before validation starts. Bootstrap separately with:
# OT_PLATFORM_REQUIREMENTS=requirements-default.txt ./install.sh
export OT_PLATFORM_REQUIREMENTS=requirements-default.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -x ./.uv/bin/uv ]]; then
    echo "missing local uv at ./.uv/bin/uv; run OT_PLATFORM_REQUIREMENTS=requirements-default.txt ./install.sh first"
    exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
    echo "missing local Python at .venv/bin/python; run OT_PLATFORM_REQUIREMENTS=requirements-default.txt ./install.sh first"
    exit 1
fi
test "$(cat .python-version)" = "3.13"

# General hygiene
git status --short --ignored . .sangoi
git diff --check
git diff HEAD --check
if rg -uu --glob '!.git/**' --glob '!.refs/**' --glob '!.uv/**' --glob '!.venv*/**' --glob '!venv*/**' -n '^(<<<<<<<|\|\|\|\|\|\|\||=======|>>>>>>>)' .; then
    echo "conflict markers found"
    exit 1
fi
bash -n install.sh run-cmd.sh start-ui.sh update.sh lib.include.sh
bash -c 'set -euo pipefail; source lib.include.sh; source lib.include.sh'

# Doctrine artifact and generated-artifact ignore policy.
git check-ignore -v .sangoi/plans/2026-05-24-master-update-sangoi-mod-cleanup.md && exit 1 || true
git check-ignore -v .sangoi/task-logs/2026-05-24-master-update-sangoi-mod-cleanup.md && exit 1 || true
git check-ignore -v .sangoi/reports/2026-05-24-master-update-diff-patrol.md
git check-ignore -v .sangoi/.tools/patrol_report.lock
rg -n '^/training_presets$|^/\.sangoi$' .gitignore && exit 1 || true
for ignored_path in \
    '.git/' '.refs/' '.uv/' '.venv*/' 'venv*/' '.sangoi/reports/' '.sangoi/.tools/' \
    'workspace*/' 'models*/' 'training_concepts/' 'training_samples/' 'training_deltas/' \
    'training_user_settings/' 'external/' 'config.json' 'secrets.json'; do
    grep -Fx -- "$ignored_path" .dockerignore
done
run_current_source_doc_scans

source lib.include.sh

# Tooling and metadata checks run before behavioral smokes so any auto-fixes happen first.
./.uv/bin/uv tool run --python 3.13 --from pre-commit pre-commit validate-config
grep -q 'repo-local uv tools' requirements-dev.txt
set +e
./.uv/bin/uv tool run --python 3.13 --from pre-commit pre-commit run --all-files
pre_commit_status=$?
set -e
if [ "$pre_commit_status" -ne 0 ]; then
    if ! git diff --quiet; then
        git diff --check
        git diff HEAD --check
        run_removed_name_scans
        ./.uv/bin/uv tool run --python 3.13 --from pre-commit pre-commit run --all-files
    else
        exit "$pre_commit_status"
    fi
fi

run_python_in_active_env - <<'PY'
from pathlib import Path
import tomllib
pyproject = tomllib.loads(Path("pyproject.toml").read_text())
assert pyproject["project"]["requires-python"] == ">=3.10,<3.14"
assert pyproject["tool"]["uv"]["python-preference"] == "managed"
assert pyproject["tool"]["uv"]["python-downloads"] == "manual"
assert Path(".python-version").read_text().strip() == "3.13"
print("uv metadata smoke ok")
PY

grep -Fx -- "-r requirements-global.txt" requirements.txt
grep -Fx -- "-r requirements-cuda.txt" requirements.txt
rg -n '^--extra-index-url' requirements-default.txt && exit 1 || true
grep -Fx -- "torch==2.9.1+cpu" requirements-default.txt
grep -Fx -- "torchvision==0.24.1+cpu" requirements-default.txt
grep -Fq -- '--torch-backend cpu' lib.include.sh
rg -n -F -- '-e git+' requirements-global.txt && exit 1 || true
grep -Fx -- 'diffusers @ git+https://github.com/huggingface/diffusers.git@0f1abc4' requirements-global.txt
grep -Fx -- 'mgds @ git+https://github.com/Nerogar/mgds.git@9320a69' requirements-global.txt
grep -Fx -- 'muon-optimizer @ git+https://github.com/KellerJordan/Muon.git@f90a42b' requirements-global.txt
grep -Eq '^rich .*Unpinned:' requirements-global.txt
grep -Eq '^pytorch-msssim .*Unpinned:' requirements-global.txt

grep -Fq 'source "$(dirname -- "${BASH_SOURCE[0]}")/lib.include.sh"' install.sh
grep -Fq 'source "$(dirname -- "${BASH_SOURCE[0]}")/lib.include.sh"' run-cmd.sh
grep -Fq 'source "$(dirname -- "${BASH_SOURCE[0]}")/lib.include.sh"' start-ui.sh
grep -Fq 'source "$(dirname -- "${BASH_SOURCE[0]}")/lib.include.sh"' update.sh
grep -Fq 'prepare_runtime_environment upgrade' install.sh
grep -Fq 'UV_TOOL_DIR="${SCRIPT_DIR}/.uv/tools"' lib.include.sh
grep -Fq 'UV_TOOL_BIN_DIR="${SCRIPT_DIR}/.uv/tool-bin"' lib.include.sh
grep -n 'git pull' update.sh
test "$(grep -Fn 'source "$(dirname -- "${BASH_SOURCE[0]}")/lib.include.sh"' update.sh | cut -d: -f1)" -lt "$(grep -n 'git pull' update.sh | cut -d: -f1)"
rg -n 'OT_PYTHON_VENV' lib.include.sh LAUNCH-SCRIPTS.md AGENTS.md
rg -n 'export OT_PYTHON_VENV|\$\{OT_PYTHON_VENV\}|\$OT_PYTHON_VENV' lib.include.sh && exit 1 || true
rg -n 'OT_PYTHON_VENV_PATH' lib.include.sh AGENTS.md LAUNCH-SCRIPTS.md README.md docs modules resources && exit 1 || true

grep -Fq 'COPY . /OneTrainer' resources/docker/NVIDIA-UI.Dockerfile
grep -Fq 'COPY . /OneTrainer' resources/docker/RunPod-NVIDIA-CLI.Dockerfile
grep -Fq 'COPY . /OneTrainer' resources/docker/Vast-NVIDIA-CLI.Dockerfile
grep -Fq 'wget curl vim git ca-certificates' resources/docker/NVIDIA-UI.Dockerfile
grep -Fq 'curl ca-certificates git' resources/docker/RunPod-NVIDIA-CLI.Dockerfile
grep -Fq 'curl ca-certificates git' resources/docker/Vast-NVIDIA-CLI.Dockerfile
grep -Fq './install.sh' resources/docker/NVIDIA-UI.Dockerfile
grep -Fq './install.sh' resources/docker/RunPod-NVIDIA-CLI.Dockerfile
grep -Fq './install.sh' resources/docker/Vast-NVIDIA-CLI.Dockerfile
grep -Fq 'OT_PLATFORM_REQUIREMENTS=requirements-cuda.txt' resources/docker/NVIDIA-UI.Dockerfile
grep -Fq 'OT_PLATFORM_REQUIREMENTS=requirements-cuda.txt' resources/docker/RunPod-NVIDIA-CLI.Dockerfile
grep -Fq 'OT_PLATFORM_REQUIREMENTS=requirements-cuda.txt' resources/docker/Vast-NVIDIA-CLI.Dockerfile
grep -Fq './.uv/bin/uv cache clean' resources/docker/RunPod-NVIDIA-CLI.Dockerfile
grep -Fq './.uv/bin/uv cache clean' resources/docker/Vast-NVIDIA-CLI.Dockerfile
grep -Fq -- '-f resources/docker/RunPod-NVIDIA-CLI.Dockerfile' resources/docker/RunPod-NVIDIA-CLI.Dockerfile
grep -Fq -- '-f resources/docker/Vast-NVIDIA-CLI.Dockerfile' resources/docker/Vast-NVIDIA-CLI.Dockerfile
rg -n 'git clone https://github.com/Nerogar/OneTrainer' resources/docker && exit 1 || true
grep -Fq 'resources/docker/NVIDIA-UI.Dockerfile' docs/DockerImage.md
grep -Fq -- '-f resources/docker/NVIDIA-UI.Dockerfile' docs/DockerImage.md
grep -Fq 'example, not validated training infrastructure' docs/DockerImage.md
rg -n ' -f Dockerfile \.' docs/DockerImage.md && exit 1 || true
rg -n '/venv\b|/venv/' modules/cloud/LinuxCloud.py && exit 1 || true
rg -n '/\.venv/bin/python|/\.uv/bin/uv' modules/cloud/LinuxCloud.py
grep -q 'install_cmd", ""' modules/util/config/CloudConfig.py
grep -q 'UPSTREAM_ONETRAINER_SOURCE_PATTERN' modules/cloud/LinuxCloud.py
grep -Fq 'github\.com[:/]+nerogar/onetrainer' modules/cloud/LinuxCloud.py
grep -q 'git -C' modules/cloud/LinuxCloud.py
grep -q 'remote -v' modules/cloud/LinuxCloud.py
grep -q 'Cloud install command points at upstream OneTrainer' modules/cloud/LinuxCloud.py
grep -q 'Cloud remote OneTrainer directory points at upstream OneTrainer' modules/cloud/LinuxCloud.py
grep -q 'Cloud install command is required' modules/cloud/LinuxCloud.py
rg -n 'cloud\.install_cmd' modules/ui/CloudTab.py
grep -q 'Required when automatic install is enabled' modules/ui/CloudTab.py
run_python_in_active_env - <<'PY'
from modules.cloud.LinuxCloud import _uses_upstream_onetrainer_source
bad_sources = [
    "git clone https://github.com/Nerogar/OneTrainer.git",
    "git clone git@github.com:Nerogar/OneTrainer.git",
    "https://github.com/nerogar/onetrainer",
    "ssh://git@github.com/Nerogar/OneTrainer",
]
good_sources = [
    "git clone https://github.com/Nerogar/OneTrainerSangoi.git",
    "git clone git@github.com:lucas/OneTrainer.git",
]
assert all(_uses_upstream_onetrainer_source(source) for source in bad_sources)
assert not any(_uses_upstream_onetrainer_source(source) for source in good_sources)
print("cloud upstream-source rejection smoke ok")
PY

run_python_in_active_env -m compileall modules scripts
run_python_in_active_env - <<'PY'
import rich
import pytorch_msssim
print("runtime dependency import smoke ok")
PY

# Scheduler helper and factory wiring.
run_python_in_active_env - <<'PY'
import torch
from torch.optim import SGD
from modules.util.config.TrainConfig import TrainConfig
from modules.util.create import create_lr_scheduler
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.lr_scheduler_util import lr_lambda_parabolic
config = TrainConfig.default_values()
optimizer = SGD([torch.nn.Parameter(torch.tensor([1.0]))], lr=1.0)
optimizer.param_groups[0]["initial_lr"] = 1.0
fn = lr_lambda_parabolic(scheduler_steps=10, num_cycles=1.0, min_factor=0.25)
assert abs(fn(0) - 1.0) < 1e-9
assert abs(fn(5) - 0.25) < 1e-9
assert abs(fn(10) - 1.0) < 1e-9
scheduler = create_lr_scheduler(
    config=config,
    optimizer=optimizer,
    learning_rate_scheduler=LearningRateScheduler.PARABOLIC,
    warmup_steps=0,
    num_cycles=1.0,
    min_factor=0.25,
    num_epochs=1,
    batch_size=1,
    approximate_epoch_length=10,
    gradient_accumulation_steps=1,
)
assert scheduler.lr_lambdas[0](5) == fn(5)
print("parabolic factory smoke ok")
PY

# Config ownership and removed-name scan.
run_python_in_active_env - <<'PY'
from modules.util.config.TrainConfig import TrainConfig
config = TrainConfig.default_values()
assert config.aspect_bucketing_quantization_override == 0
assert config.lora_key_export_path == ""
assert not hasattr(config, "bucket_ratio")
assert not hasattr(config, "gen_lora_keys")
assert not hasattr(config, "full_vae_mf")
assert not hasattr(config, "save_predictions")
assert not hasattr(config, "debugoi")
print("config ownership smoke ok")
PY
run_removed_name_scans

# Aspect bucketing override behavior: default setup quantization, positive override, negative rejection.
run_python_in_active_env - <<'PY'
from modules.util.config.TrainConfig import TrainConfig
import modules.dataLoader.mixin.DataLoaderText2ImageMixin as dl_module

class SpyAspectBucketing:
    def __init__(self, *, quantization, **kwargs):
        self.quantization = quantization

class SpyModule:
    def __init__(self, **kwargs):
        pass

class ConcreteText2ImageMixin(dl_module.DataLoaderText2ImageMixin):
    def _preparation_modules(self, *args, **kwargs):
        return []

    def _cache_modules(self, *args, **kwargs):
        return []

    def _output_modules(self, *args, **kwargs):
        return []

    def _debug_modules(self, *args, **kwargs):
        return []

original_aspect = dl_module.AspectBucketing
original_calc = dl_module.CalcAspect
original_single = dl_module.SingleAspectCalculation
try:
    dl_module.AspectBucketing = SpyAspectBucketing
    dl_module.CalcAspect = SpyModule
    dl_module.SingleAspectCalculation = SpyModule
    mixin = ConcreteText2ImageMixin()
    config = TrainConfig.default_values()
    config.aspect_ratio_bucketing = True
    config.aspect_bucketing_quantization_override = 0
    assert mixin._aspect_bucketing_in(config, 64)[1].quantization == 64
    config.aspect_bucketing_quantization_override = 32
    assert mixin._aspect_bucketing_in(config, 64)[1].quantization == 32
    config.aspect_bucketing_quantization_override = -1
    try:
        mixin._aspect_bucketing_in(config, 64)
    except ValueError:
        pass
    else:
        raise AssertionError("negative aspect_bucketing_quantization_override was accepted")
finally:
    dl_module.AspectBucketing = original_aspect
    dl_module.CalcAspect = original_calc
    dl_module.SingleAspectCalculation = original_single
print("aspect bucketing override smoke ok")
PY

# Prodigy fused support must be withdrawn for plain Prodigy, while normal stats remain explicit.
run_python_in_active_env - <<'PY'
import importlib.util
import torch
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.config.TrainConfig import TrainConfig
from modules.util.create import create_optimizer
from modules.util.enum.Optimizer import Optimizer

assert not Optimizer.PRODIGY.supports_fused_back_pass()
config = TrainConfig.default_values()
config.optimizer.optimizer = Optimizer.PRODIGY
config.optimizer.fused_back_pass = True
collection = NamedParameterGroupCollection()
collection.add_group(NamedParameterGroup("unet_lora", [torch.nn.Parameter(torch.ones(1))], config.learning_rate))
try:
    create_optimizer(collection, None, config)
except ValueError as error:
    assert "fused_back_pass" in str(error)
else:
    raise AssertionError("plain Prodigy fused_back_pass was accepted")

if importlib.util.find_spec("prodigyopt") is not None:
    from modules.util.optimizer.prodigy_extensions import ProdigyStatsBuffer
    stats = ProdigyStatsBuffer()
    stats.push(group_idx=0, name="unet_lora", step=3, d_num=1.0, d_den=2.0, dlr=0.5)
    assert stats.pop_all() == [{"group_idx": 0, "name": "unet_lora", "step": 3, "d_num": 1.0, "d_den": 2.0, "dlr": 0.5}]
    assert stats.pop_all() == []
print("prodigy fused fail-loud and stats smoke ok")
PY
rg -n 'pop_stats|ProdigyStatsBuffer|patch_prodigy' modules/util/optimizer/prodigy_extensions.py modules/util/create.py modules/trainer/GenericTrainer.py

# torch_util duplicate/undefined-name and nested data smoke.
run_python_in_active_env - <<'PY'
import ast
from pathlib import Path
import torch
from modules.util.torch_util import get_tensor_data, add_dummy_grad_fn_, has_grad_fn
path = Path("modules/util/torch_util.py")
tree = ast.parse(path.read_text(), filename=str(path))
functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
for name in ["get_tensor_data", "add_dummy_grad_fn_", "has_grad_fn"]:
    assert functions.count(name) == 1, (name, functions.count(name))
for node in ast.walk(tree):
    if isinstance(node, ast.Name) and node.id == "get_tensors":
        raise AssertionError(f"stale get_tensors reference at line {node.lineno}")
nested = {"a": [torch.ones(1), (torch.ones(2),)]}
assert len(get_tensor_data(nested)) == 2
with_grad = add_dummy_grad_fn_(torch.ones(1))
assert has_grad_fn(with_grad)
print("torch_util smoke ok")
PY

# GenerateLosses call signature/progress source contract.
run_python_in_active_env - <<'PY'
import ast
from pathlib import Path
path = Path("modules/module/GenerateLossesModel.py")
tree = ast.parse(path.read_text(), filename=str(path))
progress_update_lines = []
loop_spans = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "calculate_loss":
        assert len(node.args) == 4, f"calculate_loss must receive model,batch,data,config at line {node.lineno}"
    if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        loop_spans.append((node.lineno, node.end_lineno))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update":
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "progress":
            progress_update_lines.append(node.lineno)
source = path.read_text()
assert "self.model.train_progress" in source
assert progress_update_lines, "missing progress.update call"
assert any(start <= line <= end for line in progress_update_lines for start, end in loop_spans), progress_update_lines
assert "tqdm(" not in source
print("GenerateLosses progress loop contract ok")
PY

# Dependency ownership across POSIX/Windows/Docker/manual install surfaces.
test "$(grep -Ec '^pytorch-msssim([[:space:]#<>=]|$)' requirements-global.txt)" -eq 1
if rg -n '^import rich|from rich\b' modules scripts; then grep -Eq '^rich([[:space:]#<>=]|$)' requirements-global.txt; fi
test ! -e requirements-sangoi.txt
run_removed_name_scans

# Launch and Git policy.
git ls-files -s install.sh run-cmd.sh start-ui.sh update.sh | awk '{print $1, $4}'
test "$(git ls-files -s install.sh | awk '{print $1}')" = "100755"
test "$(git ls-files -s run-cmd.sh | awk '{print $1}')" = "100755"
test "$(git ls-files -s start-ui.sh | awk '{print $1}')" = "100755"
test "$(git ls-files -s update.sh | awk '{print $1}')" = "100755"
rg -n 'script_imports\(allow_zluda=False\)' scripts/train_ui.py && exit 1 || true
run_python_in_active_env - <<'PY'
import ast
from pathlib import Path
tree = ast.parse(Path("scripts/train_ui.py").read_text(), filename="scripts/train_ui.py")
calls = [
    node for node in ast.walk(tree)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "script_imports"
]
assert len(calls) == 1, len(calls)
assert calls[0].args == [] and calls[0].keywords == [], ast.dump(calls[0])
print("train_ui script_imports default smoke ok")
PY

# LoRA export and removed side effects.
rg -n 'gen_lora_keys|generate_keys_by_block_file\(' modules scripts && exit 1 || true
rg -n 'lora_key_export_path|build_lora_key_manifest|export_lora_key_manifest' modules
run_python_in_active_env - <<'PY'
import json
from pathlib import Path
import tempfile
import torch
from modules.module.LoRAModule import LoRAModuleWrapper, export_lora_key_manifest
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import PeftType

with tempfile.TemporaryDirectory() as tmpdir:
    config = TrainConfig.default_values()
    config.workspace_dir = tmpdir
    config.lora_key_export_path = ""
    config.peft_type = PeftType.LORA
    config.lora_rank = 1
    config.lora_alpha = 1.0
    wrapper_a = LoRAModuleWrapper(torch.nn.Sequential(torch.nn.Linear(2, 2)), "unet", config)
    wrapper_b = LoRAModuleWrapper(torch.nn.Sequential(torch.nn.Linear(2, 2)), "text_encoder", config)
    manifest_path = export_lora_key_manifest("keys.json", [wrapper_a, wrapper_b], tmpdir)
    assert manifest_path == Path(tmpdir) / "keys.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["kind"] == "sangoi_lora_key_manifest"
    keys_by_block = payload["keys_by_block"]
    assert list(keys_by_block) == sorted(keys_by_block)
    all_keys = [key for keys in keys_by_block.values() for key in keys]
    assert all_keys == sorted(all_keys)
    assert all(key.startswith(("unet.", "text_encoder.")) for key in all_keys)
    assert any(key.endswith((".lora_down.weight", ".lora_up.weight")) for key in all_keys)
print("LoRA export manifest smoke ok")
PY

# Priority sampling runtime/source contract.
rg -n 'loss_per_sample' modules/modelSetup modules/trainer
rg -n 'update_priorities\([^\n]*loss\.detach\(\)' modules/trainer/GenericTrainer.py && exit 1 || true
run_python_in_active_env - <<'PY'
import ast
from pathlib import Path
import torch
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TimestepDistribution import TimestepDistribution

def is_loss_per_sample_subscript(node):
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "model_output_data"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "loss_per_sample"
    )

loss_tree = ast.parse(Path("modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py").read_text())
assert any(
    any(is_loss_per_sample_subscript(target) for target in node.targets)
    for node in ast.walk(loss_tree)
    if isinstance(node, ast.Assign)
), "diffusion loss producer does not assign model_output_data['loss_per_sample']"

trainer_tree = ast.parse(Path("modules/trainer/GenericTrainer.py").read_text())
priority_calls = [
    node for node in ast.walk(trainer_tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "update_priorities"
]
assert priority_calls, "missing update_priorities call"
for call in priority_calls:
    assert any(is_loss_per_sample_subscript(arg) for arg in call.args), ast.dump(call)

class Dummy(ModelSetupNoiseMixin):
    pass
setup = Dummy()
config = TrainConfig.default_values()
config.timestep_distribution = TimestepDistribution.PRIORITY_SAMPLING
setup._priority = torch.ones(10)
for bad_loss in [None, torch.tensor(1.0), torch.tensor([1.0]), torch.tensor([float('nan'), 1.0]), torch.tensor([float('inf'), 1.0]), torch.tensor([-1.0, 1.0])]:
    try:
        setup.update_priorities(torch.tensor([2, 3]), bad_loss, config)
    except ValueError:
        pass
    else:
        raise AssertionError(f"bad priority loss accepted: {bad_loss}")
setup.update_priorities(torch.tensor([2, 3]), torch.tensor([1.0, 2.0]), config)
assert setup._priority[2] != setup._priority[3]
print("priority sampling smoke ok")
PY

# Sangoi loss support boundary: flow matching rejects Sangoi loss mode/weight, and UI filters unsupported choices.
run_python_in_active_env - <<'PY'
from pathlib import Path
import torch
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LossMode import LossMode
from modules.util.enum.LossWeight import LossWeight

class Dummy(ModelSetupDiffusionLossMixin):
    pass

setup = Dummy()
config = TrainConfig.default_values()
config.loss_mode_fn = LossMode.SANGOI
try:
    setup._flow_matching_losses({}, {}, config, torch.device("cpu"))
except ValueError:
    pass
else:
    raise AssertionError("LossMode.SANGOI was accepted by flow matching losses")

config = TrainConfig.default_values()
config.loss_weight_fn = LossWeight.SANGOI
try:
    setup._flow_matching_losses({}, {}, config, torch.device("cpu"))
except ValueError:
    pass
else:
    raise AssertionError("LossWeight.SANGOI was accepted by flow matching losses")

training_tab_source = Path("modules/ui/TrainingTab.py").read_text()
assert "x == LossMode.SANGOI" in training_tab_source
assert "x.supports_flow_matching() == self.train_config.model_type.is_flow_matching()" in training_tab_source
print("Sangoi flow support boundary smoke ok")
PY

# TrainGPS/DataRecorder schema owners.
rg -n 'TrainGPSPenaltyMetric|train_gps_penalty_metric' modules
rg -n 'schema_version|sangoi_train_gps_delta_log|validate_train_gps_support' modules/sangoi/TrainGPS.py
rg -n 'schema_version|sangoi_data_recorder' modules/sangoi/DataRecorder.py
rg -n 'torch\.norm\(' modules/sangoi/TrainGPS.py && exit 1 || true
rg -n 'set_current_step|self\.current_step|\.set_current_step\(' modules/sangoi/DataRecorder.py modules/trainer/GenericTrainer.py && exit 1 || true
run_python_in_active_env - <<'PY'
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import torch
from modules.sangoi.DataRecorder import DataRecorder
from modules.sangoi.TrainGPS import TrainGPS, validate_train_gps_support
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.enum.TrainGPSPenaltyMetric import TrainGPSPenaltyMetric

def make_train_gps_inputs(group_count=2):
    model = torch.nn.Linear(2, 2)
    model.train_progress = SimpleNamespace(epoch=1)
    collection = NamedParameterGroupCollection()
    collection.add_group(NamedParameterGroup("unet_lora", [model.weight], 1.0))
    if group_count > 1:
        collection.add_group(NamedParameterGroup("text_encoder_lora", [model.bias], 1.0))
    return model, collection

config = TrainConfig.default_values()
config.model_type = ModelType.STABLE_DIFFUSION_XL_10_BASE
config.training_method = TrainingMethod.LORA
config.train_gps_use_it = True
config.train_gps_path = "reference.json"
validate_train_gps_support(config)
for model_type, training_method in [(ModelType.STABLE_DIFFUSION_15, TrainingMethod.LORA), (ModelType.STABLE_DIFFUSION_XL_10_BASE, TrainingMethod.FINE_TUNE)]:
    config.model_type = model_type
    config.training_method = training_method
    try:
        validate_train_gps_support(config)
    except ValueError:
        pass
    else:
        raise AssertionError(f"unsupported TrainGPS config accepted: {model_type}, {training_method}")
save_only_config = TrainConfig.default_values()
save_only_config.model_type = ModelType.STABLE_DIFFUSION_15
save_only_config.training_method = TrainingMethod.LORA
save_only_config.train_gps_save_it = True
try:
    validate_train_gps_support(save_only_config)
except ValueError:
    pass
else:
    raise AssertionError("unsupported TrainGPS save-only config was accepted")

with tempfile.TemporaryDirectory() as tmpdir:
    metric_paths = {}
    empty_save_gps = TrainGPS(*make_train_gps_inputs(group_count=2), penalty_metric=TrainGPSPenaltyMetric.MSE)
    try:
        empty_save_gps.save_log_to_file(str(Path(tmpdir) / "empty_uninitialized.json"))
    except ValueError:
        pass
    else:
        raise AssertionError("TrainGPS saved an empty uninitialized delta log")
    empty_save_gps.setup_for_save()
    try:
        empty_save_gps.save_log_to_file(str(Path(tmpdir) / "empty_initialized.json"))
    except ValueError:
        pass
    else:
        raise AssertionError("TrainGPS saved an initialized delta log with no epoch records")

    for metric in [TrainGPSPenaltyMetric.MSE, TrainGPSPenaltyMetric.COSINE]:
        gps_path = Path(tmpdir) / f"gps_{metric.value}.json"
        model, collection = make_train_gps_inputs(group_count=2)
        gps = TrainGPS(model, collection, penalty_metric=metric)
        gps.setup_for_save()
        with torch.no_grad():
            model.weight.add_(1.0)
            model.bias.add_(2.0)
        gps.log_epoch_deltas(1)
        gps.save_log_to_file(str(gps_path))
        payload = json.loads(gps_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["kind"] == "sangoi_train_gps_delta_log"
        assert payload["metric"] == metric.value
        assert payload["records"] and all(record["epoch"] == 1 for record in payload["records"])
        metric_paths[metric] = gps_path

        use_model, use_collection = make_train_gps_inputs(group_count=2)
        use_gps = TrainGPS(use_model, use_collection, penalty_metric=metric)
        use_gps.setup_for_use(str(metric_paths[metric]))
        penalty = use_gps.compute_penalty(1.0)
        assert torch.isfinite(penalty), penalty
    gps_path = metric_paths[TrainGPSPenaltyMetric.MSE]

    mismatch_gps = TrainGPS(*make_train_gps_inputs(group_count=2), penalty_metric=TrainGPSPenaltyMetric.COSINE)
    try:
        mismatch_gps.setup_for_use(str(gps_path))
    except ValueError:
        pass
    else:
        raise AssertionError("TrainGPS reference metric mismatch was accepted")

    missing_epoch_model, missing_epoch_collection = make_train_gps_inputs(group_count=2)
    missing_epoch_model.train_progress.epoch = 2
    missing_epoch_gps = TrainGPS(missing_epoch_model, missing_epoch_collection, penalty_metric=TrainGPSPenaltyMetric.MSE)
    missing_epoch_gps.setup_for_use(str(gps_path))
    try:
        missing_epoch_gps.compute_penalty(1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("TrainGPS reference missing the current epoch was accepted")

    one_group_model, one_group_collection = make_train_gps_inputs(group_count=1)
    one_group_path = Path(tmpdir) / "gps_COSINE_one_group.json"
    one_group_save_gps = TrainGPS(one_group_model, one_group_collection, penalty_metric=TrainGPSPenaltyMetric.COSINE)
    one_group_save_gps.setup_for_save()
    with torch.no_grad():
        one_group_model.weight.add_(1.0)
    one_group_save_gps.log_epoch_deltas(1)
    one_group_save_gps.save_log_to_file(str(one_group_path))
    one_group_model, one_group_collection = make_train_gps_inputs(group_count=1)
    one_group_gps = TrainGPS(one_group_model, one_group_collection, penalty_metric=TrainGPSPenaltyMetric.COSINE)
    one_group_gps.setup_for_use(str(one_group_path))
    try:
        one_group_gps.compute_penalty(1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("one-group TrainGPS cosine penalty was accepted")

    old_path = Path(tmpdir) / "old.json"
    old_path.write_text(json.dumps({"epoch_1": {"unet_lora": 1.0}}), encoding="utf-8")
    try:
        TrainGPS(*make_train_gps_inputs(group_count=2), penalty_metric=TrainGPSPenaltyMetric.MSE).setup_for_use(str(old_path))
    except ValueError:
        pass
    else:
        raise AssertionError("old TrainGPS schema was accepted")

    bad_path = Path(tmpdir) / "bad.json"
    bad_path.write_text(json.dumps({"schema_version": 1, "kind": "sangoi_train_gps_delta_log", "metric": "MSE", "records": [{"epoch": 1, "group": "unet_lora", "delta_norm": float("nan")}]}), encoding="utf-8")
    try:
        TrainGPS(*make_train_gps_inputs(group_count=2), penalty_metric=TrainGPSPenaltyMetric.MSE).setup_for_use(str(bad_path))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite TrainGPS delta was accepted")

    missing_group_path = Path(tmpdir) / "missing_group.json"
    missing_group_path.write_text(json.dumps({
        "schema_version": 1,
        "kind": "sangoi_train_gps_delta_log",
        "metric": "MSE",
        "records": [
            {"epoch": 1, "group": "unet_lora", "delta_norm": 1.0}
        ],
    }), encoding="utf-8")
    missing_group_gps = TrainGPS(*make_train_gps_inputs(group_count=2), penalty_metric=TrainGPSPenaltyMetric.MSE)
    missing_group_gps.setup_for_use(str(missing_group_path))
    try:
        missing_group_gps.compute_penalty(1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("TrainGPS reference missing a live group was accepted")

recorder = DataRecorder()
recorder.log_metrics_step(step=7, group="unet_lora", d_num_pdgy=1.0, d_den_pdgy=2.0, dlr_pdgy=0.3, grad_norm=4.0)
recorder.log_metrics_step(step=8, group="text_encoder_lora")
try:
    recorder.log_metrics_step(step=9, group="bad", d_num_pdgy=float("nan"))
except ValueError:
    pass
else:
    raise AssertionError("DataRecorder accepted a non-finite metric")
with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "recorder.json.gz"
    recorder.dump(str(path))
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
assert payload["schema_version"] == 1
assert payload["kind"] == "sangoi_data_recorder"
assert payload["records"] == [
    {"step": 7, "group": "unet_lora", "d_num_pdgy": 1.0, "d_den_pdgy": 2.0, "dlr_pdgy": 0.3, "grad_norm": 4.0},
    {"step": 8, "group": "text_encoder_lora", "d_num_pdgy": None, "d_den_pdgy": None, "dlr_pdgy": None, "grad_norm": None},
]
print("TrainGPS and DataRecorder schema smokes ok")
PY

# UI callback ownership.
run_python_in_active_env - <<'PY'
import ast
from pathlib import Path
path = Path("modules/ui/TrainUI.py")
tree = ast.parse(path.read_text(), filename=str(path))
path_entry_workspace_commands = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "path_entry":
        args = [getattr(arg, "value", None) for arg in node.args]
        if "workspace_dir" in args:
            for keyword in node.keywords:
                if keyword.arg == "command":
                    path_entry_workspace_commands.append(node.lineno)
assert not path_entry_workspace_commands, path_entry_workspace_commands
source = path.read_text()
assert source.count('add_var_trace("workspace_dir"') == 1
assert "def _on_workspace_dir_change(" not in source

expected_threadsafe = {
    "handle_pause_request_accepted_threadsafe": "_handle_pause_request_accepted_ui",
    "handle_pause_initiated_threadsafe": "_handle_pause_initiated_ui",
    "handle_resume_started_threadsafe": "_handle_resume_started_ui",
    "handle_resume_completed_threadsafe": "_handle_resume_completed_ui",
}
functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
for wrapper_name, ui_handler_name in expected_threadsafe.items():
    wrapper = functions[wrapper_name]
    assert any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "after"
        and len(call.args) >= 2
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == 0
        and isinstance(call.args[1], ast.Attribute)
        and call.args[1].attr == ui_handler_name
        for call in ast.walk(wrapper)
    ), wrapper_name
toggle = functions["toggle_pause"]
toggle_source = ast.get_source_segment(source, toggle)
assert "request_pause()" in toggle_source and "request_resume()" in toggle_source
assert ".is_paused" not in toggle_source and "pause_requested_at_epoch_end" not in toggle_source

callbacks_source = Path("modules/util/callbacks/TrainCallbacks.py").read_text()
trainer_source = Path("modules/trainer/GenericTrainer.py").read_text()
for callback in ["on_pause_request_accepted", "on_pause_initiated", "on_resume_started", "on_resume_completed"]:
    assert callback in callbacks_source
    assert f"callbacks.{callback}()" in trainer_source
print("UI callback and pause/resume ownership smoke ok")
PY

# Tooling ownership.
test ! -e .style.yapf
test ! -e modules/util/loss/dynamic_loss_strength.py
test ! -e scripts/loadTime.py
test ! -e modules/sangoi/Logger.py
test ! -e modules/util/TensorBoardManager.py
run_python_in_active_env - <<'PY'
import ast
from pathlib import Path
text = Path('.editorconfig').read_text().splitlines()
assert text[0].strip() == 'root = true'
assert text.count('[*]') == 1

pre_commit = Path(".pre-commit-config.yaml").read_text()
assert "id: ruff-check" in pre_commit
assert "id: ruff-format" in pre_commit
assert "id: ruff\n" not in pre_commit

tree = ast.parse(Path("modules/util/NamedParameterGroup.py").read_text())
class_node = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "NamedParameterGroup")
assert "set_requires_grad" not in {node.name for node in class_node.body if isinstance(node, ast.FunctionDef)}
for node in ast.walk(class_node):
    if isinstance(node, ast.Attribute) and node.attr == "is_enabled":
        raise AssertionError("NamedParameterGroup.is_enabled still exists")
print('tooling/deletion source smoke ok')
PY
run_removed_name_scans

git diff --check
git diff HEAD --check
run_python_in_active_env -m compileall modules scripts
run_removed_name_scans
```

Expected validation note: forbidden-reference scans intentionally exclude `.sangoi` historical evidence and intentionally include current maintainer docs/tests/examples/prompts when those roots exist. Earlier `.sangoi` plans and task logs may mention old names as historical execution evidence only; any current-maintainer doc updated by this cleanup must describe current behavior only.

## Rollback Strategy

- Keep each implementation batch staged and inspected separately before commit.
- If a high-priority runtime repair fails validation, revert only that batch before moving to medium/low cleanup.
- Do not keep partial compatibility shims as rollback. Rollback means restore the previous canonical owner or stop with a blocker.
- If environment bootstrap fails, do not continue with unverified runtime claims.

## Stop Conditions

- Any implementation step requires preserving old config names for compatibility.
- Prodigy fused-back-pass becomes a hard requirement; that requires a separate exact implementation plan.
- Priority sampling cannot obtain per-sample loss without the side payload and would require changing the scalar `calculate_loss` return contract across all setups.
- TrainGPS is needed for non-SDXL or non-LoRA training in this pass.
- A planned deletion candidate has a live runtime importer outside the deletion batch.
- Validation requires network/bootstrap and dependency acquisition fails.
- Repository validation starts before `./.uv/bin/uv` and `.venv/bin/python` exist.
- Local WSL validation would require CUDA/GPU execution or CUDA requirements without explicit user confirmation.
- Dockerfiles remain in scope while cloning upstream instead of building the current source tree.

## Assumptions And Defaults

- Existing user-local config JSON compatibility is not preserved in this pass.
- SDXL LoRA is the primary local training target for TrainGPS and LoRA-rule cleanup.
- Removing broken advertised surfaces is acceptable when the alternative is implementing a new large feature not explicitly requested.
- `.sangoi/plans`, `.sangoi/task-logs`, and `.sangoi/CHANGELOG.md` remain maintainable project artifacts even if generated Patrol reports stay ignored.
- `requirements-global.txt` is the dependency owner for Sangoi runtime Python packages; `requirements.txt` remains a fan-out installer surface and must not reference removed Sangoi-specific requirement files.
