# Selected Upstream PR Port Task Log

Date: 2026-05-25
Branch: `master-update`
Mode: Doctrine mode
Plan: `.sangoi/plans/2026-05-25-selected-upstream-pr-port.md`

## Objective

Manually port selected stale upstream OneTrainer PR intent into the current Sangoi fork without merging PR branches, while preserving local training behavior and current uv/WSL constraints.

## Gate History

- `Senior Plan Auditor`: completed with `APPROVE_WITH_FIXES`.
- Plan fixes were applied before runtime edits: validation timestep ledger, accumulator resume proof, attention backend semantics, guarded-file checks, strict config/no-alias checks, Sangoi invariants, and shared-file fan-in requirements.
- First `Senior Code Reviewer` pass: `NOT_READY`.
- First reviewer remediation applied: Flux2 edit custom-conditioning mapping, Hunyuan Comfy transformer-only export, prefetch eligibility/device-boundary waits, fixed validation timestep fail-loud behavior, accumulator global-step restore check, and planned validator check-name coverage.
- Second `Senior Code Reviewer` pass: `NOT_READY`.
- Second reviewer remediation applied: strict raw config validation before `BaseConfig.from_dict()` and canonical `ValidationMetrics` ownership for validation TensorBoard writes and patience.
- Third `Senior Code Reviewer` pass: `NOT_READY`.
- Third reviewer remediation applied: Flux2 sample UI/sampler now obey the canonical `custom_conditioning_image` switch, prefetch workers close through a consumer-loop `finally`, and source guards scan tracked plus untracked runtime paths.
- Fourth `Senior Code Reviewer` pass: `NOT_READY`.
- Fourth reviewer remediation applied: `SampleConfig.from_train_config()` now returns the mutated config for `ConfigList` add/load chaining, `PrefetchIterator.wait_until_idle()` no longer waits on queue-full producer puts, and source guards cover the blocked/deferred PR residue matrix.
- Fifth `Senior Code Reviewer` pass: `NOT_READY`.
- Fifth reviewer remediation applied: epoch validation no longer consumes the validation timer twice, and epoch backup/save actions now persist `last_action_epoch` when `start_at_zero=False`.
- Sixth `Senior Code Reviewer` pass: `NOT_READY`.
- Sixth reviewer remediation applied: loaders no longer backfill missing `last_action_epoch`, and `GenerateLossesModel` now stages deterministic validation timestep metadata for STRATIFIED calculate-loss runs.
- Seventh `Senior Code Reviewer` pass: `NOT_READY`.
- Seventh reviewer remediation applied: BETA timestep sampling now supports default, alpha=1, beta=1, and arbitrary positive alpha/beta paths, and accumulator fingerprints now cover material config plus full ordered concept settings with mismatch restore smokes.
- Eighth `Senior Code Reviewer` pass: `NOT_READY`.
- Eighth reviewer remediation applied: BETA timestep sampling now uses the supplied `torch.Generator` for arbitrary alpha/beta paths, accumulator fingerprints cover the full normalized train settings plus full ordered concept settings, and source guards cover the blocked/deferred matrix terms with a narrow `set_attention_backend` exemption.
- Ninth `Senior Code Reviewer` pass: `NOT_READY`.
- Ninth reviewer remediation applied: final changed-artifact inventory now includes `modules/modelSampler/BaseModelSampler.py`, the sampler hunk is classified under the Flux2 target-resolution lane, and the plan-required final review checklist now has an observed validation artifact path.
- Tenth `Senior Code Reviewer` pass: `NOT_READY`.
- Tenth reviewer remediation applied: per-lane PR refs in the final review artifact checklist now match the approved plan PR lock table, including Flux2 `#1301/#1460`, prefetch/cache `#1461/#1134`, timestep/noise/perturbation `#1225/#1124/#1260/#1235/#1435/#1227`, resume/validation `#1470/#1465/#1422/#1408/#1416/#1436/#821`, and adapter/conversion `#1315/#1335/#1464/#1360/#1471/#1374`.
- Final `Senior Code Reviewer` pass: `READY_WITH_NITS`.
- Final reviewer nit applied: stale ordinal in the final checklist status text was updated after the gate result.

## Implementation Summary

- Added canonical config/enums/UI for validation timestep policy, attention backend selection, TensorBoard run resume, patience stop, prefetching, immiscible noise, SDXL CEP/CIOP controls, LoRA load scaling, LoKr, Scaled OFT, and DoRA-OFT.
- Added `modules/util/validation_timestep.py` and routed all current validation-timestep-producing setup owners through deterministic AUTO/FIXED/STRATIFIED validation behavior.
- Added BETA and SPEED timestep distributions with fail-loud validation and preview support.
- Added immiscible noise oversampling and SDXL CEP/CIOP runtime hooks in the current SDXL setup owner.
- Added diffusers attention backend routing through `AttentionMechanism.SDP` and `AttentionMechanism.FLASH`, using the active diffusers `AttentionBackendName` values and failing loud when a component cannot select FLASH.
- Added trainer resume improvements: TensorBoard subdir metadata reuse, validation patience best-state restore, stop-on-no-trainable-components, prefetch iterator support, epoch-action resume state, and mid-accumulation accumulator/RNG/fingerprint state save/load.
- Added canonical `ValidationMetrics` ownership so validation TensorBoard writes and patience consume one returned metric value.
- Added target-resolution image cache variation identity.
- Added LoKr, Scaled OFT, DoRA-OFT, and LoRA text-encoder/main-weight load scaling while preserving Sangoi LoRA rank/alpha rules, blacklist, and deterministic manifest export.
- Added Flux 2 edit-conditioning support through current Flux2 dataloader, setup, sampler, sample config, and sample UI owners, gated by `custom_conditioning_image`.
- Fixed HiDream tokenizer ownership for tokenizer 2 and optional tokenizer 4 loading.
- Added HunyuanVideo LoRA preprocessing and ComfyUI export support through current Hunyuan loader/saver/conversion owners.
- Added `scripts/validate_selected_upstream_pr_ports.py` to lock validation timestep ownership, resume action state, mid-accum resume state, TensorBoard continuity, patience, component-stop behavior, timestep/noise/perturbation behavior, Flux2 edit, HiDream tokenizer loading, Hunyuan conversion, adapter contracts, prefetch lifecycle, attention backend semantics, guarded files, strict config behavior, Sangoi invariants, and LoRA manifest determinism.

## Key Files

- `modules/util/config/TrainConfig.py`
- `modules/util/validation_timestep.py`
- `modules/modelSetup/mixin/ModelSetupNoiseMixin.py`
- `modules/trainer/GenericTrainer.py`
- `modules/module/LoRAModule.py`
- `modules/module/oft_utils.py`
- `modules/util/lokr_utils.py`
- `modules/modelSetup/BaseModelSetup.py`
- `modules/modelSetup/BaseStableDiffusionXLSetup.py`
- `modules/modelSetup/BaseFlux2Setup.py`
- `modules/dataLoader/Flux2BaseDataLoader.py`
- `modules/modelSampler/Flux2Sampler.py`
- `modules/modelSampler/BaseModelSampler.py`
- `modules/util/config/SampleConfig.py`
- `modules/util/PrefetchIterator.py`
- `modules/ui/SampleFrame.py`
- `modules/ui/SampleWindow.py`
- `modules/ui/SamplingTab.py`
- `modules/modelLoader/hunyuanVideo/HunyuanVideoLoRALoader.py`
- `modules/modelSaver/hunyuanVideo/HunyuanVideoLoRASaver.py`
- `modules/util/convert/lora/convert_hunyuan_video_lora.py`
- `modules/ui/TrainUI.py`
- `modules/ui/TrainingTab.py`
- `modules/ui/LoraTab.py`
- `scripts/validate_selected_upstream_pr_ports.py`
- `.sangoi/task-logs/2026-05-25-selected-upstream-pr-port-validation.md`

## Validation

Completed successfully with repo-local uv and CPU-only WSL Python:

Observed validation output artifact: `.sangoi/task-logs/2026-05-25-selected-upstream-pr-port-validation.md`

```bash
set -euo pipefail
export OT_PLATFORM_REQUIREMENTS=requirements-default.txt
export PYTHONPATH="$PWD"
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

Targeted runtime smokes covered:

- `TrainConfig.default_values()`, `validate_for_training()`, strict stale-key rejection, and `to_dict()` / `from_dict()` roundtrip.
- LoRA, LoHa, LoKr, DoRA-OFT wrapper construction and hook/unhook on CPU `nn.Linear`.
- LoRA load scaling split between text encoder and main adapter keys.
- HunyuanVideo LoRA preprocessing and transformer-only Comfy output key formation.
- BETA/SPEED timestep sampling, BETA same-generator determinism independent of global RNG, flow-model rejection, and deterministic validation timestep resolution with fixed out-of-range failure.
- Flux2 edit conditioning source contract and Flux2 dropout preflight rejection.
- Flux2 sample config/UI/sampler gating through `custom_conditioning_image`, including sample add/load chaining.
- Prefetch iterator producer success, producer-exception propagation, consumer-failure cleanup, queue-full `wait_until_idle()` behavior, config eligibility rejection, and trainer device-boundary wait ownership.
- Epoch validation emits `ValidationMetrics` after the due check, and epoch backup/save actions persist `last_action_epoch` across resume.
- Missing `last_action_epoch` metadata loads as the current default action map, while persisted validate/sample/backup/save maps survive loader fixtures.
- Calculate-loss deterministic prediction stages validation timestep index/count/seed metadata so STRATIFIED validation timestep policy remains usable outside `GenericTrainer` validation.
- BETA timestep distribution smokes cover default, beta=1, alpha=1, arbitrary alpha/beta, supplied-generator determinism, and invalid negative config rejection.
- Accumulator resume smokes verify material config, learning-rate, optimizer, trainable-part, and concept-setting fingerprint changes and reject restore on mismatch.
- Accumulator metadata, gradient/RNG ownership, and global-step restore mismatch rejection.
- Source guards scan tracked and untracked runtime paths for the blocked/deferred PR residue matrix, prove each denylist term with a validator fixture, and exclude the validator's own denylist evidence file.

## Final Review Artifact Checklist

- Plan path and status: `.sangoi/plans/2026-05-25-selected-upstream-pr-port.md`; implementation complete, mandatory final code-review gate completed with `READY_WITH_NITS`.
- Final diff anchor: base `HEAD=57d0a21f8e2b831d9e1158e9ce27bdca8bef97f3`, branch `master-update`, uncommitted cumulative diff, `git diff --stat HEAD` reports 54 tracked files changed with 2318 insertions and 334 deletions.
- Validation artifact path: `.sangoi/task-logs/2026-05-25-selected-upstream-pr-port-validation.md`, containing the command and observed `VALIDATION_STATUS=0`.
- PR status lock table: plan tables in `.sangoi/plans/2026-05-25-selected-upstream-pr-port.md` remain canonical for `IMPLEMENT`, `IMPLEMENT_PARTIAL`, `BLOCKED`, `DEFERRED`, and forbidden duplicate-owner decisions.

### Per-Lane Implementation Ledger

| Lane | PR refs / intent | Touched files | Intentional deviations | Local smokes |
| --- | --- | --- | --- | --- |
| Validation, stop policy, resume state | #1470, #1465, #1422, #1408, #1416, #1436, #821 | `modules/trainer/GenericTrainer.py`, `modules/model/BaseModel.py`, `modules/modelLoader/BaseModelLoader.py`, `modules/modelLoader/mixin/InternalModelLoaderMixin.py`, `modules/modelSaver/mixin/InternalModelSaverMixin.py`, `modules/util/TimedActionMixin.py`, `modules/util/TrainProgress.py`, `modules/module/GenerateLossesModel.py`, `modules/util/dataset_fingerprint.py` | Full train settings and full ordered concepts are fingerprinted instead of a narrow handpicked subset; missing `last_action_epoch` is not backfilled. | `resume-actions`, `timed-actions`, `accumulator`, `accumulator-mid-step`, `tensorboard`, `patience`, `component-stop`, `validation-timesteps` |
| Timestep, noise, perturbation, and setup routing | #1225, #1124, #1260, #1235, #1435, #1227 | `modules/util/config/TrainConfig.py`, `modules/util/enum/TimestepDistribution.py`, `modules/util/enum/ValidationTimestepMode.py`, `modules/util/enum/AttentionMechanism.py`, `modules/util/validation_timestep.py`, `modules/modelSetup/mixin/ModelSetupNoiseMixin.py`, `modules/util/immiscible_diffusion.py`, all touched `modules/modelSetup/Base*.py`, `modules/modelSetup/StableDiffusionLoRASetup.py`, `modules/modelSetup/StableDiffusionXLLoRASetup.py`, `modules/ui/TimestepDistributionWindow.py`, `modules/ui/TrainingTab.py`, `modules/modelSetup/BaseModelSetup.py` | `attention_mechanism` is the only backend config owner; `attention_backend` remains forbidden. Flow families reject BETA/SPEED. | `timestep-distributions`, `flow-timestep-rejection`, `immiscible`, `perturbations`, `attention-backend`, `strict-config` |
| Flux2 edit conditioning and sampler resolution | #1301, with incidental same-surface #1460 UI layout repair | `modules/dataLoader/Flux2BaseDataLoader.py`, `modules/dataLoader/mixin/DataLoaderText2ImageMixin.py`, `modules/model/Flux2Model.py`, `modules/modelSampler/BaseModelSampler.py`, `modules/modelSampler/Flux2Sampler.py`, `modules/modelSetup/BaseFlux2Setup.py`, `modules/util/config/SampleConfig.py`, `modules/ui/SampleFrame.py`, `modules/ui/SampleWindow.py`, `modules/ui/SamplingTab.py` | No duplicate `FLUX_2_EDIT` model type; base image path is accepted only when `custom_conditioning_image` is enabled. `BaseModelSampler.quantize_resolution(..., mode=\"floor\")` is limited to Flux2 conditioning-image preprocessing. | `flux2-edit`, `flux2-dropout-block` |
| Prefetch and cache identity | #1461, #1134 | `modules/util/PrefetchIterator.py`, `modules/trainer/GenericTrainer.py`, data/cache touchpoints above | Worker exceptions propagate; consumer cleanup is owned by trainer `finally`; queue-full wait does not deadlock. | `prefetch-cache`, `prefetch-device-boundary` |
| Adapter, LoRA, HiDream, and Hunyuan conversion | #1315, #1335, #1464, #1360 partial, #1471, #1374 | `modules/module/LoRAModule.py`, `modules/module/oft_utils.py`, `modules/util/lokr_utils.py`, `modules/util/NamedParameterGroup.py`, `modules/modelLoader/mixin/LoRALoaderMixin.py`, `modules/modelSaver/mixin/LoRASaverMixin.py`, `modules/modelLoader/hiDream/HiDreamModelLoader.py`, `modules/modelLoader/hunyuanVideo/HunyuanVideoLoRALoader.py`, `modules/modelSaver/hunyuanVideo/HunyuanVideoLoRASaver.py`, `modules/util/convert/lora/convert_hunyuan_video_lora.py`, `modules/util/convert_util.py`, `modules/ui/ConvertModelUI.py`, `modules/ui/LoraTab.py`, `modules/util/enum/ModelType.py` | Sangoi rank/alpha rules, blacklist, and key export remain canonical; Hunyuan Comfy output is transformer-only. | `hidream-tokenizer`, `hunyuan-comfy`, `adapters`, `lora-manifest` |
| Governance, source guards, and documentation | Doctrine and blocked/deferred guardrail implementation | `.sangoi/CHANGELOG.md`, `.sangoi/plans/2026-05-25-selected-upstream-pr-port.md`, `.sangoi/task-logs/2026-05-25-selected-upstream-pr-port.md`, `.sangoi/task-logs/2026-05-25-selected-upstream-pr-port-validation.md`, `scripts/validate_selected_upstream_pr_ports.py` | Validator scans tracked and untracked runtime paths, but excludes its own denylist evidence file. | `source-guards`, guarded-file diff checks, `git diff --check`, `git diff HEAD --check` |

### Cumulative Touched-File Inventory By Lane

- Governance and validation artifacts: `.sangoi/CHANGELOG.md`, `.sangoi/plans/2026-05-25-selected-upstream-pr-port.md`, `.sangoi/task-logs/2026-05-25-selected-upstream-pr-port.md`, `.sangoi/task-logs/2026-05-25-selected-upstream-pr-port-validation.md`, `scripts/validate_selected_upstream_pr_ports.py`.
- Validation, resume, and trainer state: `modules/model/BaseModel.py`, `modules/modelLoader/BaseModelLoader.py`, `modules/modelLoader/mixin/InternalModelLoaderMixin.py`, `modules/modelSaver/mixin/InternalModelSaverMixin.py`, `modules/trainer/GenericTrainer.py`, `modules/util/TimedActionMixin.py`, `modules/util/TrainProgress.py`, `modules/module/GenerateLossesModel.py`, `modules/util/PrefetchIterator.py`, `modules/util/dataset_fingerprint.py`.
- Config, timestep, setup, and attention: `modules/util/config/TrainConfig.py`, `modules/util/enum/TimestepDistribution.py`, `modules/util/enum/ValidationTimestepMode.py`, `modules/util/enum/AttentionMechanism.py`, `modules/util/validation_timestep.py`, `modules/modelSetup/mixin/ModelSetupNoiseMixin.py`, `modules/util/immiscible_diffusion.py`, `modules/modelSetup/BaseChromaSetup.py`, `modules/modelSetup/BaseErnieSetup.py`, `modules/modelSetup/BaseFlux2Setup.py`, `modules/modelSetup/BaseFluxSetup.py`, `modules/modelSetup/BaseHiDreamSetup.py`, `modules/modelSetup/BaseHunyuanVideoSetup.py`, `modules/modelSetup/BaseModelSetup.py`, `modules/modelSetup/BasePixArtAlphaSetup.py`, `modules/modelSetup/BaseQwenSetup.py`, `modules/modelSetup/BaseSanaSetup.py`, `modules/modelSetup/BaseStableDiffusion3Setup.py`, `modules/modelSetup/BaseStableDiffusionSetup.py`, `modules/modelSetup/BaseStableDiffusionXLSetup.py`, `modules/modelSetup/BaseWuerstchenSetup.py`, `modules/modelSetup/BaseZImageSetup.py`, `modules/modelSetup/StableDiffusionLoRASetup.py`, `modules/modelSetup/StableDiffusionXLLoRASetup.py`.
- Flux2 edit and sampler conditioning: `modules/dataLoader/Flux2BaseDataLoader.py`, `modules/dataLoader/mixin/DataLoaderText2ImageMixin.py`, `modules/model/Flux2Model.py`, `modules/modelSampler/BaseModelSampler.py`, `modules/modelSampler/Flux2Sampler.py`, `modules/util/config/SampleConfig.py`, `modules/ui/SampleFrame.py`, `modules/ui/SampleWindow.py`, `modules/ui/SamplingTab.py`.
- Adapter, conversion, and UI surfaces: `modules/modelLoader/hiDream/HiDreamModelLoader.py`, `modules/modelLoader/hunyuanVideo/HunyuanVideoLoRALoader.py`, `modules/modelLoader/mixin/LoRALoaderMixin.py`, `modules/modelSaver/hunyuanVideo/HunyuanVideoLoRASaver.py`, `modules/modelSaver/mixin/LoRASaverMixin.py`, `modules/module/LoRAModule.py`, `modules/module/oft_utils.py`, `modules/util/lokr_utils.py`, `modules/util/NamedParameterGroup.py`, `modules/util/convert/lora/convert_hunyuan_video_lora.py`, `modules/util/convert_util.py`, `modules/ui/ConvertModelUI.py`, `modules/ui/LoraTab.py`, `modules/ui/TimestepDistributionWindow.py`, `modules/ui/TrainUI.py`, `modules/ui/TrainingTab.py`, `modules/util/enum/ModelType.py`.

### Shared-File Hunk Classification

- `modules/util/config/TrainConfig.py`: config/default/validation hunks for validation timestep, TensorBoard resume, patience, prefetch, attention mechanism, immiscible/CEP/CIOP, BETA/SPEED, Flux2 custom conditioning, LoRA scaling, and Sangoi LoRA rule invariants. No blocked/deferred config owner is added.
- `modules/trainer/GenericTrainer.py`: hunks cover validation metrics/patience/TensorBoard writes, prefetch lifecycle/device-boundary handling, component-stop, validation timestep metadata, and mid-accumulation accumulator/RNG/fingerprint state. No unrelated trainer policy hunk is present.
- `modules/util/TimedActionMixin.py`: hunk records epoch actions into `last_action_epoch` even when `start_at_zero=False`.
- `modules/util/TrainProgress.py`: hunk adds canonical `last_action_epoch` resume state.
- `modules/ui/TrainingTab.py`: hunks expose current config owners for validation timestep, timestep/noise, attention, prefetch, Flux2 conditioning, and SDXL perturbation controls.
- `modules/util/enum/ModelType.py`: hunk supports landed model-family/conversion routing only; no blocked duplicate `FLUX_2_EDIT` identity is added.
- `modules/module/LoRAModule.py`: hunks cover LoKr/Scaled OFT/DoRA-OFT/load-scaling plus Sangoi manifest/rank/alpha/blacklist preservation.
- Flux2 owners: `DataLoaderText2ImageMixin.py`, `Flux2BaseDataLoader.py`, `Flux2Model.py`, `BaseFlux2Setup.py`, `Flux2Sampler.py`, `SampleConfig.py`, `SampleFrame.py`, `SampleWindow.py`, and `SamplingTab.py` hunks are Flux2 edit-conditioning or its sample UI plumbing only.
- Cache/data-loader/sampler owners: `BaseModelSampler.py` hunk adds explicit `round`/`floor` quantization modes and is consumed by Flux2 conditioning-image preprocessing; `PrefetchIterator.py` owns worker/queue lifecycle; cache fingerprinting uses `dataset_fingerprint.py`.
- Setup owners: each touched `Base*Setup.py` hunk routes the validation timestep or attention/noise contract through the current setup owner; no setup file imports blocked XPU/SmartDiskCache/split-attention runtime code.
- Loader/saver/conversion owners: loader/saver hunks serialize or restore current metadata and landed adapter/conversion contracts; missing new optional metadata uses current defaults and does not add dual-read compatibility glue.

### Config/UI Schema Freeze

- New or reaffirmed defaults: `tensorboard_resume_run=True`, `patience=False`, `patience_epochs=5`, `validation_timestep_mode=ValidationTimestepMode.AUTO`, `validation_timestep_values=\"500\"`, `validation_timestep_seed=0`, `prefetch_next_batch=False`, `attention_mechanism=AttentionMechanism.SDP`, `k_noise_sampling=1`, `cep_enabled=False`, `cep_gamma=1.0`, `ciop_noise_weight=0.0`, `ciop_p=0.8`, `timestep_distribution=TimestepDistribution.UNIFORM`, `noising_weight=0.0`, `noising_bias=0.0`, `custom_conditioning_image=False`, `lora_te_scale=1.0`, `lora_unet_scale=1.0`, `lora_modules_rank_rules=[]`, `lora_modules_alpha_rules=[]`, `lora_layers_blacklist=[]`, `lora_key_export_path=\"\"`.
- Sample defaults: `SampleConfig.custom_conditioning_image=False`.
- Forbidden schema owners: no `attention_backend` config/UI/schema field, no `FLUX_2_EDIT` model type, no blocked SmartDiskCache/split-attention/XPU/Windows-ROCm/activation-quantization config surface.

### Acceptance And Invariant Mapping

- Validation timestep and stop policy: owner files are `GenericTrainer.py`, setup owners, `validation_timestep.py`, `TrainConfig.py`, `TrainUI.py`, and `TrainingTab.py`; evidence is `validation-timesteps`, `patience`, `tensorboard`, and `component-stop`.
- Accumulator resume: owner files are `GenericTrainer.py`, loader/saver metadata owners, `dataset_fingerprint.py`, and `TrainProgress.py`; evidence is `accumulator`, `accumulator-mid-step`, and the material mutation smokes for resolution, learning rate, optimizer beta1, trainable flags, and concept image settings.
- BETA/SPEED distributions: owner files are `ModelSetupNoiseMixin.py`, `TimestepDistribution.py`, `TrainConfig.py`, and `TimestepDistributionWindow.py`; evidence is `timestep-distributions` and `flow-timestep-rejection`.
- Attention backend: owner files are `AttentionMechanism.py`, `BaseModelSetup.py`, `TrainConfig.py`, and setup owners; evidence is `attention-backend` plus source guards against `attention_backend` aliases.
- Flux2 edit conditioning: owner files are Flux2 dataloader/model/setup/sampler/sample UI files plus `BaseModelSampler.py`; evidence is `flux2-edit` and `flux2-dropout-block`.
- Prefetch and cache behavior: owner files are `PrefetchIterator.py` and `GenericTrainer.py`; evidence is `prefetch-cache` and `prefetch-device-boundary`.
- Adapter/conversion behavior: owner files are LoRA/LoKr/OFT/Hunyuan/HiDream conversion and UI owners; evidence is `adapters`, `hidream-tokenizer`, `hunyuan-comfy`, and `lora-manifest`.
- Sangoi invariants: owner files are `TrainConfig.py`, `GenericTrainer.py`, `LoRAModule.py`, Sangoi-preserved UI/config surfaces, and guarded bootstrap/cloud/docker paths; evidence is `sangoi-invariants`, guarded diff checks, and source guards.

### Blocked/Deferred And Platform Guards

- Blocked/deferred source evidence: `scripts/validate_selected_upstream_pr_ports.py` scans tracked and untracked runtime paths for the blocked matrix, including SmartDiskCache, split-attention terms, TheRock/Windows ROCm, `LinearA8`, `attention_backend`, XPU/ipex, Flux2 caption dropout, and fixed-resolution MGDS residue.
- Dependency/platform guard result: validation passed `git diff --exit-code HEAD -- requirements.txt requirements-default.txt requirements-cuda.txt requirements-rocm.txt requirements-global.txt requirements-dev.txt pyproject.toml .python-version` and guarded bootstrap/cloud/docker diffs, plus guarded untracked scan.
- CPU-only WSL caveat: validation is intentionally local CPU smoke coverage only; CUDA/GPU/XPU/ROCm hardware, Docker/cloud execution, rendered Tk UI, full training, and benchmarks were not run.

## Not Run

- Full training runs or benchmarks.
- CUDA/GPU/XPU/ROCm validation.
- Docker/cloud execution.
- Rendered Tk UI inspection.

## Remaining

- None for this tranche before handoff.
