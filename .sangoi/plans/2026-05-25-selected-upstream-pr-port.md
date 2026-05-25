# Selected Upstream PR Port Plan

Difficulty: time-consuming
Status: approved with fixes; fifth `Senior Plan Auditor` returned `APPROVE_WITH_FIXES`, local fixes applied
Date: 2026-05-25
Branch: `master-update`
Current head: `57d0a21f8e2b`
Upstream PR refs: `refs/pr/upstream/<number>` fetched from `Nerogar/OneTrainer`

## Summary

Port useful stale upstream OneTrainer PR intent into this fork without throwing away local Sangoi work. The first shortlist remains the foundation: the broader model-family/edit-training analysis may add candidates and may block technically unsupported candidates, but it must not replace or silently forget the original selection.

This is not a merge/cherry-pick wave. Upstream PRs are source evidence. Runtime edits must be manual intent ports through current `master-update` owners, preserving Sangoi SDXL behavior, TrainGPS/DataRecorder, local LoRA manifest export, repo-local uv bootstrap, Docker/cloud source guards, and current fail-loud config contracts. No compatibility aliases, old-name translation layers, dual-read paths, fallback adapters, or migration shims are allowed.

## User Intent

The user asked to implement useful stale upstream PRs, excluding the initial "leave quiet" group. I initially weighted SDXL/LoRA heavily because the fork historically focused on SDXL. The user clarified that the fork now also needs broader newer-model training help, and then clarified that "keep the PRs already selected" means the new analysis must not discard the first analysis.

Therefore this plan locks two rules:

1. The first shortlist is retained as a selected-intent floor.
2. Expanded analysis can add candidates and can mark candidates blocked/deferred with concrete technical evidence, but cannot erase the first shortlist.

## Current Evidence

- Worktree was clean on `master-update` at `57d0a21f8e2b` before this plan file was created.
- Root `AGENTS.md` requires local uv validation through `./.uv/bin/uv run --python .venv/bin/python --no-sync ...`, fixed `.venv`, `PYTHONPATH="$PWD"`, and CPU-only WSL validation with `OT_PLATFORM_REQUIREMENTS=requirements-default.txt`.
- Open upstream PR metadata was collected on 2026-05-25 with `gh pr list` / `gh pr view` into `/tmp/onetrainer-upstream-open-prs.json` and `/tmp/onetrainer-pr-details/*.json`.
- Upstream PR heads were fetched into `refs/pr/upstream/<number>` and conflict forecasts were produced with `git merge-tree --write-tree HEAD refs/pr/upstream/<number>`.
- Search for open/closed upstream PRs matching `qwen edit`, `Qwen Image Edit`, and `edit training` found no open Qwen Edit PR. It found #1301 `Flux2 Edit training` as the open edit-training PR. Qwen base support is already present in this checkout (`modules/model/QwenModel.py`, `modules/modelSetup/BaseQwenSetup.py`, `modules/modelSetup/QwenFineTuneSetup.py`, `modules/modelSetup/QwenLoRASetup.py`), and upstream Qwen base PR #1007 is already merged upstream.
- Merged upstream #812 `Add support for custom conditioning image` appears already represented by conditioning-image code paths in current data loaders. Do not re-port it as an open stale PR.
- Active dependency evidence from local uv on 2026-05-25:
  - `mgds: 0.1.dev164+g9320a6963`
  - `diffusers: 0.38.0.dev0`
  - `torch: 2.9.1+cpu`
  - `torchvision: 0.24.1+cpu`
  - `transformers: 4.57.6`
  - `smart-disk-cache` and `SmartDiskCache`: missing from package metadata
  - `mgds.pipelineModules.SmartDiskCache`, `LoadSmartCache`, `SaveSmartCache`: not importable
  - `mgds.pipelineModules.AspectBucketing`: importable
- Active hardware evidence from local uv on 2026-05-25:
  - `torch.cuda.is_available() == False`
  - `hasattr(torch, "xpu") == True`
  - `torch.xpu.is_available() == False`
- `requirements-rocm.txt` is currently Linux ROCm 6.3 oriented. Do not replace it with Windows ROCm requirements in this wave.
- No known-good runtime baseline exists for the selected upstream-PR aggregate because these features are not implemented together in this fork yet. Use `57d0a21f8e2b` as the source baseline, then prove the new aggregate through per-lane smokes and the final validation artifact.

## PR Status Lock

This table is the execution contract. A PR marked `BLOCKED` or `DEFERRED` remains part of the analysis record but must not land runtime code in this wave.

| PR | Source set | Status | Runtime action | Evidence / lock reason |
| --- | --- | --- | --- | --- |
| #1470 | first shortlist | IMPLEMENT | Port mid-epoch validation-resume action state. | High-value resume bugfix; conflicts are current-owner conflicts, not dependency blockers. |
| #1465 | first shortlist | IMPLEMENT | Port accumulator/gradient/RNG restore with fail-loud fingerprint checks. | High-value long-run resume robustness; no external dependency blocker. |
| #1422 | first shortlist | IMPLEMENT | Port TensorBoard resume continuity behind canonical config. | Small continuity fix; add explicit current field instead of unconditional behavior. |
| #1408 | first shortlist | IMPLEMENT | Port validation-loss patience after validation timestep owner exists. | Useful only when validation metric owner is canonical. |
| #1416 | first shortlist | IMPLEMENT | Implement stop-on-no-remaining-components through current limit owner. | Clean merge forecast, but upstream `requires_grad` approach is not accepted as contract owner. |
| #1436 | first shortlist | IMPLEMENT | Main validation-timestep design source. | Stronger than #821 for deterministic per-sample validation policy. |
| #821 | first shortlist | IMPLEMENT | Fold fixed timestep input into #1436 resolver; do not create a second mechanism. | Keeps first shortlist while avoiding duplicate validation ownership. |
| #1225 | first shortlist | IMPLEMENT | Add diffusion-only `BETA` timestep distribution. | Small training-quality feature; reject flow families unless later supported. |
| #1124 | first shortlist | IMPLEMENT | Add `SPEED` timestep distribution only in current noise owner. | Training-quality feature; no full noise-flow fork. |
| #1260 | first shortlist | IMPLEMENT | Add immiscible diffusion with bounded oversampling. | Useful noise assignment feature; validate shape/dtype/device. |
| #1235 | first shortlist | IMPLEMENT | Add CEP controls and runtime helper. | Training regularization feature; keep separate from CIOP. |
| #1315 | first shortlist | IMPLEMENT | Add Scaled OFT modifier. | Adapter quality feature; must preserve manifest export. |
| #1335 | first shortlist | IMPLEMENT | Add DoRA-OFT modifier. | Adapter quality feature; sequence after Scaled OFT. |
| #1464 | first shortlist | IMPLEMENT | Add real LoKr PEFT type and config/UI/runtime support. | New adapter support; must not masquerade as LoRA/OFT. |
| #1360 | first shortlist | IMPLEMENT_PARTIAL | Land only `lora_te_scale` and `lora_unet_scale`. | Full CFG distillation is out of scope and forbidden from import in this wave. |
| #1461 | first shortlist | IMPLEMENT | Add `PrefetchIterator` / `prefetch_next_batch`, default disabled. | Performance feature; validate producer exception propagation. |
| #1411 | first shortlist | BLOCKED | No SmartDiskCache runtime code. | Current `mgds` pin lacks `SmartDiskCache` modules and package metadata; no approved dependency update. |
| #1301 | expanded analysis | IMPLEMENT | Port Flux2 Edit training through current Flux2 owners. | Open edit-training PR found; Qwen Edit PR by name was not found. |
| #1471 | expanded analysis | IMPLEMENT | Port HiDream tokenizer mixup fix. | Small newer-model correctness fix. |
| #1460 | expanded analysis | IMPLEMENT | Apply only as an incidental same-surface Flux2 UI layout repair while porting #1301. | Not a standalone UI scope and not a rendered redesign. |
| #1387 | expanded analysis | BLOCKED | Do not expose or enable Flux2 caption dropout in this wave. | Current `Flux2Model.encode_text()` raises `NotImplementedError` when dropout is greater than zero; UI-only exposure would create a broken control. |
| #1374 | expanded analysis | IMPLEMENT | Port Hunyuan Video LoRA ComfyUI converter path. | Converter/export support only; no Hunyuan training semantics change. |
| #1435 | expanded analysis | IMPLEMENT | Add CIOP controls and runtime helper. | Training regularization feature; coordinate with CEP/noise path. |
| #1134 | expanded analysis | IMPLEMENT | Add `target_resolution` to image cache key. | Cache correctness fix independent from blocked SmartDiskCache. |
| #1227 | expanded analysis | IMPLEMENT | Add attention backend selection only through current native/diffusers/PyTorch APIs. | No `requirements*.txt` drift, no split-attention import, no XPU/ROCm/Windows platform coupling. |
| #1337 | expanded analysis | BLOCKED | No fixed-resolution/quantization MGDS runtime code. | Auditor lock plus local evidence: current approved `mgds` support is not proven; no dependency update approved. |
| #1218 | expanded analysis | BLOCKED | No split attention runtime code. | Active diffusers/torch support and backend semantics are unproven without dependency/platform work. |
| #1413 | expanded analysis | BLOCKED | No XPU runtime/submodule code. | Local XPU unavailable; `ipex_to_cuda`/submodule not approved; source-only path not enough for this wave. |
| #1385 | expanded analysis | BLOCKED | Do not replace ROCm requirements. | Current ROCm file targets Linux ROCm 6.3; Windows ROCm/Python 3.12 exception needs separate platform plan. |
| #1108 | expanded analysis | BLOCKED | No activation-quantization runtime code. | Enabled-path support matrix and trainable backward smoke are not available in CPU-only validation. |
| #1353 | expanded analysis | DEFERRED | No delta-transfer runtime code. | Draft experimental method; would need a separate canonical training-method design, not hidden trainer branch. |

## Explicitly Out Of Scope For This Wave

- #1244 pixi installation. It conflicts with this fork's repo-local uv-managed bootstrap contract.
- #1472 / #1285 dependency upgrade wave. Dependency upgrades can break runtime and need a separate standards/dependency plan.
- #1445 / #1446 Qt6/UI rewrite and CTK decoupling.
- #1403 DPO/RLHF.
- #1052 Diff2Flow.
- #1462 split latent caching. It can be reconsidered only after #1461 and cache-key correctness are proven.
- Full wholesale CFG distillation from #1360.
- Qwen Edit support by name. No matching open upstream PR was found. If the user provides a PR/branch, it becomes a separate model-family plan or explicit scope expansion.
- Full CUDA/GPU/XPU/ROCm validation from this WSL checkout.
- Full training quality benchmarks.

#1460 is not part of this rejected/deferred list. It remains allowed only as same-surface collateral while implementing Flux2 Edit UI changes, not as a standalone UI polish wave.

## Owner Matrix

| Lane | Implemented PRs | Canonical owners | Blocked/deferred PRs in same area |
| --- | --- | --- | --- |
| Resume and continuity | #1470, #1465, #1422 | `modules/util/TrainProgress.py`, `modules/util/TimedActionMixin.py`, model loader/saver mixins, `modules/model/BaseModel.py`, `modules/trainer/BaseTrainer.py`, `modules/trainer/GenericTrainer.py` | none |
| Validation and stop policy | #1436, #821, #1408, #1416 | `modules/trainer/GenericTrainer.py`, `modules/ui/TrainUI.py`, `modules/util/config/TrainConfig.py`, model setup validation paths, new validation timestep utility | none |
| Timestep/noise/regularization quality | #1225, #1124, #1260, #1235, #1435 | `modules/util/enum/TimestepDistribution.py`, `modules/modelSetup/mixin/ModelSetupNoiseMixin.py`, base model setup files, `modules/ui/TrainingTab.py`, `modules/util/config/TrainConfig.py`, new utility modules | none |
| Newer model/edit support | #1301, #1471, #1460 | Flux2 data loader/model/sampler/setup, HiDream model loader, `modules/ui/TrainingTab.py`, `modules/ui/SampleFrame.py`, `modules/util/enum/ModelType.py` | #1387; Qwen Edit by name, no concrete upstream PR found |
| Hunyuan conversion | #1374 | Hunyuan Video LoRA loader/saver, LoRA converter utilities, `modules/ui/ConvertModelUI.py` | none |
| LoRA/OFT/adapter expansion | #1315, #1335, #1464, #1360 partial | `modules/module/LoRAModule.py`, `modules/module/oft_utils.py`, new `modules/module/lokr_utils.py`, `modules/ui/LoraTab.py`, `modules/util/config/TrainConfig.py`, `modules/util/enum/ModelType.py`, loader/saver mixins | CFG distillation portion of #1360 |
| Cache/performance/correctness | #1461, #1134 | data loaders, `modules/dataLoader/mixin/DataLoaderText2ImageMixin.py`, `modules/trainer/GenericTrainer.py`, `modules/ui/TrainUI.py`, `modules/util/config/TrainConfig.py` | #1411, #1337 |
| Hardware/backend support | #1227 current-API-only | `modules/util/enum/AttentionMechanism.py`, `modules/modelSetup/BaseModelSetup.py`, all setup owners listed in the Attention Backend Scope section, `modules/util/torch_util.py` | #1218, #1413, #1385 |
| Experimental transfer/quantization | none | none in this wave | #1353, #1108 |

## Config / UI Schema Freeze

Add each runtime field once in `modules/util/config/TrainConfig.py`. UI bindings must use these exact current names. There are no old-name aliases, dual-read loaders, or compatibility sanitizers.

| Field / enum | Default | UI surface | Runtime consumer | PR source | Validation |
| --- | --- | --- | --- | --- | --- |
| `tensorboard_resume_run: bool` | `True` | `modules/ui/TrainUI.py` TensorBoard section | `GenericTrainer` / backup loader TensorBoard writer setup | #1422 | Config default + resume metadata smoke |
| `patience: bool` | `False` | `modules/ui/TrainUI.py` validation section | `GenericTrainer` validation-loss early stop | #1408 | Synthetic validation-loss ticks |
| `patience_epochs: int` | `5` | `modules/ui/TrainUI.py` validation section | `GenericTrainer` patience counter | #1408 | Best-state restore source smoke |
| `ValidationTimestepMode` enum | `AUTO` | validation section | new validation timestep resolver | #1436/#821 | Enum import + resolver matrix |
| `validation_timestep_mode: ValidationTimestepMode` | `AUTO` | validation section | new validation timestep resolver | #1436/#821 | AUTO/FIXED/STRATIFIED matrix |
| `validation_timestep_values: str` | `"500"` | validation section | fixed/stratified parser | #821 folded into #1436 | Parser rejects invalid values |
| `validation_timestep_seed: int` | `0` | validation section | stratified assignment helper | #1436 | Determinism smoke |
| `timestep_distribution` enum values `BETA`, `SPEED` | existing field default remains `UNIFORM` | `modules/ui/TrainingTab.py` timestep distribution selector | `ModelSetupNoiseMixin` | #1225/#1124 | Distribution enum + math smokes |
| `k_noise_sampling: int` | `1` | `TrainingTab.py` noise section | immiscible diffusion helper | #1260 | Oversampling 1 disabled-equivalent, invalid values fail |
| `cep_enabled: bool` | `False` | `TrainingTab.py` regularization/noise section | conditional embedding perturbation helper | #1235 | Disabled unchanged, enabled helper smoke |
| `cep_gamma: float` | `1.0` | `TrainingTab.py` regularization/noise section | conditional embedding perturbation helper | #1235 | Range/shape smoke |
| `ciop_noise_weight: float` | `0.0` | `TrainingTab.py` regularization/noise section | CIOP perturbation helper | #1435 | Disabled unchanged, enabled helper smoke |
| `ciop_p: float` | `0.8` | `TrainingTab.py` regularization/noise section | CIOP perturbation helper | #1435 | Probability bounds smoke |
| `scaled_oft: bool` | `False` | `modules/ui/LoraTab.py` OFT section | OFT adapter construction | #1315/#1335 | Adapter matrix smoke |
| `dora_oft: bool` | `False` | `modules/ui/LoraTab.py` OFT section | OFT adapter construction | #1335 | Adapter matrix smoke |
| `PeftType.LOKR` | n/a | `LoraTab.py` adapter type selector | LoKr adapter construction/load/save | #1464 | PEFT enum + state dict smoke |
| `lokr_dim: int` | `16` | `LoraTab.py` LoKr section | `lokr_utils.py` / `LoRAModule.py` | #1464 | Construction smoke |
| `lokr_decompose_both: bool` | `False` | `LoraTab.py` LoKr section | LoKr construction | #1464 | Construction smoke |
| `lokr_decompose_factor: int` | `-1` | `LoraTab.py` LoKr section | LoKr construction | #1464 | Invalid-factor rejection |
| `lokr_use_tucker: bool` | `False` | `LoraTab.py` LoKr section | LoKr construction | #1464 | Construction smoke |
| `lokr_weight_decompose: bool` | `False` | `LoraTab.py` LoKr section | LoKr construction | #1464 | Construction smoke |
| `lokr_dora_on_output: bool` | `True` | `LoraTab.py` LoKr section | LoKr construction | #1464 | Construction smoke |
| `lokr_full_matrix: bool` | `False` | `LoraTab.py` LoKr section | LoKr construction | #1464 | Construction smoke |
| `lokr_vec_trick: bool` | `True` | `LoraTab.py` LoKr section | LoKr construction | #1464 | Construction smoke |
| `lora_te_scale: float` | `1.0` | `modules/ui/LoraTab.py` LoRA advanced section | LoRA load/build path | #1360 partial | TE scale load smoke |
| `lora_unet_scale: float` | `1.0` | `modules/ui/LoraTab.py` LoRA advanced section | LoRA load/build path | #1360 partial | UNet scale load smoke |
| `prefetch_next_batch: bool` | `False` | `TrainUI.py` cache/dataloader section | `PrefetchIterator` wrapping trainer/dataloader iteration | #1461 | Normal/exception/early-stop smokes |
| `attention_mechanism: AttentionMechanism` | `SDP` | `TrainingTab.py` performance/attention section | model setup attention calls | #1227 | Optional backend missing smoke |

Forbidden schema names/imports in this wave:

- `distillation`, `DistillationConfig`, `DistillationCacheMode`, `DistillationLossType`, `DistillationTargetMode`, `ParentModelWrapper`, `cfg_distillation`, and any CFG-distillation cache/loss enum from #1360.
- Empty/placeholder config field names from upstream #1464's bad `data.append(("", ...))` line.
- Runtime imports of `SmartDiskCache`, `LoadSmartCache`, or `SaveSmartCache` while #1411 is blocked.
- Runtime imports of #1337 fixed-resolution/quantization MGDS APIs while #1337 is blocked. Upstream #1337's observed blocked symbol is `resolution_quantization`; current local `aspect_bucketing_quantization_override` remains an existing supported field and must not be renamed in this wave.
- `ipex_to_cuda` imports/submodules or XPU runtime activation while #1413 is blocked.
- New Flux2 caption-dropout UI/runtime behavior from #1387 while #1387 is blocked.
- `attention_backend` as a config/UI/schema field or alias. Calling active-stack methods named `set_attention_backend()` is allowed only from the canonical `attention_mechanism` owner.
- Any old-name translation for newly added optional metadata. Missing new optional metadata loads as absent/current default only; no old-name translation.

## Validation Timestep Contract

One owner resolves validation timestep policy before model-family setup code consumes it.

- New owner: `modules/util/validation_timestep.py`.
- Inputs: `validation_timestep_mode`, `validation_timestep_values`, `validation_timestep_seed`, scheduler timestep count, sample/concept identity where needed.
- Outputs: resolved timestep indices/noise seeds; model setup code consumes resolved policy only.
- `AUTO` preserves current validation behavior.
- `FIXED` parses `validation_timestep_values` as explicit integer timesteps.
- `STRATIFIED` uses deterministic sample/concept assignment seeded by `validation_timestep_seed`.
- Invalid values fail loud before training starts.
- #821's fixed-timestep intent is folded into this resolver; no parallel `validation_timesteps` owner is added.
- Per-setup ledger: every setup that currently calls `_get_timestep_discrete()` or `_get_timestep_continuous()` must be evaluated and either wired to `modules/util/validation_timestep.py` for deterministic validation or explicitly rejected with a local reason. Current owners are `BaseStableDiffusionSetup.py`, `BaseStableDiffusionXLSetup.py`, `BaseStableDiffusion3Setup.py`, `BaseFluxSetup.py`, `BaseFlux2Setup.py`, `BaseChromaSetup.py`, `BaseErnieSetup.py`, `BaseHiDreamSetup.py`, `BaseHunyuanVideoSetup.py`, `BasePixArtAlphaSetup.py`, `BaseQwenSetup.py`, `BaseSanaSetup.py`, `BaseWuerstchenSetup.py`, and `BaseZImageSetup.py`.
- The validation script's `--check validation-timesteps` command must prove that every listed owner consumes the canonical resolver during deterministic validation and does not parse `validation_timestep_values` independently.

## Strict Config Load Contract

Current `BaseConfig.from_dict()` suppresses field-level conversion exceptions and can leave defaults in place. New runtime config fields therefore need an explicit fail-loud owner.

- Canonical owner: override `TrainConfig.from_dict()` in `modules/util/config/TrainConfig.py` to perform raw-dict validation before `BaseConfig.from_dict()` and post-object validation after it.
- Missing newly added optional fields may load as current defaults.
- Present but invalid values fail loud with `ValueError` before training/model loading starts.
- No old-name translation, alias lookup, dual-read, or migration shim is allowed.
- Existing pre-task config migrations remain only for existing historical config versions. New fields from this plan must not be added to `config_migrations`, hidden alias maps, old-name lookups, migration shims, or dual-read/fallback branches.
- Entry-point propagation: training/UI load paths must use the strict `TrainConfig.from_dict()` behavior. Relevant current callers include `scripts/train.py`, `scripts/train_remote.py`, `scripts/calculate_loss.py`, `modules/ui/TopBar.py`, `modules/trainer/MultiTrainer.py`, and `modules/trainer/GenericTrainer.py` as a final pre-train guard.
- Validation checks include enum values, range checks, exact defaults after default construction, and roundtrip behavior for every new field.

## Adapter Contract

- `modules/util/enum/ModelType.py` `PeftType` owns adapter identity.
- `PeftType.LOKR` is a real adapter kind.
- `scaled_oft` and `dora_oft` are OFT modifiers only.
- LoKr parameters are LoKr-only and must not affect LoRA/OFT/LoHa paths.
- `lora_te_scale` and `lora_unet_scale` are numeric load/build modifiers only.
- Local `build_lora_key_manifest` / `export_lora_key_manifest` APIs must remain deterministic for all existing adapter paths.
- Unsupported adapter combinations fail loud instead of falling back to a different adapter type.

## Flux2 Edit Contract

#1301 lands as an edit-mode capability on current `ModelType.FLUX_2`; do not add a second Flux2 model type unless implementation evidence proves the current owner cannot represent the behavior.

- User-facing switch: reuse the existing canonical `custom_conditioning_image` field as the Flux2 Edit enable flag.
- Reference file rule: when `custom_conditioning_image=True` for Flux2, each training image must have a sibling PNG using the existing `-condlabel.png` postfix rule. Missing reference images fail loud during dataset preparation.
- Standard Flux2 disabled behavior: when `custom_conditioning_image=False`, current Flux2 training must not request, cache, sort, output, sample, or debug `conditioning_image` / `latent_conditioning_image`.
- Dataloader keys when enabled: `conditioning_image`, `latent_conditioning_image_distribution`, `latent_conditioning_image`, and debug `decoded_conditioning_image`.
- Cache/sort/output contract: Flux2 adds `latent_conditioning_image` to split/output names only when the switch is enabled, and uses the same image path as the primary image for ordering.
- Sampler/model contract: `Flux2Model.prepare_latent_image_ids()` accepts an explicit image index for reference latents; sampler/model setup concatenate conditioning latents before transformer prediction and discard conditioning predictions afterward.
- `ModelType.has_conditioning_image_input()` must not globally make all Flux2 runs conditioning-image runs unless the implementation gates every Flux2 conditioning path by `custom_conditioning_image`.
- Validation smokes: standard Flux2 disabled path has no conditioning keys; enabled path requires `-condlabel.png`, emits `latent_conditioning_image`, and missing reference image fails before training.

## Flux2 Caption Dropout Decision

#1387 is blocked in this wave.

- Current `Flux2Model.encode_text()` raises `NotImplementedError` when `text_encoder_dropout_probability > 0.0`.
- Do not expose a new Flux2 caption-dropout control and do not document Flux2 caption dropout as supported.
- Existing generic text-encoder dropout fields remain current for model families that already support them.
- `ModelType.FLUX_2` with `text_encoder.dropout_probability > 0` must fail in `TrainConfig.validate_for_training()` or an equivalent pre-model-loading guard, before dataset/model execution can reach `Flux2Model.encode_text()`.
- If Flux2 dropout is promoted later, the new plan must define real dropped-conditioning semantics and an enabled-path smoke that no longer raises.

## Prefetch Lifecycle Contract

#1461 may land only as a bounded iterator wrapper, not as a trainer-control rewrite.

- Canonical owner: `modules/util/PrefetchIterator.py` plus the current `GenericTrainer` training dataloader loop.
- Default: `prefetch_next_batch=False`.
- Allowed path: wrap only the main training dataloader iterator after dataset creation and before the per-batch training loop.
- Forbidden paths: validation dataloader, sampling, backup/save execution, `only_cache` cache generation, task-log/changelog code, and any path that moves the model between devices while the producer may still use model-owned preparation modules.
- Eligibility: prefetch is enabled only when `latent_caching=True`, `only_cache=False`, and the current dataloader path does not need live VAE/text-encoder preparation during iteration.
- Device/offload boundary: backup/save/sample commands may run only after the current batch has completed and the prefetch worker is either waiting on cached data or has been stopped; no model device move can race a producer that uses model modules.
- Cleanup: the iterator must join/close its worker on normal exhaustion, producer exception, consumer exception, and early consumer stop.
- Error contract: producer exceptions propagate on the consumer's next `next()` call; they are never converted to sentinel objects or logged-only warnings.
- Validation smokes: normal iteration, producer `RuntimeError` propagation, early close, disabled-default no-op behavior, and sample/backup/save/offload device-boundary safety while the worker is active or closing.

## Validation Metric And Patience Contract

#1408 depends on a canonical validation metric; it cannot read TensorBoard writes as the source of truth.

- Canonical owner: `GenericTrainer.__validate()` returns a small `ValidationMetrics` value or `None`.
- Metric shape: per-concept average loss keyed by concept seed/label plus `total_average_loss`.
- Single-concept behavior: `total_average_loss` is the single concept average.
- Multi-concept behavior: `total_average_loss` is weighted by validation batch counts, matching the current TensorBoard total-average calculation.
- Empty validation behavior: return `None`; if `patience=True`, this is a configuration/runtime error, not a silent no-op.
- TensorBoard writes consume `ValidationMetrics`; they do not compute independent metrics.
- Patience owner: `GenericTrainer` tracks best `total_average_loss`, patience counter, and best-state artifact.
- Best-state source: use the current internal model save/load path or an explicit lightweight best-state artifact under the workspace; restore it before final save when early stopping fires.
- Validation smokes: no-validation disabled behavior, empty-validation failure with patience enabled, single-concept metric, multi-concept weighted total, patience trigger, and best-state restore source path.

## Hunyuan Comfy LoRA Contract

#1374 lands through one current conversion owner.

- Canonical format: existing `ModelFormat.COMFY_LORA`.
- Canonical UI surface: `ConvertModelUI.py` exposes `ComfyUI LoRA` only when `model_type == ModelType.HUNYUAN_VIDEO` and `training_method == TrainingMethod.LORA`; other model/method selections must reset or hide the format.
- Canonical saver path: route through `HunyuanVideoLoRASaver.save()` and current LoRA conversion helpers. Generic LoRA saving may gain a `ModelFormat.COMFY_LORA` case only when it remains format-owner-only and does not change non-Hunyuan outputs.
- Key contract: Hunyuan Comfy export filters/fuses only intended Hunyuan transformer block LoRA keys. Bundled embedding keys, text-encoder keys, and non-Hunyuan keys are explicitly omitted from Comfy export rather than silently rewritten.
- Required fixture: input includes double-block split Q/K/V LoRA keys, single-block split Q/K/V/MLP LoRA keys, a bundled embedding key, a text-encoder key, and a non-Hunyuan key. Expected output contains fused `transformer.double_blocks.<n>.img_attn_qkv.*` / `txt_attn_qkv.*` and `transformer.single_blocks.<n>.linear1.*` keys only, and omits bundled/text/non-Hunyuan keys.
- Validation smoke: synthetic Hunyuan LoRA key set maps to expected Comfy keys; bundled/text/non-Hunyuan keys are absent; non-Hunyuan `COMFY_LORA` selection is rejected or hidden.

## Attention Backend Scope

#1227 is implemented only if it can stay inside current stack APIs.

- Canonical owner: `modules/util/enum/AttentionMechanism.py` plus `TrainConfig.attention_mechanism`.
- Exact values: `AttentionMechanism.SDP` maps to active diffusers `AttentionBackendName.NATIVE` via `component.set_attention_backend("native")`; `AttentionMechanism.FLASH` maps to active diffusers `AttentionBackendName.FLASH` via `component.set_attention_backend("flash")`.
- `AttentionMechanism.FLASH` means the external diffusers `flash` / `flash-attn` backend. It does not mean PyTorch native `_native_flash`, and this wave must not expose `_native_flash` as a user config value.
- Setup owners that must be evaluated and either wired or explicitly rejected before coding: `BaseStableDiffusionXLSetup.py`, `BaseFlux2Setup.py`, `BaseFluxSetup.py`, `BaseErnieSetup.py`, `BaseHunyuanVideoSetup.py`, `BaseStableDiffusion3Setup.py`, `BaseQwenSetup.py`, `BaseZImageSetup.py`, `BaseStableDiffusionSetup.py`, `BasePixArtAlphaSetup.py`, `BaseHiDreamSetup.py`, `BaseChromaSetup.py`, `BaseSanaSetup.py`, and `BaseWuerstchenSetup.py`.
- Per-owner ledger: every setup owner above must record whether `SDP` and `FLASH` are applied to a concrete component or explicitly rejected with the local reason. A selected non-default backend must never silently no-op.
- Source guard: only call `set_attention_backend()` on components that expose that method in the active diffusers/PyTorch stack. A selected unsupported backend fails loud during setup; it is not silently remapped.
- No `requirements*.txt` changes, no `ipex_to_cuda`, no XPU support, no Windows ROCm support, no split attention, and no submodules.
- Missing optional backend behavior: fail loud only when the user selects that backend. Default `SDP` remains importable in CPU-only validation, but an owner may no-op only when the ledger proves no compatible attention component exists for that setup.
- If source inspection shows #1227 requires dependency/platform drift, mark #1227 blocked before runtime edits.

## Implementation Strategy

Implement in dependency order. After each lane, run targeted local smokes before moving to the next lane. If evidence contradicts a status lock, stop and update the plan before runtime edits.

### Step 1: Gate This Plan

- Submit this patched plan to `Senior Plan Auditor` before runtime edits.
- Apply only auditor-approved local plan fixes before implementation.
- Done when the auditor returns `APPROVE` or `APPROVE_WITH_FIXES` and required plan fixes are applied.

### Step 2: Resume And Continuity

- Port #1470 persisted epoch-action firing state into `TrainProgress` and backup metadata load/save paths.
- Port #1470 action-state semantics through `TimedActionMixin`, not ad hoc trainer checks.
- Persist only epoch/step action identity needed to prevent duplicate or missed `validate`, `sample`, `backup`, and `save` actions after restart.
- Do not persist `time.monotonic()` values or wall-clock timer state. SECOND/MINUTE/HOUR actions restart from the new process start and remain volatile.
- `TimedActionMixin.repeating_action_needed()` owns elapsed action checks for EPOCH/STEP/SECOND/MINUTE/HOUR, and `TrainProgress` owns durable counters.
- Port #1465 accumulator/gradient/RNG state artifacts under backup data with stable parameter-group keys and a dataset/config fingerprint.
- Refuse accumulator restore on incompatible fingerprint, missing parameter groups, or changed `gradient_accumulation_steps`.
- #1465 validation must include a mid-accumulation resume with `gradient_accumulation_steps > 1`, pending nonzero gradients, restored optimizer/scaler/RNG state, matching dataset/config fingerprint success, mismatched fingerprint failure, and changed accumulation-step failure.
- Port #1422 by persisting TensorBoard subdir and adding `tensorboard_resume_run` behavior.
- Preserve optional TensorBoard behavior and Sangoi tensorboard scalar logging.
- Done when synthetic resume metadata smokes prove missing new optional metadata loads as current defaults, new metadata roundtrips, validation/backup/save action state resumes correctly, time-unit behavior is volatile and not persisted, and incompatible accumulator state fails loud.

### Step 3: Validation And Stop Policy

- Build the validation timestep resolver defined above from #1436/#821 intent.
- Wire validation timestep resolution into every model setup that currently computes validation loss.
- Complete the validation-timestep per-setup ledger before changing setup owners; the ledger must cover the fourteen setup files named in the Validation Timestep Contract.
- Port #1408 validation-loss patience and best-checkpoint restore only when validation is enabled.
- Port #1416 stop-on-no-remaining-components through current training-part/limit semantics, not `requires_grad`.
- Done when validation timestep matrix, patience ticks, best-state restore source path, and component-limit stop owner are validated.

### Step 4: Timestep / Noise / Regularization Quality

- Add `TimestepDistribution.BETA` from #1225 with diffusion-only support checks.
- Add `TimestepDistribution.SPEED` from #1124 only inside current `ModelSetupNoiseMixin` sampling logic.
- Flow-family model setups must reject `BETA` and `SPEED` before training; do not silently remap them to `UNIFORM`.
- Add `k_noise_sampling` and immiscible diffusion helper from #1260 with explicit shape/dtype/device checks.
- Port #1235 CEP as `cep_enabled` / `cep_gamma` and runtime helper.
- Port #1435 CIOP as `ciop_noise_weight` / `ciop_p` and runtime helper.
- Keep `PRIORITY_SAMPLING`, Sangoi loss, TrainGPS, and flow rejection behavior intact.
- Done when disabled behavior is unchanged, enabled features route distinctly, and invalid combinations fail loud.

### Step 5: Newer Model / Edit Support

- Port #1471 HiDream tokenizer fixes directly in `HiDreamModelLoader.py` or current HiDream loader owner.
- Port #1301 Flux2 Edit training through current Flux2 data loader/model/sampler/setup and `ModelType` owners.
- Keep or improve the one-reference-image/companion-file contract. If still required, document and validate the exact current companion filename rule.
- Do not port #1387 caption dropout behavior in this wave.
- Apply #1460 only as an incidental layout fix while editing the same Flux2 `TrainingTab.py` surface for #1301.
- Done when Flux2 Edit config/model-type smokes prove discoverability and standard Flux2 training remains unaffected when edit mode is disabled.

### Step 6: Hunyuan Conversion

- Port #1374 Hunyuan Video LoRA to ComfyUI conversion.
- Keep conversion isolated to Hunyuan Video LoRA converter/loader/saver and `ConvertModelUI.py` output format choices.
- Validate that non-Hunyuan conversion formats remain unchanged and Hunyuan ComfyUI output maps only intended keys.

### Step 7: LoRA / OFT / Adapter Expansion

- Port #1315 Scaled OFT as `scaled_oft` current config/UI field and current OFT math.
- Port #1335 DoRA-OFT as `dora_oft` current config/UI field and OFT/DoRA integration.
- Port #1464 LoKr as `PeftType.LOKR`, `lokr_utils.py`, config fields, UI controls, state dict/load/save handling, and validation.
- Port only #1360 `lora_te_scale` and `lora_unet_scale`; do not port CFG distillation, parent model wrappers, distillation cache modes, or distillation loss enums.
- Preserve current LoRA key manifest export APIs.
- Done when synthetic wrapper smokes cover Scaled OFT, DoRA-OFT, LoKr construction/state dict, TE/UNet scale load behavior, and manifest export.

### Step 8: Cache / Performance / Correctness

- Port #1461 `PrefetchIterator` with `prefetch_next_batch` defaulting to `False`.
- Enable prefetch only where cache/offload/sampling boundaries are safe.
- Port #1134 target-resolution cache-key correctness with explicit cache invalidation behavior.
- Do not port #1411 SmartDiskCache or #1337 MGDS runtime code in this wave.
- Done when prefetch smokes cover normal iteration, producer exception propagation, early consumer stop, and CPU-only operation; cache-key smoke proves target resolution affects image cache identity.

### Step 9: Attention Backend

- Port #1227 attention backend selection through one canonical `AttentionMechanism` owner and optional dependency checks.
- Create the per-owner #1227 ledger before code edits touch setup files; include `BaseWuerstchenSetup.py` and document every applied or rejected owner in the task log.
- Do not port #1218 split attention in this wave.
- Do not port #1413 XPU runtime/submodule code or #1385 Windows ROCm requirements.
- Do not modify `requirements*.txt` for #1227.
- Done when CPU validation still passes, selected optional attention backends fail loud if dependencies are unavailable, and the per-owner ledger proves no selected non-default backend silently no-ops.

### Step 10: Blocked / Deferred Accounting

- Record #1411, #1337, #1218, #1413, #1385, #1108, #1387 as blocked in the task log with evidence from this plan and current local checks.
- Record #1353 as deferred because it needs a separate canonical training-method design.
- Do not leave partial imports, dead fields, or docs for blocked/deferred PRs.

### Step 11: Documentation, Changelog, And Task Log

- Update `.sangoi/CHANGELOG.md` with user/maintainer-visible landed changes only.
- Create `.sangoi/task-logs/2026-05-25-selected-upstream-pr-port.md` with selected PRs, landed PRs, blocked/deferred PR evidence, validation run, and out-of-scope surfaces.
- Update user docs only where current behavior changes operator expectations.
- Task logs may record blocked/deferred PR evidence; user docs, examples, presets, prompts, and comments must not describe blocked behavior as current behavior.
- Do not add historical migration notes or old-field compatibility lists.
- Done when docs describe only current behavior and task log names validation artifacts exactly.

## Stop Conditions

Stop before runtime edits or before continuing a lane if any condition appears:

- `./.uv/bin/uv` or `.venv/bin/python` is missing or not executable.
- A lane requires global/system `python`, `pip`, or `uv` for validation.
- A blocked PR's runtime imports, config fields, docs, or examples appear in current code.
- #1411 SmartDiskCache becomes necessary for #1134 or #1461; current evidence says it is missing, so do not synthesize a fallback.
- #1337 requires an unapproved `mgds` pin or API shape.
- #1108 cannot produce an enabled-path backward smoke for trainable weights.
- #1413 implementation would require `ipex_to_cuda`, a submodule, or hardware success claims.
- #1385 would replace current Linux ROCm requirements instead of a separate approved Windows platform plan.
- A new config field needs an old-name alias, dual-read, or migration shim to pass validation.
- Flux2 Edit work discovers a Qwen/Edit PR or branch not in this plan; record it and ask before scope expansion.

## Blocked / Deferred Residue Matrix

The final validation denylist is rooted in this matrix. Task logs may mention these PRs as evidence; runtime code, docs, examples, presets, requirements, scripts, and comments must not expose them as landed behavior.

| PR | Status | Forbidden runtime/doc residue |
| --- | --- | --- |
| #1411 | BLOCKED | `SmartDiskCache`, `LoadSmartCache`, `SaveSmartCache`, `smart-disk-cache`, persistent smart-cache config/UI/docs. |
| #1337 | BLOCKED | new MGDS fixed-resolution/quantization API imports, `resolution_quantization`, fixed-resolution MGDS config/UI/docs, dependency pins for unapproved MGDS PR support. |
| #1218 | BLOCKED | `split_attention`, `native_split`, `flash_split`, split-attention backend docs/config/UI. |
| #1413 | BLOCKED | `ipex_to_cuda`, XPU runtime activation, XPU submodule/config/docs, Intel XPU hardware success claims. |
| #1385 | BLOCKED | Windows ROCm requirements replacement, TheRock/Windows ROCm bootstrap docs, Python 3.12 ROCm exception paths. |
| #1108 | BLOCKED | activation-quantization training controls, `LinearA8` activation-training plumbing, trainable activation quantization support claims. |
| #1353 | DEFERRED | delta-transfer training-method/config/UI/docs, hidden trainer branches for model delta transfer. |
| #1387 | BLOCKED | Flux2 caption-dropout support claims or UI/runtime behavior that permits dropout greater than zero. |
| #1360 full distillation | OUT_OF_SCOPE | `distillation`, `DistillationConfig`, `DistillationCacheMode`, `DistillationLossType`, `DistillationTargetMode`, `ParentModelWrapper`, CFG-distillation cache/loss docs. |
| #1436/#821 duplicate owner | FORBIDDEN | old `validation_timesteps` field, parallel validation timestep owner, old-name migration or alias. |
| #1301 duplicate model identity | FORBIDDEN | `FLUX_2_EDIT` model type or separate Flux2 Edit enum instead of current `ModelType.FLUX_2` plus `custom_conditioning_image`. |
| #1227 duplicate backend owner | FORBIDDEN | `attention_backend` field or backend alias in addition to canonical `attention_mechanism`. |

## Validation Plan

Run the final validation from repository root after implementation. The final artifact must include the formal command and final status marker. Implementation must add `scripts/validate_selected_upstream_pr_ports.py` for lane-specific smokes; do not replace executable checks with prose.

```bash
set -euo pipefail
export OT_PLATFORM_REQUIREMENTS=requirements-default.txt
export PYTHONPATH="$PWD"

if [ ! -x ./.uv/bin/uv ]; then
    echo "missing repo-local uv" >&2
    exit 1
fi
if [ ! -x .venv/bin/python ]; then
    echo "missing repo .venv python" >&2
    exit 1
fi

./.uv/bin/uv run --python .venv/bin/python --no-sync python --version
./.uv/bin/uv run --python .venv/bin/python --no-sync python -m compileall modules scripts

./.uv/bin/uv run --python .venv/bin/python --no-sync python - <<'PY'
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.AttentionMechanism import AttentionMechanism
from modules.util.enum.ValidationTimestepMode import ValidationTimestepMode
config = TrainConfig.default_values()
required_fields = [
    "tensorboard_resume_run",
    "patience",
    "patience_epochs",
    "validation_timestep_mode",
    "validation_timestep_values",
    "validation_timestep_seed",
    "k_noise_sampling",
    "cep_enabled",
    "cep_gamma",
    "ciop_noise_weight",
    "ciop_p",
    "scaled_oft",
    "dora_oft",
    "lokr_dim",
    "lokr_decompose_both",
    "lokr_decompose_factor",
    "lokr_use_tucker",
    "lokr_weight_decompose",
    "lokr_dora_on_output",
    "lokr_full_matrix",
    "lokr_vec_trick",
    "lora_te_scale",
    "lora_unet_scale",
    "prefetch_next_batch",
    "attention_mechanism",
]
missing = [field for field in required_fields if not hasattr(config, field)]
if missing:
    raise SystemExit(f"missing TrainConfig fields: {missing}")

expected_defaults = {
    "tensorboard_resume_run": True,
    "patience": False,
    "patience_epochs": 5,
    "validation_timestep_mode": ValidationTimestepMode.AUTO,
    "validation_timestep_values": "500",
    "validation_timestep_seed": 0,
    "k_noise_sampling": 1,
    "cep_enabled": False,
    "cep_gamma": 1.0,
    "ciop_noise_weight": 0.0,
    "ciop_p": 0.8,
    "scaled_oft": False,
    "dora_oft": False,
    "lokr_dim": 16,
    "lokr_decompose_both": False,
    "lokr_decompose_factor": -1,
    "lokr_use_tucker": False,
    "lokr_weight_decompose": False,
    "lokr_dora_on_output": True,
    "lokr_full_matrix": False,
    "lokr_vec_trick": True,
    "lora_te_scale": 1.0,
    "lora_unet_scale": 1.0,
    "prefetch_next_batch": False,
    "attention_mechanism": AttentionMechanism.SDP,
}
for field, expected in expected_defaults.items():
    actual = getattr(config, field)
    if actual != expected:
        raise SystemExit(f"{field} default is {actual!r}, expected {expected!r}")

roundtrip = TrainConfig.default_values().from_dict(config.to_dict())
roundtrip.validate_for_training()

def expect_config_error(mutator, message):
    invalid = config.to_dict()
    mutator(invalid)
    try:
        TrainConfig.default_values().from_dict(invalid)
    except ValueError:
        return
    raise SystemExit(message)

expect_config_error(
    lambda invalid: invalid.__setitem__("validation_timestep_values", "not-an-int"),
    "invalid validation_timestep_values did not fail loud",
)
expect_config_error(
    lambda invalid: invalid.__setitem__("attention_mechanism", "MISSING_BACKEND"),
    "invalid attention_mechanism did not fail loud",
)

for stale_key in [
    "validation_timesteps",
    "attention_backend",
    "FLUX_2_EDIT",
    "distillation",
    "SmartDiskCache",
    "activation_quantization",
    "delta_transfer",
    "caption_dropout_probability",
    "resolution_quantization",
]:
    expect_config_error(
        lambda invalid, stale_key=stale_key: invalid.__setitem__(stale_key, True),
        f"stale/blocked config key {stale_key!r} did not fail loud",
    )

expect_config_error(
    lambda invalid: invalid.__setitem__("model_type", "FLUX_2_EDIT"),
    "invalid model_type FLUX_2_EDIT did not fail loud",
)

def set_flux2_dropout(invalid):
    invalid["model_type"] = "FLUX_2"
    text_encoder = invalid.setdefault("text_encoder", {})
    if not isinstance(text_encoder, dict):
        raise SystemExit("TrainConfig.to_dict() text_encoder shape is not a dict")
    text_encoder["dropout_probability"] = 0.1

expect_config_error(
    set_flux2_dropout,
    "Flux2 text_encoder.dropout_probability > 0 did not fail before training",
)
PY

./.uv/bin/uv run --python .venv/bin/python --no-sync python - <<'PY'
from modules.util.enum.TimestepDistribution import TimestepDistribution
for name in ["UNIFORM", "PRIORITY_SAMPLING", "BETA", "SPEED"]:
    if not hasattr(TimestepDistribution, name):
        raise SystemExit(f"missing timestep distribution {name}")
PY

./.uv/bin/uv run --python .venv/bin/python --no-sync python - <<'PY'
from modules.util.enum.ModelType import PeftType
for name in ["LORA", "LOHA", "OFT_2", "LOKR"]:
    if not hasattr(PeftType, name):
        raise SystemExit(f"missing PeftType {name}")
PY

./.uv/bin/uv run --python .venv/bin/python --no-sync python - <<'PY'
import ast
from pathlib import Path

source = Path("modules/util/config/TrainConfig.py").read_text()
tree = ast.parse(source)
train_config = next(
    node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrainConfig"
)
default_values = next(
    node for node in train_config.body if isinstance(node, ast.FunctionDef) and node.name == "default_values"
)
fields = []
for node in ast.walk(default_values):
    if not isinstance(node, ast.Call):
        continue
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "append":
        continue
    if not node.args or not isinstance(node.args[0], ast.Tuple) or not node.args[0].elts:
        continue
    first = node.args[0].elts[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        fields.append(first.value)
if "" in fields:
    raise SystemExit("empty TrainConfig.default_values field name")
duplicates = sorted({field for field in fields if fields.count(field) > 1})
if duplicates:
    raise SystemExit(f"duplicate TrainConfig.default_values fields: {duplicates}")
if "attention_backend" in fields:
    raise SystemExit("forbidden attention_backend TrainConfig field")
PY

./.uv/bin/uv run --python .venv/bin/python --no-sync python - <<'PY'
from pathlib import Path

checked_roots = [Path("modules/ui"), Path("modules/util/config")]
for root in checked_roots:
    for path in root.rglob("*.py"):
        source = path.read_text()
        if '"attention_backend"' in source or "'attention_backend'" in source:
            raise SystemExit(f"forbidden attention_backend config/UI alias in {path}")
PY

./.uv/bin/uv run --python .venv/bin/python --no-sync python - <<'PY'
from pathlib import Path

denylisted_new_keys = [
    "validation_timesteps",
    "attention_backend",
    "FLUX_2_EDIT",
    "distillation",
    "SmartDiskCache",
    "activation_quantization",
    "delta_transfer",
    "caption_dropout_probability",
    "resolution_quantization",
]
checked_paths = [
    Path("modules/util/config/TrainConfig.py"),
    Path("scripts/train.py"),
    Path("scripts/train_remote.py"),
    Path("scripts/calculate_loss.py"),
    Path("modules/ui/TopBar.py"),
    Path("modules/trainer/MultiTrainer.py"),
    Path("modules/trainer/GenericTrainer.py"),
]
for path in checked_paths:
    source = path.read_text()
    for key in denylisted_new_keys:
        if f'"{key}"' in source or f"'{key}'" in source:
            raise SystemExit(f"denylisted key {key!r} appears in source path {path}")

train_config_source = Path("modules/util/config/TrainConfig.py").read_text()
for forbidden_snippet in [
    "legacy_",
    "alias_",
    "compat_",
    "dual_read",
    "dual_write",
    "fallback_adapter",
]:
    if forbidden_snippet in train_config_source:
        raise SystemExit(f"forbidden compatibility pattern {forbidden_snippet!r} in TrainConfig.py")
PY

./.uv/bin/uv run --python .venv/bin/python --no-sync python - <<'PY'
from modules.util.PrefetchIterator import PrefetchIterator

def producer():
    yield 1
    yield 2

if list(PrefetchIterator(producer())) != [1, 2]:
    raise SystemExit("normal prefetch iteration failed")

def failing_producer():
    yield 1
    raise RuntimeError("prefetch boom")

iterator = iter(PrefetchIterator(failing_producer()))
if next(iterator) != 1:
    raise SystemExit("prefetch first item failed")
try:
    next(iterator)
except RuntimeError:
    pass
else:
    raise SystemExit("producer RuntimeError did not propagate")
PY

./.uv/bin/uv run --python .venv/bin/python --no-sync python - <<'PY'
import importlib.util
for module in [
    "mgds.pipelineModules.SmartDiskCache",
    "mgds.pipelineModules.LoadSmartCache",
    "mgds.pipelineModules.SaveSmartCache",
]:
    if importlib.util.find_spec(module):
        raise SystemExit(f"blocked SmartDiskCache module unexpectedly importable: {module}")
PY

./.uv/bin/uv run --python .venv/bin/python --no-sync python - <<'PY'
from modules.sangoi.DataRecorder import DataRecorder  # noqa: F401
from modules.sangoi.TrainGPS import TrainGPS  # noqa: F401
from modules.module.LoRAModule import build_lora_key_manifest, export_lora_key_manifest  # noqa: F401
from modules.util.enum.TimestepDistribution import TimestepDistribution
if not hasattr(TimestepDistribution, "PRIORITY_SAMPLING"):
    raise SystemExit("missing Sangoi PRIORITY_SAMPLING distribution")
PY

./.uv/bin/uv run --python .venv/bin/python --no-sync python scripts/validate_selected_upstream_pr_ports.py \
    --check resume-actions \
    --check timed-actions \
    --check accumulator \
    --check accumulator-mid-step \
    --check tensorboard \
    --check patience \
    --check component-stop \
    --check validation-timesteps \
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

conflict_roots=(
    modules
    scripts
    docs
    README.md
    LAUNCH-SCRIPTS.md
    AGENTS.md
    training_presets
    requirements.txt
    requirements-default.txt
    requirements-cuda.txt
    requirements-rocm.txt
    requirements-global.txt
    requirements-dev.txt
    pyproject.toml
    .python-version
    install.sh
    lib.include.sh
    run-cmd.sh
    start-ui.sh
    update.sh
    .dockerignore
)
if rg -n '^(<<<<<<<|=======|>>>>>>>)' "${conflict_roots[@]}"; then
    exit 1
fi

blocked_residue_pattern='cfg_distillation|distillation|DistillationConfig|DistillationCacheMode|DistillationLossType|DistillationTargetMode|ParentModelWrapper|SmartDiskCache|LoadSmartCache|SaveSmartCache|smart-disk-cache|ipex_to_cuda|split_attention|native_split|flash_split|TheRock|Windows ROCm|LinearA8|activation_quantization|delta_transfer|validation_timesteps|FLUX_2_EDIT|caption_dropout|resolution_quantization|compatibility shim|dual-read|dual-write|fallback adapter'
if rg -n --glob '!scripts/validate_selected_upstream_pr_ports.py' "$blocked_residue_pattern" modules scripts docs README.md LAUNCH-SCRIPTS.md AGENTS.md training_presets requirements*.txt; then
    exit 1
fi

if [ -f .sangoi/CHANGELOG.md ] && rg -n "$blocked_residue_pattern" .sangoi/CHANGELOG.md; then
    exit 1
fi

echo "VALIDATION_STATUS=0"
```

## Final Code Review Bundle Checklist

Before final handoff, assemble a `Senior Code Reviewer` bundle with:

- Plan path and final plan status.
- Final diff anchor: base commit, head commit or uncommitted diff state, and exact `git diff --stat`.
- Per-lane implementation ledger: for each lane, record touched files, upstream PR refs used, intentional deviations from upstream code, and local smokes run before moving to the next lane.
- Cumulative diff stat and touched-file inventory by lane.
- Hunk-level shared-file classification for `TrainConfig.py`, `TimedActionMixin.py`, `TrainProgress.py`, `GenericTrainer.py`, `TrainingTab.py`, `ModelType.py`, `LoRAModule.py`, Flux2 owners, and cache/data loader owners.
- Hunk-level shared-file classification for every file touched by more than one implementation lane, including but not limited to `TrainConfig.py`, `TimedActionMixin.py`, `TrainProgress.py`, `GenericTrainer.py`, `TrainingTab.py`, `ModelType.py`, `LoRAModule.py`, Flux2 owners, and cache/data loader owners.
- PR status lock table copied or referenced with landed/blocked/deferred result.
- Config/UI schema freeze result, including exact field defaults.
- Acceptance-criteria mapping showing each implemented PR's invariant, owner file(s), and validation evidence.
- Invariant proof matrix for Sangoi TrainGPS/DataRecorder, `PRIORITY_SAMPLING`, Sangoi loss fields, cloud source guards, Docker current-source guards, LoRA manifest export, repo-local uv bootstrap, and CPU-only WSL assumptions.
- Explicit evidence that blocked PR symbols/imports were not landed.
- Dependency/platform manifest guard result proving `requirements.txt`, `requirements*.txt`, `.python-version`, and `pyproject.toml` did not drift unless explicitly approved.
- Validation artifact path containing command and `VALIDATION_STATUS=0`.
- CPU-only WSL / no hardware validation caveat.
- Task log and changelog paths.
- Known-risk section for any source-only support where hardware/rendered UI/full training was not validated.
- Known unvalidated surfaces: Docker build, cloud remote, CUDA/GPU/XPU/ROCm hardware, full training benchmarks, rendered Tk UI inspection unless actually performed.

Do not hand off until the reviewer returns `READY`, `READY_WITH_NITS`, or `APPROVE_WITH_FIXES` with all required local non-substantive fixes applied. If the reviewer returns `NOT_READY`, fix the blocker and rerun the gate.

## Acceptance Criteria

- The original shortlist remains represented and is not displaced by broader model-family analysis.
- Every selected PR has an explicit `IMPLEMENT`, `IMPLEMENT_PARTIAL`, `BLOCKED`, or `DEFERRED` status.
- Only `IMPLEMENT` and approved partial surfaces land runtime code.
- Blocked/deferred PRs leave no runtime imports, fields, docs, examples, or dead code.
- No out-of-scope PRs are partially imported.
- Current Sangoi SDXL behavior, TrainGPS/DataRecorder, repo-local uv bootstrap, Docker/cloud source guards, and LoRA manifest export remain intact.
- Validation artifact includes commands and final `VALIDATION_STATUS=0` marker.
- Worktree is clean after commit if the user asks to commit/push.
