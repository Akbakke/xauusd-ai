# XAUUSD Direction Repair Handover - 2026-07-14

## Continuation Goal

Continue the XAUUSD-only direction repair until the live/replay/training stack proves that the model learns to abstain or go long in bull/rising-support regimes, rather than selecting confident SHORT. Do not use non-XAU project artifacts. Do not promote any XAU bundle live until fresh XAU datasets, parity, live-like replay, calibration, and direction-pocket audits all pass.

## Current State

- Repo: `/home/andre2/src/GX1_ENGINE`
- Data root: `/home/andre2/GX1_DATA`
- Disk: `/dev/sdd` has about `838G` free after the 2026-07-15 cleanup round.
- Runtime: no `python`/`python3` training/eval jobs were running after the latest 2026-07-15 public-FLAT-from-hierarchy smoke failed closed and its failed-run artifacts were deleted.
- Non-XAU project artifacts: removed from the working machine except for fail-closed XAU isolation guards.
- Worktree: verify clean with `git status --short` before clean-git gates; latest transformer-entry source repair commit is `0e89eaa1 Require XAU hierarchy trade accuracy edge`.
- Canonical Python: `/home/andre2/venvs/gx1/bin/python`, pytest `9.0.2`, `lightgbm 4.6.0`.

## 2026-07-15 22:46 CEST Source Update - Hierarchy Ctx Prior Adapter

- Latest executed smoke before this source update was `SMART_SEQ520_XAU_SMOKE_HIERSIDEACCEDGE_E6_20260715`.
  - It failed closed on `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`; no bundle directory was produced.
  - Best epoch was `3`: `dir_acc=0.353516`, public pred LONG `0.470703`, SHORT `0.221354`, FLAT `0.307943`, `13` slice failures, `12` accuracy failures, `1` pred-rate failure.
  - Hierarchy side edge was not zero (`side_acc=0.552684`), and hierarchy trade probability was near target (`trade_prob=0.670141` vs target `0.654948`), but public hard direction still failed active-slice gates.
  - Conclusion: more epochs or more scalar pressure on the same hierarchy-side recipe is not justified.
- Implemented the next source-level formulation repair, not a fallback:
  - Added optional `hierarchical_ctx_prior_adapter` to `EntryV10CtxHybridTransformer`.
  - The adapter uses `ctx_cat` embeddings to add a learned bias to `trade_logit` and `side_logits` before hierarchical public direction composition.
  - Defaults remain OFF for legacy strict-load compatibility.
  - Smart XAU repair now requires `ENTRY_HIER_CTX_PRIOR_ADAPTER=1` and `ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE` in `[0.25, 1.00]`; wrappers/readiness/manifest/sweep/rebuild default to scale `0.50`.
  - Bundle loading fails closed if adapter weights exist but adapter scale metadata is missing.
  - This is model-native learning/inference, not a live rule and not an advisory fallback.
- Updated trainer preflight, metadata, failure evidence, bundle audit, candidate/replay readiness, smoke/trainability readiness, smart enablement, sweep lint, smoke/candidate wrappers, rebuild defaults, and tests for the new adapter contract.
- Validation completed before any heavy training:
  - `env PYTHONPATH=. pytest -q tests/test_entry_v10_train_defaults.py tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_v10_6yr_rebuild_direction_repair_contract.py tests/test_xau_direction_repair_sweep.py tests/test_entry_smart_seq520_smoke_train_enablement.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_entry_candidate_readiness.py tests/test_entry_replay_readiness.py` passed.
  - `git diff --check` passed.
  - `env PYTHONPATH=. python3 -m py_compile ...` on all changed Python modules passed.
- Committed the source repair as `3be7c69e Require XAU hierarchy ctx prior adapter`.
- Clean-git readiness after the commit:
  - `scripts/entry_next_edge_control.sh smart-smoke-readiness --quiet` passed.
  - `scripts/entry_next_edge_control.sh smart-trainability-readiness` returned exit `0` with `blockers=[]`; `candidate_training_allowed=false` remains expected until a passing candidate bundle exists.
- Ran one bounded smoke:
  - Vedtak: `SMART_SEQ520_XAU_SMOKE_HIERCTXPRIOR_E6_20260715`.
  - Pre-train manifest: `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T204719Z.json`.
  - Intended bundle: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T204719Z`.
  - Recipe log confirmed `hier_ctx_prior_adapter=1` and `hier_ctx_prior_adapter_scale=0.500`.
  - Epoch `1`: balance guard OK, `dir_acc=0.376953`, public pred LONG `0.280599`, SHORT `0.382161`, FLAT `0.337240`, `10` slice failures (`8` accuracy, `2` pred-rate), hierarchy `trade_pred=0.933594`, `trade_prob=0.667095`, side-edge acc `0.570577`. This was the best observed point, but still failed the hard slice contract.
  - Epoch `2`: hard-red regression, balance guard failed, public SHORT `0.614583`, `29` slice failures.
  - Epoch `3`: recovered balance guard but stayed hard-red and FLAT-heavy, `dir_acc=0.380859`, public pred LONG `0.142578`, SHORT `0.252604`, FLAT `0.604818`, `13` slice failures.
  - Training was manually stopped during epoch `4` because the run was not converging toward the slice contract and continuing would waste compute. No bundle was produced.
  - The aborted pre-train manifest was deleted after extracting the evidence; no bundle dir and no matching memmap dir existed.
- Resource state at this update:
  - `/home/andre2/GX1_DATA`: about `838G` free.
  - RAM: about `36GiB` available after stop, swap `0B` used.
  - No active transformer train/eval `python3` job was left running.
- Next action:
  1. Do not rerun `HIERCTXPRIOR_E6` unchanged. It improved epoch-1 slice count versus the latest side-edge smoke, but still failed hard and then oscillated.
  2. Candidate/replay/IQL/shadow/live remain closed until a fresh XAU transformer bundle passes hard slice and class-balance gates.
  3. The next repair should be another formulation/input change, not a scalar-only retune or more epochs on this recipe.
  4. Likely next angle: make hierarchy trade/flat hard predictions learn the public FLAT coverage directly per active ctx slice, or change label/input staging so ctx-slice prior calibration is learned before side competition dominates public argmax.

## 2026-07-16 Source Update - Hierarchy Trade Accuracy Edge

- After the `HIERCTXPRIOR_E6` smoke, the failure signature was not "more IQL needed" and not "train longer":
  - public slice gates still failed hard;
  - hierarchy `trade_prob` was near target, but hard `trade_pred` was around `0.93`;
  - therefore the next repair targets the hierarchy trade/flat hard-threshold surface directly.
- Implemented and committed `0e89eaa1 Require XAU hierarchy trade accuracy edge`.
  - Added `ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT` and `ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN`.
  - The trainer now penalizes active ctx slices where differentiable hard trade/flat correctness does not beat slice majority plus margin.
  - Strict smart XAU repair requires weight `>=4.00` and margin `>=0.02`; missing or weak config fails preflight/audit/readiness.
  - The recipe is passed through smoke/candidate wrappers, rebuild defaults, smart readiness, smoke manifest, enablement, sweep lint, candidate/replay readiness, bundle audit, metadata, failure evidence, and tests.
  - This is not fallback. It is a hard model-training objective and a fail-closed contract.
- Validation before clean-git readiness:
  - `git diff --check` passed.
  - `py_compile` passed for changed Python modules.
  - `bash -n` passed for changed shell wrappers.
  - Targeted pytest passed for trainer defaults, smoke/candidate wrappers, rebuild contract, smart enablement/readiness, sweep, bundle audit, candidate readiness, replay readiness, and smoke readiness.
- Clean-git post-commit gates passed:
  - `smart-smoke-readiness --quiet`
  - `smart-trainability-readiness --quiet`
  - `smart-smoke-train-enablement --vedtak SMART_SEQ520_XAU_SMOKE_HIERTRADEACCEDGE_E6_20260716 --epochs 6 --batch-size 64 --quiet`
  - Enablement confirmed the new trade accuracy-edge env and kept `candidate_training_allowed=false`, `iql_allowed=false`, `replay_allowed=false`, `promotion_shadow_live_allowed=false`, and `trainer_started=false`.
- Ran one bounded smoke:
  - Vedtak: `SMART_SEQ520_XAU_SMOKE_HIERTRADEACCEDGE_E6_20260716`.
  - Pre-train manifest was `ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260716T050651Z.json`.
  - Intended bundle was `v10_entry_smart_seq520_smoke_20260716T050651Z`.
  - Result: hard fail on `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`; no bundle directory was written.
  - Best epoch was `4`: `dir_acc=0.366536`, `balance_guard_ok=1`, public pred LONG `0.350260`, SHORT `0.326172`, FLAT `0.323568`, `10` slice failures, `10` accuracy failures, `0` pred-rate failures, `direction_slice_ckpt_score=0.031214`.
  - Best epoch hierarchy evidence: `trade_pred=0.998047`, `trade_prob=0.656814`, `side_acc_edge=0.527833`.
  - Final epoch `6` regressed to `14` slice failures (`12` accuracy, `2` pred-rate), `dir_acc=0.373047`, public pred LONG `0.255208`, SHORT `0.512370`, FLAT `0.232422`, `trade_pred=0.985026`.
  - Interpretation: the repair materially improved global class balance and eliminated pred-rate failures at best epoch, but it did not solve active slice accuracy and it did not keep hierarchy trade hard-pred calibrated. This is progress, not a candidate.
  - Failed-run manifest and failure sidecar were deleted after extracting the evidence; no bundle dir or memmap dir existed.
- Post-smoke resource state:
  - `/home/andre2/GX1_DATA`: about `838G` free.
  - RAM: about `36GiB` available, swap `0B`.
  - No active `python3` training/eval process.
- Candidate/replay/IQL/shadow/live remain closed until a fresh XAU transformer bundle passes hard slice and class-balance gates.

## Always-Active Operating Rules

- No fallback, no advisory pass, no soft continuation. If a required artifact, dataset, feature, dependency, audit, parity check, contract, or gate is missing/stale/invalid, the program must fail closed. Either it works under the declared contract, or it does not.
- Slice and pocket failures may be logged as hard failure evidence only. They must never be converted into fallback/advisory paths that let training, candidate readiness, replay, parity, or launch continue.
- Monitor disk before/during heavy rebuild, train, sweep, replay, and prediction materialization. Current cleanup threshold: when available space on `/home/andre2` or `/home/andre2/GX1_DATA` approaches or drops below 700 GB, run an explicit cleanup round for obsolete failed/superseded runs, tmp dirs, and stale reports before launching more heavy jobs.
- Cleanup must preserve ACTIVE contract artifacts and evidence still needed for the current failing gate. Delete only artifacts that are clearly obsolete, failed, superseded, or reproducible.

## 2026-07-15 State Update

- Cleaned obsolete/superseded runs immediately when available space was near `800G`; `/home/andre2` and `/home/andre2/GX1_DATA` now show about `838G` available.
- Fresh XAU direction-repair dataset exists:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/v10_dataset_6yr_smartctx_xau_direction_repair`
  with train/val/test parquets and manifests for stem `v10_6yr_dataset__HOLD_03B`.
- Fresh XAU smoke dataset exists:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/v10_dataset_6yr_smartctx_xau_direction_repair_smoke`
  with train/val/test parquets and manifests for stem `v10_smart_seq520_smoke__HOLD_03B`.
- Latest XAU pretrain audit is `PASS`:
  `/home/andre2/GX1_DATA/reports/xau_direction_repair_pretrain_audit_20260713_v1/XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_latest.json`.
- Latest smart smoke readiness is `READY_FOR_SMART_SEQ520_SMOKE_MANIFEST_REVIEW`; latest smart trainability readiness is `READY_FOR_SMART_SEQ520_TRAINABILITY_REVIEW`.
- 2026-07-15 13:58 Oslo status check: no `python3` train/eval processes were running, GPU utilization was `0%`, `/home/andre2` and `/home/andre2/GX1_DATA` had about `838G` free, RAM had about `37G` available, and swap use was `0B`.
- 2026-07-15 17:14 Oslo status after manually stopping the hard-red hierarchical-composition smoke: no host `python3` train/eval processes were running, `/home/andre2/GX1_DATA` still had about `838G` free, RAM had about `36GiB` available, and swap use was `0B`.
- 2026-07-15 17:54 Oslo status after manually stopping the hard-red hierarchy side-slice smoke: no transformer train/eval process was left running, GPU utilization was `0%`, `/home/andre2/GX1_DATA` still had about `838G` free, RAM had about `36GiB` available, and swap use was `0B`.
- Latest trainability readiness still has `candidate_training_allowed=false`, `iql_allowed=false`, `replay_allowed=false`, `shadow_live_promotion_allowed=false`, and `execution_allowed_now=false`. Entry-IQL is therefore closed. Do not run IQL until a fresh XAU transformer candidate bundle first passes the hard direction slice contract and the required candidate/replay gates.
- Broad XAU/replay/readiness test suite passed under canonical env after `lightgbm` validation and the no-fallback slice-balanced CE hardening.
- Smart XAU repair train recipe now requires `ENTRY_DIRECTION_SLICE_BALANCED_CE_*`; smoke/candidate wrappers, manifest/readiness contracts, trainer metadata, and bundle audit fail closed if this recipe is missing or too weak.
- Smart XAU smoke training with slice-balanced CE ran fail-closed and refused to write a bundle because the best checkpoint still failed active direction slice accuracy (`[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`).
- Added a stricter smart XAU slice true-class margin repair: `ENTRY_DIRECTION_SLICE_TRUE_MARGIN_*` now contributes to train/val loss and is required by trainer preflight, wrappers, readiness contracts, manifests, sweep lint, metadata, and bundle audit.
- Follow-up smoke training evidence after true-margin repair remained hard-red:
  - `SMART_SEQ520_XAU_SMOKE_TRAIN_SLICEMARGIN_20260715`: fail-closed on `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`, no bundle written.
  - `SMART_SEQ520_XAU_SMOKE_TRAIN_SLICEMARGIN_W8_CE4_20260715`: stronger slice true-margin/balanced-CE weights destabilized and still failed slice guard, no bundle written.
  - `SMART_SEQ520_XAU_SMOKE_TRAIN_SLICEMARGIN_BS256_20260715`: larger batch reduced neither the hard slice failure enough nor wrote a bundle; best checkpoint was epoch 12 with `slice_contract_ok=0` and 21 slice failures.
- Added a hard smart XAU slice-balanced sampler contract: `ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER=1` with min rows 8 is now required by smart repair preflight, wrapper recipes, readiness contracts, manifest contracts, sweep lint, trainer metadata, and smoke bundle audit. This is not fallback; if active train slices cannot be built, training fails before epoch 1.
- Follow-up smoke training with the slice-balanced sampler also failed closed:
  - `SMART_SEQ520_XAU_SMOKE_TRAIN_SLICEBAL_SAMPLER_20260715`: manifest `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T100649Z.json`, intended bundle `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T100649Z`; failed on `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`, no bundle directory was written. Best observed slice state was epoch 2 with balance guard OK, `slice_contract_ok=0`, 6 slice failures, 0 pred-rate failures; later epochs remained hard-red.
- Added the next hard smart XAU repair: slice-loss aggregation is now `mean_max` for smart smoke/candidate wrappers, manifest/readiness contracts, sweep lint, and trainer preflight. This makes the worst active slice part of the optimized objective instead of allowing the mean slice loss to hide remaining hard-red slices. This is not fallback; smart XAU training now fails preflight unless `ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION=mean_max`.
- Follow-up smoke training with `mean_max` also failed closed:
  - `SMART_SEQ520_XAU_SMOKE_TRAIN_MEANMAX_20260715`: manifest `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T102342Z.json`, intended bundle `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T102342Z`; failed on `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`, no bundle directory was written. Best checkpoint was epoch 9 with balance guard OK, `slice_contract_ok=0`, 10 slice failures, 7 accuracy failures, 3 pred-rate failures. Later epochs still failed hard.
- Added the next hard smart XAU objective repair: `ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE` is now `0.05` in smart smoke/candidate wrappers, manifest/readiness contracts, and sweep lint, and trainer preflight rejects smart XAU repair runs above `0.05`. This makes min-pred-rate losses track the hard argmax gate more closely; it is not fallback.
- Follow-up smoke training with `mean_max` plus `0.05` argmax-aligned pred-rate temperature also failed closed:
  - `SMART_SEQ520_XAU_SMOKE_TRAIN_ARGMAXTEMP005_20260715`: manifest `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T103750Z.json`, intended bundle `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T103750Z`; failed on `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`, no bundle directory was written. Best checkpoint was epoch 9 with balance guard OK, `slice_contract_ok=0`, 8 slice failures, 6 accuracy failures, and 2 pred-rate failures. Later epochs collapsed back toward hard SHORT and still failed closed.
- Implemented the next hard smart XAU objective repair: `ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_*` now adds a direct active-slice accuracy-edge loss against the same majority-plus-margin condition that blocks bundle creation. Smart XAU smoke/candidate wrappers, manifest/readiness contracts, sweep lint, trainer preflight, metadata, and smoke bundle audit now require the recipe (`weight >= 4.0`, `margin >= 0.02`, `min_label_rate >= 0.10`, `min_rows >= 8`). This is not fallback; weak or missing config fails before a smart XAU repair run can be used.
- Added hard slice failure diagnostics for the next run: `_direction_slice_balance_stats` now returns `direction_slice_failure_details`, and training logs `ENTRY_DIR_SLICE_FAILURE` rows with ctx slice, rows, accuracy, majority, label rates, prediction rates, required rates, and pred-rate shortfall. These rows are failure evidence only.
- Committed the slice accuracy-edge repair as `19ae5d9c Require XAU slice accuracy-edge repair`, then reran smart smoke readiness and smart trainability readiness on clean git. Both passed:
  - smoke readiness: `READY_FOR_SMART_SEQ520_SMOKE_MANIFEST_REVIEW`.
  - trainability readiness: `READY_FOR_SMART_SEQ520_TRAINABILITY_REVIEW`.
- Follow-up smoke training with slice accuracy-edge was manually stopped once it was clearly still hard-red:
  - `SMART_SEQ520_XAU_SMOKE_TRAIN_SLICEACCEDGE_20260715`: manifest `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T110209Z.json`, intended bundle `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T110209Z`; stopped at epoch 10 by operator with no bundle directory written. Best observed checkpoint was epoch 4 with global balance guard OK but `slice_contract_ok=0` and 12 slice failures; epoch 10 still had `slice_contract_ok=0` and 15 slice failures. This remains hard failure evidence, not a candidate.
- Added the next hard execution guard after the stopped run: `ENTRY_DIRECTION_SLICE_HARD_RED_STOP_*` now stops smart XAU transformer training when the best checkpoint and current validation remain slice-red after no slice-score progress. Smart smoke/candidate wrappers, manifest/readiness contracts, sweep lint, trainer preflight, metadata, and smoke bundle audit now require `ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE=3` and `ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS=6`. This is not fallback; it saves compute and still lets `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]` refuse bundle creation.
- Post-run resource state stayed safe: `/home/andre2` and `/home/andre2/GX1_DATA` had about `838G` free, RAM had about `38G` available after the process exited, and swap stayed at `0B` used throughout monitoring.
- Targeted validation passed after the true-margin repair:
  `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_entry_smart_seq520_smoke_manifest.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_xau_direction_repair_sweep.py -q`.
  After the slice-balanced sampler contract this targeted suite passed again with `124 passed`; after the `mean_max` hardening the same targeted suite passed again; after the `0.05` temperature hardening the same targeted suite passed again; after the `slice_accuracy_edge` repair the same targeted suite passed again; after the hard-red stop guard the same targeted suite passed again.
- After the hard-red stop guard, smart smoke readiness and smart trainability readiness were rerun sequentially to avoid stale `latest` report races. Both passed with no training, replay, IQL, shadow, live, or promotion side effects.
- The broad XAU/replay/readiness pytest surface was rerun under `/home/andre2/venvs/gx1` after the hard-red stop guard. It caught two non-training regressions and they were fixed fail-closed:
  - `materialize_entry_iql_student_trade_log_v1.py` now passes an explicit IQL-student score contract (`edge_score`, `edge_score`, `iql_student_validation_top_fraction`) through all shared replay-policy calls, and fails if the score column is missing.
  - IQL replay comparison current-artifact tests now accept the stricter replay/distillation evidence-identity failure names as valid red evidence; this does not open IQL or promotion.
  - The IQL student trade-log fixture now supplies continuous bid/ask open/high/low/close tape rows, so missing fill/exit prices remain hard failures rather than implicit close-price fallback.
- Validation after those fixes:
  - `python3 -m py_compile gx1/scripts/materialize_entry_iql_student_trade_log_v1.py`
  - `scripts/pytest_repo.sh tests/test_entry_iql_replay_comparison.py tests/test_entry_iql_student_trade_log.py -q` -> `17 passed`
  - Broad XAU/replay/readiness suite covering smart520 state/rank, XAU pretrain/labels, smoke/trainability readiness, wrappers, candidate replay, IQL replay contracts, live gate, parity, and sweep passed.
- Since the stopped `SLICEACCEDGE` run did not persist stdout logs, the trainer now writes a hard failure-evidence sidecar before raising `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`: `<intended_bundle_dir>__direction_slice_failure_evidence.json`. This sidecar is written next to the intended bundle directory, does not create a candidate bundle, sets `bundle_written=false` and `promotion_shadow_live_allowed=false`, and contains best/current direction slice stats, failure details, recipe knobs, train/val hashes, git commit, and hard-red-stop state. Validation:
  - `python3 -m py_compile gx1/models/entry_v10/entry_v10_ctx_train_v3.py`
  - `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py -q` -> `42 passed`
  - `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_entry_smart_seq520_smoke_manifest.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_xau_direction_repair_sweep.py -q` passed.
  - Broad XAU/replay/readiness suite passed again after this trainer change.
- Follow-up smoke training after failure-evidence sidecar hardening was manually stopped at epoch 6 because the transformer was clearly hard-red again and not worth burning more compute:
  - `SMART_SEQ520_XAU_SMOKE_TRAIN_FAILUREEVID_20260715`: manifest `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T113601Z.json`, intended bundle `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T113601Z`; stopped with `KeyboardInterrupt` during epoch 7 train after epoch 6 validation showed global balance `guard_ok=0`, `slice_contract_ok=0`, slice score `-1.937594`, 34 slice failures, 10 accuracy failures, and 24 pred-rate failures. No bundle directory and no failure sidecar were written because the process was intentionally interrupted before the trainer's final fail-closed sidecar write path.
- Read-only smoke-data diagnostic after the hard-red stop showed the same failure shape outside the transformer: a simple LightGBM sanity baseline over XAU smoke `seq/snap/ctx_cont/ctx_cat` reached val/test accuracy above global majority, but still failed the active ctx slice contract mostly by under-predicting FLAT inside slices. That means the next repair is not "train longer"; it must directly optimize per-slice prediction priors.
- Added the next hard smart XAU objective repair: `ENTRY_DIRECTION_SLICE_PRIOR_MATCH_*` now penalizes active ctx slices whose differentiable pred-rate distribution drifts away from the slice label distribution beyond tolerance. Smart smoke/candidate wrappers, manifest/readiness contracts, sweep lint, trainer preflight, metadata, failure evidence, and smoke bundle audit now require the recipe (`weight >= 3.0`, `tolerance <= 0.02`, `min_label_rate >= 0.10`, `min_rows >= 8`). This is not fallback; missing or weak prior-match config fails before a smart XAU repair run can be used.
- Validation after the prior-match repair:
  - `python -m py_compile gx1/models/entry_v10/entry_v10_ctx_train_v3.py gx1/scripts/verify_entry_smart_seq520_smoke_readiness_v1.py gx1/scripts/verify_entry_smart_seq520_trainability_readiness_v1.py gx1/scripts/materialize_entry_smart_seq520_smoke_manifest_v1.py gx1/scripts/audit_entry_foundation_smoke_bundle_v1.py gx1/scripts/sweep_entry_smart_seq520_direction_repair_v1.py`
  - `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_entry_smart_seq520_smoke_manifest.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_xau_direction_repair_sweep.py -q` passed.
- 2026-07-15 14:02 CEST post-prior-match verification:
  - `smart-smoke-readiness --quiet` and `smart-trainability-readiness --quiet` passed on clean git.
  - Latest readiness remains report-only: `training_allowed=false`, `execution_allowed_now=false`, `candidate_training_allowed=false`, `iql_allowed=false`, `replay_allowed=false`, `shadow_live_promotion_allowed=false`.
  - Broad XAU/replay/readiness pytest surface passed again after the prior-match repair, covering smart520 state/rank, XAU labels/pretrain, smoke/trainability readiness, wrappers, candidate replay, IQL replay contracts, live gate/parity, and sweep.
  - `scripts/entry_next_edge_control.sh smart-smoke-train --vedtak SMART_SEQ520_XAU_SMOKE_PRIORMATCH_DRYRUN_20260715 --require-edge-audit --dry-run` printed a bounded smart-smoke package only; it did not start a trainer or write a run manifest. The package included 22G RAM cap, 2G swap cap, `num_workers=0`, `ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT=3.00`, tolerance `0.02`, `ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE=3`, min epochs `6`, and post-smoke strict edge audit.
  - No actual transformer training, candidate training, replay, IQL, shadow, or live action was started in this verification pass.
- Added and committed `smart-smoke-train-enablement`, a report-only XAU package gate that requires clean git, an explicit `SMART_SEQ520_XAU_SMOKE_` vedtak, green smart smoke/trainability readiness, capped wrapper dry-run proof, prior-match/hard-red-stop env, and strict edge-audit proof. It starts no trainer, replay, IQL, shadow, live, or promotion paths.
- Materialized enablement package `SMART_SEQ520_XAU_SMOKE_PRIORMATCH_ENABLEMENT_20260715`:
  `/home/andre2/GX1_DATA/reports/entry_smart_seq520_smoke_train_enablement_20260715_v1/ENTRY_SMART_SEQ520_SMOKE_TRAIN_ENABLEMENT_latest.json`.
  Decision was `ENTRY_SMART_SEQ520_SMOKE_TRAIN_ENABLEMENT_READY_FOR_EXPLICIT_EXECUTION`, `smart_smoke_training_allowed_with_this_package=true`, while `training_allowed=false`, `candidate_training_allowed=false`, `replay_allowed=false`, `iql_allowed=false`, and `promotion_shadow_live_allowed=false`.
- Ran one bounded smart smoke train from that package:
  - Vedtak: `SMART_SEQ520_XAU_SMOKE_PRIORMATCH_ENABLEMENT_20260715`
  - Pre-train manifest: `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T121148Z.json`
  - Intended bundle: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T121148Z`
  - Result: hard fail on `[TRAIN_FAIL_DIRECTION_CLASS_BALANCE_GUARD]`, no bundle directory was written. Best/only epoch had `dir_acc=0.345052`, `balance_guard_ok=0`, `slice_contract_ok=0`, `direction_slice_ckpt_score=-1.807765`, 32 slice failures, 15 accuracy failures, and 17 pred-rate failures. This is hard failure evidence, not a candidate.
- After that run, trainer failure evidence was hardened again: class-balance guard failures now write the same no-bundle sidecar path before raising, so future `[TRAIN_FAIL_DIRECTION_CLASS_BALANCE_GUARD]` exits persist best/current direction stats, recipe, data hashes, git commit, and `bundle_written=false` just like slice-guard exits.
- Post-commit validation after class-balance evidence hardening:
  - `smart-smoke-readiness --quiet` and `smart-trainability-readiness --quiet` passed on clean git.
  - `smart-smoke-train-enablement --vedtak SMART_SEQ520_XAU_SMOKE_PRIORMATCH_ENABLEMENT_20260715 --quiet` passed again on clean git; latest enablement report has `smart_smoke_training_allowed_with_this_package=true` and still keeps training/candidate/replay/IQL/live fields false.
  - Broad XAU/replay/readiness pytest surface passed again, including smart520 state/rank, XAU labels/pretrain, smoke/trainability/enablement readiness, wrappers, candidate replay, IQL replay contracts, live gate/parity, and sweep.
- Implemented and committed the next formulation repair as `12f5ae00 Require XAU public flat from hierarchy`:
  - New strict smart-XAU flag: `ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE=1`.
  - When enabled, public `direction_logits` are composed from hierarchy `trade_logit` and `side_logits`, and the direction residual is made common to LONG/SHORT/FLAT so it is softmax-invariant. Public FLAT therefore comes from hierarchy no-trade instead of an independent learned FLAT residual channel.
  - Bundle loader, trainer preflight/metadata, smoke/candidate wrappers, rebuild defaults, smoke/trainability readiness, smoke manifest, enablement, sweep lint, candidate/replay readiness, and smoke bundle audit all fail closed if the strict smart-XAU contract is missing.
  - Focused validation passed: `py_compile` for changed model/trainer/readiness/audit scripts, `bash -n` for changed wrappers, and targeted pytest (`144 passed`).
- Post-commit gates on clean git passed:
  - `scripts/entry_next_edge_control.sh smart-smoke-readiness --quiet`
  - `scripts/entry_next_edge_control.sh smart-trainability-readiness --quiet`
  - `scripts/entry_next_edge_control.sh smart-smoke-train-enablement --vedtak SMART_SEQ520_XAU_SMOKE_PUBLICFLATFROMTRADE_E6_20260715 --epochs 6 --batch-size 64 --quiet`
  - Enablement decision was `ENTRY_SMART_SEQ520_SMOKE_TRAIN_ENABLEMENT_READY_FOR_EXPLICIT_EXECUTION`; `smart_smoke_training_allowed_with_this_package=true`, while `training_allowed=false`, `candidate_training_allowed=false`, `replay_allowed=false`, `iql_allowed=false`, `promotion_shadow_live_allowed=false`, and `trainer_started=false`.
- Ran one bounded smoke after the repair:
  - Vedtak: `SMART_SEQ520_XAU_SMOKE_PUBLICFLATFROMTRADE_E6_20260715`
  - Pre-train manifest was `ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T200725Z.json`.
  - Intended bundle was `v10_entry_smart_seq520_smoke_20260715T200725Z`.
  - Result: fail-closed on `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`; no bundle directory was written.
  - Epoch 1: balance guard OK, public pred LONG `0.468099`, SHORT `0.170573`, FLAT `0.361328`, `21` slice failures, hierarchy `trade_pred=0.930990`, `trade_prob=0.635537`.
  - Epoch 2 was the best checkpoint: balance guard OK, public pred LONG `0.229818`, SHORT `0.504557`, FLAT `0.265625`, `9` slice failures, `dir_acc=0.379557`, hierarchy `trade_pred=0.963542`, `trade_prob=0.662345` vs target `0.654948`.
  - Epoch 3 degraded: public FLAT-heavy (`pred_flat=0.558594`), `18` slice failures, `dir_acc=0.345703`; early stop selected epoch 2 but slice contract stayed red, so trainer refused bundle creation.
  - The failed-run manifest and failure evidence sidecar were deleted after extracting the above status; no stale bundle dir existed.
- Current interpretation:
  - The public-FLAT independent-residual fight is fixed and logged as active (`hier_compose_public_flat_from_trade=1`).
  - The failure moved back to genuine slice-level side/coverage accuracy: best smoke had fewer failures (`9`) than the latest flat-consistency run's epoch-3 `35`, but still no candidate bundle.
  - Do not run Entry-IQL, replay, candidate, shadow, live, or promotion. Entry-IQL remains closed until a fresh XAU transformer bundle passes hard slice/class-balance gates.
  - Next repair should not restore public FLAT residual or add fallback. Focus on hierarchy side/slice accuracy and FLAT coverage staging/calibration, because public composition is now constrained correctly.
- Implemented and committed the next hard hierarchy-side repair as `0f5ed3b7 Require XAU hierarchy side accuracy edge`:
  - New strict smart-XAU knobs: `ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT` and `ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN`.
  - The trainer now adds a hierarchy conditional side slice accuracy-edge loss against majority-plus-margin per active ctx slice, using the same slice set as the existing hierarchy side supervision. This is a hard optimization target, not fallback.
  - Smoke/candidate wrappers, rebuild defaults, smart readiness, smoke manifest, enablement package, sweep lint, candidate/replay readiness, bundle audit, trainer metadata, and tests all require/pass through the recipe (`weight >= 4.00`, `margin >= 0.02` for strict smart XAU).
  - Validation passed: `py_compile` on changed Python files, `bash -n` on changed wrappers, `git diff --check`, and focused pytest (`222 passed`, 2 torch warnings).
  - Clean-git gates passed after commit: `smart-smoke-readiness --quiet`, `smart-trainability-readiness --quiet`, and `smart-smoke-train-enablement --vedtak SMART_SEQ520_XAU_SMOKE_HIERSIDEACCEDGE_E6_20260715 --epochs 6 --batch-size 64 --quiet`.
- Follow-up bounded smoke after hierarchy-side accuracy-edge:
  - Vedtak: `SMART_SEQ520_XAU_SMOKE_HIERSIDEACCEDGE_E6_20260715`
  - Pre-train manifest was `ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T202455Z.json`.
  - Intended bundle was `v10_entry_smart_seq520_smoke_20260715T202455Z`.
  - Result: hard fail on `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`; no bundle directory was written.
  - Hard-red stop triggered at epoch 6 with best epoch 3 and no slice-contract pass.
  - Best epoch 3: balance guard OK, `dir_acc=0.353516`, public pred LONG `0.470703`, SHORT `0.221354`, FLAT `0.307943`, `13` slice failures, `12` accuracy failures, `1` pred-rate failure, hierarchy `trade_pred=0.985026`, `trade_prob=0.670141` vs target `0.654948`, hierarchy side edge acc `0.552684`.
  - Last epoch 6: balance guard OK, public pred LONG `0.164714`, SHORT `0.623047`, FLAT `0.212240`, `16` slice failures, `13` accuracy failures, `3` pred-rate failures, hierarchy `trade_pred=0.994792`, `trade_prob=0.679593`, hierarchy side edge acc `0.545726`.
  - The failed-run manifest and failure evidence sidecar were deleted after extracting the status above; no stale bundle dir existed.
- Updated interpretation after the hierarchy-side smoke:
  - This commit made the hierarchy side repair explicit and fail-closed, but it did not improve the final public slice gate versus the previous public-FLAT-from-trade best (`9` slice failures). Best observed in this run was `13` slice failures.
  - More epochs are not the answer; hard-red stop did its job. More scalar loss stacking on the same inputs is now weak evidence unless it changes the optimization surface materially.
  - Next direction should be a formulation/input change: staged hierarchy training, better slice-conditioned calibration, or additional slice-relevant input/label structure for FLAT/trade coverage. Do not open IQL/replay/candidate/live.
- Latest resource state after cleanup: `/home/andre2/GX1_DATA` about `838G` free, RAM about `37GiB` available, swap `0B`, and no Python training/eval process running.

## What Was Done

### XAU Label/Model Learning Surface

- Added/extended XAU-only structural utility repair and hierarchical target surface:
  - `gx1/scripts/repair_entry_xau_structural_utility_labels_v1.py`
  - `gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py`
  - `gx1/models/entry_v10/entry_v10_ctx_train_v3.py`
- Dataset now carries/validates side-specific utility, bad-path, expected MAE, first-N MFE/MAE, trade/flat and conditional side labels.
- Direction repair focuses on learned abstention/side confidence, not runtime hand-rules.
- New/updated tests cover hierarchical loss, structural repair, XAU label contract, and train defaults.

### Smart520 Train/Serve Contract Hardening

- Added rank-reference materializer:
  - `gx1/scripts/materialize_smart520_rank_reference_v1.py`
- Dataset manifests now include `smart520_state_contract` with:
  - `frame_anchor_utc`
  - `model_range_start_utc`
  - `rank_reference_end_utc`
  - `rank_reference_npz`
  - `rank_reference_npz_sha256`
  - sidecar/source metadata
- `v12_smart520_state_live.py` no longer silently falls back to legacy July smart520 state contract. Direct helpers require an explicit `Smart520StateContract`; legacy is only explicit historical selftest mode.
- Live state contract now validates rank-reference SHA, sidecar SHA, stale markers, and timestamp ordering. It rejects `frame_anchor_utc > rank_reference_end_utc`.
- Bucket recompute now pins only rows inside the rank-reference window; tail bars digitize against frozen distributions.

### Launch/Parity/Promotion Gates

- `v12_smart_entry_live.py` launch gate now requires:
  - expected-utility mode
  - XAU-only dataset/prediction paths
  - no stale markers including `julyext`
  - matching bundle/parity `smart520_state_contract`
  - rank-reference SHA/sidecar verification
  - non-empty direction pocket metrics, not just pocket names
  - per-pocket selected rows, bad-side rate, and positive proxy pnl
- Empty `{}` pocket audit entries now fail.
- Direction pocket audit remains a promotion gate, not a live trading rule.

### Rebuild/Sweep/Readiness Guards

- `scripts/v10_6yr_rebuild_20260626.sh` now:
  - materializes fresh smart520 rank reference
  - passes `--smart520-rank-reference-npz`
  - checks train/val/test split manifests, not just one train file
  - checks exact rank-ref path and SHA
  - keeps XAU direction-repair dataset gate fail-closed
- Dataset build proof now includes `xgb_bridge_source` and `tape_root` after tape-root resolution, fixing an internal proof/gate mismatch before long rebuilds.
- `scripts/run_entry_foundation_seq146_candidate_train.sh` now blocks `julyext` as stale for smart XAU repair.
- XAU sweep script exists:
  - `gx1/scripts/sweep_entry_smart_seq520_direction_repair_v1.py`
  - XAU-only, Latin-hypercube/bounded dry-run, strict repair knobs, no runtime rules.
  - It is serial today; Optuna/parallel jobs are still future work after gates are green.

### Replay/Live Forensics Implemented or Started

- Replay/live fill parity improvements already present:
  - `gx1/scripts/replay_entry_tabular_no_xgb_policy_v1.py`
  - `gx1/scripts/materialize_entry_candidate_replay_trade_log_v1.py`
  - `gx1/execution/v12_paper_runner.py`
- Live journal/debug fields were extended for expected utility and side evidence in prior worktree changes.
- Subagents found additional live/replay mismatches that are not fully patched yet. See "Still To Do".

### Test/Env Work

- `scripts/pytest_repo.sh` now points to one canonical shared env:
  - `/home/andre2/venvs/gx1/bin/python`
- This was done to avoid installing pytest/python packages repeatedly in multiple repo-local envs.
- Focused tests passed under canonical env:

```bash
scripts/pytest_repo.sh \
  tests/test_smart520_state_contract.py \
  tests/test_smart520_rank_reference.py \
  tests/test_v12_smart_entry_live_gate.py \
  tests/test_xau_direction_repair_pretrain_audit.py \
  tests/test_v10_6yr_rebuild_direction_repair_contract.py \
  tests/test_entry_v10_train_defaults.py -q
```

Result at handover: `58 passed`.

Broad XAU/replay/readiness suite passed under canonical env on 2026-07-15.

### 2026-07-15 Global Prior-Match Repair

- Latest bounded smart smoke train after the prior-match enablement package failed hard before writing a bundle:
  - failure: `[TRAIN_FAIL_DIRECTION_CLASS_BALANCE_GUARD]`
  - key symptom: global FLAT prediction collapse (`pred_flat=0.005859` vs `label_flat=0.345052`)
  - LONG/SHORT were over-predicted (`pred_long=0.417969`, `pred_short=0.576172`)
  - IQL, replay, shadow, and live remain closed.
- Implemented a new fail-closed transformer objective/contract:
  - `ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT`
  - `ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE`
  - `ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE`
- Smart XAU smoke/candidate auto-recipes now set global prior-match to `8.00 / 0.02 / 0.10`.
- Trainer metadata, class-balance/slice failure evidence, bundle audit, smoke manifest, smoke/trainability readiness, enablement package, and sweep lint now require/report the global prior-match contract.
- Validation completed after the change:
  - `py_compile` for trainer/readiness/audit/sweep scripts: passed.
  - Focused pytest for trainer/wrappers/readiness/enablement/bundle-audit/sweep: passed.
  - Broad XAU/replay/IQL contract suite: passed with expected skips.
- The first bounded transformer smoke after this repair is recorded below. Before any further smoke train, clean git and regenerate the readiness/enablement proof.

### 2026-07-15 Bounded Global-Prior Smoke Result

- Commit tested: `c3dc51e8 Require XAU global prior match`.
- Readiness/enablement:
  - `smart-smoke-readiness --quiet`: passed.
  - `smart-trainability-readiness --quiet`: passed.
  - enablement vedtak: `SMART_SEQ520_XAU_SMOKE_GLOBAL_PRIOR_ENABLEMENT_20260715`, passed without starting trainer.
- Bounded smoke train launched with the same vedtak, `--require-edge-audit`, 1 epoch, `MemoryMax=22G`, `MemorySwapMax=2G`, `num_workers=0`.
- Result: hard fail closed, no bundle directory written.
  - failure: `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`
  - evidence: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T122916Z__direction_slice_failure_evidence.json`
  - intended bundle dir absent: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T122916Z`
  - `best_epoch=1`, `last_epoch=1`, `best_dir_acc=0.338542`
  - `best_direction_balance_guard_ok=True`
  - `best_direction_slice_contract_ok=False`
  - `direction_slice_failure_count=29`
  - `direction_slice_accuracy_failure_count=15`
  - `direction_slice_pred_rate_failure_count=14`
  - `direction_slice_pred_rate_shortfall=0.392463`
  - `direction_slice_ckpt_score=-0.997118`
- Interpretation:
  - Global prior-match repaired the immediate global FLAT-collapse failure enough for class-balance guard to pass.
  - The blocker moved back to per-slice direction behavior: too many audited context slices still fail majority accuracy and/or active-class pred-rate coverage.
  - Do not continue by extending epochs on the same recipe. Next repair should target slice-level objective/data composition, using the recorded `ENTRY_DIR_SLICE_FAILURE` rows.

### 2026-07-15 XAU Source Preflight Repair And Current Direction

- 2026-07-15 14:46 CEST status:
  - No `python3` transformer/IQL training was running.
  - GPU was idle (`0%`, 307 MiB used).
  - `/home/andre2` and `/home/andre2/GX1_DATA` still had about `838G` free.
  - RAM stayed safe during checks and swap stayed at `0B` used.
- Fixed smart rebuild preflight after cleanup removed the old legacy foundation source:
  - `materialize_entry_smart_seq520_rebuild_preflight_v1.py` now defaults the source dataset to the XAU smart direction-repair dataset:
    `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/v10_dataset_6yr_smartctx_xau_direction_repair`
  - Missing/deleted split manifests now become explicit failed checks in a blocked preflight report, not a traceback and not a fallback.
  - The live preflight report is green on the XAU smart520 source with `source_dataset_mode=xau_smart_seq520_built_source`, `source_active_seq_snap_width=520`, `training_allowed=false`, and no failures:
    `/home/andre2/GX1_DATA/reports/entry_smart_seq_rebuild_preflight_20260630_v1/ENTRY_SMART_REBUILD_PREFLIGHT_latest.json`
- Guardrail policy was updated so `smart_smoke_train_enablement` is part of the exact readiness command contract and remains blocked without explicit vedtak/clean git. It starts no trainer, replay, IQL, shadow, or live path.
- Read-only feature metadata check on the XAU smart520 train manifest showed the dataset already contains many relevant XAU structure inputs:
  - wick/rejection, SMC sweep/BOS/CHOCH, support/resistance proximity, SR memory, MTF confluence, H1/H4/D1 regime, rising-support rail pressure, falling-resistance rail pressure, and trap-pressure features.
  - Therefore the next serious repair is not random new data or IQL. It is a targeted slice/input/target diagnostic: prove whether the red slices ignore existing rail/SR/wick/regime fields, whether labels are noisy/non-separable in those slices, or whether anti-short/anti-long hierarchy/utility caps must be made explicit.
- Validation after the preflight/guardrail change:
  - `py_compile` passed for `materialize_entry_smart_seq520_rebuild_preflight_v1.py` and `verify_entry_foundation_guardrails_v1.py`.
  - `scripts/pytest_repo.sh tests/test_entry_smart_rebuild_preflight.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_dataset_post_rebuild_readiness.py tests/test_entry_foundation_guardrails.py -q` passed (`26 passed`).

### 2026-07-15 Red-Slice Separability Audit

- 2026-07-15 14:52 CEST status:
  - Broad XAU/replay/readiness suite passed on the current commit with expected skips:
    `scripts/pytest_repo.sh tests/test_smart520_state_contract.py tests/test_smart520_rank_reference.py tests/test_v12_smart_entry_live_gate.py tests/test_xau_direction_repair_pretrain_audit.py tests/test_v10_6yr_rebuild_direction_repair_contract.py tests/test_repair_entry_xau_structural_utility_labels.py tests/test_entry_v10_train_defaults.py tests/test_entry_foundation_smoke_dataset.py tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_entry_smart_seq520_smoke_manifest.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_entry_smart_seq520_smoke_train_enablement.py tests/test_entry_smart_dataset_post_rebuild_readiness.py tests/test_entry_candidate_readiness.py tests/test_entry_candidate_replay_trade_log.py tests/test_entry_candidate_replay_evidence.py tests/test_entry_candidate_selective_edge.py tests/test_entry_replay_readiness.py tests/test_entry_iql_replay_comparison.py tests/test_entry_iql_student_trade_log.py tests/test_entry_iql_replay_evidence.py tests/test_entry_iql_replay_slice_audit.py tests/test_entry_iql_distillation_contract.py tests/test_entry_iql_distill_wrapper.py tests/test_build_entry_iql_v1.py tests/test_iql_adapter_emitter_parity.py tests/test_circuit_breaker_parity.py tests/test_xau_direction_repair_sweep.py tests/test_entry_smart_rebuild_preflight.py tests/test_entry_foundation_guardrails.py -q`
  - No training/IQL process was running; disk stayed about `838G` free and swap stayed `0B` used.
- Added read-only red-slice separability audit:
  - `gx1/scripts/audit_xau_red_slice_separability_v1.py`
  - Test: `tests/test_xau_red_slice_separability_audit.py`
  - The audit reads fail-closed slice evidence and matching XAU `val_data`, rejects non-XAU `val_data`, excludes the first seven XGB anchor snap fields, and measures LONG-vs-SHORT separability in existing XAU domain features.
  - It is report-only and keeps `training_allowed=false`, `candidate_training_allowed=false`, `replay_allowed=false`, `iql_allowed=false`, and `shadow_live_promotion_allowed=false`.
- Live audit command:
  - `/home/andre2/venvs/gx1/bin/python -m gx1.scripts.audit_xau_red_slice_separability_v1 --quiet --no-fail-on-audit-fail`
  - Report: `/home/andre2/GX1_DATA/reports/xau_red_slice_separability_audit_20260715_v1/XAU_RED_SLICE_SEPARABILITY_AUDIT_latest.json`
  - Decision: `XAU_RED_SLICE_SEPARABILITY_AUDIT_COMPLETE`
  - Evidence source: latest `TRAIN_FAIL_DIRECTION_SLICE_GUARD` sidecar from `v10_entry_smart_seq520_smoke_20260715T122916Z`.
  - XAU val data: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/v10_dataset_6yr_smartctx_xau_direction_repair_smoke/v10_smart_seq520_smoke__HOLD_03B_val.parquet`
  - Existing domain feature count: `247`; missing required rail features: `0`.
  - Red slice detail count audited: `15`; weak required-rail-feature slice count: `1` (`6.7%`).
  - Several red slices have clear utility/feature evidence despite the failed transformer prediction-rate behavior:
    - `vol_regime_id=2` / `atr_bucket=2`: `LONG` label rate `0.463`, `SHORT` label rate `0.171`, mean long-minus-short utility `+37.53 bps`, required rail feature separation `0.597` std.
    - `vol_regime_id=3` / `atr_bucket=3`: `LONG` label rate `0.369`, `SHORT` label rate `0.234`, mean utility delta `+23.31 bps`, required rail separation `0.256` std.
    - `session_id=1`: `FLAT` majority `0.435`, but mean long-minus-short utility is still `+6.09 bps`; the failed transformer was over-SHORT in this slice.
  - Interpretation: the next repair should not start with random new data or IQL. Existing XAU rail/SR/wick/regime features are present and usually separable in the red slices. The likely next code repair is transformer objective/cap hardening that forces the direction/hierarchy heads to respect these per-slice utility/rail signals, while separately inspecting the one weak required-rail slice (`session_id=2`).

### 2026-07-15 Direction Utility-Margin Repair

- Implemented the next transformer objective repair from the red-slice separability audit:
  - `ENTRY_DIRECTION_UTILITY_MARGIN_*` now adds a direct direction-head utility-margin loss against row-level `y_long_path_utility_bps - y_short_path_utility_bps`.
  - If LONG utility is ahead by at least `15 bps`, the loss penalizes SHORT logits above LONG-or-FLAT. If SHORT utility is ahead by at least `15 bps`, it penalizes LONG logits above SHORT-or-FLAT.
  - This still allows FLAT/abstain and is not a live hand-rule or fallback.
- Smart XAU repair preflight now rejects missing/weak utility-margin settings:
  - `ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT >= 4.00`
  - `ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS <= 15.0`
  - `ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN >= 0.10`
- Smart smoke/candidate wrappers, smoke/readiness/manifest contracts, train enablement, sweep lint, bundle audit, candidate readiness, replay readiness, and the direct `v10_6yr_rebuild_20260626.sh` XAU train path now carry the same utility-margin contract.
- Validation before commit:
  - `python3 -m py_compile` passed for the trainer and all touched Python gates/scripts.
  - `bash -n` passed for `run_entry_foundation_seq146_smoke_train.sh`, `run_entry_foundation_seq146_candidate_train.sh`, and `v10_6yr_rebuild_20260626.sh`.
  - Focused pytest passed:
    `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_entry_smart_seq520_smoke_manifest.py tests/test_entry_smart_seq520_smoke_train_enablement.py tests/test_xau_direction_repair_sweep.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_entry_candidate_readiness.py tests/test_entry_replay_readiness.py tests/test_v10_6yr_rebuild_direction_repair_contract.py -q`
- Clean-git readiness after commit `9b054f74 Add XAU direction utility-margin repair`:
  - `smart-smoke-readiness --quiet` passed.
  - `smart-trainability-readiness --quiet` passed.
  - `smart-smoke-train-enablement --vedtak SMART_SEQ520_XAU_SMOKE_UTILITYMARGIN_E8_20260715 --epochs 8 --batch-size 64 --quiet` passed with `trainer_started=false`, `iql_allowed=false`, and `ENTRY_DIRECTION_UTILITY_MARGIN_*` present in the capped dry-run command.
- First bounded utility-margin smoke attempt:
  - Vedtak: `SMART_SEQ520_XAU_SMOKE_UTILITYMARGIN_E8_20260715`
  - Pre-train manifest: `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T130609Z.json`
  - Intended bundle: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T130609Z`
  - Result: failed before a real epoch result because `validate()` was missing the new `total_direction_utility_margin` accumulator initialization. No bundle directory and no failure sidecar were written. This is a code bug, not model evidence.
- Immediate fix after that failed attempt:
  - `validate()` now initializes `total_direction_utility_margin = 0.0`.
  - `tests/test_entry_v10_train_defaults.py` now has a static guard for this accumulator.
  - Validation: `python3 -m py_compile gx1/models/entry_v10/entry_v10_ctx_train_v3.py` and `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py -q` passed.
  - Candidate training, replay, IQL, shadow, live, and promotion remain closed.
- Second bounded utility-margin smoke attempt after commit `ecfeb72d Fix XAU utility-margin validation accumulator`:
  - Vedtak: `SMART_SEQ520_XAU_SMOKE_UTILITYMARGIN_FIX_E8_20260715`
  - Pre-train manifest: `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T131023Z.json`
  - Intended bundle: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T131023Z`
  - Result: hard fail on `[TRAIN_FAIL_DIRECTION_CLASS_BALANCE_GUARD]`; no bundle directory was written.
  - Failure evidence sidecar was written:
    `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T131023Z__direction_slice_failure_evidence.json`
  - Sidecar confirms `bundle_written=false`, `promotion_shadow_live_allowed=false`, `decision=FAIL_DIRECTION_CLASS_BALANCE_GUARD`, `git_commit=ecfeb72d22936ff22ad60ab934855b55df674aa0`.
  - Best checkpoint: epoch `6`, `dir_acc=0.393229`, `best_dir_ckpt_score=-1.592945`, `best_direction_balance_guard_ok=false`, `best_direction_slice_contract_ok=false`, `24` slice failures (`6` accuracy failures, `18` pred-rate failures over `17` audited slices).
  - Last epoch: epoch `8`, still red with `31` slice failures (`14` accuracy failures, `17` pred-rate failures).
  - Important observation: the model repeatedly drove `direction_pred_rate_flat` to `0.000000` on validation. Utility-margin improved some side accuracy but did not solve abstention/FLAT collapse. Next repair should directly prevent FLAT starvation in the direction head while preserving utility side pressure; do not move to IQL.

### 2026-07-15 Direction FLAT-Starvation Repair

- Implemented the next transformer objective repair after the utility-margin smoke failure:
  - `ENTRY_DIRECTION_FLAT_STARVATION_*` now penalizes global and active-slice FLAT prediction starvation when FLAT labels are materially present.
  - The loss also adds a direct FLAT logit-margin term on FLAT-labeled rows, so a model that drives `direction_pred_rate_flat` to zero pays a hard training/validation loss.
  - This is not fallback, not a live hand-rule, and not an IQL path. It is a transformer training objective that must pass the existing hard class-balance and direction-slice gates.
- Smart XAU repair preflight now rejects missing/weak FLAT-starvation settings:
  - `ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT >= 8.00`
  - `ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE >= 0.10`
  - `ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS >= 8`
  - `ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION >= 0.50`
  - `ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR >= 0.10`
  - `ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN >= 0.10`
- Smart smoke/candidate wrappers, smoke/readiness/manifest contracts, train enablement, sweep lint, bundle audit, candidate readiness, replay readiness, and the direct `v10_6yr_rebuild_20260626.sh` XAU train path now carry the same FLAT-starvation contract.
- Validation before commit:
  - `python3 -m py_compile` passed for the trainer and all touched Python gates/scripts.
  - `bash -n` passed for `run_entry_foundation_seq146_smoke_train.sh`, `run_entry_foundation_seq146_candidate_train.sh`, and `v10_6yr_rebuild_20260626.sh`.
  - Focused pytest passed:
    `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_entry_smart_seq520_smoke_manifest.py tests/test_entry_smart_seq520_smoke_train_enablement.py tests/test_xau_direction_repair_sweep.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_entry_candidate_readiness.py tests/test_entry_replay_readiness.py tests/test_v10_6yr_rebuild_direction_repair_contract.py -q`
- 2026-07-15 15:33 Oslo resource check before commit:
  - `/home/andre2` and `/home/andre2/GX1_DATA` had about `838G` free.
  - RAM had about `37GiB` available, swap use was `0B`, no `python3` training/eval jobs were running, and GPU use was idle/low (`2%`, 307 MiB used).
  - Entry-IQL, replay, candidate training, shadow, live, and promotion remain closed.
- Clean-git gates after commit `c5a03cc4 Add XAU flat-starvation repair`:
  - `smart-smoke-readiness --quiet`: passed.
  - `smart-trainability-readiness --quiet`: passed when rerun sequentially. A parallel smoke/trainability readiness attempt raced on the `latest` smoke-readiness artifact and returned exit `2`; the sequential rerun had `blockers=[]`.
  - `smart-smoke-train-enablement --vedtak SMART_SEQ520_XAU_SMOKE_FLATSTARVE_E8_20260715 --epochs 8 --batch-size 64 --quiet`: passed with `trainer_started=false`, `iql_allowed=false`, `replay_allowed=false`, and `promotion_shadow_live_allowed=false`.
- Bounded FLAT-starvation smoke result:
  - Vedtak: `SMART_SEQ520_XAU_SMOKE_FLATSTARVE_E8_20260715`
  - Pre-train manifest: `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T133510Z.json`
  - Intended bundle: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T133510Z`
  - Result: hard-red-stopped at epoch `6`, then failed closed on `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`; no bundle directory was written.
  - Failure evidence sidecar:
    `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T133510Z__direction_slice_failure_evidence.json`
  - Sidecar confirms `decision=FAIL_DIRECTION_SLICE_GUARD`, `failure_code=TRAIN_FAIL_DIRECTION_SLICE_GUARD`, `bundle_written=false`, `promotion_shadow_live_allowed=false`, `hard_red_stopped=true`, `git_commit=c5a03cc4f81faef2fa9bee66f2d81106e08867a2`.
  - Best checkpoint: epoch `2`, `best_dir_acc=0.347005`, `best_dir_ckpt_score=-0.433291`, `best_direction_balance_guard_ok=true`, `best_direction_slice_contract_ok=false`.
  - Best slice stats: `18` slice failures over `17` audited slices, `15` accuracy failures, `3` pred-rate failures. Prediction rates were LONG `0.197917`, SHORT `0.516927`, FLAT `0.285156` versus label rates LONG `0.322917`, SHORT `0.332031`, FLAT `0.345052`.
  - Last epoch `6`: `26` slice failures, `8` accuracy failures, `18` pred-rate failures; FLAT collapsed again to `direction_pred_rate_flat=0.002604` while SHORT rose to `0.788411`.
  - Interpretation: FLAT-starvation repair changed the failure mode and briefly restored global balance at epoch 2, but did not solve the hard per-slice direction contract. Training longer on this recipe is not justified.

### 2026-07-15 Direction Side-Utility-Conviction Repair

- Follow-up report-only red-slice analysis on the FLAT-starvation sidecar showed no obvious global label/utility contradiction:
  - side-label rows did not show opposite clear utility in the audited red slices (`LONG label + clear SHORT utility = 0`, `SHORT label + clear LONG utility = 0` in the weighted slice check).
  - Some FLAT-labeled rows have clear side utility, but the existing utility-margin objective deliberately allowed FLAT/abstain and therefore did not force row-level side assignment.
  - The blocker is row-level side/FLAT discrimination inside active context slices, not IQL and not more epochs on the same recipe.
- Implemented a new hard transformer objective/contract:
  - `ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT`
  - `ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS`
  - `ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN`
- Semantics:
  - Only applies when `y_direction` is LONG or SHORT and `y_long_path_utility_bps - y_short_path_utility_bps` supports that same side by the configured gap.
  - For clear LONG rows, the LONG logit must beat both SHORT and FLAT by the margin.
  - For clear SHORT rows, the SHORT logit must beat both LONG and FLAT by the margin.
  - FLAT-labeled rows and unclear utility-gap rows are not forced. This is a learned transformer loss, not a live hand-rule and not fallback.
- Smart XAU repair preflight now rejects missing/weak side-utility-conviction settings:
  - `ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT >= 6.00`
  - `ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS <= 15.0`
  - `ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN >= 0.10`
- Smart smoke/candidate wrappers, smoke/readiness/manifest contracts, train enablement, sweep lint, bundle audit, candidate readiness, replay readiness, and the direct `v10_6yr_rebuild_20260626.sh` XAU train path now carry the same contract.
- Validation before commit:
  - `python3 -m py_compile` passed for the trainer and all touched Python gates/scripts.
  - `bash -n` passed for `run_entry_foundation_seq146_smoke_train.sh`, `run_entry_foundation_seq146_candidate_train.sh`, and `v10_6yr_rebuild_20260626.sh`.
  - Focused pytest passed:
    `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_entry_smart_seq520_smoke_manifest.py tests/test_entry_smart_seq520_smoke_train_enablement.py tests/test_xau_direction_repair_sweep.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_entry_candidate_readiness.py tests/test_entry_replay_readiness.py tests/test_v10_6yr_rebuild_direction_repair_contract.py tests/test_xau_red_slice_separability_audit.py -q`
- 2026-07-15 15:57 Oslo resource check during implementation:
  - `/home/andre2` and `/home/andre2/GX1_DATA` had about `838G` free.
  - RAM had about `35GiB` available, swap use was `0B`, and no `python3` training/eval jobs were running.
  - No transformer training, candidate training, replay, IQL, shadow, live, or promotion path was started by this repair.
- Clean-git gates after commit `800cb7cb Add XAU side utility conviction repair`:
  - `smart-smoke-readiness --quiet`: passed.
  - `smart-trainability-readiness --quiet`: passed.
  - `smart-smoke-train-enablement --vedtak SMART_SEQ520_XAU_SMOKE_SIDEUTIL_E8_20260715 --epochs 8 --batch-size 64 --quiet`: passed with no trainer start and no replay/IQL/live/promotion side effects.
- Bounded side-utility-conviction smoke result:
  - Vedtak: `SMART_SEQ520_XAU_SMOKE_SIDEUTIL_E8_20260715`
  - Pre-train manifest: `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T135909Z.json`
  - Intended bundle: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T135909Z`
  - Result: hard-red-stopped at epoch `6`, then failed closed on `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`; no bundle directory was written.
  - Failure evidence sidecar:
    `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T135909Z__direction_slice_failure_evidence.json`
  - Sidecar confirms `decision=FAIL_DIRECTION_SLICE_GUARD`, `failure_code=TRAIN_FAIL_DIRECTION_SLICE_GUARD`, `bundle_written=false`, `promotion_shadow_live_allowed=false`, `hard_red_stopped=true`, `git_commit=800cb7cb7e498e21dde57078fe269df44eea0121`.
  - Best checkpoint: epoch `2`, `best_dir_acc=0.367839`, `best_dir_ckpt_score=-0.484030`, `best_direction_balance_guard_ok=true`, `best_direction_slice_contract_ok=false`.
  - Best slice stats: `21` slice failures, `12` accuracy failures, `9` pred-rate failures. Prediction rates were LONG `0.147786`, SHORT `0.441406`, FLAT `0.410807` versus label rates LONG `0.322917`, SHORT `0.332031`, FLAT `0.345052`.
  - Last epoch `6`: `27` slice failures, `8` accuracy failures, `19` pred-rate failures; SHORT rose to `0.818359` and FLAT fell to `0.005208`.
  - Interpretation: side-utility-conviction did not solve the hard slice direction contract. It briefly restored FLAT coverage at the best checkpoint but still under-predicted LONG and failed too many active context slices. Extending epochs on this same recipe is not justified.
- Post-run evidence traceability fix:
  - The sideutil smoke logs confirm `side_utility_conviction` was active, but that pre-fix failure sidecar did not include the side-utility/FLAT-starvation/utility-margin fields inside its `train_recipe` block.
  - The trainer failure-evidence blocks now record the active utility-margin, side-utility-conviction, and FLAT-starvation recipe fields for both class-balance and slice-guard failures.
  - Validation: `python3 -m py_compile gx1/models/entry_v10/entry_v10_ctx_train_v3.py` passed, and `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py -q` passed (`49 passed`).

### 2026-07-15 Direction Utility-Trade-Conviction Repair

- Ran the report-only XAU red-slice separability audit against the latest sideutil failure sidecar:
  `/home/andre2/GX1_DATA/reports/xau_red_slice_separability_audit_20260715_v1/XAU_RED_SLICE_SEPARABILITY_AUDIT_latest.json`
  - Decision: `XAU_RED_SLICE_SEPARABILITY_AUDIT_COMPLETE`.
  - Evidence source: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T135909Z__direction_slice_failure_evidence.json`.
  - `domain_feature_count=247`, missing required XAU rail direction features `0`, weak required-rail-feature slice rate `1/15`.
  - Key blocker shape: best checkpoint under-predicted LONG in positive-utility slices (`vol_regime_id=2` / `atr_bucket=2` had LONG label rate `0.463`, predicted LONG `0.057`, long-minus-short utility `+37.53 bps`). Later epochs collapsed back to SHORT dominance.
- Added the next hard transformer objective:
  - `ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT`
  - `ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS`
  - `ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS`
  - `ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH`
  - `ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN`
- Semantics:
  - Applies independent of the hard `y_direction` class, but only when side utility is tradable enough: side utility must be at least the configured utility floor, side bad-path must be under the configured cap, and side utility must beat the opposite side by the configured gap.
  - For clear tradable LONG rows, LONG logit must beat both SHORT and FLAT.
  - For clear tradable SHORT rows, SHORT logit must beat both LONG and FLAT.
  - Rows with only relative utility but no tradable side edge are ignored. This is a training objective, not a live hand-rule and not fallback.
- Smart XAU repair preflight now rejects missing/weak utility-trade-conviction settings:
  - `ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT >= 8.00`
  - `ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS <= 15.0`
  - `ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS <= 0.0`
  - `ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH <= 0.50`
  - `ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN >= 0.10`
- Smart smoke/candidate wrappers, smoke/readiness/manifest contracts, train enablement, sweep lint, bundle audit, candidate readiness, replay readiness, failure evidence, metadata, and direct `v10_6yr_rebuild_20260626.sh` XAU train path now carry the same contract.
- Validation before commit:
  - `python3 -m py_compile` passed for the trainer, touched gate scripts, and touched tests.
  - `bash -n` passed for `run_entry_foundation_seq146_smoke_train.sh`, `run_entry_foundation_seq146_candidate_train.sh`, and `v10_6yr_rebuild_20260626.sh`.
  - Focused pytest passed:
    `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_entry_smart_seq520_smoke_manifest.py tests/test_entry_smart_seq520_smoke_train_enablement.py tests/test_xau_direction_repair_sweep.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_entry_candidate_readiness.py tests/test_entry_replay_readiness.py tests/test_v10_6yr_rebuild_direction_repair_contract.py -q`
- No transformer training, candidate training, replay, IQL, shadow, live, or promotion path was started by this repair. Next heavy action still requires clean git, sequential readiness, enablement proof, and resource check.

## 2026-07-15 Utility-Trade Smoke Failure And Utility-Triad-CE Repair

- Ran one bounded smart XAU transformer smoke train after the utility-trade-conviction repair:
  `scripts/entry_next_edge_control.sh smart-smoke-train --vedtak SMART_SEQ520_XAU_SMOKE_UTILITYTRADE_E8_20260715 --require-edge-audit --epochs 8 --early-stop-patience 8`
- Result: fail-closed on epoch `6` with `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`; hard-red-stop refused to burn epochs `7-8`.
  - Evidence sidecar: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T142650Z__direction_slice_failure_evidence.json`
  - `decision=FAIL_DIRECTION_SLICE_GUARD`, `failure_code=TRAIN_FAIL_DIRECTION_SLICE_GUARD`, `bundle_written=false`, `hard_red_stopped=true`.
  - Intended bundle dir was not created: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T142650Z`
  - Best checkpoint was still epoch `2`: `best_dir_acc=0.364583`, global balance OK, but `best_direction_slice_contract_ok=false`, `17` slice failures.
  - Last epoch `6` remained SHORT/FLAT-starved: pred LONG `0.204427`, pred SHORT `0.789063`, pred FLAT `0.006510`, `24` slice failures.
- Read-only parquet diagnostic after the failure showed the input labels are not the immediate blocker:
  - train utility masks at gap `15 bps`, utility `>=0`, bad-path `<=0.5`: LONG `1469`, SHORT `1437`, NONE `1189`.
  - val utility masks: LONG `508`, SHORT `532`, NONE `496`.
  - Mask purity is high: val LONG-mask label LONG `0.935`, val SHORT-mask label SHORT `0.929`, val NONE-mask label FLAT `0.925`.
- Conclusion: do not add random new input or move to IQL yet. The failure is still in transformer direction learning, especially row-level NONE/FLAT retention under utility pressure.
- Added the next hard transformer objective:
  - `ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT`
  - `ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS`
  - `ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS`
  - `ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH`
  - `ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP`
- Semantics:
  - Builds a training-only target from the same XAU utility labels: clear tradable long-edge -> LONG, clear tradable short-edge -> SHORT, all remaining no-edge rows -> FLAT.
  - Uses class-balanced CE inside the batch with a capped class weight.
  - This is not fallback and not a live hand-rule; it is a fail-closed training contract.
- Smart XAU repair preflight now rejects missing/weak triad-CE settings:
  - `ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT >= 8.00`
  - `ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS <= 15.0`
  - `ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS <= 0.0`
  - `ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH <= 0.50`
  - `ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP >= 2.0`
- Updated surfaces: trainer env/preflight/loss logging/failure evidence/metadata, smoke and candidate wrappers, rebuild script, smoke readiness, trainability readiness, smoke manifest, train enablement, sweep lint, smoke bundle audit, candidate readiness, replay readiness, and focused tests.
- Validation after triad-CE implementation:
  - `python3 -m py_compile` passed for the trainer and touched gate scripts.
  - `bash -n` passed for `run_entry_foundation_seq146_smoke_train.sh`, `run_entry_foundation_seq146_candidate_train.sh`, and `v10_6yr_rebuild_20260626.sh`.
  - Focused pytest passed:
    `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_entry_smart_seq520_smoke_manifest.py tests/test_entry_smart_seq520_smoke_train_enablement.py tests/test_xau_direction_repair_sweep.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_entry_candidate_readiness.py tests/test_entry_replay_readiness.py tests/test_v10_6yr_rebuild_direction_repair_contract.py -q`
- No candidate training, replay, IQL, shadow, live, or promotion path was started by the implementation commit.

## 2026-07-15 Utility-Triad-CE Smoke Failure And Strategy Pivot

- Commit tested: `e0cf39cd Add XAU utility triad CE repair`.
- Clean-git readiness and enablement were rerun before training:
  - `smart-smoke-readiness --quiet`: passed.
  - `smart-trainability-readiness --quiet`: passed.
  - `smart-smoke-train-enablement --vedtak SMART_SEQ520_XAU_SMOKE_TRIADCE_E8_20260715 --epochs 8 --batch-size 64 --quiet`: passed without starting trainer and kept candidate/replay/IQL/live flags false.
- Ran one bounded smart XAU transformer smoke train:
  `scripts/entry_next_edge_control.sh smart-smoke-train --vedtak SMART_SEQ520_XAU_SMOKE_TRIADCE_E8_20260715 --require-edge-audit --epochs 8 --early-stop-patience 8`
- Result: fail-closed on epoch `6` with `[TRAIN_FAIL_DIRECTION_CLASS_BALANCE_GUARD]`; hard-red-stop refused to burn epochs `7-8`.
  - Evidence sidecar: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T144900Z__direction_slice_failure_evidence.json`
  - `decision=FAIL_DIRECTION_CLASS_BALANCE_GUARD`, `failure_code=TRAIN_FAIL_DIRECTION_CLASS_BALANCE_GUARD`, `bundle_written=false`, `hard_red_stopped=true`.
  - Intended bundle dir was not created: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T144900Z`
  - Best checkpoint was epoch `2`: `best_dir_acc=0.355469`, `best_direction_balance_guard_ok=false`, `best_direction_slice_contract_ok=false`, `direction_slice_ckpt_score=-1.581540`, `28` slice failures, `12` accuracy failures, `16` pred-rate failures.
  - Best checkpoint prediction rates were LONG `0.039063`, SHORT `0.664714`, FLAT `0.296224` versus labels LONG `0.322917`, SHORT `0.332031`, FLAT `0.345052`; it preserved some FLAT but starved LONG in multiple slices.
  - Last epoch `6` was still hard-red: pred LONG `0.116536`, pred SHORT `0.791016`, pred FLAT `0.092448`, `33` slice failures, `15` accuracy failures, `18` pred-rate failures.
- Resource state after exit stayed safe: `/home/andre2/GX1_DATA` had about `838G` free, RAM had about `37GiB` available, and swap use was `0B`.
- Conclusion: do not keep tuning the same single 3-class direction softmax/loss stack. The system has now failed closed across CE, slice true-margin, slice-balanced sampler, mean-max aggregation, argmax-temperature, slice-accuracy edge, prior-match, global-prior match, utility-margin, flat-starvation, side-utility conviction, utility-trade conviction, and utility-triad CE. The next repair should change the learning formulation, not add more scalar pressure.
- Preferred next formulation: split learned direction into two coupled transformer heads:
  - `edge_or_flat`: learned TRADE versus FLAT/abstain from utility/no-edge rows.
  - `long_or_short_given_edge`: learned LONG versus SHORT only on clear tradable edge rows.
  - Compose the public 3-class direction logits from these heads and keep the existing hard class-balance/slice gates. This is a model/training contract, not fallback and not a live hand-rule.
- Do not start candidate training, replay, IQL, shadow, live, or promotion until that new transformer bundle passes hard direction slice and class-balance gates.

## 2026-07-15 Hierarchical Direction Composition Smoke Stop

- Implemented the formulation pivot as commit `9bf89003 Add XAU hierarchical direction composition`.
- New smart XAU contract:
  - `ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION=1` is required by smart repair preflight, smoke/candidate wrappers, readiness/manifest contracts, enablement, sweep lint, bundle audit, candidate readiness, replay readiness, and the direct XAU train path.
  - The model composes public `direction_logits` from hierarchy heads:
    `P(LONG)=P(TRADE)*P(LONG|TRADE)`, `P(SHORT)=P(TRADE)*P(SHORT|TRADE)`, `P(FLAT)=P(FLAT)`.
  - This is a learned model/training contract, not fallback and not a live hand-rule.
- Validation before training:
  - `python3 -m py_compile` passed for touched model/trainer/gate scripts.
  - `bash -n` passed for touched smoke/candidate/rebuild shell scripts.
  - Focused pytest passed:
    `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_entry_smart_seq520_smoke_manifest.py tests/test_entry_smart_seq520_smoke_train_enablement.py tests/test_xau_direction_repair_sweep.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_entry_candidate_readiness.py tests/test_entry_replay_readiness.py tests/test_v10_6yr_rebuild_direction_repair_contract.py -q`
  - `smart-smoke-readiness --quiet`, `smart-trainability-readiness --quiet`, and `smart-smoke-train-enablement --vedtak SMART_SEQ520_XAU_SMOKE_HIERCOMPOSE_E8_20260715 --epochs 8 --batch-size 64 --quiet` passed with no candidate/replay/IQL/live side effects.
- Ran one bounded smart XAU transformer smoke train:
  `scripts/entry_next_edge_control.sh smart-smoke-train --vedtak SMART_SEQ520_XAU_SMOKE_HIERCOMPOSE_E8_20260715 --require-edge-audit --epochs 8 --early-stop-patience 8`
- Pre-train manifest:
  `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T150722Z.json`
- Intended bundle:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T150722Z`
- Result: manually stopped at epoch `3` because it was clearly hard-red and not worth burning more compute. No bundle directory was created, and no failure sidecar was written because the operator interrupt happened before the trainer final fail-closed write path.
  - Epoch `1`: `dir_acc=0.367188`, `guard_ok=0`, `slice_contract_ok=0`, `28` slice failures, `pred_flat=0.007812`.
  - Epoch `2`: `dir_acc=0.369141`, `guard_ok=0`, `slice_contract_ok=0`, `31` slice failures, `pred_short=0.846354`, `pred_flat=0.041016`.
  - Epoch `3`: `dir_acc=0.341797`, `guard_ok=0`, `slice_contract_ok=0`, `31` slice failures, `14` accuracy failures, `17` pred-rate failures, `direction_slice_ckpt_score=-1.765918`, pred LONG `0.524089`, pred SHORT `0.458333`, pred FLAT `0.017578`.
- Interpretation: the hierarchical public-logit composition is correct as an exported contract, but this smoke still drove the trade/flat side into FLAT starvation and failed active context slices. Do not start candidate training, replay, IQL, shadow, live, or promotion from this. The next step must diagnose why the learned `TRADE` versus `FLAT/no-edge` head is collapsing inside red slices, not add random data or continue the same recipe for more epochs.
- Added post-stop hierarchy diagnostics for the next bounded smoke:
  - validation now records global `hier_trade_*`, `hier_flat_*`, and `hier_side_*` output stats when hierarchy heads are present.
  - `direction_slice_failure_details` now include the same hierarchy diagnostics per red context slice.
  - live logs now include `[ENTRY_HIER_OUTPUT]` and extended `[ENTRY_DIR_SLICE_FAILURE]` fields.
  - This is diagnostic only. It opens no gate, changes no loss, and is not fallback.
- Validation after diagnostic change:
  - `python3 -m py_compile gx1/models/entry_v10/entry_v10_ctx_train_v3.py` passed.
  - `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py -q` passed.
  - Focused gate/wrapper/readiness/audit suite passed:
    `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_entry_smart_seq520_smoke_manifest.py tests/test_entry_smart_seq520_smoke_train_enablement.py tests/test_xau_direction_repair_sweep.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_entry_candidate_readiness.py tests/test_entry_replay_readiness.py tests/test_v10_6yr_rebuild_direction_repair_contract.py -q`
- Ran one diagnostic-only bounded smoke after the hierarchy diagnostics:
  `scripts/entry_next_edge_control.sh smart-smoke-train --vedtak SMART_SEQ520_XAU_SMOKE_HIERDIAG_E6_20260715 --require-edge-audit --epochs 6 --early-stop-patience 6`
  - Pre-train manifest: `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T151842Z.json`
  - Intended bundle: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T151842Z`
  - Manually stopped after epoch `1` because the new diagnostics already proved the failure mode and further epochs would just burn compute. No bundle directory and no failure sidecar were written.
  - Epoch `1`: `dir_acc=0.367188`, `guard_ok=0`, `slice_contract_ok=0`, `28` slice failures, pred LONG `0.457031`, pred SHORT `0.535156`, pred FLAT `0.007812`.
  - New hierarchy evidence: global `hier_trade_target=0.654948` but `hier_trade_pred=1.000000`; `hier_trade_prob=0.723131`, `hier_flat_prob=0.276869`, and `hier_trade_prob_label_flat=0.722968`. Red slices showed the same pattern: `hier_trade_pred=1.000000` and `hier_trade_prob_label_flat≈0.70-0.73`.
  - Interpretation: side-head is only moderate (`side_acc_edge≈0.56`), but the immediate FLAT starvation comes from the `TRADE`/`FLAT` head. With train trade-rate `0.658120`, unweighted BCE has a constant optimum above the `0.5` threshold, so a weakly separated model predicts TRADE on every row.
- Implemented the next hard transformer repair:
  - `hier_trade_pos_weight` now uses bounded inverse-frequency with below-one weights allowed for the hierarchy `TRADE` target. For the current smoke dataset this changes the hierarchy trade BCE from forced `pos_weight=1.0` to approximately `neg/pos=0.519`.
  - Other existing positive-class heads keep their previous floor/cap behavior.
  - This is not fallback and not a live rule; it fixes the learned trade/no-trade objective so majority-positive trade labels do not make a constant TRADE prediction optimal.
- Validation after the trade-pos-weight repair:
  - `python3 -m py_compile gx1/models/entry_v10/entry_v10_ctx_train_v3.py` passed.
  - `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py -q` passed.
  - Focused gate/wrapper/readiness/audit suite passed:
    `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_entry_smart_seq520_smoke_manifest.py tests/test_entry_smart_seq520_smoke_train_enablement.py tests/test_xau_direction_repair_sweep.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_entry_candidate_readiness.py tests/test_entry_replay_readiness.py tests/test_v10_6yr_rebuild_direction_repair_contract.py -q`
- Ran one bounded smoke after the trade-pos-weight repair:
  `scripts/entry_next_edge_control.sh smart-smoke-train --vedtak SMART_SEQ520_XAU_SMOKE_HIERBAL_E6_20260715 --require-edge-audit --epochs 6 --early-stop-patience 6`
  - Pre-train manifest: `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T152512Z.json`
  - Evidence sidecar: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T152512Z__direction_slice_failure_evidence.json`
  - Intended bundle dir absent: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T152512Z`
  - Result: fail-closed on `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`, no bundle written.
  - The repair was directionally correct: `[ENTRY_HIER_BALANCE_PROOF]` reported `raw_trade_pos_weight=0.519481` and `trade_pos_weight=0.519481`; epoch `6` had global balance guard OK, `best_dir_acc=0.392578`, pred LONG `0.292969`, pred SHORT `0.473307`, pred FLAT `0.233724`, and only `1` pred-rate failure.
  - It still failed active context slice accuracy: `direction_slice_failure_count=9`, `accuracy_failures=8`, `pred_rate_failures=1`, `direction_slice_ckpt_score=-0.027847`.
  - Hierarchy evidence: `hier_trade_pred` stayed `1.000000`, but `hier_trade_prob` dropped toward `0.663552` and public FLAT recovered materially versus the previous collapse. Remaining blocker is mostly slice-level side/accuracy behavior, not global class-balance.
- Fixed evidence hygiene after the run:
  - `_direction_slice_stats_snapshot` now includes global `hier_trade_*`, `hier_flat_*`, and `hier_side_*` fields in future fail-closed sidecars, not only per-slice details.
  - `python3 -m py_compile gx1/models/entry_v10/entry_v10_ctx_train_v3.py` passed.
  - `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py -q` passed.
- Implemented the next hard transformer repair after the remaining red slices proved to be side/accuracy failures rather than a pure global-balance problem:
  - New trainer knobs:
    - `ENTRY_HIER_SLICE_SIDE_CE_WEIGHT`
    - `ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT`
    - `ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN`
    - `ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE`
    - `ENTRY_HIER_SLICE_SIDE_MIN_ROWS`
  - The new losses apply only to hierarchy side-head trade rows and only to active ctx slices with at least two side classes above min label-rate. They use the existing direction slice ctx indices and slice-loss aggregation, so smart repair keeps worst active slices in the optimized objective.
  - Smart XAU repair preflight now fails before epoch 1 unless side-slice CE is at least `4.00`, side-slice true-margin weight at least `3.00`, margin at least `0.10`, min label-rate at least `0.10`, and min rows at least `8`.
  - Smoke/candidate wrappers, direct rebuild env, smart smoke/trainability readiness, smoke manifest, smoke train enablement, bundle audit, sweep lint, metadata, and failure-evidence recipes now carry/report the same contract.
  - Validation after this source repair:
    - `python3 -m py_compile gx1/models/entry_v10/entry_v10_ctx_train_v3.py gx1/scripts/verify_entry_smart_seq520_smoke_readiness_v1.py gx1/scripts/verify_entry_smart_seq520_trainability_readiness_v1.py gx1/scripts/materialize_entry_smart_seq520_smoke_manifest_v1.py gx1/scripts/materialize_entry_smart_seq520_smoke_train_enablement_package_v1.py gx1/scripts/audit_entry_foundation_smoke_bundle_v1.py gx1/scripts/sweep_entry_smart_seq520_direction_repair_v1.py`
    - `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_entry_smart_seq520_smoke_manifest.py tests/test_entry_smart_seq520_smoke_train_enablement.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_xau_direction_repair_sweep.py -q` passed.
  - Commit: `98d590ab Require XAU hierarchy slice-side repair`.
- Readiness and enablement after commit `98d590ab`:
  - `scripts/entry_next_edge_control.sh smart-smoke-readiness --quiet` passed.
  - `scripts/entry_next_edge_control.sh smart-trainability-readiness --quiet` passed when rerun sequentially.
  - `scripts/entry_next_edge_control.sh smart-smoke-train-enablement --vedtak SMART_SEQ520_XAU_SMOKE_HIERSLICE_ENABLEMENT_20260715 --quiet` passed and kept candidate/replay/IQL/live/promotion closed.
- Ran one bounded smart smoke after the hierarchy side-slice repair and stopped it when it was clearly hard-red again:
  `scripts/entry_next_edge_control.sh smart-smoke-train --vedtak SMART_SEQ520_XAU_SMOKE_HIERSLICE_E6_20260715 --require-edge-audit --epochs 6 --early-stop-patience 6`
  - Pre-train manifest: `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T154741Z.json`
  - Intended bundle: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T154741Z`
  - Result: manually stopped with `KeyboardInterrupt` during epoch `5` train after epoch `4` validation made the failure mode clear. No bundle directory and no failure sidecar were written because the process was intentionally interrupted before the trainer's final fail-closed sidecar path.
  - Epoch `1`: balance guard failed, `slice_contract_ok=0`, `27` slice failures, pred LONG `0.464193`, pred SHORT `0.474609`, pred FLAT `0.061198`.
  - Epoch `2`: the new repair did move the model in the right direction temporarily: balance guard OK, `slice_contract_ok=0`, `20` slice failures, pred LONG `0.119141`, pred SHORT `0.505859`, pred FLAT `0.375000`, `hier_side_acc=0.5596`, `hier_slice_side_ce=6.002674`, `hier_slice_side_margin=1.306802`.
  - Epoch `3`: it swung back into a LONG-dominant failure: balance guard failed, `23` slice failures, pred LONG `0.600260`, pred SHORT `0.087891`, pred FLAT `0.311849`.
  - Epoch `4`: confirmed the run was not worth extending: balance guard failed, `31` slice failures, pred LONG `0.426432`, pred SHORT `0.036458`, pred FLAT `0.537109`, `hier_side_acc=0.5089`, `hier_trade_pred=1.000000`, and red slices still showed side/accuracy failures.
  - Important observation: side-slice CE/margin pressure is active and changed class rates, but it did not stabilize the hard direction contract. `ENTRY_RESIDUAL_MAG_PROOF` still showed `delta_abs_mean=0.000000` while hierarchy trade prediction stayed all-trade, so the next step should be diagnosis of the residual/anchor and hierarchy-composition learning surface, not more blind scalar tuning.
  - Entry-IQL, replay, candidate, shadow, live, and promotion remain closed.
- Ran the report-only red-slice separability audit against the latest completed fail-closed sidecar:
  `/home/andre2/venvs/gx1/bin/python -m gx1.scripts.audit_xau_red_slice_separability_v1 --evidence-json /home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T152512Z__direction_slice_failure_evidence.json --quiet --no-fail-on-audit-fail`
  - Report: `/home/andre2/GX1_DATA/reports/xau_red_slice_separability_audit_20260715_v1/XAU_RED_SLICE_SEPARABILITY_AUDIT_latest.json`
  - Decision: `XAU_RED_SLICE_SEPARABILITY_AUDIT_COMPLETE`
  - Domain feature count: `247`; missing required XAU rail features: `0`.
  - Red slices audited: `9`; weak required-rail-feature slice rate: `1/9`.
  - Interpretation still holds after the hierarchy balance repair: do not add random new input and do not move to IQL. Existing XAU rail/SR/wick/regime inputs are present; the next repair is model/objective mechanics.
- Implemented the next narrow source repair after confirming `delta_abs_mean=0.000000` is mechanical:
  - In `EntryV10CtxHybridTransformer.forward`, hierarchical composition now emits public `direction_logits` as:
    `logits=[log P(trade)+log P(long|trade), log P(trade)+log P(short|trade), log P(flat)] + residual_scale*delta_logits`.
  - This keeps the trade/side/flat decomposition as the base, but makes the 3-class residual `head_direction` trainable again through all public direction losses, slice losses, and class-balance gates.
  - The model now exposes `hierarchical_direction_base_logits` and `hierarchical_direction_residual_logits` for audit/debug output.
  - Bundle metadata formula now documents `+ residual_scale*delta_logits` and states that `head_direction` remains trainable through public `direction_logits`.
  - Validation after this source repair:
    - `python3 -m py_compile gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py gx1/models/entry_v10/entry_v10_ctx_train_v3.py gx1/scripts/audit_xau_red_slice_separability_v1.py`
    - `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py tests/test_entry_v10_ctx_model_shapes.py tests/test_xau_red_slice_separability_audit.py -q` passed (`65 passed`).
    - `scripts/pytest_repo.sh tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_entry_smart_seq520_smoke_manifest.py tests/test_entry_smart_seq520_smoke_train_enablement.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_entry_candidate_readiness.py tests/test_entry_replay_readiness.py tests/test_v10_6yr_rebuild_direction_repair_contract.py tests/test_xau_direction_repair_sweep.py -q` passed.
  - Commit: `a702498f Train XAU residual through hierarchy composition`.
- Readiness and enablement after commit `a702498f`:
  - `scripts/entry_next_edge_control.sh smart-smoke-readiness --quiet` passed.
  - `scripts/entry_next_edge_control.sh smart-trainability-readiness --quiet` passed.
  - `scripts/entry_next_edge_control.sh smart-smoke-train-enablement --vedtak SMART_SEQ520_XAU_SMOKE_HIERRESID_ENABLEMENT_20260715 --epochs 6 --batch-size 64 --quiet` passed and kept candidate/replay/IQL/live/promotion closed.
- Ran one bounded smart smoke after the residual-through-composition repair:
  `scripts/entry_next_edge_control.sh smart-smoke-train --vedtak SMART_SEQ520_XAU_SMOKE_HIERRESID_E6_20260715 --require-edge-audit --epochs 6 --early-stop-patience 6`
  - Pre-train manifest: `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T160114Z.json`
  - Evidence sidecar: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T160114Z__direction_slice_failure_evidence.json`
  - Intended bundle dir absent: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T160114Z`
  - Result: fail-closed on `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`, no bundle written.
  - The repair fixed the mechanical dead residual: `delta_abs_mean` became nonzero immediately and grew through training (`val` epoch `1` `0.088336`, epoch `4` `0.438854`, epoch `6` `0.948578`).
  - Best checkpoint was epoch `4`: global balance guard OK, pred LONG `0.240234`, pred SHORT `0.351562`, pred FLAT `0.408203`, `direction_slice_failure_count=11`, `accuracy_failures=9`, `pred_rate_failures=2`, `direction_slice_ckpt_score=-0.031213`.
  - Last epoch `6` over-corrected again: global balance guard failed, pred LONG `0.494141`, pred SHORT `0.485677`, pred FLAT `0.020182`, `27` slice failures, `10` accuracy failures, and `17` pred-rate failures.
  - Hierarchy trade head improved versus all-trade but remains unstable: best epoch `4` had `hier_trade_pred_rate=0.953125` and `hier_trade_prob_label_flat_mean=0.577532`; last epoch `6` had `hier_trade_pred_rate=0.739583`, but public FLAT collapsed because the residual head overrode the base composition.
  - Report-only red-slice separability audit on this new sidecar completed:
    `/home/andre2/GX1_DATA/reports/xau_red_slice_separability_audit_20260715_v1/XAU_RED_SLICE_SEPARABILITY_AUDIT_latest.json`
    Decision `XAU_RED_SLICE_SEPARABILITY_AUDIT_COMPLETE`, domain feature count `247`, missing required XAU rail features `0`, red slices `10`, weak required-rail-feature slice rate `0.10`.
  - Interpretation: residual-through-composition was the correct mechanical fix, but the residual can now overdrive FLAT out of the public logits after a temporarily good epoch. Next repair should stabilize residual/composition contribution or directly regularize residual FLAT starvation/side accuracy; do not extend this same recipe and do not move to IQL.
  - Entry-IQL, replay, candidate, shadow, live, and promotion remain closed.

### 2026-07-15 Update: Hard residual-cap source repair

- Implemented and committed a hard residual cap for hierarchical direction composition:
  - Commit: `06406498 Cap XAU hierarchical residual composition`.
  - New env: `ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP`.
  - Default is `0.0` for historical/non-smart parity.
  - Smart XAU recipe requires `0.10 <= ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP <= 0.20`; wrappers/readiness/manifest/enablement/sweep/replay/candidate gates now require `0.18`.
  - Public hierarchical direction logits now use `base_hierarchy_logits + capped(residual_scale * delta_logits)`, with `hierarchical_direction_residual_logits` exposed for audit.
  - Bundle metadata records both `hierarchical_direction_composition.residual_logit_cap` and flat `hier_compose_residual_logit_cap`; bundle audit fails smart active-head contracts if the cap is missing/weak/too high.
- Validation after source repair:
  - `python3 -m py_compile gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py gx1/models/entry_v10/entry_v10_ctx_train_v3.py gx1/models/entry_v10/entry_v10_bundle.py gx1/scripts/verify_entry_candidate_readiness_v1.py gx1/scripts/verify_entry_replay_readiness_v1.py`
  - `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py tests/test_entry_v10_ctx_model_shapes.py tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_entry_smart_seq520_smoke_manifest.py tests/test_entry_smart_seq520_smoke_train_enablement.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_xau_direction_repair_sweep.py tests/test_v10_6yr_rebuild_direction_repair_contract.py tests/test_entry_candidate_readiness.py tests/test_entry_replay_readiness.py -q` passed.
  - Pre-commit guardrails passed during commit.
- Readiness and enablement after commit `06406498`:
  - `scripts/entry_next_edge_control.sh smart-smoke-readiness --quiet` passed.
  - `scripts/entry_next_edge_control.sh smart-trainability-readiness` passed; latest JSON has `blockers=[]`, `training_allowed=false`, `candidate_training_allowed=false`, `replay_allowed=false`, `iql_allowed=false`, `shadow_live_promotion_allowed=false`.
  - `scripts/entry_next_edge_control.sh smart-smoke-train-enablement --vedtak SMART_SEQ520_XAU_SMOKE_HIERRESCAP_ENABLEMENT_20260715 --epochs 6 --batch-size 64 --quiet` passed. Enablement dry-run confirmed `ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP=0.18`, `trainer_started=false`.
- Ran one bounded smart smoke after the residual-cap repair:
  `scripts/entry_next_edge_control.sh smart-smoke-train --vedtak SMART_SEQ520_XAU_SMOKE_HIERRESCAP_E6_20260715 --require-edge-audit --epochs 6 --early-stop-patience 6`
  - Pre-train manifest: `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T162334Z.json`
  - Evidence sidecar: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T162334Z__direction_slice_failure_evidence.json`
  - Intended bundle dir absent: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T162334Z`
  - Result: fail-closed on `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`, no bundle written.
  - Hard-red stop fired at epoch `6`: best epoch `2`, best score `-0.003543`, epochs since improve `4`, patience `3`, min epochs `6`.
  - Best epoch `2`: global balance guard OK, pred LONG `0.325521`, SHORT `0.446615`, FLAT `0.227865`, `direction_slice_failure_count=9`, `accuracy_failures=8`, `pred_rate_failures=1`, `best_dir_acc=0.376302`.
  - Last epoch `6`: global balance guard OK, pred LONG `0.455078`, SHORT `0.370443`, FLAT `0.174479`, `direction_slice_failure_count=13`, `accuracy_failures=11`, `pred_rate_failures=2`.
  - Epoch `5` still showed FLAT collapse despite cap: pred FLAT `0.016276`, global balance guard failed. The cap stopped the old all-trade/near-zero-FLAT runaway from becoming the selected best epoch, but it did not solve slice side accuracy.
  - Report-only red-slice separability audit on this new sidecar completed:
    `/home/andre2/GX1_DATA/reports/xau_red_slice_separability_audit_20260715_v1/XAU_RED_SLICE_SEPARABILITY_AUDIT_latest.json`
    Decision `XAU_RED_SLICE_SEPARABILITY_AUDIT_COMPLETE`, domain feature count `247`, missing required XAU rail features `0`, red slice detail count `9`, weak required-feature slice rate `0.111111`.
  - Interpretation: current XAU inputs are still present. The blocker is not IQL-readiness and not missing required XAU rail features. The model now needs a different transformer objective/topology repair focused on slice-level side accuracy and stable conditional side separation; another scalar cap/epoch extension is not enough.
  - Entry-IQL, replay, candidate, shadow, live, and promotion remain closed.

### 2026-07-15 Update: Side-neutral hierarchy residual repair

- Implemented and committed a side-neutral residual topology for hierarchical direction composition:
  - Commit: `69db69b1 Require XAU side-neutral hierarchy residual`.
  - New env: `ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL`.
  - Default is `0` for historical/non-smart parity.
  - Smart XAU recipe now requires `ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL=1` in trainer preflight, smoke/candidate wrappers, direct XAU train path, readiness/manifest/enablement contracts, sweep lint, bundle audit, candidate readiness, and replay readiness.
  - When enabled, the hierarchical residual is projected to `[trade_residual, trade_residual, flat_residual]` before the residual cap. This lets the residual calibrate TRADE-vs-FLAT but prevents it from owning an independent LONG-vs-SHORT decision. The hierarchy side head must own side separation.
- Validation after source repair:
  - `python3 -m py_compile` passed for the touched model/trainer/bundle/readiness/audit/sweep scripts.
  - `bash -n` passed for smoke/candidate/rebuild shell wrappers.
  - Focused pytest passed:
    `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py tests/test_entry_v10_ctx_model_shapes.py tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_entry_smart_seq520_smoke_manifest.py tests/test_entry_smart_seq520_smoke_train_enablement.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_xau_direction_repair_sweep.py tests/test_v10_6yr_rebuild_direction_repair_contract.py tests/test_entry_candidate_readiness.py tests/test_entry_replay_readiness.py -q`.
  - Pre-commit guardrails passed during commit.
  - Broad XAU/replay/readiness pytest surface passed after the commit with expected skips.
- Clean-git readiness and enablement after commit `69db69b1`:
  - `scripts/entry_next_edge_control.sh smart-smoke-readiness --quiet` passed.
  - `scripts/entry_next_edge_control.sh smart-trainability-readiness --quiet` passed.
  - `scripts/entry_next_edge_control.sh smart-smoke-train-enablement --vedtak SMART_SEQ520_XAU_SMOKE_SIDENEUTRAL_ENABLEMENT_20260715 --epochs 6 --batch-size 64 --quiet` passed with `trainer_started=false`, `candidate_training_allowed=false`, `replay_allowed=false`, `iql_allowed=false`, and dry-run proof `ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL=1`.
- Ran one bounded smart smoke after the side-neutral residual repair:
  `scripts/entry_next_edge_control.sh smart-smoke-train --vedtak SMART_SEQ520_XAU_SMOKE_SIDENEUTRAL_E6_20260715 --require-edge-audit --epochs 6 --early-stop-patience 6`
  - Pre-train manifest: `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T164634Z.json`
  - Evidence sidecar: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T164634Z__direction_slice_failure_evidence.json`
  - Intended bundle dir absent: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T164634Z`
  - Result: fail-closed on `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`, no bundle written.
  - Sidecar confirms `bundle_written=false`, `promotion_shadow_live_allowed=false`, `hier_compose_residual_side_neutral=true`, and cap `0.18`.
  - Best/last epoch was epoch `6`: global balance guard OK, pred LONG `0.218750`, SHORT `0.518880`, FLAT `0.262370`, `best_dir_acc=0.384766`, `direction_slice_failure_count=7`, `accuracy_failures=6`, `pred_rate_failures=1`, `direction_slice_ckpt_score=-0.038300`.
  - Hierarchy evidence at best epoch: `hier_trade_pred_rate=0.996745`, `hier_trade_prob_mean=0.589720`, `hier_trade_prob_label_flat_mean=0.588493`, `hier_side_acc_on_edge=0.574553`.
  - Interpretation: side-neutral residual improved the final slice count versus the previous residual-cap smoke, but did not pass. The remaining blocker is now primarily hierarchy side-head LONG/SHORT discrimination and slice-level side accuracy, not residual side fighting and not missing XAU rail input.
- Report-only red-slice separability audit on the side-neutral sidecar completed:
  - Report: `/home/andre2/GX1_DATA/reports/xau_red_slice_separability_audit_20260715_v1/XAU_RED_SLICE_SEPARABILITY_AUDIT_latest.json`
  - Decision `XAU_RED_SLICE_SEPARABILITY_AUDIT_COMPLETE`.
  - Domain feature count `247`; missing required XAU direction features `0`.
  - Red slices audited `7`; weak required-feature slice rate `1/7`.
  - Training, candidate training, replay, IQL, shadow, live, and promotion remain closed.
- 2026-07-15 post-run resource state:
  - No `python3` train/eval jobs were running.
  - GPU idle (`0%`, 296 MiB used).
  - `/home/andre2` and `/home/andre2/GX1_DATA` still had about `838G` free.
  - RAM had about `36GiB` available and swap use was `0B`.
- Next source repair should target hierarchy side-head priors/contrast directly:
  - Do not rerun the same side-neutral recipe with more epochs.
  - Candidate direction: add a hard smart XAU side-head global/slice prior-match or supervised contrast/ranking objective on trade rows so `long_or_short_given_edge` cannot stay SHORT-biased inside active red slices.
  - Keep residual cap and side-neutral residual enabled.
  - Do not move to IQL, replay, candidate, shadow, live, or promotion until a fresh XAU transformer bundle passes hard direction slice/class-balance gates.

### 2026-07-15 Update: Hierarchy side-prior repair tested

- Implemented and committed hard smart-XAU hierarchy side prior matching:
  - Commit: `ccc9a230 Require XAU hierarchy side prior match`.
  - New smart-XAU trainer contract requires hierarchy side-head global prior match and per-ctx-slice side prior match on trade rows.
  - Required smart recipe values are `hier_side_global_prior_match_weight=4.0`, tolerance `0.02`, min label-rate `0.10`, plus `hier_slice_side_prior_match_weight=4.0`, tolerance `0.02`, min label-rate `0.10`, and min rows `8`.
  - Smart smoke/trainability/readiness, smoke manifest, smoke train enablement, bundle audit, candidate readiness, replay readiness, sweep lint, smoke/candidate wrappers, and rebuild wrapper now require/report this contract. Stale bundles without the side-prior contract stay closed.
- Validation after source repair:
  - `python3 -m py_compile` passed for the touched trainer/readiness/manifest/enablement/audit/sweep scripts.
  - `bash -n` passed for touched smoke/candidate/rebuild wrappers.
  - Focused pytest passed for trainer defaults, wrappers, smart smoke/readiness/manifest/enablement, bundle audit, sweep, rebuild contract, candidate readiness, and replay readiness.
  - A broad XAU/readiness pytest subset had three current-artifact/environment failures because an old foundation seq146 dataset directory is missing after cleanup; the remaining broad subset passed. This does not open any gate.
  - Pre-commit guardrails passed during commit.
- Readiness and dry-run enablement after commit `ccc9a230`:
  - `scripts/entry_next_edge_control.sh smart-smoke-readiness --quiet` passed.
  - `scripts/entry_next_edge_control.sh smart-trainability-readiness --quiet` passed.
  - `scripts/entry_next_edge_control.sh smart-smoke-train-enablement --vedtak SMART_SEQ520_XAU_SMOKE_SIDEPRIOR_ENABLEMENT_20260715 --epochs 6 --batch-size 64 --quiet` passed with `trainer_started=false`, `has_hier_side_prior=true`, and candidate/replay/IQL/live/promotion closed.
- Ran one bounded smart smoke after the side-prior repair:
  `scripts/entry_next_edge_control.sh smart-smoke-train --vedtak SMART_SEQ520_XAU_SMOKE_SIDEPRIOR_E6_20260715 --require-edge-audit --epochs 6 --early-stop-patience 6`
  - Pre-train manifest: `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T171108Z.json`
  - Evidence sidecar: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T171108Z__direction_slice_failure_evidence.json`
  - Intended bundle dir absent: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T171108Z`
  - Result: fail-closed on `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`, no bundle written.
  - Best/last epoch was epoch `6`: global direction balance guard OK, `best_dir_acc=0.398438`, pred LONG `0.397135`, SHORT `0.460286`, FLAT `0.142578`, labels LONG `0.322917`, SHORT `0.332031`, FLAT `0.345052`.
  - Slice contract still failed with `12` red-slice failures: `8` accuracy failures and `4` pred-rate failures. Accuracy deficit was `0.169332`; pred-rate shortfall was `0.038735`.
  - Hierarchy side-head improved materially: `hier_side_pred_long_rate_on_edge=0.495030`, `hier_side_acc_on_edge=0.596421`.
  - Remaining failure moved back to trade-vs-flat: `hier_trade_pred_rate=1.000000`, and `hier_trade_prob_label_flat_mean=0.607661`, so the hierarchy still scores flat-labelled rows as trade too often.
- Report-only red-slice separability audit on the side-prior sidecar completed:
  - Report: `/home/andre2/GX1_DATA/reports/xau_red_slice_separability_audit_20260715_v1/XAU_RED_SLICE_SEPARABILITY_AUDIT_latest.json`
  - Decision `XAU_RED_SLICE_SEPARABILITY_AUDIT_COMPLETE`.
  - Domain feature count `247`; missing required XAU direction features `0`.
  - Red slices audited `8`; weak required-feature slice rate `1/8`.
  - Example red-slice failure mode: `session_id=1` has FLAT label-rate `0.434911` but FLAT pred-rate `0.133136`; `spread_bucket=4` has FLAT label-rate `0.355705` but FLAT pred-rate `0.140940`.
  - Interpretation: side-prior repair is useful and should stay, but the next repair is hierarchy trade/flat calibration and flat-abstain prior pressure, not side-prior again and not IQL.
- 2026-07-15 post-audit resource state:
  - No `python3` train/eval jobs were running.
  - GPU was idle/near-idle.
  - `/home/andre2/GX1_DATA` still had about `838G` free.
  - RAM had about `39GiB` available and swap use was `0B`.
  - No cleanup was required under the active disk threshold.

### 2026-07-15 Update: Hierarchy trade-prior repair tested

- Implemented and committed hard smart-XAU hierarchy trade/flat prior matching:
  - Commit: `34a6342e Require XAU hierarchy trade prior match`.
  - New smart-XAU trainer contract requires hierarchy trade-head global prior match and per-ctx-slice trade/flat prior match.
  - Required smart recipe values are `hier_trade_global_prior_match_weight=4.0`, tolerance `0.02`, min label-rate `0.10`, plus `hier_slice_trade_prior_match_weight=4.0`, tolerance `0.02`, min label-rate `0.10`, and min rows `8`.
  - Smart smoke/trainability/readiness, smoke manifest, smoke train enablement, bundle audit, candidate readiness, replay readiness, sweep lint, smoke/candidate wrappers, and rebuild wrapper now require/report this contract. Stale bundles without the trade-prior contract stay closed.
- Validation after source repair:
  - `python3 -m py_compile` passed for the touched trainer/readiness/manifest/enablement/audit/sweep scripts.
  - `bash -n` passed for touched smoke/candidate/rebuild wrappers.
  - Focused pytest passed for trainer defaults, wrappers, smart smoke/readiness/manifest/enablement, bundle audit, sweep, rebuild contract, candidate readiness, and replay readiness.
  - Pre-commit guardrails passed during commit.
- Clean-git readiness and enablement after commit `34a6342e`:
  - `scripts/entry_next_edge_control.sh smart-smoke-readiness --quiet` passed.
  - `scripts/entry_next_edge_control.sh smart-trainability-readiness --quiet` passed. A parallel `--quiet` trainability call briefly returned exit `2`, but the sequential rerun returned exit `0` with `blockers=[]`; avoid parallel readiness calls that race on `latest` artifacts.
  - `scripts/entry_next_edge_control.sh smart-smoke-train-enablement --vedtak SMART_SEQ520_XAU_SMOKE_TRADEPRIOR_ENABLEMENT_20260715 --epochs 6 --batch-size 64 --quiet` passed with no trainer start and candidate/replay/IQL/live/promotion closed.
- Ran one bounded smart smoke after the trade-prior repair:
  `scripts/entry_next_edge_control.sh smart-smoke-train --vedtak SMART_SEQ520_XAU_SMOKE_TRADEPRIOR_E6_20260715 --require-edge-audit --epochs 6 --early-stop-patience 6`
  - Pre-train manifest: `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T185315Z.json`
  - Evidence sidecar: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T185315Z__direction_slice_failure_evidence.json`
  - Intended bundle dir absent: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T185315Z`
  - Result: hard-red-stopped at epoch `6`, then failed closed on `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`; no bundle written.
  - Best checkpoint remained epoch `1`: global balance guard OK, `best_dir_acc=0.330729`, `best_dir_ckpt_score=-0.516330`, `best_direction_slice_contract_ok=false`.
  - Best slice stats: `18` slice failures over `17` audited slices (`15` accuracy failures, `3` pred-rate failures), pred LONG `0.567708`, SHORT `0.153646`, FLAT `0.278646` versus labels LONG `0.322917`, SHORT `0.332031`, FLAT `0.345052`.
  - Last epoch `6` was still hard-red: `32` slice failures (`15` accuracy, `17` pred-rate), pred LONG `0.438802`, SHORT `0.515625`, FLAT `0.045573`.
  - Trade-prior changed the soft trade probability in the intended direction, but not the hard argmax: last epoch had `hier_trade_target_rate=0.654948`, `hier_trade_prob_mean=0.632771`, but `hier_trade_pred_rate=0.998698` and `hier_flat_pred_rate=0.001302`.
  - Side-prior still helped the side head at the end: `hier_side_pred_long_rate_on_edge=0.483101`, `hier_side_acc_on_edge=0.522863`; the immediate blocker is trade/flat argmax and public FLAT starvation, not side-prior absence.
- Report-only red-slice separability audit on the trade-prior sidecar completed:
  - Report: `/home/andre2/GX1_DATA/reports/xau_red_slice_separability_audit_20260715_v1/XAU_RED_SLICE_SEPARABILITY_AUDIT_latest.json`
  - Decision `XAU_RED_SLICE_SEPARABILITY_AUDIT_COMPLETE`.
  - Evidence source: trade-prior sidecar above; best epoch `1`, last epoch `6`, hard-red-stopped.
  - Domain feature count `247`; missing required XAU direction features `0`.
  - Red slice detail count `16`; weak required-feature slice rate `1/16`.
  - Interpretation: existing XAU rail/SR/wick/regime features are present. The next repair is not random new input and not IQL; it must address the hierarchy trade/flat decision threshold/logit separation so soft FLAT probability becomes hard FLAT prediction where labels/utility require abstain.
- 2026-07-15 post-trade-prior resource state:
  - No `python3` train/eval jobs were running.
  - GPU was idle/near-idle after the capped run exited.
  - `/home/andre2/GX1_DATA` still had about `838G` free.
  - RAM had about `39GiB` available and swap use was `0B`.
  - No cleanup was required under the active disk threshold.

## 2026-07-15 Flat-Logit Repair Status

- Implemented and committed hard hierarchy trade/flat logit-margin repair:
  - Commit: `2a5fe00e Require XAU hierarchy flat logit margin`
  - Added fail-closed trainer env/metadata/loss terms:
    - `ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT`
    - `ENTRY_HIER_FLAT_LOGIT_MARGIN`
    - `ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE`
    - `ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT`
    - `ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN`
    - `ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE`
    - `ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS`
  - Smart-XAU preflight now requires global and slice flat-logit-margin weights `>=8.0`, margin `>=0.10`, min label rate `>=0.10`, and slice rows `>=8`.
  - Wrapper/readiness/manifest/enablement/audit/sweep contracts now require the same proof. Old soft trade-prior-only runs cannot pass as current smart-XAU proof.
  - Focused validation passed:
    - `python3 -m py_compile ...`
    - `bash -n scripts/run_entry_foundation_seq146_smoke_train.sh scripts/run_entry_foundation_seq146_candidate_train.sh scripts/v10_6yr_rebuild_20260626.sh`
    - `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_v10_6yr_rebuild_direction_repair_contract.py tests/test_xau_direction_repair_sweep.py tests/test_entry_smart_seq520_smoke_train_enablement.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_entry_smart_seq520_smoke_manifest.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_entry_candidate_readiness.py tests/test_entry_replay_readiness.py -q`
- Clean-git smart readiness/enablement passed after the commit:
  - `scripts/entry_next_edge_control.sh smart-smoke-readiness --quiet`
  - `scripts/entry_next_edge_control.sh smart-trainability-readiness --quiet`
  - Enablement report: `/home/andre2/GX1_DATA/reports/entry_smart_seq520_smoke_train_enablement_20260715_v1/ENTRY_SMART_SEQ520_SMOKE_TRAIN_ENABLEMENT_20260715T191602Z.json`
  - Enablement decision: `ENTRY_SMART_SEQ520_SMOKE_TRAIN_ENABLEMENT_READY_FOR_EXPLICIT_EXECUTION`
  - Proof: `has_hier_flat_logit_margin=true`, `iql_allowed=false`, `replay_allowed=false`, `trainer_started=false`.
- Ran one bounded smart smoke with the new flat-logit repair:
  - Vedtak: `SMART_SEQ520_XAU_SMOKE_FLATLOGIT_E6_20260715`
  - Command: `scripts/entry_next_edge_control.sh smart-smoke-train --vedtak SMART_SEQ520_XAU_SMOKE_FLATLOGIT_E6_20260715 --require-edge-audit --epochs 6 --early-stop-patience 6`
  - It was manually stopped after epoch `5` because epoch `5` was hard red and regressed to `hier_trade_pred=1.000000`. No candidate bundle was produced.
  - The only generated pre-train manifest for the aborted run was deleted:
    - `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T191642Z.json`
  - Resource state after stop: `/home/andre2/GX1_DATA` about `838G` free, RAM about `38GiB` available, swap `0B`, GPU back near idle.
- Smoke evidence summary:
  - Epoch `1`: balance guard OK, `18` slice failures, public pred LONG `0.238281`, SHORT `0.502604`, FLAT `0.259115`; hierarchy `trade_pred=0.927083`, `trade_prob=0.586075`.
  - Epoch `2`: hierarchy trade/flat improved toward target with `trade_pred=0.773438`, `trade_prob=0.546685`, but public direction collapsed LONG to `0.002604`; `31` slice failures and balance guard failed.
  - Epoch `3` was the best checkpoint observed: `dir_acc=0.361328`, balance guard OK, `13` slice failures, public pred LONG `0.222005`, SHORT `0.171875`, FLAT `0.606120`; hierarchy `trade_pred=0.862630`, `trade_prob=0.543640`.
  - Epoch `4` regressed to `27` slice failures, balance guard failed, `trade_pred=0.899089`.
  - Epoch `5` was hard red: `42` slice failures, balance guard failed, public pred LONG `0.792318`, SHORT `0.112630`, FLAT `0.095052`; hierarchy `trade_pred=1.000000`, `trade_prob=0.605607`.
- Interpretation:
  - This is real progress versus the trade-prior-only smoke: hard hierarchy trade/flat can now move away from all-trade for some epochs (`0.773438` at epoch `2` versus previous `0.998698` at failed epoch `6`).
  - It is still not solved. The new pressure competes with the public 3-class direction head and the model oscillates between FLAT-heavy and LONG-heavy collapse.
  - Do not continue this exact recipe for more epochs. It already showed the mechanism and then went hard red.
  - Next repair should decouple or stage the public 3-class direction head from hierarchy trade/flat calibration, or add an explicit consistency term between public FLAT and hierarchy flat so the two heads do not fight.

## 2026-07-15 Public/Hierarchy Flat-Consistency Repair Status

- Implemented and committed the explicit consistency repair between the public 3-class direction head and hierarchy trade/flat head:
  - Commit: `dcdcde91 Require XAU public hierarchy flat consistency`
  - Added global and active-slice loss/contract terms:
    - `ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT`
    - `ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE`
    - `ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT`
    - `ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE`
    - `ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS`
  - Smart-XAU preflight now requires global/slice consistency weights `>=4.0`, min label rate `>=0.10`, and slice rows `>=8`.
  - Trainer metadata, loss summaries, failure evidence, wrappers, rebuild script, smoke/readiness/manifest contracts, enablement, bundle audit, candidate readiness, replay readiness, and sweep lint now require/report the same contract.
- Validation before commit:
  - `python3 -m py_compile` passed for the trainer and touched Python gate scripts.
  - `bash -n` passed for touched smoke/candidate/rebuild wrappers.
  - Focused pytest passed:
    `scripts/pytest_repo.sh tests/test_entry_v10_train_defaults.py tests/test_entry_foundation_smoke_train_wrapper.py tests/test_entry_candidate_train_wrapper.py tests/test_v10_6yr_rebuild_direction_repair_contract.py tests/test_xau_direction_repair_sweep.py tests/test_entry_smart_seq520_smoke_train_enablement.py tests/test_entry_smart_seq520_smoke_readiness.py tests/test_entry_smart_seq520_trainability_readiness.py tests/test_entry_smart_seq520_smoke_manifest.py tests/test_entry_foundation_smoke_bundle_audit.py tests/test_entry_candidate_readiness.py tests/test_entry_replay_readiness.py -q`
  - Pre-commit guardrails passed during commit.
- Clean-git readiness/enablement after commit:
  - `scripts/entry_next_edge_control.sh smart-smoke-readiness --quiet` passed.
  - `scripts/entry_next_edge_control.sh smart-trainability-readiness --quiet` passed. A first parallel quiet trainability call returned exit `2`, then the non-quiet call returned `blockers=[]` and the repeated quiet call returned exit `0`; avoid parallel readiness calls that race on `latest` artifacts.
  - Enablement vedtak `SMART_SEQ520_XAU_SMOKE_FLATCONSIST_ENABLEMENT_20260715` passed without starting trainer:
    `/home/andre2/GX1_DATA/reports/entry_smart_seq520_smoke_train_enablement_20260715_v1/ENTRY_SMART_SEQ520_SMOKE_TRAIN_ENABLEMENT_20260715T195001Z.json`
- Ran one bounded smart smoke after the consistency repair:
  `scripts/entry_next_edge_control.sh smart-smoke-train --vedtak SMART_SEQ520_XAU_SMOKE_FLATCONSIST_E6_20260715 --require-edge-audit --epochs 6 --early-stop-patience 6`
  - Pre-train manifest was written, then deleted after manual stop because the run was aborted and stale:
    `/home/andre2/GX1_DATA/reports/entry_foundation_smoke_train_manifests_20260628_v1/ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_20260715T195017Z.json`
  - Intended bundle dir was not created:
    `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_20260715T195017Z`
  - The run was manually stopped during epoch `4` train after epoch `3` validation was still hard-red. No bundle and no failure sidecar were written because the operator interrupted before final trainer failure handling.
- Smoke evidence summary:
  - Epoch `1`: global balance guard OK, `14` slice failures, public pred LONG `0.380859`, SHORT `0.352865`, FLAT `0.266276`; hierarchy `trade_pred=0.936849`, `trade_prob=0.582468`; consistency losses were active but small (`hier_public_flat_consistency=0.024263`, slice `0.049824`).
  - Epoch `2`: hierarchy trade improved to `trade_pred=0.830729`, but public direction collapsed LONG to `0.005859`; global balance guard failed and slice failures rose to `32`.
  - Epoch `3`: hierarchy trade improved further to `trade_pred=0.750651`, closer to target `0.654948`, but public direction collapsed FLAT-heavy with pred LONG `0.130208`, SHORT `0.096354`, FLAT `0.773438`; global balance guard still failed and slice failures rose to `35`.
  - This was stopped immediately after epoch `3` evidence because continuing to epoch `6` would burn compute without a plausible bundle path.
- Interpretation:
  - The new consistency surface is wired and active, and it helps the hierarchy trade/flat hard argmax move toward target.
  - It is still not sufficient. The public 3-class head still oscillates between LONG/SHORT/FLAT collapse while the hierarchy trade/flat head improves.
  - The next repair should not add another scalar pressure or run more epochs. It should change the public direction formulation/staging so public FLAT is not independently fighting hierarchy no-trade.
  - Candidate training, replay, IQL, shadow, live, and promotion remain closed.
- Post-stop resource state:
  - No relevant `python`/`python3` training/eval process running.
  - `/home/andre2/GX1_DATA` still about `838G` free.
  - RAM about `38GiB` available, swap `0B`.
  - The aborted run manifest was deleted; no stale bundle directory existed.

## Current Blockers

1. Current direction pocket audit is red/stale and must not be used as promotion proof.
   - Latest old audit has:
     - `intraday_bull selected SHORT rate 0.885`
     - `intraday_bull__htf_bull selected SHORT rate 0.948`
     - `rising_channel_support_touch selected SHORT rate 0.840`
   - It also points at stale July/pathutil artifacts.

2. Latest executed smart XAU smoke after public-FLAT-from-hierarchy repair failed closed on `[TRAIN_FAIL_DIRECTION_SLICE_GUARD]`. No candidate bundle was produced and no failed bundle should be used as evidence.
   - The public-FLAT independent-residual fight is fixed and active: `ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE=1` was present in the trainer recipe, and public residual is now softmax-invariant when strict smart-XAU composition is enabled.
   - Best checkpoint was epoch `2`: balance guard OK, `dir_acc=0.379557`, public pred LONG `0.229818`, SHORT `0.504557`, FLAT `0.265625`, and `9` slice failures.
   - Epoch `3` degraded to FLAT-heavy public predictions (`pred_flat=0.558594`) and `18` slice failures. Trainer early-stopped and refused bundle creation because the slice contract stayed red.
   - The active blocker is now genuine slice-level side/coverage accuracy under the hierarchy-composed public direction output, not missing public-FLAT composition.
   - The blocker is not missing required XAU rail input and not IQL-readiness; the latest separability audit still found domain feature count `247`, missing required XAU direction features `0`, and only `1/16` weak required-feature red slices.
   - Until a fresh XAU transformer candidate bundle passes hard direction-slice and class-balance gates, candidate training, replay, IQL, shadow, live, and promotion remain closed.

3. No promoted XAU candidate yet proves the required bull/rising-support, bear/falling-resistance, calibration, replay, parity, and launch gates.

## Highest-Priority Next Steps

1. Do not extend epochs on the old side-utility-conviction, utility-trade-conviction, utility-triad-CE, hierarchical-composition, trade-pos-weight, hierarchy side-slice, residual-through-composition, residual-cap, side-neutral residual, side-prior, trade-prior, or flat-logit-margin recipe. They already hard-red-stopped, failed closed, or were manually stopped with no candidate bundle.

2. Next action should be a new small source repair, not another heavy run on the same recipe:
   - keep residual-through-composition, the hard residual cap, side-neutral residual, public-FLAT-from-hierarchy composition, side-prior contract, and trade-prior contract; they are useful guardrails.
   - do not spend another run tuning only scalar caps/weights/epochs. The latest smoke shows public composition fixed the independent FLAT fight but did not solve active-slice side/coverage accuracy.
   - the next repair should target hierarchy side/slice accuracy and FLAT coverage staging/calibration under the composed public output.
   - likely next source-level options:
     - stage/anneal public direction class-balance/slice losses versus hierarchy trade/flat losses instead of applying all pressures at full strength from epoch 1.
     - split public-side logits from public-FLAT/no-trade logits so side learning cannot starve abstain.
     - add direct hierarchy side active-slice accuracy/coverage pressure tied to public LONG/SHORT slice failures.
   - keep side-prior enabled while targeting remaining slice-level accuracy; feature audit still says required XAU rail inputs are present.
   - after a source repair, rerun focused tests, then clean-git readiness/enablement, then only one bounded smoke with hard-red stop.
   - do not add random new input, do not move to IQL, and do not tune another scalar weight blindly.

3. Keep clean-git/readiness discipline before any heavy job:
   - `git status --short` must be clean.
   - `smart-smoke-readiness --quiet` and `smart-trainability-readiness --quiet` must pass sequentially.
   - Disk/RAM must remain above the active safety thresholds.

4. After a candidate bundle passes hard audits:
   - materialize expected-utility predictions
   - run live-like direction pocket audit
   - run serve parity
   - do not promote unless all launch gates pass.

## Still To Implement After Current Guards

These came from the five subagent audits and should be handled before any serious live promotion:

1. Replay/live admission parity:
   - candidate replay is still effectively single-position/cooldown while live can be cap-3 with same-side cap.
   - Implement live-style open-book replay: max open 3, same-side cap 2, opposing-side block, drawdown breaker parity.

2. Fill/latency parity:
   - replay still uses idealized T+5 M1 open.
   - live uses actual quote/OANDA fill after polling latency.
   - Journal and replay decision available time, fill time, fill price, bid/ask, spread, slippage.

3. Exit parity:
   - fixed-horizon entry replay should not be used as final promotion metric.
   - use exit-stack replay/V3/Exit-IQL parity for launch evidence.

4. Decision-bar parity:
   - hard parity has historically excluded latest decision bar even though live trades on it.
   - either make state causal and retrain/rematerialize, delay decision by required closed-bar horizon, or hard-fail on decision-bar side/take flips and material geometry/swing diffs.

5. Expected-utility hierarchy calibration:
   - final direction odds cap is not enough.
   - calibrate and gate hierarchy heads used by expected utility: trade logit, side logits, side validity, side bad path, utility scale, uncertainty penalty.
   - add rising-support anti-short and falling-resistance anti-long slice caps on hierarchy/utility outputs.

6. Prediction provenance:
   - parity/audit should require pinned predictions from the same fresh XAU bundle/dataset.
   - include prediction parquet hash and bundle/dataset provenance in launch gate.

7. Sweep efficiency:
   - current sweep is safe and XAU-only, but serial.
   - after gates pass, add bounded `--jobs` and/or Optuna SQLite ask/tell with objective weighted toward bad-side pocket rate, coverage, pocket proxy pnl, and selective-edge metrics.

## Do Not Do

- Do not use non-XAU files/configs/models/artifacts.
- Do not add live hand-rules forcing long/short direction.
- Do not promote current July/pathutil/utilityrepair artifacts.
- Do not treat global winrate as proof. Promotion must pass live-like bull/bear and rising/falling rail pockets.
- Do not bypass dirty-git gates for rebuild/training.

## Resume Checklist

Start with:

```bash
cd /home/andre2/src/GX1_ENGINE
ps -C python3 -o pid,etime,cmd
git status --short
df -h /home/andre2 /home/andre2/GX1_DATA
/home/andre2/venvs/gx1/bin/python -c "import sys, pytest; print(sys.executable); print(pytest.__version__)"
/home/andre2/venvs/gx1/bin/python -c "import importlib.util; print(importlib.util.find_spec('lightgbm') is not None)"
```

Then continue from the highest-priority next steps above.
