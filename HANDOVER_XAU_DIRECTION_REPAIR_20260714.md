# XAUUSD Direction Repair Handover - 2026-07-14

## Continuation Goal

Continue the XAUUSD-only direction repair until the live/replay/training stack proves that the model learns to abstain or go long in bull/rising-support regimes, rather than selecting confident SHORT. Do not use non-XAU project artifacts. Do not promote any XAU bundle live until fresh XAU datasets, parity, live-like replay, calibration, and direction-pocket audits all pass.

## Current State

- Repo: `/home/andre2/src/GX1_ENGINE`
- Data root: `/home/andre2/GX1_DATA`
- Disk: `/dev/sdd` has about `838G` free after the 2026-07-15 cleanup round.
- Runtime: no `python`/`python3` training/eval jobs were running after the latest 2026-07-15 hard-red smart smoke stop.
- Non-XAU project artifacts: removed from the working machine except for fail-closed XAU isolation guards.
- Worktree: verify clean with `git status --short` before clean-git gates; current smart XAU slice-loss repairs are intended source changes, not ad hoc run overrides.
- Canonical Python: `/home/andre2/venvs/gx1/bin/python`, pytest `9.0.2`, `lightgbm 4.6.0`.

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

## Current Blockers

1. Current direction pocket audit is red/stale and must not be used as promotion proof.
   - Latest old audit has:
     - `intraday_bull selected SHORT rate 0.885`
     - `intraday_bull__htf_bull selected SHORT rate 0.948`
     - `rising_channel_support_touch selected SHORT rate 0.840`
   - It also points at stale July/pathutil artifacts.

2. Latest executed smart XAU smoke after the residual-through-composition repair still failed hard on direction-slice accuracy/stability. No fallback path and no failed bundle should be used as evidence.
   - The source repair worked mechanically: `delta_abs_mean` is no longer stuck at zero, and epoch `4` reached a much healthier global balance with pred LONG `0.240234`, SHORT `0.351562`, FLAT `0.408203`.
   - It still failed the active slice contract at best epoch `4` with `11` slice failures (`9` accuracy, `2` pred-rate), then over-corrected by epoch `6` and collapsed FLAT to `0.020182`.
   - The new blocker is not missing XAU input and not IQL-readiness. It is residual/composition stability plus remaining slice-level side accuracy.
   - Until a fresh XAU transformer candidate bundle passes hard direction-slice and class-balance gates, candidate training, replay, IQL, shadow, live, and promotion remain closed.

3. No promoted XAU candidate yet proves the required bull/rising-support, bear/falling-resistance, calibration, replay, parity, and launch gates.

## Highest-Priority Next Steps

1. Do not extend epochs on the old side-utility-conviction, utility-trade-conviction, utility-triad-CE, hierarchical-composition, trade-pos-weight, hierarchy side-slice, or residual-through-composition recipe. They already hard-red-stopped, failed closed, or were manually stopped with no candidate bundle.

2. Next action should be a small source repair, not another heavy run:
   - keep residual-through-composition, because it fixed the dead residual path.
   - prevent the residual branch from overdriving the hierarchy base logits after the temporarily good epoch `4`; candidates include bounded residual contribution, residual norm/flat-starvation regularization, or a schedule/gate that fails closed when residual destroys FLAT coverage.
   - continue targeting remaining slice-level side accuracy; feature audit still says required XAU rail inputs are present.
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
