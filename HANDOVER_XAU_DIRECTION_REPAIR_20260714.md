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

## Current Blockers

1. Current direction pocket audit is red/stale and must not be used as promotion proof.
   - Latest old audit has:
     - `intraday_bull selected SHORT rate 0.885`
     - `intraday_bull__htf_bull selected SHORT rate 0.948`
     - `rising_channel_support_touch selected SHORT rate 0.840`
   - It also points at stale July/pathutil artifacts.

2. Latest smart XAU smoke training attempts failed hard on direction slice guard. No fallback path and no slice-failed bundle should be used as evidence.
   - True-margin standard, stronger W8/CE4, batch-size 256, slice-balanced sampler, `mean_max` aggregation, `0.05` argmax-aligned pred-rate temperature, and slice accuracy-edge all failed/stopped hard without writing a bundle.
   - Current evidence says the objective can now see hard argmax collapse and log exact slice failures, but it still cannot keep all active direction slices above the hard gate. The next repair should change the objective/data contract, not rerun the same transformer recipe longer.

3. No promoted XAU candidate yet proves the required bull/rising-support, bear/falling-resistance, calibration, replay, parity, and launch gates.

## Highest-Priority Next Steps

1. Do not relaunch the same `SLICEACCEDGE` transformer recipe just to burn more epochs. Use the `ENTRY_DIR_SLICE_FAILURE` rows from the stopped run to implement the next hard objective/data repair. Current leading evidence: global balance can pass while ctx slices still miss majority accuracy, and the model oscillates between missing FLAT/SHORT/LONG pred-rate across slices.

2. Keep clean-git/readiness discipline before any heavy job:
   - `git status --short` must be clean.
   - `smart-smoke-readiness --quiet` and `smart-trainability-readiness --quiet` must pass sequentially.
   - Disk/RAM must remain above the active safety thresholds.

3. After a candidate bundle passes hard audits:
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
