# XAUUSD Direction Repair Handover - 2026-07-14

## Continuation Goal

Continue the XAUUSD-only direction repair until the live/replay/training stack proves that the model learns to abstain or go long in bull/rising-support regimes, rather than selecting confident SHORT. Do not use non-XAU project artifacts. Do not promote any XAU bundle live until fresh XAU datasets, parity, live-like replay, calibration, and direction-pocket audits all pass.

## Current State

- Repo: `/home/andre2/src/GX1_ENGINE`
- Data root: `/home/andre2/GX1_DATA`
- Disk: `/dev/sdd` has about `829G` free.
- Runtime: no `python3` training/eval jobs were running at handover.
- Non-XAU project artifacts: removed from the working machine except for fail-closed XAU isolation guards.
- Worktree: dirty with many XAU/entry/live/replay changes and new tests/scripts. Do not assume clean-git gates can run until this is committed/stashed intentionally.
- Canonical Python: `/home/andre2/venvs/gx1/bin/python`, pytest `9.0.2`.
- Canonical Python currently lacks `lightgbm`. Broad tests that import replay/readiness modules fail on `ModuleNotFoundError: No module named 'lightgbm'`.

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

Broad XAU/replay/readiness suite currently fails during collection because canonical env lacks `lightgbm`.

## Current Blockers

1. Fresh XAU direction-repair dataset is missing.
   - Expected dataset dir:
     `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/v10_dataset_6yr_smartctx_xau_direction_repair`
   - Missing:
     - `v10_6yr_dataset__HOLD_03B_train.parquet`
     - `v10_6yr_dataset__HOLD_03B_val.parquet`
     - `v10_6yr_dataset__HOLD_03B_test.parquet`
   - Latest pretrain audit: `FAIL`.

2. Sweep is blocked until the fresh dataset passes pretrain audit.
   - Latest sweep plan: `BLOCKED_PLAN_ONLY`
   - Reason: missing fresh XAU parquets.

3. Current direction pocket audit is red and stale.
   - Latest old audit has:
     - `intraday_bull selected SHORT rate 0.885`
     - `intraday_bull__htf_bull selected SHORT rate 0.948`
     - `rising_channel_support_touch selected SHORT rate 0.840`
   - It also points at stale July/pathutil artifacts.

4. Smoke/trainability readiness are blocked.
   - Smoke readiness points at old smart candidate dataset/audits and reports dirty git.
   - Trainability readiness is blocked by smoke readiness.

5. Canonical env needs `lightgbm` installed once.
   - Do this in `/home/andre2/venvs/gx1`, not repo `.venv`.

6. Worktree is dirty.
   - Rebuild/training wrappers enforce clean git.
   - Commit or otherwise intentionally preserve current changes before running clean-git gates.

## Highest-Priority Next Steps

1. Install missing dependency once in the canonical env:

```bash
/home/andre2/venvs/gx1/bin/python -m pip install lightgbm
```

2. Rerun broad relevant tests:

```bash
scripts/pytest_repo.sh \
  tests/test_repair_entry_xau_structural_utility_labels.py \
  tests/test_entry_v10_train_defaults.py \
  tests/test_xau_direction_repair_pretrain_audit.py \
  tests/test_entry_replay_mfe_protect.py \
  tests/test_entry_candidate_replay_trade_log.py \
  tests/test_v12_smart_entry_live_gate.py \
  tests/test_xau_direction_repair_sweep.py \
  tests/test_entry_candidate_readiness.py \
  tests/test_entry_replay_readiness.py \
  tests/test_entry_foundation_smoke_bundle_audit.py \
  tests/test_entry_smart_seq520_smoke_readiness.py \
  tests/test_entry_smart_seq520_trainability_readiness.py \
  tests/test_entry_smart_seq520_smoke_manifest.py \
  tests/test_entry_candidate_train_wrapper.py \
  tests/test_entry_foundation_smoke_train_wrapper.py \
  tests/test_entry_smart_dataset_post_rebuild_readiness.py \
  tests/test_v10_6yr_rebuild_direction_repair_contract.py \
  tests/test_smart520_rank_reference.py \
  tests/test_smart520_state_contract.py -q
```

3. Get to a clean, intentional source state.
   - Review diff.
   - Commit/stash intentionally.
   - Do not revert unrelated user work.

4. Rebuild the fresh XAU dataset with training off.
   - Use `scripts/v10_6yr_rebuild_20260626.sh` only after clean-git requirements are satisfied.
   - Ensure it produces:
     - fresh rank-reference `.npz` + sidecar
     - train/val/test parquets
     - split manifests with `smart520_state_contract.rank_reference_npz_sha256`
     - `DATASET_BUILD_PROOF.json` with neutral bridge and XAU tape root.

5. Run XAU pretrain audit:

```bash
/home/andre2/venvs/gx1/bin/python -m gx1.scripts.audit_xau_direction_repair_pretrain_v1 \
  --dataset-dir /home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/v10_dataset_6yr_smartctx_xau_direction_repair \
  --stem v10_6yr_dataset__HOLD_03B \
  --out-dir /home/andre2/GX1_DATA/reports/xau_direction_repair_pretrain_audit_20260713_v1 \
  --data-splits train,val,test \
  --require-rail-features \
  --fail-on-audit-fail
```

6. Only after pretrain audit passes:
   - run smoke/readiness gates
   - run bounded XAU sweep dry-run
   - execute limited XAU sweep or candidate train
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
