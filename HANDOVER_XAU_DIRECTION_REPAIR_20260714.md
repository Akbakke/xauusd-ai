# GX1 XAUUSD handover

Updated 2026-08-21. `scripts/gx1_handover.sh` is the executable status owner and
outranks this file — run it before relying on anything here. `GX1_RULES.md` is
binding scope; `CLAUDE.md` is the process constitution.

Chronological attempt and repair history was cut; git holds it. **Budget: 2,400
words** because the chain-binding checklist and split derivation are
load-bearing. This is a map, not a diary; cut history, never a checklist.

## Current verdict

Launch is `BLOCK`. There is **no admitted dataset, no model, no calibration, no
untouched-TEST result, no PnL and no win-rate proof**.

This repository is **offline-only**: no change, rebuild, audit or result here
authorizes paper, demo or live trading.

**V40 completed GREEN on 2026-08-21.** It wrote TRAIN/VAL/sealed TEST
(283,787 / 76,577 / 6,556 rows), seven compact lifecycle files in 29.0 MB and
passed preflight, cross-surface v3, full-input liveness and pretrain v6. The
report-only post-rebuild gate is READY and did not open TEST bytes. Foundation
feature, target-v4 and specialist audits pass on exact TRAIN/VAL bytes: all
eight specialists, every routed feature and every serialized active target are
finite and live. The exact output surface is fitted-Q plus sizing, dip,
forecast, timing, tail-risk, volatility, side-MAE and trendline-event heads;
legacy direction/tradable/path/bad-path/MTF heads are blocked. A stale audit
split that simultaneously advertised those retired heads and blocked
`mtf_direction` while naming it independent was removed; trainer and gates now
reject it. On 2026-08-23, after the committed report-only control-cap repair
`aeed9a77`, V40 published a fresh immutable smoke manifest, smoke readiness,
trainability readiness and smoke recipe for distinct run
`V40_SMOKE_20260823T153000Z`. All four are green; the recipe is `PASS` with
`execution_allowed=true`, but `activation_authority=false`, and the exact
wrapper `--dry-run` passed. Its only permitted next execution is the bounded
research smoke: CUDA, one epoch, 10,000 TRAIN rows, batch 64, 4G RAM and
512M swap. One explicit `--execute` attempt on 2026-08-23 reached the capped
runner but stopped at its preflight (exit 75), before a Python trainer or CUDA
context was started, and created no bundle. A recipe pass is not machine
telemetry: on this WSL host the current trainer preflight fails closed before
the trainer starts because the RTX 3090 reports `temperature.memory=N/A` and
its configured power limit is 390 W, above the 250 W policy. Commit
`0b5cde21` resolves WSL's system-owned `nvidia-smi` path but intentionally
does not weaken either canonical condition. The canonical smoke is therefore
not executable until there is a real telemetry/power solution. Following an
explicit operator decision on 2026-08-23, the source also contains one
separate `model-native-attended-smoke-train` route for an operator-present
diagnostic only: CUDA smoke only, 300-second hard wall, 75 C core cutoff,
250 W **actual draw** cutoff, 1-second polling, and all existing 4G/512M,
one-job and one-physical-core protections. It may accept only the literal WSL
memory value `N/A` while all other telemetry remains numeric; it permits a
configured limit up to the observed 390 W only because the 250 W actual-draw
kill remains active. It must be invoked through that dedicated route, never
overnight or unattended. Its bundle records `execution_tier=attended_only` in
both metadata and lock; smoke-bundle audit rejects that tier, so it cannot
enter candidate, TEST, promotion, paper or live paths. The attended route is
not a hardware solution and does not make the canonical route safe.

The first attended train smoke on 2026-08-23 proved the guard but did **not**
reach CUDA model work: before its 300-second wall-clock stop it began building
the complete 26.3 GB TRAIN nested-array memmap, because the immutable full-TRAIN
normalization fit is correctly upstream of the 10k optimization subsample. It
published no bundle and its PID-bound scratch was removed. Do not relabel that
as a GPU-training result or weaken the full-TRAIN normalization contract merely
to make a smoke faster. `entry_v10_ctx_train_v3.py` now translates the attended
guard's `SIGTERM` into normal Python unwinding so its `TemporaryDirectory`
cleans that regenerable scratch before the guard's KILL fallback.

For an actual CUDA architecture/thermal check without that full-data I/O,
`model-native-attended-hardware-smoke` is a separately named, source-bound,
operator-present route. It constructs the exact Entry architecture and
specialist routing and executes one deterministic CUDA forward/backward/AdamW
step on contract-valid **synthetic** tensors. It reads no TRAIN/VAL/TEST
parquet, writes no files, uses the same 4G/512M, five-minute, 75 C,
250 W actual-draw and one-second guard, and has `authority=none`. It cannot
stand in for a data smoke, normalization fit, candidate, edge claim, TEST,
promotion, paper or live evidence.

The prerequisite for a compact, data-backed smoke was independently verified
on 2026-08-23 by the source-bound full scanner
`audit_entry_sequence_roll_v1`: TRAIN proof
`sequence_roll_audit/ENTRY_SEQUENCE_ROLL_AUDIT_20260823T172949Z.json` passed
for all 283,787 rows (`chain_sha256`
`02b03e8cce2d1dd736efedacb2d14547d410501a2e2bc73b8d2efb1b4c409ae1`) and VAL
proof `sequence_roll_audit/ENTRY_SEQUENCE_ROLL_AUDIT_VAL_20260823T173400Z.json`
passed for all 76,577 rows (`chain_sha256`
`f9f2650c512cf0229aa3d3078e8cc95eb9acee6fdc554e2cfa5b9c34ffc2f9ea`).
Each proof binds its exact parquet and manifest SHA-256 and verifies every
finite float32 value, every `seq[-1] == snap` equality, every adjacent roll
and every Arrow-batch boundary. This permits a future smoke-only loader to
reconstruct the identical sequence view from the 96-row prefix plus snapshots;
it does not authorize a candidate shortcut or reduce the full-TRAIN
normalization population.

The separately named V40 adoption-candidate report is intentionally
`BLOCKED_MODEL_NATIVE_ADOPTION_REVIEW`: the fitted-Q target is explicitly
gross, spread-inclusive, research-only and cannot gain production authority
until causal executable bid/ask, commission, slippage, financing, portfolio
replay and related economics evidence exists. This blocks activation, edge
claims and all paper/live use; it does not make a research smoke invalid. V40
is therefore not admitted for a candidate, calibration, TEST evaluation or
trading, and the smoke must not start without explicit operator approval.
V34–V39 remain invalid.

Everything below the chain is unchanged: no model has ever been trained on this
substrate, and `train==serve` has never been proven (see below).

The evaluation reference is the coin-flip null and it is **substrate-specific**.
The −13.16 / −18.58 bps pair carried until 2026-08-19 was measured on the retired
V27 snapshot and does not transfer. The V31 figures null **−1.87 bps**, oracle
**+23.84**, skill **+25.71** are *reported, not re-derived*: no hash-bound
artifact carries them and no coin-flip owner exists in source. **Re-measuring
them on the admitted rebuild is a precondition of the pre-registered test**, not
an afterthought.

- **Source authority**: pair generation
  `53cba4593471be7532b03a165243506b1add8453886b37a01aca7fb7da4668f7`, published
  2026-08-20 under the existing `CANONICAL_V3_BASE28_BUILDER_V7_20260818T153858Z_GENERATIONS`
  root. It exists because the v34 rename made the previous generation unusable:
  `ema20_slope` / `_v1_ema3_ema6_spread_frac` are gone from its `canonical_v3`
  and the `_atr` spellings are present. Its `base28.parquet` is bit-identical to
  the previous generation's — only the derived surface moved. The previous
  generation `1f9424d8…` is still on disk and is the parent of the V31/V32
  chains; do not reclaim either without a hand-built parent-pointer proof
  (rule 9 — the retention owner cannot do it, see below).
- **Retired, no authority**: the V28 (513) and V29J (592) datasets, and every
  V31/V32 chain root. Nothing was ever trained on any of them.
- **Seed variance flips collapse direction.** Single-seed judging is invalid: one
  three-seed run produced no-collapse, FLAT-drift and LONG-collapse from the same
  recipe. Treat >=5-seed agreement as a gate before any edge claim.

## Current feature architecture

The same eight feature owners, one implementation each, run independently at native M5 for
Entry and native M1 for Exit. Entry reads a local M5 sequence plus closed
M15/H1/H4/D1 context; Exit reads a 480-bar M1 sequence plus closed M5/M15/H1/H4/D1, the
frozen Entry-decision token and its causal in-trade path. The architecture is in
`SYSTEM_MAP.md`; it is not repeated here.

**No width or schema version is restated in this file** (rule 4/13). The counts it
used to carry were stale by 88 fields within two days, and the surface moved again
on 2026-08-19/20. Read them:

```bash
.venv/bin/python -c "import gx1.contracts.entry_model_native_signal_v1 as s, gx1.features.htf_features as h; \
  print('signal', s.MODEL_NATIVE_SIGNAL_DIM, s.MODEL_NATIVE_SIGNAL_SCHEMA_VERSION); \
  print('ctx', s.MODEL_NATIVE_CTX_CONT_DIM, s.MODEL_NATIVE_CTX_CAT_DIM); \
  print('mandatory', s.MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT); \
  print('candidates', s.MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT); \
  print('per_tf', h.MULTI_TF_FEATURE_COUNT_V4, h.HTF_V4_MATRIX_CONTRACT); \
  print('cache', h.HTF_V4_CACHE_SCHEMA_VERSION); \
  [print(' fam', n, len(f)) for n, f in s.MODEL_NATIVE_MANDATORY_FAMILY_FEATURES]"
```

One UTC trading-session clock phases H4 bins on 22/02/06/10/14/18 UTC and D1 at
22:00 UTC. Relevance is learned — no handwritten confluence vote or timeframe
weight exists.

## What is implemented

- **The v34 surface generation** (`b11ec2b2`) repaired eleven fidelity defects.
  Classes worth retaining: producer-emitted retired duplicates; features that
  could not express their names; denominators leaking direction; and mislabeled
  quantities. Nine volatility-coupled fields moved from IQR ratios 1.28–1.94 to
  0.966–1.064. Exact measurements live in the commit, not here.
- **Three owner-divergence blockers closed:** stale trainer mode/width literals,
  a chain/readiness status-version split and a parity call with an invalid null
  bundle. Tests now derive values from owners instead of pinning restatements.
- **Two crash classes removed.** Sample-inferred binaryness could classify a
  partially observed ternary as binary and reject its third value at serve time;
  that branch is gone and nonzero legacy `binary_mask` fails closed. Chain warm-up
  now covers the 252-bar daily receptive field instead of only 96 M5 rows.
- **The objective is fitted-Q in basis points, not classification.** Decision
  authority is masked raw-bps MSE and unique `entry_action_q_bps` argmax; task
  weights are learned. One BCE remains only on the `trendline_event` auxiliary.
- **The trendline registry was entirely dead and is now alive** (`55148a3b`):
  29 of 31 fields were constant or all-NaN on every lane, now 0 of 31. A full
  field catalogue exists, every verdict attacked by an independent refuter — 143
  were overturned by that attack.
- **The cross-surface audit now uses the dataset's exact decision population.**
  V35 exposed that a full physical surface includes valid warm-up rows before
  the signal manifest's research history. Contract v2 validates those rows but
  excludes them from local/MTF equality hashes and projection. Regression tests
  reproduce the original insufficient-D1-history failure and fail closed on a
  tampered population boundary.
- **The model-native scalar owner now survives disk publication exactly.** Cache
  v30 stores a hash-bound float32 scalar matrix per TF. The M1 producer no longer
  recomputes Wilder/EMA state from a shorter history, and policy v3 declares only
  owner-derived current-value/history-lane aliases on both local input paths.
- **The native enriched producer now consumes its Group-A warmup contract.** It
  trims only the declared whole-row causal prefix after attach and before the
  all-finite output gate; a later non-finite row fails closed.
- **Lifecycle storage is compact and source-derived.** Episode rows persist
  scalar M1 pointers, not five repeated 512-state Python lists; validation
  re-derives every state clock and hashes the compact population contract.

## What remains empirically unproven or unadmitted

- No admitted dataset, model, calibration, edge, PnL or win-rate. V40's bytes
  pass the current foundation audits and the current-source report-only smoke
  chain, but its bounded research smoke is unexecuted and production admission
  is deliberately incomplete. The M5
  diagnostics covered 477,229 rows; all 67 TRAIN-fitted candidates were finite,
  live and non-duplicate on 283,902 TRAIN rows, with top absolute diagnostic
  Spearman only 0.023966. This is weak univariate signal, not an edge verdict.
- **Known and deliberately left un-repaired**, recorded so a later FAIL cannot be
  blamed on them retroactively: `mtf_level_bars_since_break` is an exact
  duplicate of `|…_signed|` on all six lanes; the level registry breaks with **no
  confirmation band** (`if close > centre`) and fires on ~19% of bars on every
  clock from M1 to D1 — a property of `SWING_LOOKBACK=3`, not of the market, and
  the rule systematically deletes the levels nearest to price;
  `volatility.squeeze_active` occupies ~87% of bars with no `var0<var1` guard in
  the admission gate; and `level_recurrence_threshold_atr` spans four orders of
  magnitude across seven lanes, with the D1 value fitted on 14 observations.
  The V34 unit repair also made **49 columns bitwise identical** between the
  local M5 surface and the per-TF M5 lane. This is no longer invisible: the
  pre-build cross-surface full scan hashes every active Entry-M5 and Exit-M1
  input against its actually routed MTF last-closed values. Entry excludes M5
  from its MTF route, so those 49 pairs are reported as inactive physical
  overlap; any undeclared duplicate on an active route fails closed. V36's v2
  RED motivated the repair; V40 supplies the fresh v3 pass.
- **Six-clock squeeze**: the 2026-08-15 artifacts were absorbing under the
  runtime decoder on all six clocks — M1 emitted **one** release in 352,193 TRAIN
  bars. The cause was the decoder, not the parameters; fit and serve now share one
  causal forward filter and the old files fail closed at load. The artifacts are
  **TRAIN-window-bound and pair-bound**, so they refit whenever either moves; that
  is why five refits happened on 2026-08-19/20. Never name a squeeze path or hash
  in a document — this file did, and pointed at a superseded set while the chains
  bound a different one. Read the binding from the run's own V4 cache manifest.
  Before admitting a refit, check what the gate does **not**:
  `variances[low] < variances[high]` on every clock, and that the high state is
  not absorbing. The current set passes both, with 1–2 orders of margin and
  high-state runs of 19–26 bars.
- **train==serve is unproven and stays unproven.** Zero
  `MODEL_NATIVE_SERVE_PARITY` events exist (measured 2026-08-19/20). The gate's
  `bundle_dir=None` defect is fixed, so it can now reach a verdict — but there is
  no bundle and no prediction event for it to have a verdict *about*, so rule 6
  belongs in "not examined", never in "proven consistent". The three known
  source divergences are now closed in code, but not yet empirically admitted:
  (a) live no longer overwrites the canonical Wilder ATR with a partial-window
  SMA; (b) the three long-lookback HTF `ctx_cont` fields now delegate to the
  canonical V4 scalar owner and its last-closed projection, rather than a
  private SMA/`ewm`/epsilon formula; and (c) the live Entry MTF builder casts
  OHLCV to the cache's float32 convention before it calls the shared V4 owner.
  The next fresh bundle must still produce a real parity event over these exact
  bound bytes; source agreement is a repair, not evidence of a served model.
- Whether every static magnitude in the trainer is data-derived is **not
  examined**: the objective contract declares the handwritten-weight flags
  False, but nobody has swept the trainer.

## Coarse history

Several apparently good builds were later invalidated by substrate defects; one
early backtest's 74.7 bps EV did not transfer. v9–v19 retired roughly 280
handwritten votes and composites for raw primitives. Direction edge has been
**refuted four times**: June's information ceiling; an August walk-forward with
1/5 folds and −19.48 bps utility regression; a GBM with 0/5 folds over coin flip
and worse OOS log-loss than a constant prior; and a horizon sweep with no passing
horizon. All measured all-bar averages, which the frozen selective-edge test is
designed to correct. The durable assets are the rules and fail-closed gates.

## Machine and process safety

Every heavy producer, audit, train or replay enters through
`scripts/gx1_capped_run.sh`: one job at a time, CPU affinity 0-1, 512 MiB swap,
4G for audits and tests, at most 20G for the heavy dataset producers
(`--class producer`) and 20G for the canonical trainer — this file said 10G for
producers until 2026-08-19; `scripts/gx1_capped_run.sh` is the authority. A cap
kill or partial directory is failed evidence.

The 2026-08-23 smoke preflight additionally measured that WSL exposes CPU 0-1
as the two hyperthreads of **one** physical core, and the runner pins every
heavy job there, sets common numerical libraries to one thread, and lowers CPU
and I/O priority. WSL exposes neither `lm-sensors`, `/sys/class/thermal` nor
CPU frequency control here, so there is **no CPU-temperature cutoff** to claim.
CUDA trainer work does have a fail-closed 20-minute wall-clock plus 2-second
GPU telemetry guard (78 C core, 90 C memory and 250 W configured power limit);
missing telemetry terminates its process group. Treat the unobservable CPU
temperature as a residual machine risk, not as evidence that it is safe.
The current RTX 3090/WSL telemetry is specifically insufficient: core
temperature is available, but `temperature.memory` is `N/A`, and the driver
reports a 390 W configured limit. The guard has been proven to stop before
trainer launch on this condition. Do not bypass it, claim the 90 C VRAM limit
is observed, or alter the host power limit without an explicit operator
decision and new evidence. On 2026-08-23 the operator explicitly authorized a
250 W setting attempt; `/usr/lib/wsl/lib/nvidia-smi --id=0 --power-limit=250`
returned `Insufficient Permissions`, and the readback remained 390 W. That is
not a successful cap change and does not authorize weakening the canonical
guard. The only narrowly-scoped exception is the separate, source-bound,
operator-present attended-smoke route described above; it tolerates exactly
`temperature.memory=N/A`, keeps a 250 W actual-draw kill and is permanently
blocked from downstream evidence. NVIDIA's CUDA-on-WSL documentation confirms that
NVIDIA's CUDA-on-WSL documentation confirms that
WSL NVML does not support all queries and identifies `/usr/lib/wsl/lib/nvidia-smi`
as the WSL path. The Windows-host `nvidia-smi.exe` was also read-tested from
WSL on 2026-08-23 and failed with `UtilAcceptVsock` rather than yielding sensor
data, so it is not a bridge. A later remedy must provide independently
measurable host-side VRAM telemetry to the guard; fabricated, cached or
caller-selected readings are forbidden.

The attended route now has one additional, source-bound **storage-only**
optimization. It accepts one immutable full-split rolling-identity audit for
each of TRAIN and VAL and re-hashes both the parquet and split manifest before
using it. Only when the proof says that every 96-bar `seq` window rolls exactly
from the preceding `snap` does the loader build a zero-copy sliding view over
the first 95 bars plus every snapshot, rather than writing a regenerable
~26 GB TRAIN sequence memmap (and the corresponding VAL mirror). The normalizer
still reads the complete TRAIN population; there is no sampling, feature
transformation or model shortcut. This is valid only with
`model-native-attended-smoke-train` plus explicit
`--train-sequence-roll-audit-json` and `--val-sequence-roll-audit-json` paths.
Canonical smoke and candidate/full training reject those flags and retain the
materialized route. A proof may never be treated as model, OOS, PnL, candidate,
TEST, promotion, paper or live evidence. Because the trainer and wrapper are
source-bound by the recipe audit, create a fresh immutable recipe audit after
this code change before using the attended route.

**Observed attended data-smoke result, 2026-08-23:** source commit `8107e13f`
passed its hook and the fresh recipe audit
`train_recipe_attended_data_20260823T174924Z/...174944002159Z.json` passed.
The real CUDA attended run then passed guard/preflight and logged trainer
configuration, but did **not** reach a GPU model batch: a read at 3:10 showed
47 C core, 34.75 W draw, 5% GPU utilization and 485 MiB VRAM. At 4:14 its
isolated cgroup was at 4,279,123,968 / 4,294,967,296 bytes with 1,483
`memory.events:max` events (no OOM event at that instant); it ended before a
manual SIGTERM could be delivered and emitted no bundle. The final exit reason
was not captured, so call this **memory-pressure termination/suspected cap
failure**, not a proven OOM. It did not alter V40 bytes and left no V40
sequence-memmap scratch. Do not rerun it unchanged: full Exit/M1 lifecycle and
normalization preparation still exceed the 4G attended envelope before GPU
optimization. The next engineering task is a source-bound, full-population
streaming representation for that remaining lifecycle/normalization surface;
never relax the cgroup or skip the full population to make a smoke green.
Deletions under `/home/andre2/GX1_DATA` go through the retention owner only.

## What will stop you when you run a chain

Six chain attempts on 2026-08-19/20 failed before any compute. Every one was a
binding, not a computation, and every one failed closed in seconds. Check all of
these **before** launching, not after:

1. **Squeeze TRAIN window** must equal the chain's `--train-start/--train-end`.
   Move TRAIN, refit the six clocks. No exception.
2. **Squeeze pair binding** must equal the pair you pass — and the check compares
   the recorded `pair_manifest_artifact` **path**, resolved, not just its hash.
   Fit the squeeze against the *generation-local* `PAIR_MANIFEST.json`, never a
   copy you placed elsewhere, even when the bytes are identical.
3. **Pair manifest** passed to the chain must be the generation-local one.
4. **Worktree must be clean.** The chain binds HEAD; commit first.
5. **Event root must be empty.** A part-built root is never resumed (rule 7).
6. **The pair's `canonical_v3` must carry the current field names.** A base-block
   rename invalidates the pair. Check with `pyarrow` before spending two hours.

The runnable pre-flight is: replicate the gate's own comparison, field for field,
from its source. Verifying the three fields you assumed it checks is how attempt
five failed after attempt four's "ALL BINDINGS OK".

`--registry-fit-train-end` defaults to `--train-end`: one origin. Do not pass it.
`--vedtak` is **inherited** from the native tapes' `explicit_vedtak_id`, never
chosen. A canonical-pair rebuild needs `GX1_V10_MULTI_TF_V4_CACHE_DIR` pointing
at a cache whose manifest carries frozen v29 registry constants — it has no
default by design; the env propagates because `gx1_capped_run.sh` uses
`systemd-run --scope`.

## Next implementation sequence

1. ~~Land the repair wave as one surface generation.~~ Done 2026-08-20,
   `b11ec2b2` (v34 surface: fidelity repairs, three chain blockers, doc truth
   pass) and `e69ab0fb` (sealed-JSON bound derived from the tape).
2. ~~Rebuild the canonical pair on the v34 owners.~~ Done, `53cba459…`.
3. ~~Run the fresh audit-v6 successor chain; never resume V34–V39.~~ V40 is
   terminal GREEN. Its foundation audits and current-source report-only smoke
   chain now pass; the exact smoke wrapper also passed `--dry-run`. Do not
   rebuild V40 merely to change a report-only consumer. Keep actual GPU
   training stopped until the explicitly bounded smoke is approved.
4. **Split, and why it is what it is.** TRAIN `2021-06-01 → 2025-05-31` (4y),
   VAL `2025-06-01 → 2026-06-30` (13 months), TEST `2026-07-01 → 2026-08-04T07:50`.
   Four years is a floor, not a preference: below two years the normalization fit
   raises `[ENTRY_INPUT_NORMALIZATION_UNSCALEABLE]` on seven constant D1 fields
   and the trainer cannot start. 2020 is excluded — spread p90 12.46 bps against
   1.78–2.82 everywhere else. **VAL is 13 months because 1 month could not answer
   the question**: 30 days is ~480 independent label windows, 1σ ≈ 2.3pp, against
   an effect size of ~1.6pp. Thirteen months gives ~6,200 windows, 1σ ≈ 0.64pp.
   VAL also spans the 2025–2026 volatility expansion (median M5 bar range ~2.5 →
   ~5.1 USD) by design, so it tests regime transfer, not just fit.
   Derivations: `docs/TRAIN_WINDOW_WIDENING_20260819.md`.
5. ~~Re-measure every field and target against real TRAIN/VAL bytes.~~ Feature,
   target-v4 and specialist audits pass. The report-only smoke gates and
   wrapper dry-run also pass; the next step is the single approved bounded
   research smoke, not a candidate or full training run.
6. **Run the pre-registered test in
   `docs/PREREGISTERED_DIRECTION_TEST_20260820.md`.** It was
   written before the dataset existed and must not be edited after seeing a
   number. Its central correction: all four previous refutations measured
   *average accuracy over all bars*, which is nearly guaranteed to answer "no"
   whether or not an edge exists — a model that abstains on 92% of bars and has
   real edge on the remaining 8% is invisible in that average. The test asks for
   a selective-edge curve against a re-derived coin-flip null and an
   autocorrelation-preserving (circular-shift) floor, with the decision rule
   fixed in advance.

## Takeover

```bash
bash scripts/gx1_handover.sh --check
bash scripts/gx1_handover.sh
git status --short --untracked-files=all
git worktree list --porcelain
```

Then read `GX1_RULES.md`, `AGENTS.md`, `SYSTEM_MAP.md` and
`docs/DATA_CONTRACT.md`. Do not infer state from old run directories.
