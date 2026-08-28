# GX1 XAUUSD handover

Updated 2026-08-28. `scripts/gx1_handover.sh` is the executable status owner and
outranks this file — run it before relying on anything here. `GX1_RULES.md` is
binding scope; `CLAUDE.md` is the process constitution.

Chronological attempt and repair history was cut; git holds it. **Budget: 2,400
words** because the chain-binding checklist and split derivation are
load-bearing. This is a map, not a diary; cut history, never a checklist.

## Current verdict

Launch is `BLOCK` for admission, demo, paper and live operation. There is **no
admitted dataset, no model, no calibration, no untouched-TEST result, no PnL
and no win-rate proof**.

This repository is **offline-only**: no change, rebuild, audit or result here
authorizes paper, demo or live trading.

**Current V46 status, reconciled 2026-08-28:** `V46_20260825T170935Z` is the
current hash-bound research dataset in `PROJECT_STATE_xau_direction_launch.json`.
It has PASS evidence for all eight feature families, source-backed sequences,
TRAIN-only normalization and the repaired M5-decision-to-M1-fill causality
audit. Its adoption review is deliberately BLOCKED by fitted-Q production
economics. That blocks production-net edge claims and every paper/demo/live
route. A gross-research candidate/OOS path exists in source, but is not yet
available in practice: it still requires a successful canonical smoke bundle,
a fresh candidate and an untouched TEST replay.

**Current execution status, 2026-08-28:** canonical V46 smoke preflight passed
all TRAIN/VAL, eight-family and five-timeframe checks. The old batch-64 attempt
reached 24,277 MiB VRAM; commit `4754853a` repaired that allocation mismatch by
using the existing eight-row streamed Exit-episode path for canonical CUDA. A
fresh batch-32 recipe (`…canonical_cuda_smoke32_20260828T070400Z`, source
`c82ca0dc`) then repeated the complete preflight and reached the first forward
with only 8,873 MiB VRAM at 65 C. The native guard stopped it at 250.48 W
against the strict 250 W actual-draw boundary, before an optimizer step or a
bundle. This is a safe stop, not an OOM or a PC/WSL crash. The new
hash-bound trade-path reporter is ready for a later full candidate TEST replay,
but no PnL, win rate, MAE/MFE, drawdown, candidate or TEST result exists yet.

**Active CUDA safety truth:** `scripts/gx1_capped_run.sh` and
`scripts/gx1_guarded_trainer_exec.sh` are authoritative. They use the pinned
native WSL `nvidia-smi` executable, accept the observed 390 W *configured*
driver limit, and poll once per second. Canonical CUDA stops above 70 C core,
250 W actual draw or 12 GiB resident VRAM; WSL memory-junction `N/A` is logged
as unobserved rather than fabricated. The legacy host-bridge design is retired
as a canonical prerequisite. Later historical paragraphs that require a
physical 250 W driver setting or bridge-provided VRAM temperature do not
describe the active guard.

**Historical V42 status, verified 2026-08-25:** the explicit
`current_audited_dataset_evidence` binding in
`PROJECT_STATE_xau_direction_launch.json` rehashes the V42 rebuild terminal,
post-rebuild readiness, full input and feature-surface liveness, all three
foundation audits, smoke manifest/readiness, trainability, recipe and adoption
review. All data/feature reports named there pass, including exact routing for
all eight specialists. This establishes that V42 is an **audited research
dataset**, not an admitted dataset: its distinct adoption review is deliberately
blocked by the fitted-Q production-economics contract. Causal executable
bid/ask, commission, slippage, financing, gap/terminal treatment and portfolio
constraints remain unbound. Therefore no edge claim, candidate, calibration,
TEST evaluation, paper/demo/live action or activation is authorized. The V42
recipe's `execution_allowed=true` means only that its immutable research recipe
is internally coherent; it does not grant operator authorization to run it.
The older sizing-chain joint replay that subtracted a fixed 1 bp
commission/slippage proxy was retired on 2026-08-25: bid/ask quote delta remains
available only as a labelled research diagnostic, cannot return an economics
`PASS`, and cannot produce or load an adoption proof. The sizing authority now
also fails before artifact loading while the fitted-Q production-economics gate
is red. A successor needs immutable broker cost and financing facts plus a
shared-portfolio replay; no replacement cost number may be invented.
The local OANDA fill journal now preserves explicitly returned `pl`,
`financing`, `commission`, `halfSpreadCost` and guaranteed-execution-fee
components as broker observations. A missing/malformed field is recorded as
incomplete rather than zero, and half-spread is never double-counted against an
executed fill price. No such observed fill ledger exists for V42 historical
rows, so this is only the collection/verification instrument for a later,
separately authorized demo-evidence phase—not a V42 economics admission.

**Execution-causality verdict, verified 2026-08-25:** a separate bounded,
manifest-only audit now distinguishes the two Entry supervision paths before a
trainer may allocate a GPU. V42's frozen fitted-Q Entry-to-Exit lifecycle
passes its own M1 timeline: the decision is available at the closed M5 bar,
the entry uses the next authoritative M1 open, and the first Exit decision
comes only after one completed post-fill M1 bar. But the still-active
diagnostic and position-size auxiliaries share the older M5 ranking policy,
whose declared prices are `ask_close_t0` / `bid_close_t0`. They therefore lack
an exact decision-time-to-M1-fill binding. The immutable V42 report is
`audit/ENTRY_EXECUTION_CAUSALITY_AUDIT_20260825T052916Z.json`: `BLOCK`,
`entry_fitted_q_m1_fill_lifecycle_bound=true`,
`active_auxiliary_targets_m1_fill_bound=false` and
`future_causal_rebuild_required=true`. The canonical smoke/candidate launch
contract now requires a fresh PASS causality report and rejects this state
before it constructs a trainer or CUDA context. Do **not** train V42 merely
because its old report-only recipe was internally coherent. A successor must
rebuild every active auxiliary from exact M1 fill bid/ask quotes, bind the
entry and exit quote times in each split's evidence, then re-run this audit.
This discovery produces no model, epoch, PnL, win rate or edge result.

**Successor causal-label implementation, 2026-08-25:** commits `224beaec`,
`8db0c0fc` and `b2bccf24` add exact M5-decision-to-M1-fill/exit primitives, a
TRAIN-only direction policy, selected-side sizing ECDF, and the split-level
binding of those auxiliaries. The successor integration
migrates the ranker, signal manifest, split builder, preflight and
launch/audit consumers to this source: ranker now requires the pair-bound M1
parquet, a split drops labels whose exact M1 exit would cross its end, and the
causality audit refuses a legacy sizing payload whenever the ranking claims a
causal M1 contract. A causal candidate must also carry the complete
hash-checked ranker policy, the exact M1 diagnostic projection in each split,
and a sizing policy bound to the same M5 source, tape provenance, M1 source
and direction-policy hash. Unit and bounded real-source smoke checks pass. **No
dataset rebuild, training, CUDA work, TEST read, PnL or edge evaluation has
been run from this implementation.** V42 remains historical BLOCK evidence;
V46 is its fresh successor and has completed the bounded rebuild/audit chain
with the required PASS causality report. It is still not training-authorized
for a candidate because production economics remains unbound.

**Historical V40 completed GREEN on 2026-08-21.** It wrote TRAIN/VAL/sealed TEST
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
low-VRAM diagnostic only: CUDA smoke only, a 600-second data preflight followed
by a 300-second model wall, 70 C core cutoff, 180 W **actual draw** cutoff,
12 GiB NVML-use cutoff, 1-second polling, and all existing 4G/512M, one-job
and one-physical-core protections. It may accept only the literal WSL
memory value `N/A` while all other telemetry remains numeric; it permits a
configured limit up to the observed 390 W only because the 180 W actual-draw
kill remains active. It must be invoked through that dedicated route, never
overnight or unattended. Its bundle records `execution_tier=attended_only` in
both metadata and lock; smoke-bundle audit rejects that tier, so it cannot
enter candidate, TEST, promotion, paper or live paths. The attended route is
not a hardware solution and does not make the canonical route safe.

**2026-08-24 source-status correction:** a read-only V40 audit found that the
TRAIN-only level-registry fit could select a near/far threshold with a thin
branch (M5: 56 versus 46,199 fit observations; D1: 3 versus 165). The source
contract now requires every near/far branch to have at least the ceiling of the
square root of its own inner-fit and inner-selection population. This is a
population-scaled anti-tail-fit gate, not a fixed tuning value. The old V40
registry payloads intentionally fail the new schema validator, so **V40 is
blocked from any new train/smoke attempt until a fresh, immutable rebuild
produces branch-supported registry artifacts**. No rebuild was started by this
change.

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
parquet, writes no files, uses the same 4G/512M, five-minute, 70 C,
180 W actual-draw, 12 GiB NVML-use and one-second guard, and has
`authority=none`. It cannot
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

**V46 source-backed sequence repair, 2026-08-26:** the causal M1 lifecycle
correctly omits un-supervisable decision rows, so V46's emitted TRAIN/VAL rows
are not an uninterrupted snapshot chain and must not use the older
`sequence_roll` storage shortcut. This is not a missing-feature or target
problem: the immutable `m5_feature_base.parquet` is the original 238-signal,
causal M5 timeline from which every V46 `seq` and `snap` was built. The new
`model-native-sequence-source-reconstruction-audit` exhaustively binds the
split parquet, split manifest, M5 feature-surface parquet and its manifest,
then compares every stored 96×238 sequence and snapshot byte-for-byte with the
source surface. V46 proof results are TRAIN
`audit/ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_TRAIN_20260826T000000Z.json`
(248,028 rows; chain `b460ccdc891017a392d964c22c381557b5c474b0b719c6a0b30bb2d941dc9d54`)
and VAL
`audit/ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_VAL_20260826T000000Z.json`
(70,880 rows; chain `a61e194d30044d02f8ab4b45f3cdf4a9868b0e320b7843c624c04b43227d423b`).
Both are PASS. The attended smoke loader may use those proofs to obtain
windows from the source surface without writing the former 20+ GiB temporary
sequence memmap; it preserves the full-TRAIN normalization population and has
`data_reconstruction_only` authority. It is not a training, edge, TEST, paper
or live result.

**First V46 attended source smoke, 2026-08-26:** commit `13803b50` was run
through the attended-only CUDA guard with the observed 390 W configured limit,
an independent 180 W *actual-draw* stop, 70 C core stop, 12 GiB VRAM stop,
10 GiB cgroup memory limit, batch size 8, one epoch, and a 10,000-row
TRAIN-only smoke subset. The ten-minute data preflight passed: both source
proofs revalidated; TRAIN avoided a 22.67 GiB sequence memmap, VAL avoided a
6.48 GiB sequence memmap; the normalization fit used 1,659,115 physical
TRAIN context rows and zero VAL/TEST rows; all five 176-column MTF surfaces
were present. The guarded model phase built the model and reached its first
backward pass, then stopped cleanly at the actual-draw boundary before an
optimizer step completed (`power_draw` guard stop). This is a successful data
and storage-path smoke, not a completed training epoch, checkpoint, PnL,
win-rate, MAE/MFE, OOS, candidate, TEST, paper, or live result. The small
attended-session contract was retained for forensic continuity; the ordinary
smoke bundle was not produced. Do not raise the actual-draw stop implicitly:
any future model-phase experiment requires an explicit thermal/power decision
and a fresh immutable recipe.

**Current V46 attended 390 W guard verification, 2026-08-26:** the later,
explicit operator-present 390 W amendment was executed from fresh recipe audit
`train_recipe_audit_attended_390w_guardlog_20260826T083608Z/ENTRY_MODEL_NATIVE_SEQ513_TRAIN_RECIPE_AUDIT_20260826T083609772346Z.json`
against source commit `cc137327`. It passed the full source-bound preflight in
420 seconds (both sequence-source proofs, five 176-column causal MTF surfaces,
all eight specialist groups and TRAIN-only normalization over 1,659,115
physical context rows), then completed exactly two FP32 CUDA optimizer steps
at batch 8. The trainer recorded a 5,717 MiB CUDA peak; the sidecar recorded a
normal `child_status=0` exit and no guard brake. The private two-slot session
has `complete_optimizer_steps=2`; its declared authority is strictly
`research_trainability_only`, and the declared bundle path is absent. Thus it
is evidence that the repaired V46 data-to-model path can run within the
attended guard, **not** evidence of validation, a completed epoch, candidate,
edge, PnL, win rate, MAE/MFE, OOS, TEST, paper or live performance.

The matching canonical CUDA batch-64 recipe
`train_recipe_audit_canonical_cuda_smoke_20260826T084616Z/ENTRY_MODEL_NATIVE_SEQ513_TRAIN_RECIPE_AUDIT_20260826T084618615360Z.json`
also passed its exact dry-run. An intentional real canonical guard probe then
returned exit 75 *before* Python/data/model launch and left its bundle path
absent: the current WSL NVML read has `Memory Current Temp: N/A` while the
driver power limit is 390 W. Canonical execution therefore requires **both** a
physical driver limit at or below 250 W and a trusted host-side VRAM-temperature
telemetry bridge (or another execution environment with an equivalent real
reading). Do not make the attended exception canonical, accept a cached or
caller-controlled reading, or retry canonical training until both conditions
are actually evidenced.

**V46 bounded CPU recovery smoke, 2026-08-26:** rather than raise the GPU
actual-draw boundary after the CUDA guard stop, commit `c953f6fa` added an
explicit CPU-only attended recovery tier. It retains the exact V46 source
proofs, 10 GiB memory cap, 512 MiB swap cap, two-core cgroup, staged
600-second data / 300-second model limits, and the same non-promotable
authority. Its immutable recipe was
`train_recipe_audit_attended_cpu_20260826T073849Z/ENTRY_MODEL_NATIVE_SEQ513_TRAIN_RECIPE_AUDIT_20260826T073850991678Z.json`.
The full data preflight passed again and reached the exact model-native
forward/loss/backward/optimizer path. It completed exactly one CPU optimizer
step, saved the resumable session checkpoint
`ATTENDED_RESEARCH_SESSION_ACTIVE.json` (state SHA-256
`1d3714f23c16cd4451433e3961b7e045eaa85c5bdefea0819ed9de466d3e9d9d`),
then paused normally. Peak trainer RSS was 5.33 GiB. This proves one true
optimizer boundary without CUDA load, but the one-step session is explicitly
partial and cannot be treated as a bundle, validation, PnL, edge, OOS,
candidate, TEST, paper or live result. The cache may expose read-only MTF
NumPy views; the loader now copies only each per-sample MTF window to writable
float32 before PyTorch conversion, with a deterministic read-only test.

**V46 canonical source-representation repair, 2026-08-26:** the two V46
source-reconstruction PASS proofs are now mandatory for every model-native
TRAIN/VAL launch, not merely attended smokes. The immutable training recipe
binds their exact bytes; the wrapper passes them on every smoke/candidate
route; and the trainer re-hashes them before loading the source surface. This
is only a storage representation repair: the 238-signal, 96-bar sequences,
targets, split boundaries, full-TRAIN normalization, five MTF surfaces and
eight specialist families are unchanged. It closes the prior canonical path
that would otherwise allocate the old 22.67 GiB TRAIN plus 6.48 GiB VAL
materialised sequence views. Candidate authority still requires its separate
readiness and smoke-bundle gates; no canonical full training, candidate bundle,
PnL, win rate, MAE/MFE, OOS, TEST, paper or live result has been produced.
The V46 proof-bound canonical smoke recipe is
`train_recipe_audit_canonical_source_20260826T080554Z/ENTRY_MODEL_NATIVE_SEQ513_TRAIN_RECIPE_AUDIT_20260826T080705100625Z.json`
(PASS; SHA-256 `a06519cccfda1a92a85ba5d4235f92732206b85e9d25dca0a65d62cef62bcc76`).
Its wrapper dry-run passed with `execution_tier=canonical`, source-audit hashes
in the trainer environment, the exact V46 TRAIN/VAL manifests and no output
bundle created. This is an admission/identity check only, not an optimizer,
validation or economic result.

The separately named historical V40 adoption-candidate report is intentionally
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

- No admitted dataset, model, calibration, edge, PnL or win-rate. V46's bytes
  pass the current foundation audits, source-backed sequence causality and
  report-only readiness chain, but its separately hash-bound adoption review is
  blocked on production economics. Its recipe therefore has no activation authority. The M5
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
`temperature.memory=N/A`, keeps the explicitly approved 390 W actual-draw
kill, adds a 12 GiB
NVML-use stop and a 50% CUDA allocator fence, and is permanently blocked from
downstream evidence. NVIDIA's CUDA-on-WSL documentation confirms that
WSL NVML does not support all queries and identifies `/usr/lib/wsl/lib/nvidia-smi`
as the WSL path. The Windows-host `nvidia-smi.exe` was also read-tested from
WSL on 2026-08-23 and failed with `UtilAcceptVsock` rather than yielding sensor
data, so it is not a bridge. A later remedy must provide independently
measurable host-side VRAM telemetry to the guard; fabricated, cached or
caller-selected readings are forbidden. The required nonce-bound,
signature-verified admission contract is
[`docs/CANONICAL_HOST_GPU_TELEMETRY_BRIDGE_CONTRACT.md`](docs/CANONICAL_HOST_GPU_TELEMETRY_BRIDGE_CONTRACT.md);
it is design-only until a separately authorized host-side implementation is
installed and verified.

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

**Pending bounded-loader repair, 2026-08-23:** the shared M1 feature-surface
loader now flushes and discards clean pages from its temporary disk-backed
maps in bounded increments, and scans categorical domains in bounded rows.
It preserves the complete source-bound float/int matrices and every existing
validation; it is not sampling or a changed feature representation. The full
test suite passed under the 4G/one-core audit cap. This repair has not yet been
measured on the V40 attended route, so it does not authorize a retry by itself:
create a fresh source-bound recipe audit and retain the existing 4G/300-second
guard before any future attended execution.

The recipe's exact source-binding set now includes the shared
`entry_exit_feature_surface_v1.py` owner as a distinct dependency of the
unified Exit lifecycle. A recipe created before that binding is insufficient:
the new audit must bind this loader's exact bytes, not merely the lifecycle
module that imports it.

**Second attended data-smoke, 2026-08-23:** after source commits `7ca258a5`
and `db9fe42d`, the fresh recipe audit
`train_recipe_attended_m1pager_20260823T182110Z/...183206467050Z.json` passed
and the wrapper dry-run proved the attended-only path. The controlled execute
reached the full `MULTI_TF_DECISION_WINDOW_COVERAGE` contract, which the first
attempt had not logged, and then stopped normally at its 300-second wall while
rehashing the complete 6.62 GB TRAIN parquet for the immutable sequence-roll
proof. The cgroup recorded reclaim events but zero OOM/oom-kill events; GPU
telemetry remained 48 C and about 35 W, so no CUDA model batch occurred. It
published no bundle, changed no V40 dataset/lifecycle bytes and removed its
new temporary M1 maps during SIGTERM unwinding. This is stronger proof of the
bounded M1-loader repair, not a training or hardware-success result. Do not
weaken the sequence hash or extend the 300-second guard implicitly: a later
staged preflight must retain the exact source identity before a distinct GPU
phase may consume it.

**Observed staged attended data/model smoke, 2026-08-24:** source commits
`11c6e63e` (verified V4 cache memory mapping/reuse) and `e8d26eaa` (pre-model
RSS telemetry) passed their hooks and the fresh immutable recipe audit
`train_recipe_attended_cache_rss_20260824T065600Z/ENTRY_MODEL_NATIVE_SEQ513_TRAIN_RECIPE_AUDIT_20260824T045627627308Z.json`
is PASS, source-bound to `e8d26eaa`. The cache repair preserves the exact
byte/hash checks but streams verification into read-only mappings, discards
clean pages after exhaustive validation, and passes the already verified cache
into full-TRAIN normalization rather than loading a second copy. A direct V40
measurement reported 0.46 GiB mapped cache backing and 0.17 GiB RSS after
load/validation.

The approved exact route completed the entire 600-second-capable
`data_preflight` in 405 seconds: immutable TRAIN/VAL hashes, decision-window
coverage, full-TRAIN normalization over 1,694,883 declared rows, both
sequence-roll proofs, all contracts and all eight specialist routes passed.
Its pre-model RSS was 3.22 GiB, versus about 4.05 GiB in the previous attempt.
The token-bound private FIFO then moved the same cgroup into the separate
300-second `model_smoke` phase. With the unchanged batch size of 64, four
complete Exit/Entry forward-backward-optimizer steps finished; RSS was 3.55 GiB
after the first full loss, 3.58 GiB during steps two/three and 3.63 GiB when
batch five began. Observed guard heartbeats were 54–62 C and 105–192 W, below
the fixed 75 C and 250 W actual-draw stops. The guard ended the live process
group only at `stage_model_smoke_wall_clock_limit_300s` (exit 75); there was no
cgroup OOM, temperature or power stop, no output bundle, no active trainer,
no attended FIFO/scratch directory and no V40 data write. This is a successful
bounded *trainability/safety* result, not a completed epoch, candidate, edge,
OOS, PnL, TEST, paper or live result.

One WSL kernel line, `dxgkio_make_resident: Ioctl failed: -12`, was logged
during the scope. It did not produce a trainer exception or override the
guard's time-limit termination, but it means GPU/WSL residency capacity remains
an observed platform risk. Do not enlarge batch size, duration, cgroup limits
or power limits on the basis of this smoke. The next work is to make the
bounded run resumable/observable and then assess a separately audited training
budget; it is not permission for full/candidate training.

**Implemented attended-session continuation, first bounded session observed:** the
trainer now keeps deterministic FP32 and the same V40 model/data/objective, but
the attended-only smoke route ends itself after a fixed four complete optimizer
steps rather than waiting for the five-minute guard to kill it. Every completed
step is written atomically to an inactive one of two local, hash-validated
session slots. The static session contract binds source commit, exact
TRAIN/VAL/M5/lifecycle bytes, normalization, smoke configuration and intended
bundle name; it refuses any source/data/recipe/output mismatch on a later
attended invocation. It persists online and frozen target models, optimizer,
EMA/scheduler, exact remaining batch permutation and all relevant RNG state,
but only at a completed optimizer boundary. The session lives beside the
still-absent bundle directory and is marked `research_trainability_only`; it
runs no VAL, selection or export and has no candidate/TEST/promotion/paper/live
authority. This is an implementation result only: no new V40 CUDA run, model,
edge, OOS, PnL or trading result exists. The first-batch logs now also include
complete Entry online/target time, Exit time, post-Exit backward time and peak
CUDA allocation; no BF16, TF32, autocast, compile, batch-size, cgroup, power or
timeout relaxation was introduced.

**Second observed V40 attended session, 2026-08-24:** the fresh V40 recipe
audit bound to source `44a253c6` passed before model construction. Its
research-only private session wrote four complete, hash-verified checkpoint
states and its active pointer recorded `complete_optimizer_steps=4`,
`next_batch_offset=4` and `complete=false`; the intended bundle directory
remained absent. This proves only durable bounded trainability progress: no
VAL, checkpoint selection, candidate, edge, OOS, PnL, TEST, promotion, paper
or live result exists. At the observed ten-minute status the trainer was below
the 4 GiB cgroup ceiling (about 3.64 GiB RSS), core temperature was 63 C and
actual draw 191 W, but reported VRAM was 24,260 MiB (near the 24 GiB device
ceiling). A further WSL/DXG `dxgkio_make_resident: Ioctl failed: -12` line was
then present in the kernel log. Treat that as a residency-risk event even
though the process exited and checkpoint state verified. The next source-bound
attended configuration therefore limits each session to two complete optimizer
steps, CUDA batch size 8, streams the Exit loss in 8-episode groups, fences the
PyTorch allocator at 50% of device VRAM, stops at 12 GiB observed NVML use and
stages checkpoint loads on CPU before CUDA restore. It must use a fresh output
path and recipe audit; the older four-step state is intentionally not resumed
under changed memory behavior. Do not increase batch size, session duration,
cgroup, power limit or GPU utilization from either observed session.

**Attended hardware smoke, 2026-08-23:** the separate no-data CUDA route
passed under the same attended guard. It constructed the exact Entry
architecture and specialist routing, completed one deterministic
forward/backward/AdamW step in 1.211 seconds and reported 163 MiB peak CUDA
allocation. Its output declares `authority=none`, `data_authority=none` and
all candidate/TEST/promotion/live flags false. It confirms only the guarded
GPU architecture path; it must never be cited as a data-smoke, edge, model,
normalization or trading result.

**Low-VRAM V40 historical measurement, 2026-08-24:** source `cc245139` and
the immutable recipe audit
`train_recipe_attended_lowvram_20260824T212239Z/...212240905051Z.json` passed.
The full V40 TRAIN/VAL identity and normalization preflight reached the
pre-model boundary at 3.83 GiB RSS. The attended-only CUDA fence was set to
50% (12,287 MiB), the exact batch-8 model constructed, and the first Entry and
8-episode Exit loss completed its logged forward/Exit-backward profile. The
outer safety guard then sent TERM for `power_draw` during the remaining model
backward, before an optimizer step or checkpoint completed. The intended bundle
directory stayed absent; only the non-promotable static attended-session
contract exists. After exit the GPU read 49 C, 35.52 W and 399 MiB with no
active trainer. This proves the new memory fence and automatic draw stop, but
not a completed training step, model, edge, OOS, PnL, win rate or backtest.
Do not raise the 180 W stop or retry the historical CUDA lane automatically;
first obtain an independently measurable host-side power plan.

**Eight-family operativity audit, V42, re-verified 2026-08-25:** this is a
read-only check of the immutable V42 artifacts, not a rebuild or training
result. The four report hashes pinned in `PROJECT_STATE_xau_direction_launch.json`
all rehash exactly. Its full-input contract
`ENTRY_FULL_INPUT_LIVENESS_CONTRACT_20260825T031933857454Z` validates as PASS
against V42's `dataset/`: all 310 model fields (238 signal, 71 continuous,
one categorical) are finite across all 283,787 TRAIN and 76,577 VAL rows;
306 TRAIN fields are LIVE and four are explicitly allowed rare events, while
304 VAL fields are observed-variable and six are observed rare events. None is
silently zero-filled or treated as ordinary numeric variation. The five-clock
MTF contract is separately PASS.

`ENTRY_FEATURE_SURFACE_LIVENESS_20260825T050000000000Z` full-scanned the
native sources after their required history: 467,154 M5 rows (9,871 prefix
rows excluded) and 2,292,408 M1 rows. For both clocks every field in every
family is finite and live: chart/geometry 33, momentum/flow 45,
price-action/candle 25, session/regime 19, SMC/liquidity 64, structure/swing
34, trend/EMA 65 and volatility 25 (310/310). The distinct specialist-routing
audit `ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_20260825T042226Z` is PASS with
zero dead or unmapped TRAIN signal fields. Its exact one-owner training-signal
partition is structure 28, SMC/liquidity 50, trend/EMA 44, volatility 19,
momentum/flow 33, session/regime 9, chart/geometry 32 and candle/price action
23 — 238/238 in total — and each count is live in both TRAIN and VAL. The
trainer rejects missing, overlapping or incomplete specialist indices before
model construction, then iterates all eight encoders, sends their tokens
through learned cross-family attention and a dynamic gate. Context routing is
also exact-one-owner. Therefore these are real data inputs and runtime model
paths, not inert config labels. A future causal successor must repeat these
audits on its own immutable output; V42's PASS cannot be inherited by V43.
The V42 report's qualified M1 and M5 field lists were also rechecked directly:
the same 310 names occur in the same order (list SHA-256
`0ae5da7e9cac3eb40e2d96a7368f820459a16debeee6564b588acc4d082f1020`),
with identical per-family lists. The successor feature-surface audit now makes
this cross-clock harmony a fail-closed, serialized contract rather than a
manual check.

This is intentionally **not** a claim that every family already affects a
decision. `specialist_out` starts at zero so the untrained model is neutral;
the first genuine training epoch must open the branch. A trained candidate is
then required to pass the separate held-out serve-parity ablations: zeroing
each specialist encoder and each of the five-timeframe-by-eight-family routes
must change enough decision margins. Until such a trained bundle exists,
eight-family decision influence, edge, serve parity and trading remain
unproven.

**Target-semantics audit, 2026-08-23:** the superseded V3 target audit failed
on an obsolete target-policy schema and must never be cited. The bound V4
artifact `foundation_target_audit_v4/ENTRY_TARGET_FOUNDATION_AUDIT_20260821T111346Z.json`
is PASS with zero failures and binds the same V40 TRAIN/VAL parquet and
manifest hashes as the current liveness/recipe path. Its static diagnostic
horizons are exactly 19 bars in both splits, but the contract explicitly
forbids a fixed-horizon direction or path label from Entry decision authority.
The sole Entry decision target is instead LONG/SHORT/FLAT raw-bps fitted-Q:
LONG and SHORT bridge only to a frozen TRAIN-fitted Exit target at the first
authoritative post-fill state, FLAT terminates at zero, exact ties fail closed,
and VAL/TEST never update the target snapshot. Future-looking auxiliary
supervision has declared horizons up to 96 bars, excludes incomplete tail rows
before emission, and every active auxiliary head is live in both TRAIN and
VAL. This proves target presence, target population and the declared
train-only target topology; it is not a model-quality result.

The same V4 artifact makes the production boundary explicit. Its fitted-Q
economics are `gross_spread_inclusive_research_only`, and the source contract
sets `production_authority_ready`, `candidate_ready_allowed`,
`bundle_serving_admission_allowed` and `edge_claim_allowed` false. Missing
evidence includes causal next-executable bid/ask, commission, slippage,
financing/swap semantics, elapsed-time/gap classification, an immutable net
cost/gap audit, and portfolio replay with overlapping-capital constraints.
Therefore a future model can be evaluated as research after the pre-registered
gates, but it cannot honestly be called tradable or be routed to paper/live
until that separate execution-economics program exists and passes.

**Runtime-scope audit, 2026-08-23:** the only active GX1 market process is
`v12_oanda_data_collector.py` (roughly 2–3% of one CPU core, no CUDA). It
requests only completed XAU_USD M1 candles, validates canonical bid/ask/mid
rows, rejects conflicting historical overlaps, atomically persists them under
`GX1_DATA/reports/v12_live_data_strict_m1_v1`, and latches on a source or
storage contract violation. At review time its failure latch was absent. It
does **not** call an order, trade or position endpoint.

The checkout does contain OANDA mutation primitives and a paper-runner for
future controlled work, so their mere presence must not be mistaken for an
active trading route. New Entry launch is deliberately frozen at the runner
boundary: its broker-entry lease, launch lease and live-tail admission all
raise `launch, broker and live-tail admission are outside the frozen offline
scope`; the public evidence control surface rejects `shadow`, `live` and
`start-live` commands. No paper/live runner process was present during this
audit. This is a source/runtime containment result only — it does not replace
the later execution-economics, candidate, paper or operator-approval gates.

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
4. **Direct dependencies must exactly match `requirements.txt` and import.** The
   chain checks its CPython 3.10 environment before pair/data work and fails
   before a producer starts on a missing, mismatched or unimportable package.
   Check manually with
   `python -m gx1.scripts.verify_rebuild_dependency_readiness_v1 --repo /home/andre2/src/GX1_ENGINE`.
5. **Worktree must be clean.** The chain binds HEAD; commit first.
6. **Event root must be empty.** A part-built root is never resumed (rule 7).
7. **The pair's `canonical_v3` must carry the current field names.** A base-block
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
   chain now pass. The one explicitly bounded attended smoke completed its
   data preflight and four optimizer steps within every cgroup, temperature and
   actual-power stop; a later session observed near-capacity VRAM/DXG residency
   risk and tightened its future bounded configuration. It is still
   research-only and does not authorize
   candidate or full training. Do not rebuild V40 merely to change a
   report-only consumer.
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
   target-v4 and specialist audits pass. The report-only smoke gates, wrapper
   dry-run and the bounded attended data/model smoke have passed their
   safety/data phases. Historical CUDA `--research-smoke` is suspended after
   a WSL/GPU reset: it held nearly all VRAM resident under a 24-hour watchdog.
   The long-running historical route remains disabled. The first low-VRAM
   attended data measurement safely stopped at the 180 W actual-draw guard
   before an optimizer step. Do not retry it or raise its draw threshold until
   an independently measurable host-side power plan exists. Do not reuse either
   attended private checkpoint session as a bundle output.
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
