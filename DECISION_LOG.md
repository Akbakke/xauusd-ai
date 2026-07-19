# GX1 current decisions

Historical implementation narratives were removed because they repeatedly
acted as stale authority. Git history retains them. This file records only
decisions that constrain the current model-native Entry work.

## 2026-07-17 — Entry success criterion is abstention quality, not direction precision

User vedtak (explicit choice over "restore promoted chain" and "continue
unchanged"): the model-native seq513 architecture stays, but the primary
empirical admission criterion is reframed. The June 2026 falsification
campaign proved directional accuracy has a hard information ceiling (~0.62,
12+ tracks refuted strict-OOT); the proven Entry-side edge was SELECTION (the
historical Entry-IQL raw_adv gate). Therefore:

1. The learned `FLAT` class / abstention surface must match or beat the
   historical Entry-IQL gate's OOT selection quality (EV-per-take at
   comparable coverage) before any candidate can advance. This is the primary
   Entry edge gate; direction precision beyond the proven ceiling is not a
   success metric and "near-perfect direction" is not claimable.
2. Before any large rebuild/training spend, a cheap decisive diagnostic runs
   on existing immutable evidence: can a learned abstention head reproduce or
   beat the historical gate's take/skip separation OOT? A failed diagnostic
   stops the lane for re-evaluation instead of burning a full training run.
3. Flat-starvation (zero FLAT predictions — the failure mode of every
   July 8-16 smoke) is the central training problem to solve, not a slice
   detail. Entry-IQL as a separate post-model stage remains retired; its
   selection ROLE lives inside the model.

## 2026-07-17 — audit wave: zero-reachability deletions and contract hardening

User-approved (git history is the backup). Deleted verified-dead surfaces:
`gx1/execution/telemetry.py`, `gx1/execution/v12_live_features.py`,
`gx1/features/feature_manifest.py`, `gx1/utils/pnl.py`,
`gx1/contracts/signal_bridge_v2.py` (+ its sole dead consumer
`assert_canonical_v2_schema`), `gx1/runtime/column_collision_guard.py` (+ its
sole-purpose test), the dead `PrebuiltStateLoader.get_latest_row` method, the
dead `GX1_SIGNAL_BRIDGE_VERSION` env write, and all empty migration-residue
directories (runtime paths `data/`, `runs/`, `gx1/live/logs` preserved). The
absence guard test now pins these as ModuleNotFound.

Hardened in the same wave: the 305-field mandatory causal-layer prefix ORDER is
now validated at every manifest consumer (membership alone no longer passes);
the five previously unvalidated `required_*` partition constants in the launch
JSON are enforced against code constants on the ALLOW path; the 90-second Entry
latency limit has one numeric owner (the runtime evidence contract).

## 2026-07-17 — historical Exit truth_e2e evidence producers are retired from test scope

The immutable April-2026 `truth_e2e_sanity` input locks were deleted from
GX1_DATA, so `materialize_exit_hold_exit_now_mdp_reward_contract_v1.write_artifacts`
and `materialize_exit_off_policy_eval_harness_v1.write_artifacts` can never run
again as-is, and per the handover the historical producer chain must not be
restored. Their two `write_artifacts` tests are removed; the modules REMAIN as
live library owners for the retained Exit-IQL chain
(`FORBIDDEN_STATE_FIELDS_V1`, `evaluate_policy`, `_exit_index_realized_exit`).
Fresh Exit evidence requires the new exact builder
(`BLOCKED_PENDING_NEW_EXACT_BUILDER`).

## 2026-07-17 — full-stack families cannot be ranked away

The 479-field specialist surface is now exactly 305 code-owned mandatory
outputs from ten registered causal feature layers followed by 174 fields from
deterministic TRAIN-only ranking. The emitted exact order and both components
are hash-bound. This retains genuine trend, session, liquidity, structure,
volatility, momentum, price-action, support/resistance and MTF evidence while
still rejecting redundant, dead, unrouteable or future-leaking aliases.

## 2026-07-17 — rebuild authority is artifact-bound

An Entry rebuild decision must survive beyond the shell wrapper. One validated
`--vedtak` is now required by both writing Python producers and is bound into
the rank NPZ, its sidecar, the dataset build proof, the model-native state
contract and every split manifest. Missing, placeholder or unequal IDs fail
closed. This change does not authorize or run a rebuild.

## 2026-07-17 — learned sizing is required execution evidence

The learned sizing head requires immutable calibration, XAU instrument/account
capacity inputs and TEST utility/exposure/drawdown diagnostics. Label-horizon
results alone do not grant capital authority. Paper/live remains blocked until
an exact joint adopted-active-Exit sizing replay and fresh post-adoption broker
runtime parity both pass. Historical fixed 1x is a comparison baseline only
and can neither satisfy admission nor act as fallback.

## 2026-07-16 — Entry launch is blocked

No existing bundle is accepted for current XAU direction. Fresh seq513 data,
bundle, calibration, immutable prediction/replay and train==serve evidence are
required; older evidence has no compatibility or launch authority.

## 2026-07-16 — one model-native direction path

Final calibrated `LONG/SHORT/FLAT` logits and their argmax are the sole Entry
direction authority. XGB anchors, neutral bridges, Entry-IQL, hand-written
trend/session/confidence/utility filters and compatibility fallbacks are not
permitted.

## 2026-07-16 — preserve the full evidence stack

Retiring a filter does not retire its information. Genuine multi-timeframe
trend, structure, liquidity, volatility, momentum, session/regime, chart,
candle, path and utility evidence must remain as live model inputs, targets or
supervised heads.

## 2026-07-16 — sizing is learned but not automatically capital authority

The position-size head is mandatory, trained, parity-checked and journaled.
Its calibration and label-horizon OOS controls are diagnostic. It receives
capital authority only after a separate exact joint adopted-Exit sizing replay
and fresh post-adoption broker runtime parity pass and are explicitly admitted;
otherwise Entry emits no order.

## 2026-07-16 — exact evidence or fail closed

Every authority boundary uses explicit immutable paths, hashes, exact schemas
and newest-terminal-event precedence. Missing, stale, mutable, malformed or
mismatched evidence blocks the path. Unit tests prove source contracts only;
they do not prove trading edge.

## 2026-07-16 — continuous source cleanup

Disconnected scripts, archived code, stale configs, sole-purpose tests and
obsolete Markdown are deleted after active-call/process/evidence checks. Active
Exit behavior and persistent data processes remain outside Entry cleanup scope.

## 2026-07-16 — one exact runtime evidence contract

The model-native decision, `TradeState`, trade journal and daily review must
validate the same complete immutable snapshot. Direction/logit/probability
parity, hierarchy, path, utility, calibration, MTF, all eight specialists and
the learned size head are mandatory. No consumer may fill missing fields or
accept retired Entry overlay evidence.

## 2026-07-16 — Entry freshness is immutable

Entry consumes exactly 96 rows ending at the latest closed M5 bar. The row has
a fixed five-minute availability lag, then a 90-second decision limit; the
canonical-cutoff age cap is therefore 390 seconds. The limits have no Entry
runtime override. Missing, wrong or late state yields no model direction, not
synthetic `FLAT`, an older cached row or backlog execution. Exit freshness is a
separate contract.

## 2026-07-16 — source completion does not prove edge

Legacy Entry branches and zero-reachability adapters, critics, duplicate
journal schemas, detached feature modules and manual sizing implementations
are physically removed. Launch remains `BLOCK`: no rebuild or training was run,
and no practical-precision claim is allowed before new immutable OOS,
live-like, cost and train==serve evidence passes.
