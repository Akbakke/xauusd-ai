# GX1 project state

Updated 2026-07-21.

## Entry direction

Status: **BLOCK**.

2026-07-21 latest update: V12 rebuilt the source cascade and proved 385,677
rows x 188 columns with all 187 numeric fields live, then exercised the full
history fix: Group-A trimmed 2,207 causal warmup rows rather than V11's 13,714.
It produced train=357,519, val=5,676 and test=8,406 rows with exact 513+142+5
input schemas and complete path-quality labels. V12 was then deliberately
interrupted during full-input liveness, before any liveness PASS was written,
because the source/test cutoff was still 2026-06-14. Its terminal state is
`ABORTED`; it is diagnostic history, not a reusable dataset authority.

The running OANDA collector now supplies complete M1 candles through
2026-07-21. A read-only audit found 47,086 timestamps overlapping canonical
M1 with exact zero difference in every mid/bid/ask/volume field, 1,481
identical duplicate timestamps, no conflicting duplicates, no nonfinite data
and no bad OHLC/bid-ask geometry. V13 snapshotting exposed and rejected 13
unsupported partial M1 buckets, then was invalidated because its MTF cache used
the trimmed model-range source. V14 rebuilt fresh and now has source-cascade
PASS through 2026-07-21T17:00Z: 392,959 rows x 188 columns, all 187 numeric
fields live, exact full-v3 MTF ownership and no fallback. Its ranking/dataset
chain remains pending.

The repaired contract now supplies the full causal M5 prefix independently of
the decision slice, proves exact timestamp/OHLC inclusion, hashes that prefix
into checkpoint schema v2 and uses the same rule in live preparation. A real
January probe changed Group-A warmup from 13,714 rows to zero while preserving
finite D1 liquidity, ATR-term, dip and five-TF structure evidence. Live
HTF/REGIME_V4 is likewise computed on the complete prefix before the model
history slice. Commit `4134ca19` owns this repair; V11 remains terminal RED and
V12 remains terminal ABORTED and V13 is rejected source diagnostics only. V14
is the sole current source lineage; its fresh ranking/dataset chain is required.

The execution-path source repair is present: the immutable TRAIN-rank
reference is created before and bound into ranking, and ranking is owned by the
same chain as the dataset build. Every capped heavy command competes for one
host-wide exclusive lock, Group-A persists exact 4096-row chunks with
frame/MTF/field/run-window identity, the ranker and builder permit one exact
checkpoint retry, and terminal chain exits publish immutable schema-v4 events.
Focused, causal and full-suite tests cover the repair. V12 exercised it but is
stale/aborted, so it creates no accepted dataset or empirical authority.

The feature audit also closed four source mismatches before any new rebuild:
ranking now applies the same immutable TRAIN ECDF/ATR transform as dataset and
serve; the EMA layer uses exact common-history `close` and recomputed `atr`
with no schema fallback; signed BOS/sweep pressure is split symmetrically; and
the unprovable partial live MTF splice is removed in favor of an exact
zero-gap/full-refresh requirement. The mandatory specialist prefix is 316
fields across eleven families, including all 11 M5 EMA50/200 state/cross
fields; 163 positions remain deterministic TRAIN-only ranking-owned.

There is no accepted model-native seq513 bundle. The active source contract is
513 signals, 142 continuous context fields, 5 categorical context fields,
sequence length 96, five timeframes, eight specialists and one calibrated
`LONG/SHORT/FLAT` direction authority. Twenty-two positively supervised heads
feed one exact learned 26-group/96-value fusion (`96 -> 128 -> 3`).

The 513 signals are 34 code-owned base fields plus 479 specialist fields. The
first 316 specialist fields are every output from eleven registered causal
full-stack layers and are mandatory in exact registry order. Only the final
163 positions are chosen by deterministic TRAIN-only ranking. This prevents
feature selection from ranking away an entire trend, session, liquidity,
structure, volatility, momentum, price-action, support/resistance or MTF
evidence layer.

All earlier Entry datasets, bundles, reports and promotion records are rejected
by the current exact contract and cannot override the launch block.

Run lineage `XAU_SEQ513_REBUILD_20260718_V1` was used. The seq513 rebuild attempts
on 2026-07-19 were terminated and invalidated because their feature ranking
used TRAIN `2020-11-13..2026-03-31` while the active build requested TRAIN
`2021-03-16..2026-03-31`; the old preflight omitted that nested comparison.
No rebuild process is running now. No dataset, bundle or OOS candidate result
from those attempts is accepted, so no empirical precision or launch-readiness
claim exists. Partial outputs have no authority. The event-local
`CHAIN_STATUS.json` is now terminalized in schema v2 as `RED` with reason
`FEATURE_RANKING_TRAIN_WINDOW_MISMATCH` and bound hashes.

Preflight, wrapper and builder now require explicit ranking/manifest artifacts
and validate their lineage, Entry run ID, source hash and exact TRAIN window. This is
source-contract proof only, not proof of trading edge.

The target/evidence boundary is now stricter. Target foundation audit v2
requires all 46 canonical aux targets, including all `time_to_mfe` and
LONG/SHORT/FLAT Q targets; old v1 reports are rejected before readiness or
training. Immutable prediction evidence and smoke audit v3 require VAL/TEST
target alignment for the learned TOP/BOTTOM timing and Q/V/Advantage heads,
including supported high-precision near-turn pockets and reward-best Q
ranking. No such fresh prediction evidence exists yet, so this closes a source
pass-through without changing launch `BLOCK`.

Smoke v3 now reports only exact head-prediction, specialist-gate and strict
bundle-component liveness; the false duplicate claim that this proved absence
of pass-through was removed. Serve-parity v4 owns actual launch influence: all
eight specialists must move both raw and calibrated class margins under two
independent ablations, and every one of the 26 learned-fusion groups must move
both surfaces under immutable VAL-mean slice replacement on deterministic TEST
states. Continuous/categorical context and all five timeframes must also move
both surfaces under exact zero-mask ablation. Old parity events and any input
or group lacking raw or final movement fail
closed. No fresh v4 event exists, so launch remains `BLOCK`.

Smoke/candidate launch and the trainer now require explicit TRAIN/VAL/TEST
manifest and parquet paths. Exact hashes are emitted only by the validated
recipe, then rechecked with manifest self-path, run lineage and six-way-distinctness
contracts. VAL and TEST are never inferred from TRAIN names or directory
inventory.

The same immutable identities now bind foundation feature/target/specialist
audits, smoke/adoption review, selective-edge prediction, candidate replay,
serve parity and sizing calibration/OOS provenance. Audit parquets are resolved
only from explicitly supplied hash-bound manifests; downstream consumers read
the exact paths and SHA-256 values from their matching immutable reports. There
is no split glob, stem-derived manifest or unbound-directory fallback in the
Entry evidence chain.

No seq513 rebuild chain or training process is running. Active Entry has zero
imports from `signal_bridge_v1` and `signal_bridge_v3`;
`entry_model_native_signal_v1` is the exact owner of the 34 base, 142
continuous-context and 5 categorical-context fields. The retained V3 XGB
bridge is Exit-only and remains required by two real Exit consumers.

Live gate loading starts from the exact launch-declared event path/SHA rather
than a fixed-root newest lookup. A failed model `decide()` produces structured
direction unavailability and leaves the M5 bucket retryable; it cannot become
synthetic FLAT. Downstream pipeline/runner tests prohibit direction/action
mutation and post-model trend/session/utility/path threshold authority.

`PROJECT_STATE_xau_direction_launch.json` is the machine-readable Entry launch
decision. Both it and the artifact guard must admit the same immutable bundle
before Entry resolution can succeed.

## Evidence boundary

All real structure, trend, liquidity, volatility, momentum, session, price-
action, path-quality and utility evidence remains in the learned model. Old
post-model filters and manual sizing overlays are retired because they were
competing authorities, not because their underlying market information was
unwanted.

The eight learned specialists cover structure/swing, SMC/liquidity, trend/EMA,
volatility/compression, momentum/flow, session/regime, chart geometry and
price-action/candles. Their evidence is fused with hierarchy, MTF, path and
utility objectives before the final calibrated three-class argmax. None is a
separate live direction rule.

The learned size head is mandatory and has no implicit capital fallback. Its
logit and prediction must survive decision, state, journal and review parity.
Hash-bound calibration, the account grid and any label-horizon TEST controls
are sizing-head diagnostics only; no fresh accepted current-contract result
exists. The source admission path now exists: adoption requires a full-TEST
row-recomputed replay with a bound, contiguous per-M1 trace through the exact
registry-selected active Exit stack, and paper/live additionally requires fresh
post-adoption broker shadow parity. Artifact resolution validates both plus the
complete serve gates before returning `ALLOW`. No real joint replay, adoption
or runtime-parity event exists for a current Entry bundle, so execution remains evidence-blocked. Missing
proof means no order, never a silent multiplier `1.0`; fixed 1x is only a named
historical benchmark.

The report-only model-native abstention metadata run is
`BLOCK_ABSTENTION_EMPIRICAL_GATE`. It found balanced FLAT labels — TRAIN
`1400/4095` (`34.19%`), validation `530/1536` (`34.51%`) and TEST `516/1536`
(`33.59%`) — and positive active FLAT, utility and margin weights. It read zero
parquet and produced no learned predictions. Immutable historical selection-
benchmark bytes and exact learned-probe evidence are absent, so this proves
neither abstention quality nor direction edge and authorizes nothing.

Because the historical bytes do not exist, the unreachable pre-rebuild
abstention verifier/control route is deleted. Future abstention admission begins
with fresh immutable candidate TEST rows and requires a bound proxy comparison
plus absolute OOT/cost/live-like support; metadata cannot substitute.

The original Entry-IQL row/model benchmark was deleted incorrectly on
2026-07-07 after a dry-run protected it with abbreviated literal `...`
exclusions that the executor failed to resolve. The immutable incident facts
are recorded in `PROJECT_STATE_entry_iql_delete_incident.json`. Git can restore
the old IQL source at commit `7dc9241086f7e24ea0dad974dc58534ee158662f`,
but no Git object contains the lost model/parquet/row bytes. Salvaged JSON/MD
metadata cannot be promoted into that missing benchmark.

The corrected Super-AI source path does not resurrect the old separate
Entry-IQL policy or its hand-written live overlays. Expectile-V and full-
counterfactual `Q(s, LONG/SHORT/FLAT, K12/K48/K96)` are now positively trained
internal heads on the shared seq513 encoder. Q, V and exact `Q-V` Advantage
feed the single learned 96-value direction fusion and survive strict bundle,
serve-parity, runtime, state, journal and review contracts. This is source
completion, not empirical edge. Distillation is admissible only from a
hash-bound teacher that independently passes untouched OOT gates. Continual
adaptation is required to mean offline drift-triggered challenger training,
regime replay, shadow evaluation and explicit promotion/rollback, never live
online weight updates. That fail-closed state machine is now implemented in
source: same-bundle row-recomputed drift, replay-readiness v2 byte handoff,
offline challenger, zero-order shadow, explicit promotion and prior-incumbent
rollback are immutable transitions. Shadow promotion requires the incumbent
and challenger on identical hash-bound price paths, row-recomputed bid/ask PnL,
absolute side edge and positive lower-95% paired improvement. Launch requires a fresh
activating lifecycle event bound to the exact serve, joint Exit, learned-sizing
and runtime-parity evidence. No real lifecycle event exists, so this source
completion grants no authority and launch remains `BLOCK`.

## Runtime boundary

One exact runtime evidence contract is shared by the model-native decision,
`TradeState`, journal persistence/recovery and daily review. It accepts neither
missing auxiliary evidence nor retired overlay fields. Entry also requires an
exact 96-row window ending at the latest closed M5, a fixed 90-second decision
limit after the bar becomes available, and a fixed 390-second canonical-cutoff
age limit. Failure emits no direction and cannot be softened by an environment
override, cached row or synthetic `FLAT`.

Missing, invalid or session-inconsistent Entry context, including a fabricated
ASIA flag, yields `MODEL_NATIVE_ENTRY_CONTEXT_NO_DIRECTION`. No bridge, default
or cached context may repair that failure. Retained OANDA backfill writers are
a separate data-mutation scope with their own explicit write authorization.

## Exit

The retained Exit V3/Exit-IQL chain is a separate contract. Its XGB use and M1
exit semantics are not removed by the Entry cleanup. Shared helpers have
neutral or Exit-owned modules; active Exit math is unchanged.

The Exit-only V3 XGB bridge owns exact 7/41 field validation for two active Exit
consumers; both its import and ordered field contract fail closed. The retired
Entry-IQL registry record has `path=null` and status
`RETIRED_ARTIFACT_ABSENT`, so it cannot act as an Entry fallback.

## Next admissible milestone

The current source baseline passes the full repository suite with five skips
and zero failures. Narrow audits removed
dead stop-script branches, bound serving/train/downstream artifact identity,
and added a compact takeover fingerprint; these are source-contract changes,
not empirical edge. The missing historical row/model benchmark cannot be
reconstructed from its metadata and is no longer treated as a satisfiable
prerequisite. The next
source milestone is complete: internal Q/V/action-value heads, one final
fusion, canonical TOP/BOTTOM timing semantics, mandatory 46-target foundation
proof and exact train/export/serve target-alignment gates pass the full
repository suite. The
next empirical milestone is a fresh immutable proxy comparison plus absolute
untouched OOT,
cost/live-like, abstention-support and joint Exit/sizing gates. No historical
metadata is allowed to soften those gates. Only the complete source contract
may justify returning to the hardened seq513 rebuild runbook with a newly
matched ranking/manifest. The selected immutable lineage is
not preallocated: invalidated V1-V11 lineages cannot be reused, and the next
chain must allocate a wholly fresh immutable run ID that grants no authority;
only an accepted rebuild may advance to smoke. Zero FLAT predictions remains
hard-red. No accepted rebuild, training result or empirical precision result
exists yet.
