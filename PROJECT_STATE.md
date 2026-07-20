# GX1 project state

Updated 2026-07-20.

## Entry direction

Status: **BLOCK**.

There is no accepted model-native seq513 bundle. The active source contract is
513 signals, 142 continuous context fields, 5 categorical context fields,
sequence length 96, five timeframes, eight specialists and one calibrated
`LONG/SHORT/FLAT` direction authority. Twenty positively supervised heads feed
one exact learned 23-group/75-value fusion (`75 -> 128 -> 3`).

The 513 signals are 34 code-owned base fields plus 479 specialist fields. The
first 305 specialist fields are every output from ten registered causal
full-stack layers and are mandatory in exact registry order. Only the final
174 positions are chosen by deterministic TRAIN-only ranking. This prevents
feature selection from ranking away an entire trend, session, liquidity,
structure, volatility, momentum, price-action, support/resistance or MTF
evidence layer.

All earlier Entry datasets, bundles, reports and promotion records are rejected
by the current exact contract and cannot override the launch block.

Vedtak `XAU_SEQ513_REBUILD_20260718_V1` was issued. The seq513 rebuild attempts
on 2026-07-19 were terminated and invalidated because their feature ranking
used TRAIN `2020-11-13..2026-03-31` while the active build requested TRAIN
`2021-03-16..2026-03-31`; the old preflight omitted that nested comparison.
No rebuild process is running now. No dataset, bundle or OOS candidate result
from those attempts is accepted, so no empirical precision or launch-readiness
claim exists. Partial outputs have no authority. The event-local
`CHAIN_STATUS.json` is now terminalized in schema v2 as `RED` with reason
`FEATURE_RANKING_TRAIN_WINDOW_MISMATCH` and bound hashes.

Preflight, wrapper and builder now require explicit ranking/manifest artifacts
and validate their lineage, vedtak, source hash and exact TRAIN window. This is
source-contract proof only, not proof of trading edge.

Smoke/candidate launch and the trainer now require explicit TRAIN/VAL/TEST
manifest and parquet paths. Exact hashes are emitted only by the validated
recipe, then rechecked with manifest self-path, vedtak and six-way-distinctness
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
exists. Execution admission remains structurally blocked
until a joint sizing-only replay binds the exact adopted active Exit stack and
a fresh post-adoption broker runtime-parity event. Missing proof means no
order, never a silent multiplier `1.0`; fixed 1x is only a named historical
benchmark.

The report-only model-native abstention metadata run is
`BLOCK_ABSTENTION_EMPIRICAL_GATE`. It found balanced FLAT labels — TRAIN
`1400/4095` (`34.19%`), validation `530/1536` (`34.51%`) and TEST `516/1536`
(`33.59%`) — and positive active FLAT, utility and margin weights. It read zero
parquet and produced no learned predictions. Immutable historical selection-
benchmark bytes and exact learned-probe evidence are absent, so this proves
neither abstention quality nor direction edge and authorizes nothing.

The abstention verifier cannot accept free-standing JSON assertions. A
historical benchmark must equal the exact registered comparison artifact, and
learned rows must be rejoined one-for-one to an immutable candidate TEST
prediction event with matching report/predictions hashes, bundle, dataset,
UTC keys and recomputed model direction. Those inputs remain absent.

The original Entry-IQL row/model benchmark was deleted incorrectly on
2026-07-07 after a dry-run protected it with abbreviated literal `...`
exclusions that the executor failed to resolve. The immutable incident facts
are recorded in `PROJECT_STATE_entry_iql_delete_incident.json`. Git can restore
the old IQL source at commit `7dc9241086f7e24ea0dad974dc58534ee158662f`,
but no Git object contains the lost model/parquet/row bytes. Salvaged JSON/MD
metadata cannot be promoted into that missing benchmark.

The corrected Super-AI path does not resurrect the old separate Entry-IQL
policy or its hand-written live overlays. Its expectile-V and counterfactual
`Q(s, LONG/SHORT/FLAT, horizon)` primitives are to be ported into positively
trained internal heads on the shared seq513 encoder, then fused through the
single final learned direction layer. Distillation is admissible only from a
hash-bound teacher that independently passes untouched OOT gates; continual
adaptation means offline drift-triggered challenger training, regime replay,
shadow evaluation and explicit promotion/rollback, never live online weight
updates.

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
or cached context may repair that failure. Retained OANDA backfill writers
require an explicit `--vedtak` before any filesystem side effect.

## Exit

The retained Exit V3/Exit-IQL chain is a separate contract. Its XGB use and M1
exit semantics are not removed by the Entry cleanup. Shared helpers have
neutral or Exit-owned modules; active Exit math is unchanged.

The Exit-only V3 XGB bridge owns exact 7/41 field validation for two active Exit
consumers; both its import and ordered field contract fail closed. The retired
Entry-IQL registry record has `path=null` and status
`RETIRED_ARTIFACT_ABSENT`, so it cannot act as an Entry fallback.

## Next admissible milestone

The current committed source baseline and subsequent contract changes pass the
full repository suite with five skips and zero failures. Narrow audits removed
dead stop-script branches, bound serving/train/downstream artifact identity,
and added a compact takeover fingerprint; these are source-contract changes,
not empirical edge. The missing historical row/model benchmark cannot be
reconstructed from its metadata and is no longer treated as a satisfiable
prerequisite. The next
source milestone is the internal Q/V/action-value head contract with one final
fusion and exact train/export/serve parity. The next empirical milestone after
that is a fresh immutable proxy comparison plus absolute untouched OOT,
cost/live-like, abstention-support and joint Exit/sizing gates. No historical
metadata is allowed to soften those gates. Only the complete source contract
may justify returning to the hardened seq513 rebuild runbook with a newly
matched ranking/manifest and a **new** explicit vedtak; invalidated
`XAU_SEQ513_REBUILD_20260718_V1` cannot be reused;
only an accepted rebuild may advance to smoke. Zero FLAT predictions remains
hard-red. No accepted rebuild, training result or empirical precision result
exists yet.
