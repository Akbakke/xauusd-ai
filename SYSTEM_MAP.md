# GX1 system map

## Current feature architecture

```text
OANDA XAU_USD complete MBA candles
  M5 native source --------------------------+
  M1 native source ----+                     |
                       v                     v
             same 8 feature owners, run independently
                 M1 values              M5 values
          + M5/M15/H1/H4/D1      + M15/H1/H4/D1
                       |                     |
                       +-- exact contract ----+
                                  |
              TRAIN-only ranking + normalization
                                  |
        signal + continuous ctx + categorical ctx (widths: run the owner)
                                  |
                    shared specialist encoder
                       /                     \
        Entry M5 (96 bars)             Exit M1 (480 bars)
        LONG/SHORT/FLAT                 HOLD/EXIT_NOW
                       \                     /
                        same committed bundle
```

The eight owners are:

1. structure/swing;
2. SMC/liquidity;
3. trend/EMA;
4. volatility/compression;
5. momentum/flow;
6. session/regime;
7. chart geometry;
8. price action/candles.

Entry has local M5 evidence and closed M15/H1/H4/D1 context. Exit has local M1
evidence and closed M5/M15/H1/H4/D1 context. OHLCV is closed and aligned before
the same owners compute each timeframe; finished M1 features are never rolled
up. Relevance is learned, with no handwritten confluence vote or TF weight.

Every MTF lane has `MULTI_TF_FEATURE_COUNT_V4` ordered fields. The volume owner computes
`vol_z_20`, `vol_ratio_5_20` and `vol_pct_96` independently on every closed
timeframe after OHLCV resampling with tick volume summed. The 96-bar Entry
slice is computed from 191 native M5 rows and the 480-bar Exit slice from 575
native M1 rows: both include the required 95-row volume prefix and neither is
zero padded.

The five-field volatility-squeeze state uses the same owner independently on
local M1/M5 and every native MTF clock. One immutable six-clock manifest binds
the separate TRAIN-only parameters, source/pair/split/tape lineage, exact bar
grids and file/payload hashes; there is no default or cross-clock reuse.

The MTF matrix, cache manifest and full-input liveness contracts bind the single
UTC trading-session clock (their schema versions are printed by
`bash scripts/gx1_handover.sh`, never restated here). Its H4 bins open on 22/02/06/10/14/18 UTC and D1 opens
at 22:00 UTC, so the retired H4 00/04/... and calendar-midnight D1 axes cannot
pass as current cache identity.

The signal surface is a frozen base block + the mandatory causal families + the
complete code-owned candidate remainder. **The widths are not restated here**
(rule 4/13): the counts this paragraph used to carry were stale by 88 fields
within two days. Read them from the owner:
`gx1/contracts/entry_model_native_signal_v1.py` —
`MODEL_NATIVE_SIGNAL_DIM`, `MODEL_NATIVE_BASE_SIGNAL_DIM`,
`MODEL_NATIVE_MANDATORY_SELECTED_FIELDS`,
`MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS`,
`MODEL_NATIVE_MANDATORY_FAMILY_FEATURES`. The runnable one-liner is in
`HANDOVER_XAU_DIRECTION_REPAIR_20260714.md`.

Signal binds the same exact causal candle geometry/relation/carry owner locally
and on every TF; its width derives from
`gx1/features/entry_candle_primitives_v1.CANDLE_PRIMITIVE_FEATURE_NAMES`. The retained six-field
local SMC addition exposes displacement, sided sweep depth, one-shot sweep
events and event age as raw evidence rather than a direction score.

Exact counts and order derive from `MODEL_NATIVE_SPECIALIST_LAYER_FEATURES`;
there is no second hand-maintained schema. Every active emitted owner field is
available to the learned model; no fixed top-k/ranker has selection authority.
The historically named V29 addition introduced the retained level
registry
(`gx1/features/level_registry_v1.py`: level identity, touch counts, ages,
signed reaction history, break/retest events, round numbers) and a trendline
registry (`gx1/features/trendline_registry_v1.py`: two-point sloped lines,
≥3-touch validation, channels), plus per-timeframe EMA-cross, RSI-threshold,
divergence, regime-flip and swing-break event primitives on all five
timeframes. The per-TF width is derived from `MULTI_TF_PER_BAR_FEATURES_V4`. Registry tolerances are TRAIN-fitted
by immutable chronological inner-TRAIN competing-risk selection (no quantile
or window recipe input exists) on the declared TRAIN window — an ordered
`declared_train_window_start`/`declared_train_window_end` pair, both required
and both re-checked against the chain's own `--train-start` /
`--registry-fit-train-end` by exact timestamp equality once each lane
publishes — and frozen into the
hash-bound build manifests (M5 lanes: V4 cache manifest; Exit M1 lane: the
M1-enriched manifest) with exact fit-source provenance, including a
hash-bound `pair_manifest_artifact`/`pair_manifest_sha256` pointer to the
generation the fit read; consumers fail closed without them. The level registry's post-fit runtime-population shadow uses the
same state machine as serving and is only a nonempty-support/provenance gate,
not a duplicate registry or a shadow/live execution route.

The immutable M5 surface is Entry's sole signal/context input authority. It is
loaded once and exposed to TRAIN/VAL/TEST as exact contiguous timestamp views,
so no split rebuilds the specialist stack. The M1 surface is Exit's matching
native-resolution authority; neither surface can substitute for the other.

## Data and lifecycle

The M1 and M5 sources are one immutable generation pair. Dataset splits share
the same run ID and boundaries. Entry and Exit share TRAIN normalization and
the exact ordered signal-manifest identity, while their computed values remain
native to each clock. Exit episodes point into the hash-bound M1 surface; they
do not duplicate paths.

Exit supervision has no caller-selected lookahead. The dataset event fits one
hash-bound target policy on native TRAIN M1 only, learns its indifference band
from executable spread and selects its horizon from the observed 1..512-row
material-improvement discovery curve. VAL/TEST reuse the frozen policy and
contribute zero fit rows; corpus load recomputes the TRAIN fit from source.

The current published source authority is pair generation
`9b18e215061b0310bc0b9e962b00cfc2710f86e9484f3cee66f953f0077232cd`
(published 2026-08-09; the 2026-08-04 parent `64d62c1f…` is untouched
history). One rank artifact is fit
from its canonical M5 market fields; the final model source must prove exact
market identity through TRAIN before either ranking or dataset construction.

The Exit row clock is consecutive authoritative observed M1 rows. Weekend and
market-closure gaps are allowed only when the native OANDA manifest proves
source absence; no synthetic candle is inserted. A lifecycle episode has 480
feature rows and 512 supervised path states. Runtime/replay retains the latest
512 detailed path rows but carries all-time elapsed age and an incremental hash
over every prior row, so 512 is not a forced trade-duration limit.

Direction labels are future-outcome supervision, not live rules. Their horizon
is 24 observed M5 bars. Any M1 reconstruction resolves those M5 buckets first;
24 may never be interpreted as 24 M1 rows.

## Model-native decision path

```text
all shared evidence -> encoder -> calibrated direction logits
                                   |
                       unique argmax or failure
                         LONG / SHORT / FLAT

learned Entry-decision token + M1 features + five-TF context + path -> Exit logits
                                           |
                               unique argmax or failure
                                  HOLD / EXIT_NOW
```

Auxiliary, utility, path-quality and sizing heads train the representation and
produce evidence. They cannot vote, veto, threshold or replace the direction
argmax. Sizing cannot create an order when direction is FLAT or invalid.
Its target is an exact selected-side path-quality ECDF fitted on TRAIN
tradable rows; only explicitly masked rows train the size head, VAL/TEST use
the frozen ECDF, and the size output has no direction authority.
The Entry-decision token is a learned 609-to-128 projection of the exact ordered
local, final, MTF, raw-fusion, fusion-hidden and final-logit decision blocks.
It is frozen once at fill as exact little-endian float32 bytes. Every Exit result additionally binds the exact M1
and five-TF tensor bytes, their clocks/cache identity, side, quotes, path and
trade identity in one persisted full-input envelope.

Lifecycle TRAIN/VAL probes are chosen without looking at HOLD/EXIT_NOW labels;
future outcomes are attached only after state selection. The lifecycle owner
also exposes a bounded full-trajectory iterator over every non-tied long/short
state. Epoch selection uses probes for tractable training, then the selected
candidate checkpoint must pass a streaming evaluation of every non-tied VAL
state before bundle creation. Smoke runs cannot authorize this gate.

The five handwritten regime composites, the handcrafted `tf_agreement`
auxiliary objective/head and `signed_vol_z_20` are absent from the active
surface. Raw per-TF regime/EMA/trend-age/D1-distance evidence, genuine change
events, local return and the three unsigned volume primitives remain available
for learned fusion.

Training-objective v6 and the 46-key recipe-v5 schema use plain unweighted CE
for main/MTF/masked-side classification and plain unweighted BCE for hierarchy
binary tasks. Waves A/B retired direction and hierarchical distribution
forcing. Fixed auxiliary task weights, rank margins and gate regularization
remain for Wave C, so this is not a claim that every static objective magnitude
has been eliminated.

The TRAIN-fit squeeze owner is implemented but not connected to the production
surface. Adoption requires separately fitted artifacts for every clock plus
manifest/materializer plumbing. Exit remains a closed-M1 system; no tick-level
feature, dataset, OOS result or trading claim exists.

## Evidence sequence

```text
source pair
 -> feature/cache/liveness proofs
 -> lifecycle + split manifests
 -> smoke recipe and smoke bundle audit
 -> full candidate
 -> immutable calibration
 -> untouched TEST selective-edge report
 -> same-candidate unified Entry/Exit replay
 -> recomputable sizing/serve parity evidence
```

Failure at any arrow stops the chain. Fresh native and canonical source
exists. Historical V28/V29J datasets were retired with their superseded
feature contracts and have no training or comparison authority. Current
status is before the current-contract V30 feature-surface and dataset rebuild.
No accepted candidate exists.

## Scope boundary

The active checkout ends at offline train/OOS/replay. Historical live, paper,
collector, launch and adaptation modules are quarantined and not exposed by the
control surface. They are not evidence and cannot authorize operation.
