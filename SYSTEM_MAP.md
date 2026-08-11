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
                    592 signal + 142 cont + 5 cat
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

The signal surface is 592 ordered fields: 34 base + 425 mandatory causal + 133
TRAIN-ranked over 16 mandatory families (V29 event surface; counts derive from
the owner tuples). The V29 layer adds a level registry
(`gx1/features/level_registry_v1.py`: level identity, touch counts, ages,
signed reaction history, break/retest events, round numbers) and a trendline
registry (`gx1/features/trendline_registry_v1.py`: two-point sloped lines,
≥3-touch validation, channels), plus per-timeframe EMA-cross, RSI-threshold,
divergence, regime-flip and swing-break event primitives on all five
timeframes (per-TF V4 context width 173). Registry tolerances are TRAIN-fitted
with the explicit recipe input `--level-tol-quantile-q` and frozen into the
hash-bound build manifests (M5 lanes: V4 cache manifest; Exit M1 lane: the
M1-enriched manifest); consumers fail closed without them.

The immutable M5 surface is Entry's sole 592/142/5 input authority. It is
loaded once and exposed to TRAIN/VAL/TEST as exact contiguous timestamp views,
so no split rebuilds the specialist stack. The M1 surface is Exit's matching
native-resolution authority; neither surface can substitute for the other.

## Data and lifecycle

The M1 and M5 sources are one immutable generation pair. Dataset splits share
the same run ID and boundaries. Entry and Exit share TRAIN normalization and
the exact ordered signal-manifest identity, while their computed values remain
native to each clock. Exit episodes point into the hash-bound M1 surface; they
do not duplicate paths.

The current published source authority is pair generation
`9b18e215061b0310bc0b9e962b00cfc2710f86e9484f3cee66f953f0077232cd`
(published 2026-08-09; the 2026-08-04 parent `64d62c1f…` is untouched
history). One rank artifact is fit
from its canonical M5 market fields; the final model source must prove exact
market identity through TRAIN before either ranking or dataset construction.

The Exit row clock is consecutive authoritative observed M1 rows. Weekend and
market-closure gaps are allowed only when the native OANDA manifest proves
source absence; no synthetic candle is inserted. A lifecycle episode has 480
feature rows and may hold at most 512 causal path states.

Direction labels are future-outcome supervision, not live rules. Their horizon
is 24 observed M5 bars. Any M1 reconstruction resolves those M5 buckets first;
24 may never be interpreted as 24 M1 rows.

## Model-native decision path

```text
all shared evidence -> encoder -> calibrated direction logits
                                   |
                       unique argmax or failure
                         LONG / SHORT / FLAT

frozen Entry state + M1 features + path -> Exit logits
                                           |
                               unique argmax or failure
                                  HOLD / EXIT_NOW
```

Auxiliary, utility, path-quality and sizing heads train the representation and
produce evidence. They cannot vote, veto, threshold or replace the direction
argmax. Sizing cannot create an order when direction is FLAT or invalid.

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
exists, and the V28 dataset chain ran GREEN end to end (369,303/5,904/6,551
rows, TEST sealed) — those bytes are the frozen baseline for the
pre-registered V29-vs-V28 evaluation. Current status is before the V29 (592)
feature-surface and dataset rebuild. No accepted candidate exists.

## Scope boundary

The active checkout ends at offline train/OOS/replay. Historical live, paper,
collector, launch and adaptation modules are quarantined and not exposed by the
control surface. They are not evidence and cannot authorize operation.
