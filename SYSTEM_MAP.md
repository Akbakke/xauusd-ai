# GX1 system map

## Current feature architecture

```text
OANDA XAU_USD complete MBA candles
  M5 native source --------------------------+
  M1 native source ----+                     |
                       v                     v
              shared feature-surface owner (same formulas/order)
                 M1 surface            M5 surface
                       |                     |
                       +----- lineage -------+
                                  |
              TRAIN-only ranking + normalization
                                  |
                    513 signal + 142 cont + 5 cat
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

Each also reaches M5/M15/H1/H4/D1 through the fixed V4 grid. Relevance is
learned; there is no handwritten confluence vote or timeframe weight.

## Data and lifecycle

The M1 and M5 sources are one immutable generation pair. Dataset splits share
the same run ID and boundaries. Entry and Exit share TRAIN normalization. Exit
episodes point into the hash-bound M1 surface; they do not duplicate paths.

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

Failure at any arrow stops the chain. Current status is before a fresh source
and dataset rebuild; no accepted candidate exists.

## Scope boundary

The active checkout ends at offline train/OOS/replay. Historical live, paper,
collector, launch and adaptation modules are quarantined and not exposed by the
control surface. They are not evidence and cannot authorize operation.
