# GX1 XAUUSD model-native trading engine

## Active scope

GX1 is frozen to one offline path only:

`immutable XAUUSD snapshot -> shared featurebase -> Entry M5 -> Exit M1 -> offline train/OOS/replay`

Entry and Exit share the same eight causal feature owners, formulas,
normalization and lineage. No live/paper/demo operation, broker route, daemon,
polling, live-tail admission, promotion, drift adaptation, fallback or second
feature owner is active or permitted. Reuse exact immutable caches and append
only after overlap/hash proof. See [GX1_RULES.md](GX1_RULES.md).

GX1 is being rebuilt around one learned bundle and shared encoder for
gold/XAUUSD. It emits calibrated Entry `LONG/SHORT/FLAT` and Exit
`HOLD/EXIT_NOW` logits. Exact model argmax owns both decisions. Missing, stale,
malformed or contradictory evidence stops the path; there is no fallback
model, hand-written live rule, cached decision or synthetic `FLAT`/`HOLD`.

## Current status — 2026-08-02

**BLOCKED FOR V4 MODEL, EDGE AND LAUNCH.**

The latest empirical model evidence is V21C. It used the older incomplete
multi-timeframe surface and overfit before it produced a balanced,
generalizing checkpoint: train loss fell 36% while validation loss doubled;
the best balanced accuracy was 0.3438 against a 0.3858 majority baseline and
0.4021 from a plain diagnostic MLP on the same substrate. V21C produced no
accepted model or launch authority.

The rejected V18 trainability bundle and stale V19/V26 dataset/audit artifacts
were retired from authority and deleted on 2026-07-29. A fresh V8/V13
combined Entry/lifecycle dataset now exists and passed full-input liveness,
pretrain audit and Exit-lifecycle integrity under the capped runner. It is
offline evidence only: no checkpoint, direction bundle, candidate bundle or
launch authority is admitted.

The source/input architecture has since advanced to MTF V4:

- exact M5/M15/H1/H4/D1 surfaces;
- 111 ordered causal fields at every timeframe;
- all eight non-empty specialist families at every timeframe;
- 40 learned family×timeframe cooperation routes;
- 555 contextual feature×timeframe gates (`5 × 111`);
- recipe-owned causal history windows that become progressively coarser with
  age;
- exact V4 cache, normalization, bundle, runtime and serve-parity contracts.

The measured real-data V4 cache is frozen at
`/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v26_6yr_rebuild_20260725_seq513_model_native_v26/MULTI_TF_V4_CACHE_20260729`.
Its identity is
`ff9cac78cdf6d5d4338f4d07b77df822c95efb568ed80a1e864600580a2b361a`.
Across all five timeframes, every post-warmup field is finite and variable,
with zero constant fields and zero exact duplicate pairs.

This remains historical schema-v2 input evidence through
`2026-07-24T20:55:00Z`. The fresh V8/V13 dataset is separately bound to its
immutable source, pair manifest, V4 cache and chronological TRAIN/VAL/TEST
lineage. No trained checkpoint, calibrated bundle, immutable TEST edge or
train==serve influence event exists.

The cleanup retained the native/canonical source and frozen V4 input cache,
while removing 92 stale target groups plus 500 exact legacy files under
immutable plan/approval/quarantine evidence. Operational source and data scans
are clean; only immutable cleanup receipts retain historical deletion terms.

Live is independently blocked: the old canonical incremental daemon interface
is retired. The existing native owner now implements a parent-CAS successor
that reuses verified history and fetches only one bounded overlap chunk plus
the new tail. The pair owner publishes the candidate-bound freshness event
before moving its serving pointer; two consecutive events form one short-lived
admission. Launch stores a static pair/root/producer anchor, while every new
Entry must revalidate the newest admission against the exact pair generation
used for inference and again immediately before an order. Missing or stale
evidence blocks only new exposure, not same-bundle Exit recovery. No real
successor/admission chain is published or launch-bound yet.

## Active Entry surface

The current-bar model-native state contains:

- 513 ordered signals: 34 base plus 479 specialist fields;
- 378 mandatory outputs from twelve causal feature layers;
- 101 deterministic TRAIN-only ranked fields;
- 142 continuous and five categorical context fields;
- sequence length 96;
- 22 positively supervised Entry heads plus one unified Exit head;
- 26 evidence groups producing 96 values for one learned
  `96 -> 128 -> 3` direction fusion.

The V4 multi-resolution branch adds the exact `5 × 111` timeframe grid. Its
one-owner family partition is:

| Family | Fields per timeframe |
|---|---:|
| structure/swing | 5 |
| SMC/liquidity | 11 |
| trend/EMA | 10 |
| volatility/compression | 2 |
| momentum/flow | 4 |
| session/regime | 5 |
| chart geometry | 10 |
| price-action/candles | 64 |
| **Total** | **111** |

Every family is represented independently at every resolution. Shared family
encoders preserve semantics across timeframes; timeframe-specific
normalization, temporal position, contextual feature gates, axial attention
and learned cooperation determine influence. An engineered confluence field
is evidence only. It cannot vote, veto or choose a direction.

The complete current-tree repository contract passed on 2026-07-31
(`1962 passed, 2 skipped`; 1,964 collected). These tests prove source
contracts, not direction edge or retraining. The mutable collector is current
through complete M1 `2026-07-31T09:34:00Z` without a failure latch, but the
last immutable native pair remains frozen on 2026-07-24. A fresh pair must be
published from a clean committed producer identity before schema-v3 cache and
dataset rebuild.

The source runtime now preserves the exact 128-value shared Entry
representation, builds a hash-bound contiguous closed-M1 path, calls
`HOLD/EXIT_NOW` on the same bundle and persists/journals the decision through
TradeState v8. Broker state is bound to the exact account/environment, and a
proven `EXIT_NOW` has a per-exposure no-replay intent across restart/close
failure.
This is not empirical authorization: no current unified bundle has been
trained or replayed through that path.

The existing sizing owner also implements the candidate-bound canonical
full-TEST Entry+Exit producer. It must load the exact pre-activation candidate
commit and immutable source/rank/OOS evidence, and it reuses the production
Exit adapter and TradeState transition path. It has produced no artifact:
there is no admissible unified candidate to replay.

## Evidence standard

Source tests prove contracts, not trading edge. A candidate may advance only
when immutable evidence binds the same bytes through:

1. source and V4 cache identity;
2. complete TRAIN/VAL/TEST liveness and exact field order;
3. TRAIN-only normalization and ranking;
4. model/head/gate liveness;
5. untouched TEST direction, calibration, cost and slice evidence;
6. train==serve parity;
7. raw and calibrated class-margin movement for every retained specialist,
   timeframe, family×timeframe route and evidence group;
8. serve-parity v11 local sensitivity for 1,723 numeric routes: 513 sequence
   fields, the same 513 snapshot fields, 142 continuous-context fields and
   555 MTF cells, plus valid-category counterfactual movement for all five
   categorical fields;
9. learned sizing plus a fresh same-candidate unified-Exit replay;
10. an admitted immutable live-tail publisher, zero-order shadow, promotion
    and one-time launch approval.

Gate or attention values are diagnostics. They do not prove causal use.
Family×timeframe ablation and the final OOS result must prove that the retained
routes affect the model-native decision.

## Causal time pyramid

M5 is for recent microstructure, not distant history. As context becomes older
it is represented at M15, H1, H4 and D1 resolution. Exact window lengths are
immutable recipe inputs and must have strictly increasing wall-clock coverage.
They are selected on declared TRAIN/VAL evidence. They may not be hidden model
defaults, wrapper literals or live regime rules.

No fixed claim such as “H4 always matters more than M5” is permitted. The same
feature can matter differently by timeframe, age and regime, and that
relevance is learned per feature×timeframe.

## Start here

```bash
bash scripts/gx1_handover.sh
bash scripts/entry_next_edge_control.sh handover --check
.venv/bin/python -m json.tool PROJECT_STATE_xau_direction_launch.json
scripts/entry_next_edge_control.sh --help
```

Read, in order:

1. `AGENTS.md`
2. `PIPELINE_AUDIT_XAU_20260723.md` as historical audit context only
3. `SYSTEM_MAP.md`
4. `HANDOVER_XAU_DIRECTION_REPAIR_20260714.md`
5. `PROJECT_STATE_xau_direction_launch.json`
6. relevant code contracts/tests

The compact handover hashes the broader rules, state and documentation set;
that complete fingerprint inventory is not a second reading order.

Large data and immutable evidence live under `/home/andre2/GX1_DATA`; source
lives in this repository. Do not delete external artifacts outside the
evidence-retention owner and do not run a large rebuild/training job while
another GX1 heavy process is active.

## Next admissible work

The next lineage must first publish a fresh immutable generation-local
native/canonical pair, then rebuild and bind the V4 cache under schema v3 and
materialize fresh full-input evidence,
fit a declared capacity/regularization sweep on TRAIN/VAL, train a bounded
smoke, then train a candidate. Untouched TEST, calibration, ablation,
train==serve and unified-Exit gates remain closed until those artifacts exist.

Before any paper/live restart, execute two consecutive fresh successor
publications through the existing snapshot/pair ownership path and publish the
short-lived admission with exact freshness, lineage and health evidence. The
implementation exists; no event has been admitted. A watchdog around the
retired daemon is not a replacement.

The lifecycle materializer/loader, same-bundle `HOLD/EXIT_NOW` head, positive
trainer loss, component-movement gates and canonical candidate-bound full-TEST
producer are implemented in source. The producer is exposed only through
`model-native-sizing-produce-unified-joint-proof` and uses the same adapter,
TradeState and exact closed-M1 path transform as runtime. These are not trained
or replayed results. Exit remains blocked until fresh native/pair authority,
combined training, train==serve, producer execution and runtime parity pass.
The former separate Exit stack is deleted and may not be rebuilt.

Near-perfect practical precision is a target, not a current claim.
