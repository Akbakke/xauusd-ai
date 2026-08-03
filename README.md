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
`HOLD/EXIT_NOW` logits. One unique exact model argmax owns each decision; a
tied top logit fails closed instead of inheriting array order. Missing, stale,
malformed or contradictory evidence stops the path; there is no fallback model,
hand-written live rule, cached decision or synthetic `FLAT`/`HOLD`.

## Current status — 2026-08-03

**BLOCKED FOR V4 MODEL, EDGE AND LAUNCH.**

Takeover starts with `bash scripts/gx1_handover.sh --check` followed by
`bash scripts/gx1_handover.sh`. The current V8/V13 dataset and explicit
CPU-safe smoke recipe are offline evidence only; a new maintainer must use
the printed immutable paths and never start a second capped job, direct-run a
trainer, select a `latest` artifact or enter live/paper/demo/broker/drift
operation.

The current six-epoch V8/V13 smoke was interrupted in epoch 3 by a Windows
bugcheck recorded as `HYPERVISOR_ERROR (0x20001)` and produced no completion
bundle. This proves the host/Hyper-V failure class, not that training caused
it. An earlier bounded attempt
collapsed almost entirely to FLAT and failed path/cooperation evidence. No
current model or edge is accepted.

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

The current real-data V4 manifest is
`/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_BASE28_OFFLINE_20260801_MTF_V4/manifest.json`.
It is schema v3, has full-input liveness PASS and cache identity
`68568bf9431b1c770876a05e5051eefc252c6eccbf145ca024a9350688ca31b4`.
The V8/V13 dataset binds this exact cache and chronological TRAIN/VAL/TEST
lineage. No completed checkpoint, calibrated bundle, immutable TEST edge or
train==serve influence event exists.

The cleanup retained the native/canonical source and frozen V4 input cache,
while removing 92 stale target groups plus 500 exact legacy files under
immutable plan/approval/quarantine evidence. Operational source and data scans
are clean; only immutable cleanup receipts retain historical deletion terms.
A later 2026-08-03 exact-leaf event removed the remaining 5,447,068,479 bytes
of superseded V21/V23/V26 bulk run intermediates while preserving current
V8/V13/V4/V15 inputs and registry-bound historical receipts.

Live, paper, broker, publisher, promotion and drift work are outside the
frozen scope. Their historical source contracts cannot change the current
offline resume boundary.

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

Focused repository contracts are green. Test counts prove source behavior,
not direction edge. The current V8/V13 source/cache/dataset line is already
immutable and is the only allowed input to the next bounded smoke.

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
2. `SYSTEM_MAP.md`
3. `HANDOVER_XAU_DIRECTION_REPAIR_20260714.md`
4. `PROJECT_STATE_xau_direction_launch.json`
5. relevant code contracts/tests

The compact handover hashes the broader rules, state and documentation set;
that complete fingerprint inventory is not a second reading order.

Large data and immutable evidence live under `/home/andre2/GX1_DATA`; source
lives in this repository. Do not delete external artifacts outside the
evidence-retention owner and do not run a large rebuild/training job while
another GX1 heavy process is active.

## Next admissible work

After the next natural WSL restart, verify the configured 32 GB/4 GB VM cap,
then run the existing V8/V13 smoke under 10 GiB/512 MiB/CPU0–1. Do not rebuild
the current dataset. A valid smoke may proceed to bundle audit and one
same-bundle candidate; untouched TEST, calibration, ablation, train==serve and
unified-Exit replay remain closed until that candidate exists.

The lifecycle materializer/loader, same-bundle `HOLD/EXIT_NOW` head, positive
trainer loss, component-movement gates and candidate-bound full-TEST producer
are implemented. These are not trained or replayed results. Exit remains
blocked until combined training, train==serve and producer execution pass. The
former separate Exit stack is deleted and may not be rebuilt.

Near-perfect practical precision is a target, not a current claim.
