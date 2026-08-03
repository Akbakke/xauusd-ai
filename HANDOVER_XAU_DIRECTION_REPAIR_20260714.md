# GX1 XAUUSD handover

Updated 2026-08-03. This file is the current operational handover. Historical
chronology belongs in `DECISION_LOG.md`; rejected designs are not instructions.

> Active scope freeze: this checkout is **offline shared-featurebase only** for
> XAUUSD. Entry is M5, Exit is M1, both use the same eight feature owners, and
> only offline train/OOS/replay evidence is active. Live, paper, demo, broker,
> daemon, polling, live-tail, launch, promotion, drift and online adaptation
> are forbidden. The historical operational text below is reference only and
> cannot reopen any of those routes. See [`GX1_RULES.md`](GX1_RULES.md).

## Objective

Build one XAUUSD model-native trading system that learns tops, bottoms and
abstention from the complete causal feature stack. The same immutable bundle
and shared encoder must own:

- Entry logits ordered `LONG/SHORT/FLAT`;
- Exit logits ordered `HOLD/EXIT_NOW`;
- learned position size.

One unique exact argmax owns each decision. A tied top logit is unavailable
evidence and fails closed; array order cannot select a class. No auxiliary
model, compatibility bridge, manual confluence weight, threshold, veto,
direction flip, close overlay, cached output or synthetic `FLAT`/`HOLD` is
allowed. Missing proof is a hard error.

## Current authority

The fresh Run13 preflight passed 30/30 checks. The new V8/V13 output lineage
completed under the historical 14 GiB cgroup: train 369,303 rows, val 5,904,
test 6,071; full-input liveness PASS; pretrain audit PASS; and unified Exit
lifecycle PASS. No partial V7/V12 output was resumed or admitted. The
513/142/5 contract and all eight-family ownership remain unchanged. This
dataset PASS does not constitute model training, OOS edge proof or launch
authority.

## Immediate takeover path

Use the executable viewer before reading historical chronology:

```bash
bash scripts/gx1_handover.sh --check
bash scripts/gx1_handover.sh
```

The current offline evidence anchors are the explicit V8/V13 dataset,
`UNIFIED_EXIT_LIFECYCLE_MANIFEST.json`, the Run13/feature/specialist audits,
the MTF V4 manifest and the current timestamped train-recipe audit. The
handover prints their exact paths and reports whether the bounded smoke
process is active. The current smoke recipe is CPU-only, batch 8, six epochs,
patience 3, 512 sampled TRAIN rows, zero workers and the explicit five-timeframe windows
`16/64/96/96/252` under the hard `10G/512M/2-core` cap. This is a trainability
attempt, not an admission. A missing bundle, cap kill or audit failure keeps
the system BLOCK.

The machine-readable launch state remains the final authority. It deliberately
has no accepted model/bundle authority while the offline V8/V13 line is being
evaluated. Never turn a PASS dataset or smoke artifact into launch authority
by editing state manually.

Status is **BLOCK**.

The 2026-08-03 six-epoch CPU smoke was interrupted in epoch 3 by a Windows
bugcheck. The Windows Error Reporting record and minidump identify
`HYPERVISOR_ERROR (0x20001)`; Linux recorded no OOM kill for this incident.
This proves the host/Hyper-V failure class, not that training caused it. The
run published no completion bundle or checkpoint-admission failure event, so
all partial output is invalid. The safety authority is now `10G` job memory,
`512M` job swap, two CPU-affined cores and a `20G` available-RAM launch floor.
WSL is configured for a `32GB` VM and `4GB` swap; that setting applies at the
next WSL restart. Do not start another heavy run until handover reports the
active lower VM cap and the runner prints `[capped_run_scope_verified]`.

`PROJECT_STATE_artifacts.json` has:

- `production_admission.status = BLOCK`;
- `selection_registry_is_launch_authority = false`;
- `active = {}`.

The artifact guard rejects decision loads while launch admission is blocked.
There is no launch-admitted dataset, accepted Entry/Exit bundle or immutable
OOS edge proof. The current V8/V13 dataset remains valid offline evidence for
the bounded smoke path. The rejected V18 bundle and stale V19/V26
dataset/audit artifacts were retired and deleted through immutable cleanup
evidence on 2026-07-29.
`current_smoke_launch_evidence` is now explicitly `null`: the handover no
longer opens or validates the obsolete 162-value V26 smoke recipe as a current
takeover dependency. Historical failure chronology remains in the diagnostic
fields and `DECISION_LOG.md`, never in the active resume path.

On 2026-08-03 a second exact-leaf retention event removed 33 superseded
V21/V23/V26 bulk leaves (5,447,068,479 bytes) without targeting the current
V8/V13/V4/V15 anchors or the four still-existing historical registry
references. Its terminal `DELETE_COMPLETE` event is
`/home/andre2/GX1_DATA/reports/gx1_evidence_retention_cleanup_reports/OLD_RUNS_20260803_V1/GX1_EVIDENCE_CLEANUP_EXECUTION_20260803T153441676760Z.json`
with SHA-256
`f0e96fe751de8bcc25730d1a5bfa8939e2632a94d8523203d8ca52d932b9d99d`.
Do not search for, restore or rebuild those retired leaves.

The former tree-based decision provider and the separate V3/Exit-IQL/
Strategy-F chain are permanently retired. Their source packages, runtime
adapters, contracts, tests, registry roles, selected model artifacts and Python
packages have been removed. They must not be restored under another name.

## Current feature architecture

The current Entry surface owns:

- 513 ordered signals: 34 base plus 479 specialist fields;
- 378 mandatory fields from twelve causal feature layers;
- 101 deterministic TRAIN-only ranked fields;
- 142 continuous and five categorical context fields;
- sequence length 96;
- 22 positively supervised Entry heads plus the unified Exit head;
- 26 evidence groups producing 96 learned fusion values.

MTF V4 gives M5, M15, H1, H4 and D1 the same 111-field semantic surface:

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

The model owns 555 feature×timeframe gates and 40 family×timeframe cooperation
routes. Timeframe importance is learned from data; it is not hard-coded.
History is resolution-aware: recent context remains fine, while older context
becomes progressively coarser across M15/H1/H4/D1.

Entry's decision clock is closed M5 with a 96-bar sequence. Exit's decision
clock is closed M1 with the same ordered 513-signal surface, a 480-bar M1
sequence (`5 × 96`) and an additive closed-M1 path; it also requires the Entry
M5 representation. The M1 feature-base manifest is the single shared owner,
not a second Exit taxonomy. Both sides bind the same dataset run, split
boundaries and TRAIN normalization state.

The current offline MTF manifest is schema v3, binds complete trailing
resample buckets and has full-input liveness PASS. The launch-state checkpoint
still carries the older schema-v2 cache as historical BLOCK evidence; it is
not the cache used by the current V8/V13 smoke recipe. The executable handover
now validates the recipe, MTF and lifecycle bindings before reporting the
offline line ready.

## What is implemented

- final model-native Entry direction authority and no-rule operating point;
- exact 513/142/5 identities and full eight-family routing;
- full M5/M15/H1/H4/D1 V4 feature surface;
- family×timeframe cooperation and feature×timeframe gating;
- immutable source/cache/dataset/recipe/bundle/evidence contracts;
- atomic registry/state guard with production BLOCK enforcement;
- one-model Entry/Exit metadata boundary;
- runtime startup requirement for the same-bundle Exit head;
- exact Exit output validation: finite logits/probabilities, simplex,
  logit/probability/action argmax agreement and SHA-bound evidence;
- TradeState v8 with exact broker-account ownership, fill time, first-full
  closed-M1 semantics and an
  untruncated literal bid/ask/mid OHLCV path bound to source path/SHA;
- the exact 128-value shared Entry representation is retained in the frozen
  runtime snapshot and consumed only by the same bundle's Exit head;
- one train/replay/live owner for the 14-field Exit path transform;
- a same-bundle `decide_exit` adapter, exact path/output hashing and
  transactional one-closed-M1-bar state updates;
- durable, idempotently journaled Exit intent: a proven `EXIT_NOW` survives
  restart or broker-close failure and cannot be replaced by a later `HOLD`;
- cumulative MFE/MAE from executable-side intrabar extrema, and explicitly
  named executable-range evidence rather than mislabeled ATR;
- no silent runner conversion from `EXIT_NOW` to HOLD;
- no old Exit fields in TradeState or the live journal.
- immutable native and canonical successor publication in the existing
  source/pair owners;
- immutable live-tail publication events, two-consecutive-event admission,
  launch anchoring and per-Entry runtime revalidation;
- efficient native schema-v4 successors with exact parent-manifest CAS,
  verified historical-chunk reuse and only one bounded overlap/tail refetch;
- candidate publication PASS before canonical pointer activation, followed by
  exact inference-pair equality checks before every new Entry and order;
- static launch authority separated from short-lived Entry freshness so stale
  publication evidence never suppresses same-bundle Exit recovery;
- public control routes for both snapshot publication and admission.

## What remains empirically unproven or unadmitted

The lifecycle materializer/loader, same-bundle `head_exit_action`, positive
loss and component-movement export/load gates are implemented in source.
They remain intentionally unadmitted:

- the fresh V8/V13 native-manifest-bound combined lifecycle dataset exists and
  passed liveness, pretrain audit and Exit lifecycle integrity; it is not a
  trained-model or launch authority;
- no current trained bundle proves positive Exit loss or component movement;
- no trained candidate has crossed the new runtime snapshot/envelope schemas;
- no candidate-bound train==serve proof validates the implemented adapter
  against an immutable trained bundle;
- no candidate-bound full-TEST lifecycle replay artifact exists;
- no real live-tail successor pair or admission event has been published.

This is source-contract progress, not a retrained Entry or Exit result.

The complete current-tree repository suite passes after the unified
replay-authority and live-runtime repairs: `1961 passed, 2 skipped`
(1,963 collected) on 2026-07-31. The live collector has no
failure latch and held complete M1 through `2026-07-31T09:34:00Z`, but it is a
mutable observation source and therefore has no training authority. The last
immutable native M1/M5 pair still ends on 2026-07-24. Fresh native publication
correctly fails closed while the repository has uncommitted producer changes;
the native producer requires a clean committed source identity.

The public sizing replay producer is now implemented in the existing sizing
owner and exposed as
`model-native-sizing-produce-unified-joint-proof`. It loads the immutable
pre-activation candidate directly, revalidates its committed bundle, OOS
runtime heads, TRAIN-only rank reference and generation-local pair authority,
then runs the production `SmartEntryLiveInference.decide_exit` plus
transactional `TradeState` path over every non-FLAT TEST row. Every trace row
binds the exact closed-M1 source, logits/probabilities and four model evidence
hashes. Missing bars, any Entry mismatch, a 512-bar non-exit or byte drift
fails closed. No producer output exists because no current unified candidate
exists; this is source-contract progress, not replay evidence.

## Next implementation sequence

1. Complete the bounded integrated same-bundle Entry/Exit smoke against the
   immutable V8/V13 dataset; preserve untouched TEST.
2. Audit the produced bundle and require positive `HOLD/EXIT_NOW` loss, both
   validation classes and measured movement in the shared encoder plus every
   Exit component before export.
3. Prove that the implemented serving envelope and same-bundle adapter are
   byte/float-identical to training; runtime and canonical replay now share the
   same adapter/path transform, update TradeState once per complete bar and
   fail on a missed bar.
4. Prove movement and ablation across every required family/timeframe route for
   both Entry and Exit outputs.
5. Train one candidate from the admitted smoke boundary, run the implemented
   canonical full-TEST producer against the immutable candidate commit and
   prove offline train==serve parity.

Source publication, live-tail, paper, broker, promotion and launch work remain
outside the frozen offline scope. If the current immutable source/cache line
is invalidated, stop and establish a new explicit scope/evidence decision
before replacing it.

## Permanent engineering rules

- Reuse and extend existing owners for small changes. Do not create a new
  versioned script for every repair.
- No feature family may exist only on M5 when the retained MTF contract claims
  all timeframes.
- No fixed timeframe importance. Learn it and prove it.
- No stale artifact discovery by glob, mtime, lexical version or `latest`.
- No fallback/default/zero-fill for decision-affecting evidence.
- No TEST-driven feature, window, capacity or threshold selection.
- No heavy concurrent GX1 jobs.
- Destructive `GX1_DATA` cleanup uses only
  `gx1/scripts/cleanup_gx1_evidence_v1.py` with plan, approval, quarantine,
  verification and delete evidence.
- Historical chronology cannot override current machine-readable BLOCK.

## Takeover

Read, in order:

1. `GX1_RULES.md`
2. `AGENTS.md`
3. `SYSTEM_MAP.md`
4. this handover
5. `PROJECT_STATE_xau_direction_launch.json`
6. relevant code contracts/tests

Run:

```bash
bash scripts/entry_next_edge_control.sh handover --check
bash scripts/entry_next_edge_control.sh handover
```

The compact view prints the exact current resume stage, worktree source gate,
ordered control routes and remaining P0 register. `authority_fingerprint`
binds the ordered handover documents and launch state.
`worktree_fingerprint` separately binds HEAD, the complete tracked diff and
every untracked file byte, so equal changed-path counts cannot conceal a
different source state. Use the full handover only when authority changed; if
the worktree identity changed, inspect that diff and rerun affected contracts
before continuing. Do not start training or live services from this document;
use the single control surface and let all contracts fail closed.
