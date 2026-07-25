# Canonical Exit status

Updated 2026-07-25.

Status: **BLOCK**. `PROJECT_STATE_artifacts.json` still records the historical
XGB, V3 and Exit-IQL selections, but `ACTIVE` there is not current production
admission:

- the XGB bundle is the obsolete 80-field contract. The fresh contract removes
  non-causal `_v1_cost_bps_est`; runtime now requires bundle-owned feature and
  sanitizer contracts in exact order;
- V3 predates exact T+5 state and lacks the mandatory reproducible training
  lineage, including the exact XGB bridge identity;
- Exit-IQL declares `research_only_v1=true` and
  `iql_production_allowed_v1=false`;
- all three artifacts predate the corrected closed-M5 join, overlay placement
  and Entry distribution.

Startup therefore fails closed. No old artifact may be padded, relabeled,
mixed with a newly trained role or treated as an incumbent.

The source boundary now provides:

- one atomic canonical-v3/BASE28 pair-generation pointer;
- full-history native-M5 canonical recomputation;
- a hash-bound historical M1 provider seam through the production
  `V12Pipeline.make_exit_decision` path;
- deterministic V3 record order and full dataset/config/code/checkpoint/XGB
  lineage checks.

The post-audit implementation further requires a 173-field float32 market
matrix with per-M1 historical closed-M5 context, exact UTC-minute/time
identity, zero base trade-state slots, recomputed XGB bridge values,
contiguous overlays, exact 240-row records and terminal teacher equality.
The common M1→closed-M5 mapping, complete volume prefix and XGB
session/probability domains fail closed. Replay schema v7 keeps canonical
label-horizon outcomes separate from model decision bars and following fresh
fills. Its canonical operation owns every TEST row, explicit FLAT no-order
result and LONG/SHORT per-M1 decision trace, while binding the exact runtime
heads, SourceTape, frozen pair, active artifacts, source closure and outputs.

The immutable native OANDA M1/M5 producer and the snapshot-driven
native→canonical-v3/raw-BASE28 pair producer executed on 2026-07-24 under
vedtak `XAU_NATIVE_PAIR_BOOTSTRAP_20260724_V1`: the 2019-01-01→2026-07-24
roots are accepted and pair generation `077e5419…` is published at the
canonical pointer. The pair route binds native/source/code/
formula/timing lineage and publishes raw BASE28 with exactly 13 native M1
fields; it cannot copy an old prebuilt. XGB is cut by user vedtak 2026-07-24:
no XGB trainer, rescore or fresh XGB artifact is ever built. The successor
Exit IO contract replaces the bridge evidence with the accepted model-native
Entry bundle's calibrated outputs; the current XGB runtime/contracts/registry
entry remain only as the already-blocked historical surface and are deleted
with the V8 IO contract in the Exit-rebuild wave. Still required are the
separate immutable TRAIN-only rank reference bound through V3/Exit dataset,
bundle, replay and live (the registry intentionally has no
`train_rank_reference` entry yet), the successor Exit IO contract, a fresh
model-native V3 dataset on accepted Entry prediction evidence, production
Exit-IQL retrain, and execution of the code-proven canonical full-TEST
active-chain producer on that fresh chain.
Until their immutable OOS and live-like gates pass, Exit cannot authorize
paper, demo or live operation.
