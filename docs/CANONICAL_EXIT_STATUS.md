# Canonical Exit status

Updated 2026-07-23.

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

The immutable native OANDA M5 producer is now code-proven in the existing
owner but has not run. Still required are its explicit production run, a
complete native→canonical-v3/BASE28 bootstrap,
a fresh model-native V3 dataset produced by the now code-proven exact
writer/event, a fresh 79-field Exit-XGB, V3 rescore/retrain, production
Exit-IQL retrain, and execution of the code-proven canonical full-TEST
active-chain producer on that fresh chain. Exact SourceTape lookup, frozen-pair
loading, V3 dataset production, the Exit-only pipeline factory and full-TEST
producer now exist in source; no fresh production artifact was created.
Until their immutable OOS and live-like gates pass, Exit cannot authorize
paper, demo or live operation.
