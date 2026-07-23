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

Still required are a compliant native OANDA M5 materialization, pair bootstrap,
the exact model-native V3 dataset producer, a fresh 79-field Exit-XGB, V3
rescore/retrain, production Exit-IQL retrain, and the canonical full-TEST
active-chain replay producer. Until their immutable OOS and live-like gates
pass, Exit cannot authorize paper, demo or live operation.
