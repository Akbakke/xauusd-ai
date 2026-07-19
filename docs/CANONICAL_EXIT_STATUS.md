# Retained Exit status

Entry is currently blocked, but the separately selected Exit artifacts remain:

- V3 Exit transformer:
  `/home/andre2/GX1_DATA/runs/FASE2B_CLEAN_20260608/v3_train_out_clean/v3_exit_clean_20260608`;
- Exit-IQL:
  `/home/andre2/GX1_DATA/runs/FASE2B_CLEAN_20260608/exit_iql_deferral_20260707`;
- Exit XGB input bundle:
  `/home/andre2/GX1_DATA/models/xgb_v7_base80_20260526_cpu_PROMOTED_20260708`.

The exact machine selection and Exit environment pins live in
`PROJECT_STATE_artifacts.json`. `scripts/gx1_exit_env_pin.sh --print` must
produce that complete environment unchanged.

Removing XGB anchors and Entry-IQL from Entry does not authorize changes to
Exit XGB use, M1 cadence, V3/Exit-IQL math, artifacts or operating point.
