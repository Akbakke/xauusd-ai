#!/usr/bin/env python3
"""Train ridge IQL on V2 EXIT state with five reward variants and compare to V1.

Background
----------
Gate 6 (`EXIT_PER_BAR_SANITY_TRAINING_V1`) trained the first sanity ridge
IQL on the V1 9-feature exit state (intercept + 8 z-scored fields) under
the REALIZED_PNL_REWARD variant. The trained policy collapsed to bar-0-
exit on all three splits, identical to the ALWAYS_EXIT_NOW_AT_BAR_0
baseline. The recommended next gate, declared by gate 6, was state-
feature deepening (`DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V1`).

DEEPEN locked V2 with 51 state features (40 HAVE + 4 DERIVABLE + 7
NOT_ESTABLISHED per-bar XGB) plus 5 audit-only labels. Recovery sub-gate
promoted four ENTRY_CONTEXT_SNAPSHOT fields from NOT_ESTABLISHED to HAVE
via an offline join to `xgb_multi_horizon_predictions` per week.

This gate is named `RUN_EXIT_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1`
and lives in the exit-IQL track. A different gate with a similar name
(`RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1`) belongs to
the parallel entry-IQL research-lane track and is left untouched.

Goal
----
Run the second-stage exit-IQL research training:

  - Project the V2 state matrix from the augmented split-locked dataset,
    BASE34_M5 prebuilt, recovery LOCK, and per-trade derived running-
    state columns; train-only normalization.
  - Train one closed-form ridge IQL per reward variant (5 variants),
    each with two Q-heads (HOLD, EXIT_NOW). Reward variants are pulled
    from the augmented dataset's pre-computed reward columns
    (`reward_realized_pnl_reward_v1`, `reward_mfe_capture_reward_v1`,
    `reward_mae_penalty_reward_v1`, `reward_giveback_penalty_reward_v1`,
    `reward_transparent_combined_reward_v1`).
  - Evaluate each trained policy on train, val, test splits via the
    gate-5 harness (8 metrics, 6 baselines).
  - Compare against the V1 IQL result from gate 6 to measure the V2 lift.
  - Block training-with-V2 from production paths; pure research-only.

The seven NOT_ESTABLISHED per-bar XGB fields are NOT projected. The four
DERIVABLE running-state fields ARE projected here using the recipes
pinned in DEEPEN's `derivation_recipe_v2`.

Research-only; never promotes a policy to runtime; never modifies any
V1/V2 contract; never touches `exit_manager.py` / `live_features.py` /
`entry_manager.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import exit_iql_artifact_primitives_v1 as contract_gate
from gx1.scripts import materialize_exit_hold_exit_now_mdp_reward_contract_v1 as mdp_gate
from gx1.scripts import materialize_exit_off_policy_eval_harness_v1 as eval_gate


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "RUN_EXIT_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1"

INPUT_V2_CONTRACT_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V1_20260429T200926Z_LOCK"
)
INPUT_RECOVERY_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "RECOVER_ENTRY_SNAPSHOT_SIGNALS_FOR_EXIT_IQL_V1_20260429T200022Z_LOCK"
)
INPUT_SPLIT_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_V1_20260429T141227Z_LOCK"
)
INPUT_EVAL_HARNESS_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_OFF_POLICY_EVAL_HARNESS_V1_20260429T154407Z_LOCK"
)
INPUT_V1_TRAINING_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_PER_BAR_SANITY_TRAINING_V1_20260429T155423Z_LOCK"
)
INPUT_MDP_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_V1_20260429T103326Z_LOCK"
)
BASE34_M5_FEATURES_PATH = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/MONDAY_WEEK_EXTENSION_CANDIDATES/"
    "monday_week_prebuilt_extension_20260423_145325/"
    "xauusd_m5_BASE34_20250101_20260420_MODEL_BARS.parquet"
)

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")

ACTION_HOLD_ID = 0
ACTION_EXIT_NOW_ID = 1

RIDGE_LAMBDA = 1e-3
SEED_V1 = 20260429

REWARD_VARIANTS_V2: list[dict[str, str]] = [
    {
        "reward_id_v1": "REALIZED_PNL_REWARD",
        "reward_column_v1": "reward_realized_pnl_reward_v1",
    },
    {
        "reward_id_v1": "MFE_CAPTURE_REWARD",
        "reward_column_v1": "reward_mfe_capture_reward_v1",
    },
    {
        "reward_id_v1": "MAE_PENALTY_REWARD",
        "reward_column_v1": "reward_mae_penalty_reward_v1",
    },
    {
        "reward_id_v1": "GIVEBACK_PENALTY_REWARD",
        "reward_column_v1": "reward_giveback_penalty_reward_v1",
    },
    {
        "reward_id_v1": "TRANSPARENT_COMBINED_REWARD",
        "reward_column_v1": "reward_transparent_combined_reward_v1",
    },
]

V1_CONTINUOUS_FROM_AUGMENTED: list[str] = [
    "running_pnl_at_close_bps_v1",
    "running_mfe_bps_v1",
    "running_mae_bps_v1",
    "running_giveback_from_peak_bps_v1",
    "distance_from_peak_mfe_bps_v1",
    "atr_bps_now_v1",
    "trend_slope_ema3_v1",
    "session_volatility_pressure_v1",
    "minutes_since_session_open_v1",
    "entry_spread_bps_v1",
]
V1_LOG1P_FROM_AUGMENTED: list[str] = [
    "bars_held_v1",
    "time_since_mfe_bars_v1",
]
V1_PASSTHROUGH_FROM_AUGMENTED: list[str] = [
    "giveback_ratio_v1",
    "exit_prob_v1",
]
V1_ONEHOT_FROM_AUGMENTED: dict[str, list[str]] = {
    "session_id_v1": ["ASIA", "EU", "OVERLAP", "US"],
    "vol_regime_id_v1": ["LOW", "MEDIUM", "HIGH"],
    "side_v1": ["long", "short"],
    "entry_session_v1": ["ASIA", "EU", "OVERLAP", "US"],
}

NEW_BASE34_CONTINUOUS: list[str] = [
    "minutes_to_next_session_boundary",
    "_v1_atr_z_10_100",
    "_v1_bb_squeeze_20_2",
    "_v1_bb_bandwidth_delta_10",
    "_v1_body_share_1",
    "_v1_body_tr",
    "_v1_clv",
    "_v1_kama_slope_30",
    "_v1_ema_diff",
    "_v1_r1",
    "_v1_r12",
    "_v1_kurt_r",
    "_v1_pk_sigma20",
]
NEW_BASE34_BINARY: list[str] = [
    "session_change_flag",
    "is_ASIA",
    "_v1_is_EU",
    "_v1_is_US",
    "session_tradable",
]

DERIVED_CONTINUOUS: list[str] = [
    "pnl_velocity_v2",
    "pnl_acceleration_v2",
    "rolling_slope_pnl_5bars_v2",
    "mfe_decay_rate_3bars_v2",
]

RECOVERED_PASSTHROUGH: list[str] = [
    "p_long_entry_v1",
    "p_hat_entry_v1",
    "uncertainty_entry_v1",
    "margin_entry_v1",
]
RECOVERY_SENTINEL_VALUE = -1.0
EXIT_PROB_SENTINEL_VALUE = -1.0


ALLOWED_FINAL_STATUSES = {
    "RUN_EXIT_IQL_V2_PASS_BEST_VARIANT_BEATS_TRAIL_STOP",
    "RUN_EXIT_IQL_V2_PASS_BEST_VARIANT_BEATS_REALIZED_NOT_TRAIL_STOP",
    "RUN_EXIT_IQL_V2_PARTIAL_BEST_VARIANT_TIES_REALIZED",
    "RUN_EXIT_IQL_V2_PARTIAL_BEST_VARIANT_UNDERPERFORMS_REALIZED",
    "RUN_EXIT_IQL_V2_BLOCKED_BY_NO_SHORTCUT_FAIL",
    "RUN_EXIT_IQL_V2_BLOCKED_BY_INPUT_LOCK_MISSING",
}

ALLOWED_NEXT_ACTIONS = {
    "EXIT_PER_BAR_REWARD_VARIANT_SENSITIVITY_DEEPER_SWEEP_V1",
    "RUN_PER_BAR_XGB_REPLAY_FOR_TRANSFORMER_SIGNAL_FAMILY_V1",
    "REPAIR_EXIT_IQL_TRAINING_BEFORE_VARIANT_SENSITIVITY_V1",
    "HOLD_EXIT_IQL_RESEARCH_UNTIL_DATA_FIXED_V1",
}


_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json
_write_rows = contract_gate._write_rows
_write_report = contract_gate._write_report
_read_json = contract_gate._read_json
_file_hash = contract_gate._file_hash
_python_manifest = contract_gate._python_manifest


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_explicit_artifact_roots(paths: Iterable[Path]) -> bool:
    return contract_gate.validate_explicit_artifact_roots(paths)


def validate_no_forbidden_actions(**kwargs: Any) -> dict[str, Any]:
    return contract_gate.validate_no_forbidden_actions(**kwargs)


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def validate_no_deprecated_revival(script_path: Path) -> bool:
    text = script_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.lstrip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        for fragment in QUARANTINE_FORBIDDEN_PATH_FRAGMENTS:
            if fragment in stripped:
                raise RuntimeError("DEPRECATED_QUARANTINE_REVIVAL_FORBIDDEN")
    return True


def _load_inputs() -> dict[str, Any]:
    roots = [
        INPUT_V2_CONTRACT_ROOT,
        INPUT_RECOVERY_ROOT,
        INPUT_SPLIT_ROOT,
        INPUT_EVAL_HARNESS_ROOT,
        INPUT_V1_TRAINING_ROOT,
        INPUT_MDP_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "v2_state_contract": INPUT_V2_CONTRACT_ROOT
        / "state_feature_contract_v2.json",
        "v2_summary": INPUT_V2_CONTRACT_ROOT / "summary_v1.json",
        "recovery_per_trade": INPUT_RECOVERY_ROOT
        / "entry_snapshot_signals_per_trade_v1.parquet",
        "recovery_summary": INPUT_RECOVERY_ROOT / "summary_v1.json",
        "split_locked_dataset": INPUT_SPLIT_ROOT
        / "split_locked_augmented_dataset_v1.parquet",
        "eval_harness_summary": INPUT_EVAL_HARNESS_ROOT / "summary_v1.json",
        "eval_harness_baseline_metrics": INPUT_EVAL_HARNESS_ROOT
        / "baseline_metrics_per_split_v1.json",
        "v1_training_summary": INPUT_V1_TRAINING_ROOT / "summary_v1.json",
        "mdp_no_shortcut_axioms": INPUT_MDP_ROOT / "no_shortcut_axioms_v1.json",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_INPUT_LOCKS: {missing}")
    if not BASE34_M5_FEATURES_PATH.exists():
        raise RuntimeError(
            f"BASE34_M5_FEATURES_PATH_NOT_FOUND: {BASE34_M5_FEATURES_PATH}"
        )
    return {
        "required_paths": required,
        "v2_state_contract": _read_json(required["v2_state_contract"]),
        "v2_summary": _read_json(required["v2_summary"]),
        "recovery_summary": _read_json(required["recovery_summary"]),
        "eval_harness_baseline_metrics": _read_json(
            required["eval_harness_baseline_metrics"]
        ),
        "v1_training_summary": _read_json(required["v1_training_summary"]),
        "base34_path": BASE34_M5_FEATURES_PATH,
    }


def _per_bar_view(df: pd.DataFrame) -> pd.DataFrame:
    hold = df[df["action_id_v1"] == ACTION_HOLD_ID].copy()
    return hold.sort_values(["candidate_uid_v1", "bars_held_v1"]).reset_index(drop=True)


def _join_base34(per_bar: pd.DataFrame, base34_path: Path) -> pd.DataFrame:
    base34 = pd.read_parquet(base34_path)
    base34_cols_needed = NEW_BASE34_CONTINUOUS + NEW_BASE34_BINARY
    # The BASE34 parquet stores its M5 timestamp on a DatetimeIndex named "time".
    if "time" not in base34.columns:
        if base34.index.name == "time":
            base34 = base34.reset_index()
        else:
            raise RuntimeError("BASE34_M5_PARQUET_MISSING_TIME_COLUMN")
    missing = [c for c in base34_cols_needed if c not in base34.columns]
    if missing:
        raise RuntimeError(f"BASE34_M5_MISSING_COLUMNS: {missing}")
    base34_use = base34.loc[:, ["time", *base34_cols_needed]].copy()
    base34_use["time"] = pd.to_datetime(base34_use["time"], utc=True)
    base34_use = base34_use.sort_values("time", kind="mergesort").reset_index(drop=True)

    per_bar = per_bar.copy()
    per_bar["ts_v1"] = pd.to_datetime(per_bar["ts_v1"], utc=True)
    per_bar = per_bar.sort_values("ts_v1", kind="mergesort").reset_index(drop=True)
    joined = pd.merge_asof(
        per_bar,
        base34_use,
        left_on="ts_v1",
        right_on="time",
        direction="backward",
        tolerance=pd.Timedelta(minutes=5),
    )
    joined = joined.drop(columns=["time"], errors="ignore")
    joined = joined.sort_values(
        ["candidate_uid_v1", "bars_held_v1"], kind="mergesort"
    ).reset_index(drop=True)
    return joined


def _compute_derivatives(per_bar: pd.DataFrame) -> pd.DataFrame:
    out = per_bar.copy()
    grp = out.groupby("candidate_uid_v1", sort=False)
    out["pnl_velocity_v2"] = (
        grp["running_pnl_at_close_bps_v1"].diff().fillna(0.0).astype(float)
    )
    out["pnl_acceleration_v2"] = (
        grp["running_pnl_at_close_bps_v1"].diff().diff().fillna(0.0).astype(float)
    )
    diff4 = grp["running_pnl_at_close_bps_v1"].diff(4)
    out["rolling_slope_pnl_5bars_v2"] = (diff4 / 4.0).fillna(0.0).astype(float)
    mfe_diff3 = grp["running_mfe_bps_v1"].diff(3)
    decay = mfe_diff3.clip(upper=0.0).fillna(0.0).astype(float) / 3.0
    out["mfe_decay_rate_3bars_v2"] = decay
    return out


def _join_recovery(
    per_bar: pd.DataFrame, recovery_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rec = pd.read_parquet(recovery_path)
    needed = [
        "candidate_uid",
        "trade_uid",
        "p_long_entry_v1",
        "p_hat_entry_v1",
        "uncertainty_entry_v1",
        "margin_entry_v1",
        "recovery_status_v1",
    ]
    missing = [c for c in needed if c not in rec.columns]
    if missing:
        raise RuntimeError(f"RECOVERY_PARQUET_MISSING_COLUMNS: {missing}")
    rec_use = rec.loc[:, needed].copy()
    rec_use["candidate_uid_v1"] = rec_use["candidate_uid"].astype(str)
    rec_use = rec_use.drop(columns=["candidate_uid", "trade_uid"], errors="ignore")
    rec_use = rec_use.drop_duplicates(subset="candidate_uid_v1", keep="first")
    out = per_bar.merge(rec_use, on="candidate_uid_v1", how="left")
    not_recovered_mask = (
        out["recovery_status_v1"].fillna("MISSING")
        != "RECOVERED_FROM_XGB_PREDICTIONS"
    )
    not_recovered_trade_count = int(
        out.loc[not_recovered_mask, "candidate_uid_v1"].nunique()
    )
    for c in RECOVERED_PASSTHROUGH:
        out.loc[not_recovered_mask, c] = np.nan
    audit = {
        "audit_id_v1": "RECOVERY_JOIN_AUDIT_V2",
        "status_v1": "PASS",
        "trades_with_full_recovery_v1": int(
            out["candidate_uid_v1"].nunique() - not_recovered_trade_count
        ),
        "trades_without_recovery_v1": not_recovered_trade_count,
        "policy_v1": (
            "Trades without RECOVERED_FROM_XGB_PREDICTIONS get NaN for the four "
            "entry-snapshot fields; sentinel substitution applied at state-matrix "
            "build time. No fabricated values."
        ),
    }
    return out, audit


def _fit_train_normalization(per_bar_train: pd.DataFrame) -> dict[str, Any]:
    norm: dict[str, Any] = {}
    z_columns = (
        V1_CONTINUOUS_FROM_AUGMENTED + NEW_BASE34_CONTINUOUS + DERIVED_CONTINUOUS
    )
    for col in z_columns:
        if col not in per_bar_train.columns:
            raise RuntimeError(f"NORMALIZATION_COLUMN_MISSING: {col}")
        s = per_bar_train[col].astype(float)
        median = float(s.median())
        s_filled = s.fillna(median)
        mean = float(s_filled.mean())
        std = float(s_filled.std(ddof=0)) or 1.0
        norm[col] = {"transform": "z", "mean": mean, "std": std, "median": median}
    for col in V1_LOG1P_FROM_AUGMENTED:
        if col not in per_bar_train.columns:
            raise RuntimeError(f"NORMALIZATION_COLUMN_MISSING: {col}")
        s = per_bar_train[col].astype(float)
        median = float(s.median())
        s_filled = s.fillna(median)
        log_s = np.log1p(s_filled.clip(lower=0.0))
        norm[col] = {
            "transform": "log1p_z",
            "mean": float(log_s.mean()),
            "std": float(log_s.std(ddof=0)) or 1.0,
            "median": median,
        }
    return norm


def _zscore(values: pd.Series, cfg: dict[str, Any]) -> np.ndarray:
    s = values.astype(float).fillna(cfg["median"])
    return ((s - cfg["mean"]) / cfg["std"]).clip(-5.0, 5.0).to_numpy()


def _log1p_z(values: pd.Series, cfg: dict[str, Any]) -> np.ndarray:
    s = values.astype(float).fillna(cfg["median"]).clip(lower=0.0)
    log_s = np.log1p(s)
    return ((log_s - cfg["mean"]) / cfg["std"]).clip(-5.0, 5.0).to_numpy()


def _passthrough_with_sentinel(
    values: pd.Series, sentinel: float = EXIT_PROB_SENTINEL_VALUE
) -> np.ndarray:
    return values.astype(float).fillna(sentinel).clip(-1.0, 1.0).to_numpy()


def _passthrough_zero_one_with_sentinel(
    values: pd.Series, sentinel: float = RECOVERY_SENTINEL_VALUE
) -> np.ndarray:
    s = values.astype(float)
    return np.where(s.notna(), s.clip(0.0, 1.0).to_numpy(), sentinel).astype(float)


def _binary_passthrough(values: pd.Series) -> np.ndarray:
    return values.astype(float).fillna(0.0).clip(0.0, 1.0).to_numpy()


def _onehot(values: pd.Series, vocab: list[str]) -> np.ndarray:
    n = len(values)
    out = np.zeros((n, len(vocab)), dtype=float)
    s = values.astype(str).str.upper()
    for i, k in enumerate(vocab):
        out[:, i] = (s == k.upper()).astype(float).to_numpy()
    return out


def _build_state_matrix_v2(
    per_bar: pd.DataFrame, norm: dict[str, Any]
) -> tuple[np.ndarray, list[str]]:
    columns: list[str] = ["intercept"]
    blocks: list[np.ndarray] = [np.ones((len(per_bar), 1))]

    for col in V1_CONTINUOUS_FROM_AUGMENTED:
        columns.append(f"{col}__z")
        blocks.append(_zscore(per_bar[col], norm[col]).reshape(-1, 1))

    for col in V1_LOG1P_FROM_AUGMENTED:
        columns.append(f"{col}__log1p_z")
        blocks.append(_log1p_z(per_bar[col], norm[col]).reshape(-1, 1))

    columns.append("giveback_ratio_v1__pass")
    blocks.append(
        per_bar["giveback_ratio_v1"]
        .astype(float)
        .fillna(0.0)
        .clip(-2.0, 2.0)
        .to_numpy()
        .reshape(-1, 1)
    )
    columns.append("exit_prob_v1__sentinel")
    blocks.append(_passthrough_with_sentinel(per_bar["exit_prob_v1"]).reshape(-1, 1))

    for col, vocab in V1_ONEHOT_FROM_AUGMENTED.items():
        oh = _onehot(per_bar[col], vocab)
        for cat in vocab:
            columns.append(f"{col}__{cat.upper()}")
        blocks.append(oh)

    for col in NEW_BASE34_CONTINUOUS:
        columns.append(f"{col}__z")
        blocks.append(_zscore(per_bar[col], norm[col]).reshape(-1, 1))

    for col in NEW_BASE34_BINARY:
        columns.append(f"{col}__bin")
        blocks.append(_binary_passthrough(per_bar[col]).reshape(-1, 1))

    for col in DERIVED_CONTINUOUS:
        columns.append(f"{col}__z")
        blocks.append(_zscore(per_bar[col], norm[col]).reshape(-1, 1))

    for col in RECOVERED_PASSTHROUGH:
        columns.append(f"{col}__pass_or_sentinel")
        blocks.append(_passthrough_zero_one_with_sentinel(per_bar[col]).reshape(-1, 1))

    X = np.concatenate(blocks, axis=1)
    if not np.isfinite(X).all():
        bad_count = int((~np.isfinite(X)).sum())
        raise RuntimeError(f"STATE_MATRIX_HAS_NON_FINITE_VALUES: count={bad_count}")
    return X, columns


def _compute_targets_for_variant(
    per_bar: pd.DataFrame, reward_column: str
) -> pd.DataFrame:
    if reward_column not in per_bar.columns:
        raise RuntimeError(f"REWARD_COLUMN_MISSING_IN_AUGMENTED: {reward_column}")
    out = per_bar.copy()
    out["__target_exit_now_v1"] = out[reward_column].astype(float)
    last_bar_idx = out.groupby("candidate_uid_v1")["bars_held_v1"].idxmax()
    last_bar = out.loc[last_bar_idx, ["candidate_uid_v1", reward_column]].copy()
    last_bar = last_bar.rename(columns={reward_column: "__target_hold_v1"})
    out = out.merge(last_bar, on="candidate_uid_v1", how="left")
    return out


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray:
    a = X.T @ X + lam * np.eye(X.shape[1])
    b = X.T @ y
    return np.linalg.solve(a, b)


def _train_q_heads_for_variant(
    X_train: np.ndarray,
    target_hold: np.ndarray,
    target_exit_now: np.ndarray,
    feature_names: list[str],
    reward_id: str,
) -> dict[str, Any]:
    coef_hold = _ridge_fit(X_train, target_hold)
    coef_exit_now = _ridge_fit(X_train, target_exit_now)
    return {
        "reward_id_v1": reward_id,
        "coef_hold_v1": coef_hold.tolist(),
        "coef_exit_now_v1": coef_exit_now.tolist(),
        "feature_names_v1": list(feature_names),
        "ridge_lambda_v1": RIDGE_LAMBDA,
        "seed_v1": SEED_V1,
        "train_row_count_v1": int(X_train.shape[0]),
    }


def _exit_index_from_iql_policy(
    per_bar: pd.DataFrame,
    X: np.ndarray,
    coef_hold: np.ndarray,
    coef_exit_now: np.ndarray,
) -> pd.Series:
    q_hold = X @ coef_hold
    q_exit = X @ coef_exit_now
    pick_exit = q_exit > q_hold
    realized_idx_map = eval_gate._exit_index_realized_exit(per_bar)
    per_bar = per_bar.reset_index(drop=True)
    pick_exit = pd.Series(pick_exit, index=per_bar.index)
    out: list[tuple[str, int]] = []
    for uid, group in per_bar.groupby("candidate_uid_v1", sort=False):
        triggered = group[pick_exit.loc[group.index]]
        if not triggered.empty:
            out.append((uid, int(triggered.index[0])))
        else:
            out.append((uid, int(realized_idx_map.loc[uid])))
    return pd.Series({uid: idx for uid, idx in out})


def audit_no_shortcut_at_training_time(
    feature_names: Sequence[str], raw_columns_used: set[str]
) -> dict[str, Any]:
    forbidden = set(mdp_gate.FORBIDDEN_STATE_FIELDS_V1)
    leak = sorted(raw_columns_used & forbidden)
    if leak:
        raise RuntimeError(f"TRAINING_USES_FORBIDDEN_FIELDS: {leak}")
    audit_token_state = [
        n
        for n in feature_names
        if n.startswith("audit_") or "post_exit" in n or "exit_reason" in n
    ]
    if audit_token_state:
        raise RuntimeError(f"FORBIDDEN_TOKEN_IN_FEATURE_COLUMN: {audit_token_state}")
    return {
        "audit_id_v1": "TRAINING_NO_SHORTCUT_AUDIT_V2",
        "status_v1": "PASS",
        "model_state_columns_v1": list(feature_names),
        "raw_columns_used_v1": sorted(raw_columns_used),
        "forbidden_intersection_v1": leak,
    }


def audit_train_only_normalization(
    per_bar_full: pd.DataFrame, norm: dict[str, Any]
) -> dict[str, Any]:
    train = per_bar_full[per_bar_full["primary_split_v1"] == "train"]
    if len(train) == 0:
        raise RuntimeError("EMPTY_TRAIN_SPLIT_FOR_NORMALIZATION_AUDIT")
    sample_col = "running_pnl_at_close_bps_v1"
    cfg = norm[sample_col]
    expected_median = float(train[sample_col].astype(float).median())
    s = train[sample_col].astype(float).fillna(expected_median)
    expected_mean = float(s.mean())
    if abs(cfg["mean"] - expected_mean) > 1e-6:
        raise RuntimeError(
            f"NORMALIZATION_FIT_NOT_TRAIN_ONLY: expected_mean={expected_mean} "
            f"got={cfg['mean']}"
        )
    return {
        "audit_id_v1": "TRAIN_ONLY_NORMALIZATION_AUDIT_V2",
        "status_v1": "PASS",
        "checked_field_v1": sample_col,
        "train_mean_v1": expected_mean,
    }


def audit_split_isolation(per_bar: pd.DataFrame) -> dict[str, Any]:
    bad = (
        per_bar.groupby("candidate_uid_v1")["primary_split_v1"]
        .nunique()
        .gt(1)
        .sum()
    )
    if int(bad) > 0:
        raise RuntimeError(
            f"SPLIT_ISOLATION_VIOLATION: {bad} trades span multiple splits"
        )
    return {
        "audit_id_v1": "TRAINING_SPLIT_ISOLATION_AUDIT_V2",
        "status_v1": "PASS",
        "spanning_trade_count_v1": int(bad),
    }


def audit_policy_safety_at_inference(
    per_bar: pd.DataFrame, exit_indices: pd.Series, *, variant_id: str
) -> dict[str, Any]:
    selected = per_bar.loc[exit_indices.values]
    bars_held_max_per_trade = per_bar.groupby("candidate_uid_v1")["bars_held_v1"].max()
    selected_grouped = selected.set_index("candidate_uid_v1")["bars_held_v1"]
    bad: list[str] = []
    for uid, sel_bar in selected_grouped.items():
        if sel_bar > bars_held_max_per_trade.loc[uid]:
            bad.append(str(uid))
    if bad:
        raise RuntimeError(
            f"POLICY_SAFETY_VIOLATION[{variant_id}]: {len(bad)} trades exceed bar range"
        )
    return {
        "audit_id_v1": f"POLICY_SAFETY_AUDIT_V2_{variant_id}",
        "status_v1": "PASS",
        "out_of_range_trade_count_v1": 0,
    }


def _reproducibility_audit(
    models: list[dict[str, Any]],
    iql_results: list[dict[str, Any]],
    feature_names: list[str],
) -> dict[str, Any]:
    return {
        "layer_name": "RUN_EXIT_IQL_V2_REPRODUCIBILITY_AUDIT_V1",
        "model_v1": "CLOSED_FORM_RIDGE_TWO_HEADS_PER_REWARD_VARIANT",
        "feature_count_v1": len(feature_names),
        "feature_names_v1": list(feature_names),
        "ridge_lambda_v1": RIDGE_LAMBDA,
        "seed_v1": SEED_V1,
        "reward_variant_count_v1": len(models),
        "splits_evaluated_v1": sorted({r["split_v1"] for r in iql_results}),
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }


def _go_no_go(
    iql_results: list[dict[str, Any]],
    baseline_metrics_per_split: dict[str, list[dict[str, Any]]],
    v1_test_total_pnl: float | None,
) -> tuple[str, str, str, dict[str, Any]]:
    test_results = [r for r in iql_results if r["split_v1"] == "test"]
    if not test_results:
        raise RuntimeError("IQL_TEST_RESULTS_MISSING")
    best = max(test_results, key=lambda r: r["total_realized_pnl_bps_v1"])
    baseline_test = {
        b["policy_id_v1"]: b for b in baseline_metrics_per_split.get("test", [])
    }
    realized = baseline_test["REALIZED_EXIT_BASELINE"]["total_realized_pnl_bps_v1"]
    trail_stop = baseline_test["TRAIL_STOP_25_PCT_DD"]["total_realized_pnl_bps_v1"]
    best_total = best["total_realized_pnl_bps_v1"]
    delta_v1 = (
        best_total - v1_test_total_pnl if v1_test_total_pnl is not None else None
    )
    headline = {
        "best_variant_v1": best["reward_variant_v1"],
        "best_test_pnl_v1": float(best_total),
        "realized_v1": float(realized),
        "trail_stop_v1": float(trail_stop),
        "v1_iql_test_pnl_v1": v1_test_total_pnl,
        "delta_vs_v1_iql_v1": delta_v1,
        "best_test_mean_bars_to_exit_v1": float(best["mean_bars_to_exit_v1"]),
    }
    if best_total >= trail_stop:
        return (
            "RUN_EXIT_IQL_V2_PASS_BEST_VARIANT_BEATS_TRAIL_STOP",
            "EXIT_PER_BAR_REWARD_VARIANT_SENSITIVITY_DEEPER_SWEEP_V1",
            (
                f"Best V2 variant `{best['reward_variant_v1']}` test PNL "
                f"{best_total:.0f} >= TRAIL_STOP {trail_stop:.0f} >= REALIZED "
                f"{realized:.0f}. V2 state escapes the bar-0-collapse and "
                "matches/beats the implementable rule baseline. Next: deeper "
                "reward-variant + ridge-lambda sensitivity sweep."
            ),
            headline,
        )
    if best_total > realized:
        return (
            "RUN_EXIT_IQL_V2_PASS_BEST_VARIANT_BEATS_REALIZED_NOT_TRAIL_STOP",
            "RUN_PER_BAR_XGB_REPLAY_FOR_TRANSFORMER_SIGNAL_FAMILY_V1",
            (
                f"Best V2 variant `{best['reward_variant_v1']}` test PNL "
                f"{best_total:.0f} > REALIZED {realized:.0f} but < TRAIL_STOP "
                f"{trail_stop:.0f}. V2 lifts above the realized floor, still "
                "below the simple trail-stop rule. Next: per-bar XGB replay to "
                "fill the seven NOT_ESTABLISHED transformer-signal fields."
            ),
            headline,
        )
    if abs(best_total - realized) <= 50.0:
        return (
            "RUN_EXIT_IQL_V2_PARTIAL_BEST_VARIANT_TIES_REALIZED",
            "RUN_PER_BAR_XGB_REPLAY_FOR_TRANSFORMER_SIGNAL_FAMILY_V1",
            (
                f"Best V2 variant `{best['reward_variant_v1']}` test PNL "
                f"{best_total:.0f} ~= REALIZED {realized:.0f}. V2 features did "
                "not produce a clear lift above the realized floor. Next: "
                "per-bar XGB replay to add transformer-signal information."
            ),
            headline,
        )
    return (
        "RUN_EXIT_IQL_V2_PARTIAL_BEST_VARIANT_UNDERPERFORMS_REALIZED",
        "REPAIR_EXIT_IQL_TRAINING_BEFORE_VARIANT_SENSITIVITY_V1",
        (
            f"Best V2 variant `{best['reward_variant_v1']}` test PNL "
            f"{best_total:.0f} < REALIZED {realized:.0f}. V2 ridge IQL "
            "actively underperforms doing nothing. Investigate before any "
            "further escalation."
        ),
        headline,
    )


def _build_input_manifest(
    inputs: dict[str, Any], artifact_root: Path
) -> dict[str, Any]:
    files = [
        {
            "name_v1": name,
            "path_v1": str(path),
            "sha256_v1": _file_hash(path),
        }
        for name, path in inputs["required_paths"].items()
    ]
    files.append(
        {
            "name_v1": "base34_m5_features",
            "path_v1": str(inputs["base34_path"]),
            "sha256_v1": _file_hash(inputs["base34_path"]),
        }
    )
    return {
        "layer_name": "RUN_EXIT_IQL_V2_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "v2_state_contract_root_v1": str(INPUT_V2_CONTRACT_ROOT),
            "recovery_root_v1": str(INPUT_RECOVERY_ROOT),
            "split_root_v1": str(INPUT_SPLIT_ROOT),
            "eval_harness_root_v1": str(INPUT_EVAL_HARNESS_ROOT),
            "v1_training_root_v1": str(INPUT_V1_TRAINING_ROOT),
            "mdp_root_v1": str(INPUT_MDP_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_contract_v1": True,
        "iql_training_run_v1": True,
        "iql_production_allowed_v1": False,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "entry_manager_modified_v1": False,
        "v1_state_contract_modified_v1": False,
        "v2_state_contract_modified_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def write_artifacts(
    out_root: Path | None = None,
    *,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    inputs = _load_inputs()
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)

    validate_no_deprecated_revival(Path(__file__))
    forbidden_audit = validate_no_forbidden_actions(
        adapter=False,
        r6=False,
        iql_production=False,
        package=False,
        freeze=False,
        promo=False,
        live=False,
        optuna=False,
        broad_sweep=False,
    )
    _write_json(
        artifact_root / "input_manifest_v1.json",
        _build_input_manifest(inputs, artifact_root),
    )

    df = pd.read_parquet(inputs["required_paths"]["split_locked_dataset"])
    df["candidate_uid_v1"] = df["candidate_uid_v1"].astype(str)
    df["ts_v1"] = pd.to_datetime(df["ts_v1"], utc=True)
    per_bar_v1 = _per_bar_view(df)
    per_bar_with_b34 = _join_base34(per_bar_v1, BASE34_M5_FEATURES_PATH)
    per_bar_with_deriv = _compute_derivatives(per_bar_with_b34)
    per_bar_full, recovery_join_audit = _join_recovery(
        per_bar_with_deriv,
        INPUT_RECOVERY_ROOT / "entry_snapshot_signals_per_trade_v1.parquet",
    )

    split_isolation = audit_split_isolation(per_bar_full)

    per_bar_train = per_bar_full[per_bar_full["primary_split_v1"] == "train"]
    norm = _fit_train_normalization(per_bar_train)
    train_only_audit = audit_train_only_normalization(per_bar_full, norm)

    X_full, feature_names = _build_state_matrix_v2(per_bar_full, norm)
    train_mask = (per_bar_full["primary_split_v1"] == "train").to_numpy()
    X_train_only = X_full[train_mask]

    raw_cols_used: set[str] = set(
        V1_CONTINUOUS_FROM_AUGMENTED
        + V1_LOG1P_FROM_AUGMENTED
        + V1_PASSTHROUGH_FROM_AUGMENTED
        + list(V1_ONEHOT_FROM_AUGMENTED.keys())
        + NEW_BASE34_CONTINUOUS
        + NEW_BASE34_BINARY
        + DERIVED_CONTINUOUS
        + RECOVERED_PASSTHROUGH
    )
    no_shortcut = audit_no_shortcut_at_training_time(feature_names, raw_cols_used)

    iql_results: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    safety_audits: list[dict[str, Any]] = []
    for variant in REWARD_VARIANTS_V2:
        variant_id = variant["reward_id_v1"]
        reward_col = variant["reward_column_v1"]
        per_bar_train_with_targets = _compute_targets_for_variant(
            per_bar_train, reward_col
        )
        target_hold = (
            per_bar_train_with_targets["__target_hold_v1"].astype(float).to_numpy()
        )
        target_exit_now = (
            per_bar_train_with_targets["__target_exit_now_v1"].astype(float).to_numpy()
        )

        model = _train_q_heads_for_variant(
            X_train_only, target_hold, target_exit_now, feature_names, variant_id
        )
        models.append(model)

        coef_hold = np.array(model["coef_hold_v1"], dtype=float)
        coef_exit_now = np.array(model["coef_exit_now_v1"], dtype=float)

        for split in ["train", "val", "test"]:
            mask = (per_bar_full["primary_split_v1"] == split).to_numpy()
            per_bar_split = per_bar_full[mask].reset_index(drop=True)
            if per_bar_split.empty:
                continue
            X_split = X_full[mask]
            exit_indices = _exit_index_from_iql_policy(
                per_bar_split, X_split, coef_hold, coef_exit_now
            )
            safety_audits.append(
                audit_policy_safety_at_inference(
                    per_bar_split, exit_indices, variant_id=f"{variant_id}_{split}"
                )
            )
            metrics = eval_gate.evaluate_policy(
                per_bar_split,
                exit_indices,
                policy_id=f"IQL_V2_RIDGE_2HEAD_{variant_id}",
                split=split,
            )
            metrics["model_id_v1"] = "EXIT_IQL_V2_RIDGE_2HEAD"
            metrics["reward_variant_v1"] = variant_id
            iql_results.append(metrics)

    _write_json(
        artifact_root / "trained_models_per_variant_v1.json",
        {"variant_count_v1": len(models), "models_v1": models},
    )
    _write_json(artifact_root / "training_normalization_v1.json", norm)

    baseline_metrics_flat = inputs["eval_harness_baseline_metrics"]["rows_v1"]
    baseline_per_split: dict[str, list[dict[str, Any]]] = {}
    for row in baseline_metrics_flat:
        baseline_per_split.setdefault(row["split_v1"], []).append(row)

    comparator_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        for r in baseline_per_split.get(split, []):
            comparator_rows.append({**r, "row_kind_v1": "BASELINE"})
        for r in iql_results:
            if r["split_v1"] == split:
                comparator_rows.append(
                    {
                        **r,
                        "implementable_v1": True,
                        "uses_oracle_v1": False,
                        "row_kind_v1": "IQL_V2",
                    }
                )

    v1_summary = inputs["v1_training_summary"]
    v1_test = v1_summary.get("iql_test_v1") or {}
    v1_val = v1_summary.get("iql_val_v1") or {}
    v1_train = v1_summary.get("iql_train_v1") or {}
    for split, r in (("train", v1_train), ("val", v1_val), ("test", v1_test)):
        if not r:
            continue
        comparator_rows.append(
            {
                **r,
                "row_kind_v1": "IQL_V1_REFERENCE",
                "policy_id_v1": "IQL_V1_RIDGE_REALIZED_PNL_REFERENCE",
                "implementable_v1": True,
                "uses_oracle_v1": False,
                "split_v1": split,
            }
        )

    _write_rows(artifact_root / "iql_v2_vs_baseline_comparator_v1.csv", comparator_rows)
    _write_json(
        artifact_root / "iql_v2_vs_baseline_comparator_v1.json",
        {"row_count_v1": len(comparator_rows), "rows_v1": comparator_rows},
    )

    audits = [split_isolation, no_shortcut, train_only_audit, recovery_join_audit]
    _write_json(
        artifact_root / "training_audits_v1.json",
        {"audit_count_v1": len(audits), "audits_v1": audits},
    )
    _write_json(
        artifact_root / "policy_safety_audits_v1.json",
        {"audit_count_v1": len(safety_audits), "audits_v1": safety_audits},
    )

    repro = _reproducibility_audit(models, iql_results, feature_names)
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    v1_test_total = (
        float(v1_test["total_realized_pnl_bps_v1"]) if v1_test else None
    )
    status, next_action, recommendation, headline = _go_no_go(
        iql_results, baseline_per_split, v1_test_total
    )
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "RUN_EXIT_IQL_V2_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "model_id_v1": "EXIT_IQL_V2_RIDGE_2HEAD",
        "reward_variant_count_v1": len(REWARD_VARIANTS_V2),
        "reward_variants_v1": [v["reward_id_v1"] for v in REWARD_VARIANTS_V2],
        "ridge_lambda_v1": RIDGE_LAMBDA,
        "feature_count_v1": len(feature_names),
        "v2_have_count_v1": inputs["v2_summary"]["feature_count_have_v1"],
        "v2_derivable_count_v1": inputs["v2_summary"]["feature_count_derivable_v1"],
        "v2_not_established_count_v1": inputs["v2_summary"][
            "feature_count_not_established_v1"
        ],
        "iql_results_v1": iql_results,
        "audits_v1": {a["audit_id_v1"]: a["status_v1"] for a in audits},
        "research_only_v1": True,
        "iql_training_run_v1": True,
        "iql_production_allowed_v1": False,
        "training_blocked_v1": False,
        "next_research_gate_v1": next_action,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "entry_manager_modified_v1": False,
        "forbidden_actions_audit_v1": forbidden_audit,
    }
    _write_json(artifact_root / "summary_v1.json", summary)

    status_payload = {
        "layer_name": "RUN_EXIT_IQL_V2_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_TRAINING_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": True,
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "RUN_EXIT_IQL_V2_GO_NO_GO_V1",
        "status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
        "training_allowed_v1": True,
        "downstream_block_v1": (
            "Research-only V2 training. The trained policies are NOT promoted "
            "to production runtime. Adapter/R6/IQL production/live, freeze/"
            "promo/live, exit_manager modification, entry_manager modification "
            "all forbidden. V1 / V2 state contracts unmodified."
        ),
    }
    _write_json(artifact_root / "run_exit_iql_v2_go_no_go_v1.json", go_no_go)

    test_rows = [r for r in iql_results if r["split_v1"] == "test"]
    test_rows_sorted = sorted(
        test_rows, key=lambda r: r["total_realized_pnl_bps_v1"], reverse=True
    )
    report_lines = [
        "# Run Exit IQL With V2 State And Reward Variants V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: research-only; policies NOT promoted to runtime.",
        "",
        "## Headline",
        f"- Best V2 variant: `{headline['best_variant_v1']}`",
        f"- Best test PNL: {headline['best_test_pnl_v1']:.0f} bps",
        f"- REALIZED floor: {headline['realized_v1']:.0f} bps",
        f"- TRAIL_STOP rule: {headline['trail_stop_v1']:.0f} bps",
        f"- V1 IQL reference: {headline['v1_iql_test_pnl_v1']} bps",
        f"- Delta vs V1 IQL: {headline['delta_vs_v1_iql_v1']}",
        f"- Best mean bars to exit: {headline['best_test_mean_bars_to_exit_v1']:.1f}",
        "",
        "## Model",
        "- Algorithm: closed-form ridge regression, two Q-heads (HOLD, EXIT_NOW), one ridge per reward variant",
        f"- Reward variants: {len(REWARD_VARIANTS_V2)}",
        f"- Features (post-one-hot): {len(feature_names)}",
        f"- Ridge lambda: {RIDGE_LAMBDA}",
        "",
        "## V2 IQL test results sorted by total PNL (descending)",
        "",
        "| Reward variant | Trades | Sum PNL | Mean PNL | MFE-cap | MAE-burden | Giveback | CATA% | Bars |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in test_rows_sorted:
        report_lines.append(
            f"| `{r['reward_variant_v1']}` | {r['trade_count_v1']} | "
            f"{r['total_realized_pnl_bps_v1']:.0f} | "
            f"{r['mean_realized_pnl_bps_v1']:.2f} | "
            f"{r['mean_mfe_capture_ratio_v1']:.3f} | "
            f"{r['mean_mae_burden_bps_v1']:.1f} | "
            f"{r['mean_giveback_bps_v1']:.1f} | "
            f"{r['cata_proxy_rate_v1']*100:.1f}% | "
            f"{r['mean_bars_to_exit_v1']:.1f} |"
        )
    report_lines.extend(["", "## Audits"])
    for a in audits:
        report_lines.append(f"- `{a['audit_id_v1']}`: {a['status_v1']}")
    safety_pass = all(a["status_v1"] == "PASS" for a in safety_audits)
    report_lines.append(
        f"- Per-(variant, split) policy safety audits: {len(safety_audits)} "
        f"({'all PASS' if safety_pass else 'FAILURES'})"
    )
    report_lines.extend(["", "## Recommendation", recommendation])
    _write_report(artifact_root / "report_v1.md", report_lines)

    artifact_manifest = {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "append_only_namespace_v1": "truth_e2e_sanity",
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "status": str(artifact_root / "status_v1.json"),
            "go_no_go": str(artifact_root / "run_exit_iql_v2_go_no_go_v1.json"),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "trained_models_per_variant": str(
                artifact_root / "trained_models_per_variant_v1.json"
            ),
            "training_normalization": str(
                artifact_root / "training_normalization_v1.json"
            ),
            "iql_v2_vs_baseline_comparator_csv": str(
                artifact_root / "iql_v2_vs_baseline_comparator_v1.csv"
            ),
            "iql_v2_vs_baseline_comparator_json": str(
                artifact_root / "iql_v2_vs_baseline_comparator_v1.json"
            ),
            "training_audits": str(artifact_root / "training_audits_v1.json"),
            "policy_safety_audits": str(
                artifact_root / "policy_safety_audits_v1.json"
            ),
            "reproducibility_audit": str(
                artifact_root / "reproducibility_audit_v1.json"
            ),
            "report": str(artifact_root / "report_v1.md"),
        },
        "read_only_references_v1": True,
        "trained_model_v1": True,
        "not_controller_v1": True,
        "not_live_gate_v1": True,
    }
    _write_json(artifact_root / "manifest_v1.json", artifact_manifest)

    return {
        "artifact_root": str(artifact_root),
        "summary": summary,
        "status": status_payload,
        "go_no_go": go_no_go,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize RUN_EXIT_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1."
    )
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()
    out_root = (
        Path(args.out_root).expanduser().resolve() if args.out_root else None
    )
    result = write_artifacts(out_root=out_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
