#!/usr/bin/env python3
"""Train a meta-classifier that predicts SHOULD_HAVE_SKIPPED at trade entry.

Background
----------
The exit-IQL track (gate 6 V1, V2 training, V2 sweep) optimizes
*when to exit* given that a trade was taken. A complementary side-track
optimizes *whether to take the trade in the first place*. The V2 audit
contract defines `audit_should_have_skipped_v2`:

    1 if pnl_bps < 0 AND mae_bps <= -50 AND mfe_bps < 25; else 0

empirical distribution across the 1914-trade substrate:

  - should_skip=1 rate: 14.1% (269 trades)
  - mean pnl when should_skip=1: -132.67 bps
  - mean pnl when should_skip=0: +18.71 bps
  - total realized pnl across all trades: -4904.69 bps
  - total pnl if all should_skip=1 trades had been excluded: +30784.10 bps

Oracle skip thus lifts realized PNL by +35688 bps. A modest-quality
classifier capturing even half the lift produces ~+17000 bps - dwarfing
the +509 bps the exit-IQL V2 produces. Skip-side and exit-side are
orthogonal; their effects multiply.

This gate trains a closed-form ridge regression on the binary label
using ONLY AT_TRADE_OPEN features (recovery LOCK entry-snapshot fields
+ trade_outcomes static fields + BASE34 market state at the entry bar).
No per-bar features, no post-exit fields, no realized-outcome leakage.

It evaluates on val (used for threshold tuning) and on test (locked
report). For each threshold in {0.3, 0.4, 0.5, 0.6, 0.7}, the gate
computes counterfactual realized PNL via the gate-5 evaluation: take
trades where predicted P(should_skip) < threshold, exclude the rest,
sum the realized pnl_bps. The val-tuned threshold is then locked and
reported on test.

Research-only; the trained classifier is NOT promoted to runtime; no
exit_manager / live_features / entry_manager modification; no V1/V2
state-contract modification. The skip decision is a research
counterfactual measure, not an active gate on any trading path.
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

from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)
from gx1.scripts import (
    materialize_run_exit_iql_with_v2_state_and_reward_variants_v1 as v2_train_gate,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "LEARN_TRADE_SKIP_META_CLASSIFIER_AT_TRADE_OPEN_V1"

INPUT_RECOVERY_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "RECOVER_ENTRY_SNAPSHOT_SIGNALS_FOR_EXIT_IQL_V1_20260429T200022Z_LOCK"
)
INPUT_SPLIT_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_V1_20260429T141227Z_LOCK"
)
INPUT_V2_CONTRACT_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V1_20260429T200926Z_LOCK"
)
BASE34_M5_FEATURES_PATH = v2_train_gate.BASE34_M5_FEATURES_PATH

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")
SEED_V1 = 20260430
RIDGE_LAMBDA = 1e-3

# Label formula (must match V2 contract's audit_should_have_skipped_v2).
SHOULD_SKIP_PNL_THRESHOLD = 0.0
SHOULD_SKIP_MAE_THRESHOLD = -50.0
SHOULD_SKIP_MFE_THRESHOLD = 25.0


# AT_TRADE_OPEN feature plan
ENTRY_RECOVERY_PASSTHROUGH: list[str] = [
    "p_long_entry_v1",
    "p_hat_entry_v1",
    "uncertainty_entry_v1",
    "margin_entry_v1",
]
TRADE_OUTCOMES_CONTINUOUS: list[str] = ["entry_spread_bps"]
TRADE_OUTCOMES_ONEHOT: dict[str, list[str]] = {
    "side": ["long", "short"],
    "session": ["ASIA", "EU", "OVERLAP", "US"],
}
BASE34_AT_ENTRY_CONTINUOUS: list[str] = [
    "atr_bps",
    "_v1_close_ema_slope_3",
    "minutes_since_session_open",
    "minutes_to_next_session_boundary",
    "_v1_atr_z_10_100",
    "_v1_bb_squeeze_20_2",
    "_v1_bb_bandwidth_delta_10",
    "_v1_kama_slope_30",
    "_v1_ema_diff",
    "_v1_r12",
    "_v1_pk_sigma20",
]
BASE34_AT_ENTRY_BINARY: list[str] = [
    "is_ASIA",
    "_v1_is_EU",
    "_v1_is_US",
    "session_change_flag",
    "session_tradable",
]
BASE34_AT_ENTRY_ONEHOT: dict[str, list[str]] = {
    "_v1_atr_regime_id": ["LOW", "MEDIUM", "HIGH"],
}

THRESHOLD_GRID: list[float] = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
RECOVERY_SENTINEL_VALUE = -1.0


ALLOWED_FINAL_STATUSES = {
    "LEARN_TRADE_SKIP_META_CLASSIFIER_PASS_TUNED_THRESHOLD_LIFTS_TEST_PNL",
    "LEARN_TRADE_SKIP_META_CLASSIFIER_PARTIAL_TUNED_THRESHOLD_TIES_REALIZED",
    "LEARN_TRADE_SKIP_META_CLASSIFIER_PARTIAL_TUNED_THRESHOLD_DEGRADES_PNL",
    "LEARN_TRADE_SKIP_META_CLASSIFIER_BLOCKED_BY_LABEL_FAIL",
    "LEARN_TRADE_SKIP_META_CLASSIFIER_BLOCKED_BY_INPUT_LOCK_MISSING",
}

ALLOWED_NEXT_ACTIONS = {
    "RUN_PER_BAR_XGB_REPLAY_FOR_TRANSFORMER_SIGNAL_FAMILY_V1",
    "COMBINE_SKIP_CLASSIFIER_WITH_EXIT_IQL_V2_V1",
    "REPAIR_SKIP_CLASSIFIER_BEFORE_PROMOTION_V1",
    "HOLD_SKIP_CLASSIFIER_RESEARCH_UNTIL_DATA_FIXED_V1",
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


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------


def compute_should_skip_label(trade_row: pd.Series) -> int:
    """Mirror audit_should_have_skipped_v2 from DEEPEN V2 contract."""
    pnl = float(trade_row["pnl_bps"])
    mae = float(trade_row["mae_bps"])
    mfe = float(trade_row["mfe_bps"])
    if (
        pnl < SHOULD_SKIP_PNL_THRESHOLD
        and mae <= SHOULD_SKIP_MAE_THRESHOLD
        and mfe < SHOULD_SKIP_MFE_THRESHOLD
    ):
        return 1
    return 0


def compute_should_skip_labels_vectorized(trades: pd.DataFrame) -> np.ndarray:
    pnl = trades["pnl_bps"].astype(float).to_numpy()
    mae = trades["mae_bps"].astype(float).to_numpy()
    mfe = trades["mfe_bps"].astype(float).to_numpy()
    mask = (
        (pnl < SHOULD_SKIP_PNL_THRESHOLD)
        & (mae <= SHOULD_SKIP_MAE_THRESHOLD)
        & (mfe < SHOULD_SKIP_MFE_THRESHOLD)
    )
    return mask.astype(int)


def validate_label_formula_against_v2_contract(
    v2_contract: dict[str, Any]
) -> dict[str, Any]:
    audit_labels = v2_contract.get("audit_only_labels_v1", [])
    target = next(
        (a for a in audit_labels if a["label_name_v2"] == "audit_should_have_skipped_v2"),
        None,
    )
    if target is None:
        raise RuntimeError("AUDIT_SHOULD_HAVE_SKIPPED_V2_NOT_FOUND_IN_V2_CONTRACT")
    formula = target["formula_v2"]
    expected_tokens = ["pnl_bps < 0", "mae_bps <= -50", "mfe_bps < 25"]
    matches = all(tok in formula for tok in expected_tokens)
    return {
        "audit_id_v1": "LABEL_FORMULA_VS_V2_CONTRACT_AUDIT_V1",
        "status_v1": "PASS" if matches else "FAIL",
        "v2_formula_v1": formula,
        "expected_tokens_v1": expected_tokens,
        "all_tokens_present_v1": matches,
    }


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _load_inputs() -> dict[str, Any]:
    roots = [INPUT_RECOVERY_ROOT, INPUT_SPLIT_ROOT, INPUT_V2_CONTRACT_ROOT]
    validate_explicit_artifact_roots(roots)
    required = {
        "recovery_per_trade": INPUT_RECOVERY_ROOT
        / "entry_snapshot_signals_per_trade_v1.parquet",
        "recovery_summary": INPUT_RECOVERY_ROOT / "summary_v1.json",
        "split_locked_dataset": INPUT_SPLIT_ROOT
        / "split_locked_augmented_dataset_v1.parquet",
        "v2_state_contract": INPUT_V2_CONTRACT_ROOT / "state_feature_contract_v2.json",
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
        "recovery_summary": _read_json(required["recovery_summary"]),
        "v2_state_contract": _read_json(required["v2_state_contract"]),
        "base34_path": BASE34_M5_FEATURES_PATH,
    }


def _list_truth_weeks() -> list[Path]:
    return sorted(
        DEFAULT_REPORTS_ROOT.glob("TRUTH_MONFRI_WEEK_*"),
        key=lambda p: p.name,
    )


def _load_trade_outcomes_concat() -> pd.DataFrame:
    weeks = _list_truth_weeks()
    parts: list[pd.DataFrame] = []
    for w in weeks:
        path = w / f"trade_outcomes_{w.name}_MERGED.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df["week_name_v1"] = w.name
        parts.append(df)
    if not parts:
        raise RuntimeError("NO_TRADE_OUTCOMES_FOUND")
    full = pd.concat(parts, ignore_index=True)
    full = full.sort_values(
        ["week_name_v1", "open_ts_utc", "candidate_uid"], kind="mergesort"
    ).reset_index(drop=True)
    return full


# ---------------------------------------------------------------------------
# Feature projection
# ---------------------------------------------------------------------------


def _project_per_trade_features(
    trades: pd.DataFrame, recovery: pd.DataFrame, base34_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a per-trade frame with the AT_TRADE_OPEN feature columns and
    a should_skip label. Trades without recovery get sentinel for the four
    recovery fields.
    """
    out = trades.copy()
    out["candidate_uid_v1"] = out["candidate_uid"].astype(str)
    out["open_ts_utc"] = pd.to_datetime(out["open_ts_utc"], utc=True)

    rec = recovery.loc[
        :,
        [
            "candidate_uid",
            *ENTRY_RECOVERY_PASSTHROUGH,
            "recovery_status_v1",
        ],
    ].copy()
    rec["candidate_uid_v1"] = rec["candidate_uid"].astype(str)
    rec = rec.drop(columns=["candidate_uid"], errors="ignore").drop_duplicates(
        subset="candidate_uid_v1", keep="first"
    )
    out = out.merge(rec, on="candidate_uid_v1", how="left")
    not_recovered_mask = (
        out["recovery_status_v1"].fillna("MISSING")
        != "RECOVERED_FROM_XGB_PREDICTIONS"
    )
    not_recovered_count = int(not_recovered_mask.sum())
    for c in ENTRY_RECOVERY_PASSTHROUGH:
        out.loc[not_recovered_mask, c] = np.nan

    # BASE34 join at open_ts_utc.
    base34 = pd.read_parquet(base34_path)
    if "time" not in base34.columns:
        if base34.index.name == "time":
            base34 = base34.reset_index()
        else:
            raise RuntimeError("BASE34_M5_PARQUET_MISSING_TIME_COLUMN")
    cols_needed = list(
        set(
            BASE34_AT_ENTRY_CONTINUOUS
            + BASE34_AT_ENTRY_BINARY
            + list(BASE34_AT_ENTRY_ONEHOT.keys())
        )
    )
    missing = [c for c in cols_needed if c not in base34.columns]
    if missing:
        raise RuntimeError(f"BASE34_M5_MISSING_COLUMNS: {missing}")
    base34_use = base34.loc[:, ["time", *cols_needed]].copy()
    base34_use["time"] = pd.to_datetime(base34_use["time"], utc=True)
    base34_use = base34_use.sort_values("time", kind="mergesort").reset_index(drop=True)

    out_sorted = out.sort_values("open_ts_utc", kind="mergesort").reset_index(drop=True)
    joined = pd.merge_asof(
        out_sorted,
        base34_use,
        left_on="open_ts_utc",
        right_on="time",
        direction="backward",
        tolerance=pd.Timedelta(minutes=5),
    )
    joined = joined.drop(columns=["time"], errors="ignore")
    joined = joined.sort_values(
        ["week_name_v1", "open_ts_utc", "candidate_uid_v1"], kind="mergesort"
    ).reset_index(drop=True)

    # Compute should_skip label.
    joined["should_skip_v1"] = compute_should_skip_labels_vectorized(joined)

    audit = {
        "audit_id_v1": "PER_TRADE_FEATURE_PROJECTION_AUDIT_V1",
        "status_v1": "PASS",
        "trade_count_v1": int(len(joined)),
        "trades_without_recovery_v1": not_recovered_count,
        "should_skip_count_v1": int(joined["should_skip_v1"].sum()),
        "should_skip_rate_v1": float(joined["should_skip_v1"].mean()),
        "policy_v1": (
            "Trades without RECOVERED_FROM_XGB_PREDICTIONS get NaN for the four "
            "recovery fields, sentinel-substituted at state-matrix build."
        ),
    }
    return joined, audit


def _join_split_assignment(
    per_trade: pd.DataFrame, split_locked_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_df = pd.read_parquet(
        split_locked_path,
        columns=["candidate_uid_v1", "primary_split_v1"],
    )
    split_df["candidate_uid_v1"] = split_df["candidate_uid_v1"].astype(str)
    split_per_trade = split_df.drop_duplicates(subset="candidate_uid_v1", keep="first")
    out = per_trade.merge(split_per_trade, on="candidate_uid_v1", how="left")
    out_has_split = out[out["primary_split_v1"].notna()].copy()
    skipped_no_split = int(len(out) - len(out_has_split))
    audit = {
        "audit_id_v1": "SPLIT_JOIN_AUDIT_V1",
        "status_v1": "PASS",
        "trades_with_split_v1": int(len(out_has_split)),
        "trades_without_split_v1": skipped_no_split,
        "policy_v1": (
            "Per-trade rows without primary_split_v1 (trades not in the gate-4 "
            "split-locked dataset) are excluded from training/eval. They are "
            "candidates that did not become accepted trades or were not "
            "augmented in gate-3."
        ),
        "split_counts_v1": (
            out_has_split["primary_split_v1"]
            .value_counts()
            .to_dict()
        ),
    }
    return out_has_split, audit


# ---------------------------------------------------------------------------
# Train-only normalization + state matrix
# ---------------------------------------------------------------------------


def _fit_train_normalization(per_trade_train: pd.DataFrame) -> dict[str, Any]:
    norm: dict[str, Any] = {}
    z_columns = TRADE_OUTCOMES_CONTINUOUS + BASE34_AT_ENTRY_CONTINUOUS
    for col in z_columns:
        if col not in per_trade_train.columns:
            raise RuntimeError(f"NORMALIZATION_COLUMN_MISSING: {col}")
        s = per_trade_train[col].astype(float)
        median = float(s.median())
        s_filled = s.fillna(median)
        mean = float(s_filled.mean())
        std = float(s_filled.std(ddof=0)) or 1.0
        norm[col] = {"transform": "z", "mean": mean, "std": std, "median": median}
    return norm


def _zscore(values: pd.Series, cfg: dict[str, Any]) -> np.ndarray:
    s = values.astype(float).fillna(cfg["median"])
    return ((s - cfg["mean"]) / cfg["std"]).clip(-5.0, 5.0).to_numpy()


def _onehot(values: pd.Series, vocab: list[str]) -> np.ndarray:
    n = len(values)
    out = np.zeros((n, len(vocab)), dtype=float)
    s = values.astype(str).str.upper()
    for i, k in enumerate(vocab):
        out[:, i] = (s == k.upper()).astype(float).to_numpy()
    return out


def _binary(values: pd.Series) -> np.ndarray:
    return values.astype(float).fillna(0.0).clip(0.0, 1.0).to_numpy()


def _passthrough_zero_one_with_sentinel(values: pd.Series) -> np.ndarray:
    s = values.astype(float)
    return np.where(s.notna(), s.clip(0.0, 1.0).to_numpy(), RECOVERY_SENTINEL_VALUE).astype(float)


def _build_state_matrix(
    per_trade: pd.DataFrame, norm: dict[str, Any]
) -> tuple[np.ndarray, list[str]]:
    columns: list[str] = ["intercept"]
    blocks: list[np.ndarray] = [np.ones((len(per_trade), 1))]

    for col in ENTRY_RECOVERY_PASSTHROUGH:
        columns.append(f"{col}__pass_or_sentinel")
        blocks.append(_passthrough_zero_one_with_sentinel(per_trade[col]).reshape(-1, 1))

    for col in TRADE_OUTCOMES_CONTINUOUS:
        columns.append(f"{col}__z")
        blocks.append(_zscore(per_trade[col], norm[col]).reshape(-1, 1))

    for col, vocab in TRADE_OUTCOMES_ONEHOT.items():
        oh = _onehot(per_trade[col], vocab)
        for cat in vocab:
            columns.append(f"{col}__{cat.upper()}")
        blocks.append(oh)

    for col in BASE34_AT_ENTRY_CONTINUOUS:
        columns.append(f"{col}__z")
        blocks.append(_zscore(per_trade[col], norm[col]).reshape(-1, 1))

    for col in BASE34_AT_ENTRY_BINARY:
        columns.append(f"{col}__bin")
        blocks.append(_binary(per_trade[col]).reshape(-1, 1))

    for col, vocab in BASE34_AT_ENTRY_ONEHOT.items():
        oh = _onehot(per_trade[col], vocab)
        for cat in vocab:
            columns.append(f"{col}__{cat.upper()}")
        blocks.append(oh)

    X = np.concatenate(blocks, axis=1)
    if not np.isfinite(X).all():
        n_bad = int((~np.isfinite(X)).sum())
        raise RuntimeError(f"STATE_MATRIX_HAS_NON_FINITE_VALUES: count={n_bad}")
    return X, columns


# ---------------------------------------------------------------------------
# Ridge fit + threshold sweep
# ---------------------------------------------------------------------------


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray:
    a = X.T @ X + lam * np.eye(X.shape[1])
    b = X.T @ y
    return np.linalg.solve(a, b)


def _evaluate_threshold(
    per_trade_split: pd.DataFrame,
    p_skip: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Skip trades where p_skip >= threshold; sum realized pnl on the rest."""
    actual = per_trade_split["should_skip_v1"].astype(int).to_numpy()
    pred_skip = (p_skip >= threshold).astype(int)
    tp = int(((pred_skip == 1) & (actual == 1)).sum())
    fp = int(((pred_skip == 1) & (actual == 0)).sum())
    tn = int(((pred_skip == 0) & (actual == 0)).sum())
    fn = int(((pred_skip == 0) & (actual == 1)).sum())
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if precision and recall
        else None
    )
    pnl = per_trade_split["pnl_bps"].astype(float).to_numpy()
    pnl_taken = float(pnl[pred_skip == 0].sum())
    pnl_skipped = float(pnl[pred_skip == 1].sum())
    pnl_no_skip = float(pnl.sum())
    pnl_lift = pnl_taken - pnl_no_skip
    n_skipped = int((pred_skip == 1).sum())
    n_taken = int((pred_skip == 0).sum())
    return {
        "threshold_v1": float(threshold),
        "trade_count_v1": int(total),
        "tp_v1": tp,
        "fp_v1": fp,
        "tn_v1": tn,
        "fn_v1": fn,
        "precision_v1": precision,
        "recall_v1": recall,
        "f1_v1": f1,
        "trades_skipped_v1": n_skipped,
        "trades_taken_v1": n_taken,
        "pnl_no_skip_v1": pnl_no_skip,
        "pnl_taken_v1": pnl_taken,
        "pnl_skipped_v1": pnl_skipped,
        "pnl_lift_vs_no_skip_v1": pnl_lift,
    }


def _evaluate_oracle_skip(per_trade_split: pd.DataFrame) -> dict[str, Any]:
    actual = per_trade_split["should_skip_v1"].astype(int).to_numpy()
    pnl = per_trade_split["pnl_bps"].astype(float).to_numpy()
    pnl_taken = float(pnl[actual == 0].sum())
    pnl_no_skip = float(pnl.sum())
    return {
        "policy_v1": "ORACLE_SKIP_ALL_SHOULD_HAVE_SKIPPED",
        "trades_skipped_v1": int((actual == 1).sum()),
        "trades_taken_v1": int((actual == 0).sum()),
        "pnl_no_skip_v1": pnl_no_skip,
        "pnl_taken_v1": pnl_taken,
        "pnl_lift_vs_no_skip_v1": float(pnl_taken - pnl_no_skip),
    }


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------

FORBIDDEN_FEATURE_TOKENS = (
    "pnl_bps",
    "mae_bps",
    "mfe_bps",
    "post_exit",
    "exit_reason",
    "duration_bars",
    "running_",
    "bars_held",
    "is_terminal",
    "exit_price",
    "exit_bid",
    "exit_ask",
    "close_ts",
)


def audit_no_shortcut_at_train_time(
    feature_names: Sequence[str],
) -> dict[str, Any]:
    leak: list[str] = []
    for name in feature_names:
        for tok in FORBIDDEN_FEATURE_TOKENS:
            if tok in name:
                leak.append(name)
                break
    if leak:
        raise RuntimeError(f"SKIP_CLASSIFIER_FEATURE_LEAK: {leak}")
    return {
        "audit_id_v1": "SKIP_CLASSIFIER_NO_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS",
        "feature_count_v1": len(feature_names),
        "feature_names_v1": list(feature_names),
        "forbidden_tokens_checked_v1": list(FORBIDDEN_FEATURE_TOKENS),
        "feature_leak_v1": leak,
    }


def audit_train_only_normalization(
    per_trade_full: pd.DataFrame, norm: dict[str, Any]
) -> dict[str, Any]:
    train = per_trade_full[per_trade_full["primary_split_v1"] == "train"]
    if len(train) == 0:
        raise RuntimeError("EMPTY_TRAIN_SPLIT_FOR_NORMALIZATION_AUDIT")
    sample_col = "entry_spread_bps"
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
        "audit_id_v1": "SKIP_CLASSIFIER_TRAIN_ONLY_NORMALIZATION_V1",
        "status_v1": "PASS",
        "checked_field_v1": sample_col,
        "train_mean_v1": expected_mean,
    }


def audit_split_isolation(per_trade: pd.DataFrame) -> dict[str, Any]:
    bad = (
        per_trade.groupby("candidate_uid_v1")["primary_split_v1"]
        .nunique()
        .gt(1)
        .sum()
    )
    if int(bad) > 0:
        raise RuntimeError(
            f"SPLIT_ISOLATION_VIOLATION: {bad} trades span multiple splits"
        )
    return {
        "audit_id_v1": "SKIP_CLASSIFIER_SPLIT_ISOLATION_V1",
        "status_v1": "PASS",
        "spanning_trade_count_v1": int(bad),
    }


# ---------------------------------------------------------------------------
# go-no-go
# ---------------------------------------------------------------------------


def _go_no_go(
    test_at_locked: dict[str, Any], oracle_test: dict[str, Any]
) -> tuple[str, str, str, dict[str, Any]]:
    pnl_no_skip = test_at_locked["pnl_no_skip_v1"]
    pnl_locked = test_at_locked["pnl_taken_v1"]
    pnl_oracle = oracle_test["pnl_taken_v1"]
    lift = pnl_locked - pnl_no_skip
    headline = {
        "tuned_threshold_v1": test_at_locked["threshold_v1"],
        "test_pnl_no_skip_v1": pnl_no_skip,
        "test_pnl_with_skip_classifier_v1": pnl_locked,
        "test_pnl_lift_vs_no_skip_v1": lift,
        "test_pnl_oracle_skip_v1": pnl_oracle,
        "test_oracle_lift_v1": oracle_test["pnl_lift_vs_no_skip_v1"],
        "captured_fraction_of_oracle_lift_v1": (
            lift / oracle_test["pnl_lift_vs_no_skip_v1"]
            if oracle_test["pnl_lift_vs_no_skip_v1"]
            else None
        ),
        "test_precision_v1": test_at_locked["precision_v1"],
        "test_recall_v1": test_at_locked["recall_v1"],
        "test_f1_v1": test_at_locked["f1_v1"],
        "test_trades_skipped_v1": test_at_locked["trades_skipped_v1"],
        "test_trades_taken_v1": test_at_locked["trades_taken_v1"],
    }
    if lift > 100.0:
        return (
            "LEARN_TRADE_SKIP_META_CLASSIFIER_PASS_TUNED_THRESHOLD_LIFTS_TEST_PNL",
            "COMBINE_SKIP_CLASSIFIER_WITH_EXIT_IQL_V2_V1",
            (
                f"Tuned threshold {test_at_locked['threshold_v1']} lifts test "
                f"PNL by {lift:.0f} bps (from {pnl_no_skip:.0f} to "
                f"{pnl_locked:.0f}). Skipped {test_at_locked['trades_skipped_v1']} "
                f"of {test_at_locked['trade_count_v1']} trades; precision "
                f"{test_at_locked['precision_v1']}, recall "
                f"{test_at_locked['recall_v1']}. Captured "
                f"{(lift / oracle_test['pnl_lift_vs_no_skip_v1']*100):.1f}% of the oracle lift "
                f"({oracle_test['pnl_lift_vs_no_skip_v1']:.0f} bps). Next: "
                "combine skip classifier with exit IQL V2 in a research-only "
                "stack to measure compounded lift."
            ),
            headline,
        )
    if abs(lift) <= 50.0:
        return (
            "LEARN_TRADE_SKIP_META_CLASSIFIER_PARTIAL_TUNED_THRESHOLD_TIES_REALIZED",
            "REPAIR_SKIP_CLASSIFIER_BEFORE_PROMOTION_V1",
            (
                f"Tuned threshold {test_at_locked['threshold_v1']} produces "
                f"test PNL {pnl_locked:.0f} bps, ~= no-skip baseline "
                f"{pnl_no_skip:.0f} (lift {lift:.0f}). Skip classifier did not "
                "produce a meaningful lift; investigate features, label, or "
                "ridge formulation."
            ),
            headline,
        )
    if lift > 0:
        return (
            "LEARN_TRADE_SKIP_META_CLASSIFIER_PASS_TUNED_THRESHOLD_LIFTS_TEST_PNL",
            "COMBINE_SKIP_CLASSIFIER_WITH_EXIT_IQL_V2_V1",
            (
                f"Tuned threshold {test_at_locked['threshold_v1']} lifts test "
                f"PNL by {lift:.0f} bps (small but positive lift)."
            ),
            headline,
        )
    return (
        "LEARN_TRADE_SKIP_META_CLASSIFIER_PARTIAL_TUNED_THRESHOLD_DEGRADES_PNL",
        "REPAIR_SKIP_CLASSIFIER_BEFORE_PROMOTION_V1",
        (
            f"Tuned threshold {test_at_locked['threshold_v1']} degrades test "
            f"PNL by {-lift:.0f} bps (from {pnl_no_skip:.0f} to "
            f"{pnl_locked:.0f}). Classifier rejects more good trades than bad."
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
        "layer_name": "LEARN_TRADE_SKIP_META_CLASSIFIER_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "recovery_root_v1": str(INPUT_RECOVERY_ROOT),
            "split_root_v1": str(INPUT_SPLIT_ROOT),
            "v2_contract_root_v1": str(INPUT_V2_CONTRACT_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_contract_v1": True,
        "iql_training_run_v1": False,
        "iql_production_allowed_v1": False,
        "skip_classifier_promoted_to_runtime_v1": False,
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


# ---------------------------------------------------------------------------
# Materializer
# ---------------------------------------------------------------------------


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

    # 1. Label-formula audit
    label_formula_audit = validate_label_formula_against_v2_contract(
        inputs["v2_state_contract"]
    )
    _write_json(artifact_root / "label_formula_audit_v1.json", label_formula_audit)

    # 2. Load source data
    trades = _load_trade_outcomes_concat()
    recovery = pd.read_parquet(inputs["required_paths"]["recovery_per_trade"])
    per_trade, projection_audit = _project_per_trade_features(
        trades, recovery, BASE34_M5_FEATURES_PATH
    )
    per_trade_split, split_join_audit = _join_split_assignment(
        per_trade, inputs["required_paths"]["split_locked_dataset"]
    )
    split_isolation_audit = audit_split_isolation(per_trade_split)

    # 3. Train-only normalization
    per_trade_train = per_trade_split[per_trade_split["primary_split_v1"] == "train"]
    norm = _fit_train_normalization(per_trade_train)
    train_only_audit = audit_train_only_normalization(per_trade_split, norm)
    _write_json(artifact_root / "training_normalization_v1.json", norm)

    # 4. Build state matrix
    X_full, feature_names = _build_state_matrix(per_trade_split, norm)
    train_mask = (per_trade_split["primary_split_v1"] == "train").to_numpy()
    val_mask = (per_trade_split["primary_split_v1"] == "val").to_numpy()
    test_mask = (per_trade_split["primary_split_v1"] == "test").to_numpy()
    no_shortcut_audit = audit_no_shortcut_at_train_time(feature_names)

    # 5. Train ridge regression on binary label
    y_full = per_trade_split["should_skip_v1"].astype(float).to_numpy()
    coef = _ridge_fit(X_full[train_mask], y_full[train_mask])
    p_skip_full = X_full @ coef
    p_skip_full = np.clip(p_skip_full, 0.0, 1.0)

    model_summary = {
        "model_v1": "CLOSED_FORM_RIDGE_BINARY_REGRESSION",
        "feature_count_v1": len(feature_names),
        "feature_names_v1": feature_names,
        "ridge_lambda_v1": RIDGE_LAMBDA,
        "seed_v1": SEED_V1,
        "train_row_count_v1": int(train_mask.sum()),
        "coef_v1": coef.tolist(),
        "coef_l2_norm_v1": float(np.linalg.norm(coef)),
    }
    _write_json(artifact_root / "trained_model_v1.json", model_summary)

    # Predicted-probability distribution per split.
    pred_dist: dict[str, dict[str, float]] = {}
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        if not mask.any():
            continue
        p = p_skip_full[mask]
        pred_dist[split_name] = {
            "min_v1": float(p.min()),
            "p05_v1": float(np.quantile(p, 0.05)),
            "p25_v1": float(np.quantile(p, 0.25)),
            "p50_v1": float(np.quantile(p, 0.50)),
            "p75_v1": float(np.quantile(p, 0.75)),
            "p95_v1": float(np.quantile(p, 0.95)),
            "max_v1": float(p.max()),
            "mean_v1": float(p.mean()),
            "std_v1": float(p.std(ddof=0)),
            "n_v1": int(mask.sum()),
        }
    _write_json(
        artifact_root / "predicted_skip_probability_distribution_v1.json", pred_dist
    )

    # 6. Threshold sweep on each split
    threshold_metrics: dict[str, list[dict[str, Any]]] = {}
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        per_trade_split_sub = per_trade_split[mask].reset_index(drop=True)
        if per_trade_split_sub.empty:
            continue
        p_skip_split = p_skip_full[mask]
        rows: list[dict[str, Any]] = []
        for thr in THRESHOLD_GRID:
            m = _evaluate_threshold(per_trade_split_sub, p_skip_split, thr)
            m["split_v1"] = split_name
            rows.append(m)
        threshold_metrics[split_name] = rows
    flat_threshold_rows = [r for rows in threshold_metrics.values() for r in rows]
    _write_rows(
        artifact_root / "threshold_sweep_metrics_v1.csv", flat_threshold_rows
    )
    _write_json(
        artifact_root / "threshold_sweep_metrics_v1.json",
        {"row_count_v1": len(flat_threshold_rows), "rows_v1": flat_threshold_rows},
    )

    # 7. Tune threshold on val: pick the one that maximizes pnl_taken on val
    val_rows = threshold_metrics.get("val", [])
    if not val_rows:
        raise RuntimeError("EMPTY_VAL_SPLIT_FOR_THRESHOLD_TUNING")
    best_val = max(val_rows, key=lambda r: r["pnl_taken_v1"])
    tuned_threshold = float(best_val["threshold_v1"])

    # 8. Locked test eval at val-tuned threshold
    per_trade_test = per_trade_split[test_mask].reset_index(drop=True)
    p_skip_test = p_skip_full[test_mask]
    test_at_locked = _evaluate_threshold(
        per_trade_test, p_skip_test, tuned_threshold
    )
    test_at_locked["split_v1"] = "test"
    _write_json(
        artifact_root / "locked_test_evaluation_v1.json",
        {**test_at_locked, "tuned_threshold_v1": tuned_threshold},
    )

    # 9. Oracle skip on each split
    oracle_per_split = {
        "train": _evaluate_oracle_skip(
            per_trade_split[train_mask].reset_index(drop=True)
        ),
        "val": _evaluate_oracle_skip(
            per_trade_split[val_mask].reset_index(drop=True)
        ),
        "test": _evaluate_oracle_skip(per_trade_test),
    }
    _write_json(
        artifact_root / "oracle_skip_per_split_v1.json", oracle_per_split
    )

    # 10. Aggregate audits
    audits = [
        label_formula_audit,
        projection_audit,
        split_join_audit,
        split_isolation_audit,
        train_only_audit,
        no_shortcut_audit,
    ]
    _write_json(
        artifact_root / "training_audits_v1.json",
        {"audit_count_v1": len(audits), "audits_v1": audits},
    )

    repro = {
        "layer_name": "LEARN_TRADE_SKIP_META_CLASSIFIER_REPRODUCIBILITY_AUDIT_V1",
        "model_v1": "CLOSED_FORM_RIDGE_BINARY_REGRESSION",
        "feature_count_v1": len(feature_names),
        "ridge_lambda_v1": RIDGE_LAMBDA,
        "seed_v1": SEED_V1,
        "tuned_threshold_v1": tuned_threshold,
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status, next_action, recommendation, headline = _go_no_go(
        test_at_locked, oracle_per_split["test"]
    )
    validate_final_status(status, next_action)

    # 11. Summary / status / go-no-go / report
    summary = {
        "layer_name": "LEARN_TRADE_SKIP_META_CLASSIFIER_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "model_id_v1": "SKIP_CLASSIFIER_RIDGE_BINARY",
        "feature_count_v1": len(feature_names),
        "ridge_lambda_v1": RIDGE_LAMBDA,
        "tuned_threshold_v1": tuned_threshold,
        "test_at_locked_threshold_v1": test_at_locked,
        "oracle_per_split_v1": oracle_per_split,
        "audits_v1": {a["audit_id_v1"]: a["status_v1"] for a in audits},
        "research_only_v1": True,
        "iql_training_run_v1": False,
        "iql_production_allowed_v1": False,
        "skip_classifier_promoted_to_runtime_v1": False,
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
        "layer_name": "LEARN_TRADE_SKIP_META_CLASSIFIER_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_TRAINING_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": True,
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "LEARN_TRADE_SKIP_META_CLASSIFIER_GO_NO_GO_V1",
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
        "skip_classifier_promotion_allowed_v1": False,
        "downstream_block_v1": (
            "Research-only skip classifier. The trained classifier is NOT "
            "promoted to runtime; entry_manager / exit_manager / live_features "
            "all unmodified. Adapter/R6/IQL production/live, freeze/promo/live "
            "all forbidden."
        ),
    }
    _write_json(
        artifact_root / "learn_trade_skip_meta_classifier_go_no_go_v1.json", go_no_go
    )

    # Report
    report_lines = [
        "# Learn Trade-Skip Meta-Classifier At Trade Open V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: research-only; classifier NOT promoted to runtime.",
        "",
        "## Headline (test split)",
        f"- Tuned threshold (val-best): {tuned_threshold}",
        f"- No-skip PNL: {headline['test_pnl_no_skip_v1']:.0f} bps",
        f"- With-skip PNL: {headline['test_pnl_with_skip_classifier_v1']:.0f} bps",
        f"- Lift vs no-skip: {headline['test_pnl_lift_vs_no_skip_v1']:+.0f} bps",
        f"- Oracle (skip all should-skip) PNL: {headline['test_pnl_oracle_skip_v1']:.0f} bps",
        f"- Oracle lift: {headline['test_oracle_lift_v1']:+.0f} bps",
        f"- Captured fraction of oracle lift: {headline['captured_fraction_of_oracle_lift_v1']}",
        f"- Test precision: {headline['test_precision_v1']}",
        f"- Test recall: {headline['test_recall_v1']}",
        f"- Test F1: {headline['test_f1_v1']}",
        f"- Trades skipped: {headline['test_trades_skipped_v1']} of {test_at_locked['trade_count_v1']}",
        "",
        "## Threshold sweep (test split)",
        "",
        "| Threshold | Skip | Take | PNL no-skip | PNL with-skip | Lift | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in threshold_metrics.get("test", []):
        report_lines.append(
            f"| {r['threshold_v1']} | {r['trades_skipped_v1']} | {r['trades_taken_v1']} | "
            f"{r['pnl_no_skip_v1']:.0f} | {r['pnl_taken_v1']:.0f} | "
            f"{r['pnl_lift_vs_no_skip_v1']:+.0f} | "
            f"{r['precision_v1']} | {r['recall_v1']} | {r['f1_v1']} |"
        )
    report_lines.extend(
        [
            "",
            "## Threshold sweep (val split)",
            "",
            "| Threshold | Skip | Take | PNL no-skip | PNL with-skip | Lift | Precision | Recall | F1 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for r in threshold_metrics.get("val", []):
        report_lines.append(
            f"| {r['threshold_v1']} | {r['trades_skipped_v1']} | {r['trades_taken_v1']} | "
            f"{r['pnl_no_skip_v1']:.0f} | {r['pnl_taken_v1']:.0f} | "
            f"{r['pnl_lift_vs_no_skip_v1']:+.0f} | "
            f"{r['precision_v1']} | {r['recall_v1']} | {r['f1_v1']} |"
        )
    report_lines.extend(
        [
            "",
            "## Oracle ceiling per split",
            f"- Train oracle lift: {oracle_per_split['train']['pnl_lift_vs_no_skip_v1']:+.0f} bps",
            f"- Val oracle lift: {oracle_per_split['val']['pnl_lift_vs_no_skip_v1']:+.0f} bps",
            f"- Test oracle lift: {oracle_per_split['test']['pnl_lift_vs_no_skip_v1']:+.0f} bps",
            "",
            "## Audits",
        ]
    )
    for a in audits:
        report_lines.append(f"- `{a['audit_id_v1']}`: {a['status_v1']}")
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
            "go_no_go": str(
                artifact_root / "learn_trade_skip_meta_classifier_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "trained_model": str(artifact_root / "trained_model_v1.json"),
            "training_normalization": str(
                artifact_root / "training_normalization_v1.json"
            ),
            "threshold_sweep_metrics_csv": str(
                artifact_root / "threshold_sweep_metrics_v1.csv"
            ),
            "threshold_sweep_metrics_json": str(
                artifact_root / "threshold_sweep_metrics_v1.json"
            ),
            "predicted_skip_probability_distribution": str(
                artifact_root / "predicted_skip_probability_distribution_v1.json"
            ),
            "locked_test_evaluation": str(
                artifact_root / "locked_test_evaluation_v1.json"
            ),
            "oracle_skip_per_split": str(
                artifact_root / "oracle_skip_per_split_v1.json"
            ),
            "training_audits": str(artifact_root / "training_audits_v1.json"),
            "label_formula_audit": str(artifact_root / "label_formula_audit_v1.json"),
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
        description="Materialize LEARN_TRADE_SKIP_META_CLASSIFIER_AT_TRADE_OPEN_V1."
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
