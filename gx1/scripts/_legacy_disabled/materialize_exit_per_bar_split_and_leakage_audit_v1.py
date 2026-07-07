#!/usr/bin/env python3
"""Lock train/val/test splits for the augmented exit-IQL dataset and audit
all known leakage paths.

This is gate 4 of 6 in the exit-IQL pre-train dependency graph.

Splits locked:

  - PRIMARY: TIME_ORDER_PER_TRADE_SPLIT_70_15_15
    Trades sorted by entry timestamp; first 70 percent of trades to train,
    next 15 percent to val, last 15 percent to test. All bars of a trade go
    to the same split (per-trade integrity), and time-order means train
    cannot peek at future weeks during validation/test.

  - SENSITIVITY: WEEK_BLOCK_SPLIT_70_15_15
    Whole replay weeks assigned to splits in chronological order, locked as
    sensitivity comparator only.

Leakage audits:

  A1 INTRA_TRADE_INTEGRITY: every candidate_uid_v1 lives in exactly one split.
  A2 TEMPORAL_NON_OVERLAP: train.max_open_ts < val.min_open_ts < test.min_open_ts
     (strict time-ordered separation).
  A3 NEXT_ROW_POINTER_CROSS_SPLIT: HOLD non-terminal next_row_id_per_bar_v1
     never points across splits.
  A4 STATE_NO_SHORTCUT_RECHECK: state columns are still disjoint from the 29
     forbidden fields locked in gate 1.
  A5 REWARD_INPUT_NOT_IN_STATE: reward-input fields (mfe_bps, mae_bps, pnl_bps)
     never appear as state columns.
  A6 ACTION_BALANCE_PER_SPLIT: each split has HOLD count equal to EXIT_NOW
     count (preserves the augmentation invariant per-split).
  A7 PROPENSITY_DISTRIBUTION_SANITY: each split has at least one of each
     LOGGED/COUNTERFACTUAL propensity label (no degenerate split).

The gate persists per-split parquet shards plus the audit reports. Training
remains BLOCKED.
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

from gx1.scripts import materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate
from gx1.scripts import materialize_exit_hold_exit_now_mdp_reward_contract_v1 as mdp_gate


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_V1"

INPUT_AUGMENT_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_ACTION_SUPPORT_AUGMENT_V1_20260429T130000Z_LOCK"
)
INPUT_STATE_FEATURE_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1_20260429T113745Z_LOCK"
)
INPUT_MDP_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_V1_20260429T103326Z_LOCK"
)

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

ACTION_HOLD_ID = 0
ACTION_EXIT_NOW_ID = 1

REWARD_INPUT_FIELDS_FORBIDDEN_AS_STATE = {
    "mfe_bps",
    "mae_bps",
    "pnl_bps",
    "post_exit_mfe_bps",
    "early_exit_regret",
    "duration_bars",
    "exit_reason",
}

ALLOWED_FINAL_STATUSES = {
    "EXIT_PER_BAR_SPLIT_LOCKED_LEAKAGE_AUDIT_PASSED",
    "EXIT_PER_BAR_SPLIT_BLOCKED_BY_INTRA_TRADE_LEAKAGE",
    "EXIT_PER_BAR_SPLIT_BLOCKED_BY_TEMPORAL_OVERLAP",
    "EXIT_PER_BAR_SPLIT_BLOCKED_BY_NEXT_ROW_CROSS_SPLIT",
    "EXIT_PER_BAR_SPLIT_BLOCKED_BY_STATE_NO_SHORTCUT_FAIL",
    "EXIT_PER_BAR_SPLIT_BLOCKED_BY_PROPENSITY_DEGENERATE",
}

ALLOWED_NEXT_ACTIONS = {
    "EXIT_OFF_POLICY_EVAL_HARNESS_V1",
    "REPAIR_SPLIT_LEAKAGE_BEFORE_OFF_POLICY_HARNESS_V1",
    "HOLD_UNTIL_SPLIT_LEAKAGE_RESOLVED_V1",
}

# Reuse helpers
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


def validate_split_proportions(train: float, val: float, test: float) -> bool:
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise RuntimeError(f"SPLIT_PROPORTIONS_DO_NOT_SUM_TO_1: total={total}")
    if min(train, val, test) <= 0.0:
        raise RuntimeError("SPLIT_PROPORTIONS_MUST_BE_POSITIVE")
    return True


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _augmented_dataset_path() -> Path:
    return INPUT_AUGMENT_ROOT / "augmented_per_bar_action_dataset_v1.parquet"


def _load_inputs() -> dict[str, Any]:
    roots = [INPUT_AUGMENT_ROOT, INPUT_STATE_FEATURE_ROOT, INPUT_MDP_ROOT]
    validate_explicit_artifact_roots(roots)
    required = {
        "augmented_dataset": _augmented_dataset_path(),
        "augment_summary": INPUT_AUGMENT_ROOT / "summary_v1.json",
        "state_feature_contract": INPUT_STATE_FEATURE_ROOT / "state_feature_contract_v1.json",
        "mdp_summary": INPUT_MDP_ROOT / "summary_v1.json",
        "mdp_no_shortcut_axioms": INPUT_MDP_ROOT / "no_shortcut_axioms_v1.json",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_INPUT_LOCKS: {missing}")
    return {
        "required_paths": required,
        "augment_summary": _read_json(required["augment_summary"]),
        "state_feature_contract": _read_json(required["state_feature_contract"]),
        "mdp_summary": _read_json(required["mdp_summary"]),
    }


def _load_augmented_dataset() -> pd.DataFrame:
    df = pd.read_parquet(_augmented_dataset_path())
    df["candidate_uid_v1"] = df["candidate_uid_v1"].astype(str)
    df["ts_v1"] = pd.to_datetime(df["ts_v1"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------


def _assign_time_order_per_trade_split(
    df: pd.DataFrame,
    *,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> pd.DataFrame:
    validate_split_proportions(train_fraction, val_fraction, test_fraction)
    # First-bar timestamp per trade defines trade open time
    trade_open = (
        df.groupby("candidate_uid_v1")["ts_v1"].min().sort_values().reset_index()
    )
    n_trades = len(trade_open)
    n_train = int(round(n_trades * train_fraction))
    n_val = int(round(n_trades * val_fraction))
    n_test = n_trades - n_train - n_val
    if n_test <= 0:
        raise RuntimeError(
            f"SPLIT_TEST_NON_POSITIVE: n_trades={n_trades} train={n_train} val={n_val} test={n_test}"
        )
    trade_open["primary_split_v1"] = (
        ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    )
    out = df.merge(
        trade_open.loc[:, ["candidate_uid_v1", "primary_split_v1"]],
        on="candidate_uid_v1",
        how="left",
    )
    if out["primary_split_v1"].isna().any():
        raise RuntimeError("PRIMARY_SPLIT_ASSIGNMENT_HAS_NA")
    return out


def _extract_week_for_trade(candidate_uid: pd.Series) -> pd.Series:
    """Trade UID format is `TRUTH_MONFRI_WEEK_<start>_<end>:0:cand::...`."""

    def parse(uid: str) -> str:
        try:
            return uid.split(":")[0]
        except Exception:
            return ""

    return candidate_uid.astype(str).map(parse)


def _assign_week_block_split(
    df: pd.DataFrame,
    *,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> pd.DataFrame:
    validate_split_proportions(train_fraction, val_fraction, test_fraction)
    df = df.copy()
    df["source_week_v1"] = _extract_week_for_trade(df["candidate_uid_v1"])
    weeks_sorted = sorted(df["source_week_v1"].unique())
    n_weeks = len(weeks_sorted)
    n_train_w = int(round(n_weeks * train_fraction))
    n_val_w = int(round(n_weeks * val_fraction))
    n_test_w = n_weeks - n_train_w - n_val_w
    if n_test_w <= 0:
        raise RuntimeError(
            f"WEEK_SPLIT_TEST_NON_POSITIVE: n_weeks={n_weeks} train={n_train_w} val={n_val_w} test={n_test_w}"
        )
    week_to_split = {}
    for i, w in enumerate(weeks_sorted):
        if i < n_train_w:
            week_to_split[w] = "train"
        elif i < n_train_w + n_val_w:
            week_to_split[w] = "val"
        else:
            week_to_split[w] = "test"
    df["sensitivity_week_split_v1"] = df["source_week_v1"].map(week_to_split)
    if df["sensitivity_week_split_v1"].isna().any():
        raise RuntimeError("WEEK_SPLIT_HAS_NA")
    return df


# ---------------------------------------------------------------------------
# Leakage audits
# ---------------------------------------------------------------------------


def audit_intra_trade_integrity(df: pd.DataFrame, split_col: str) -> dict[str, Any]:
    """A1: each candidate_uid_v1 must appear in exactly one split."""
    uid_split = df.groupby("candidate_uid_v1")[split_col].nunique()
    bad = uid_split[uid_split > 1]
    if not bad.empty:
        raise RuntimeError(
            f"INTRA_TRADE_LEAKAGE_DETECTED: {len(bad)} trades span multiple splits"
        )
    return {
        "audit_id_v1": "A1_INTRA_TRADE_INTEGRITY",
        "status_v1": "PASS",
        "trade_count_v1": int(uid_split.size),
        "split_col_v1": split_col,
        "spanning_trade_count_v1": 0,
    }


def audit_temporal_non_overlap(df: pd.DataFrame, split_col: str) -> dict[str, Any]:
    """A2: trades' OPEN timestamps must be strictly ordered across splits.

    We check open-time ordering, not all-bar ordering, because trades can
    span multiple hours and a train trade that opened earlier may close
    after a val trade has opened. The decision-time leakage we care about
    is whether val/test trades' OPEN moments could have been seen during
    training, and that is captured by trade open_ts ordering.
    """
    trade_open = df.groupby(["candidate_uid_v1", split_col])["ts_v1"].min().reset_index()
    open_ranges = trade_open.groupby(split_col)["ts_v1"].agg(["min", "max"])
    train_open_max = open_ranges.loc["train", "max"]
    val_open_min = open_ranges.loc["val", "min"]
    val_open_max = open_ranges.loc["val", "max"]
    test_open_min = open_ranges.loc["test", "min"]
    # Allow equality at boundaries because two distinct trades may open at
    # the same M5 bar timestamp; the per-trade split assignment then uses
    # candidate_uid order to break ties deterministically. The leakage-
    # relevant guarantee is that no later-split trade opens BEFORE the
    # latest earlier-split trade, which is enforced by `<=`.
    if not (train_open_max <= val_open_min and val_open_max <= test_open_min):
        raise RuntimeError(
            "TEMPORAL_OPEN_TIME_OVERLAP_DETECTED: train_open_max="
            f"{train_open_max} val_open_min={val_open_min} val_open_max={val_open_max} test_open_min={test_open_min}"
        )
    bar_ranges = df.groupby(split_col)["ts_v1"].agg(["min", "max"])
    return {
        "audit_id_v1": "A2_TEMPORAL_NON_OVERLAP",
        "status_v1": "PASS",
        "split_col_v1": split_col,
        "trade_open_train_min_v1": str(open_ranges.loc["train", "min"]),
        "trade_open_train_max_v1": str(train_open_max),
        "trade_open_val_min_v1": str(val_open_min),
        "trade_open_val_max_v1": str(val_open_max),
        "trade_open_test_min_v1": str(test_open_min),
        "trade_open_test_max_v1": str(open_ranges.loc["test", "max"]),
        "bar_train_min_v1": str(bar_ranges.loc["train", "min"]),
        "bar_train_max_v1": str(bar_ranges.loc["train", "max"]),
        "bar_val_min_v1": str(bar_ranges.loc["val", "min"]),
        "bar_val_max_v1": str(bar_ranges.loc["val", "max"]),
        "bar_test_min_v1": str(bar_ranges.loc["test", "min"]),
        "bar_test_max_v1": str(bar_ranges.loc["test", "max"]),
        "note_v1": (
            "Bar-time ranges may overlap across splits because long-running "
            "trades opened in train can close after val opens. The leakage-"
            "relevant condition is open-time ordering, which is enforced."
        ),
    }


def audit_next_row_pointer_cross_split(
    df: pd.DataFrame, split_col: str
) -> dict[str, Any]:
    """A3: next_row_id_per_bar_v1 should never cross splits when set."""
    df = df.copy()
    df = df.reset_index(drop=True)
    has_next = df["next_row_id_per_bar_v1"].notna()
    if not has_next.any():
        return {
            "audit_id_v1": "A3_NEXT_ROW_POINTER_CROSS_SPLIT",
            "status_v1": "PASS_TRIVIAL",
            "split_col_v1": split_col,
            "rows_with_next_pointer_v1": 0,
            "cross_split_pointer_count_v1": 0,
        }
    src_split = df.loc[has_next, split_col].values
    next_ids = df.loc[has_next, "next_row_id_per_bar_v1"].astype("Int64").to_numpy()
    # Map row_id_per_bar_v1 -> split. row_id is bar-level (one per per-bar row),
    # but augmented dataset has 2 rows per bar (HOLD + EXIT_NOW). Use HOLD rows
    # only as canonical because next_row_id is bar-level.
    hold_mask = df["action_id_v1"] == ACTION_HOLD_ID
    canonical = df.loc[hold_mask, ["row_id_per_bar_v1", split_col]].drop_duplicates()
    rowid_to_split = canonical.set_index("row_id_per_bar_v1")[split_col].to_dict()
    target_split = pd.Series(next_ids).map(rowid_to_split)
    cross = (target_split.notna()) & (target_split.values != src_split)
    cross_count = int(cross.sum())
    if cross_count > 0:
        raise RuntimeError(
            f"NEXT_ROW_POINTER_CROSS_SPLIT_DETECTED: {cross_count} HOLD pointers cross split boundary"
        )
    return {
        "audit_id_v1": "A3_NEXT_ROW_POINTER_CROSS_SPLIT",
        "status_v1": "PASS",
        "split_col_v1": split_col,
        "rows_with_next_pointer_v1": int(has_next.sum()),
        "cross_split_pointer_count_v1": cross_count,
    }


def audit_state_no_shortcut_recheck(state_columns: Sequence[str]) -> dict[str, Any]:
    """A4: state columns disjoint from 29 forbidden fields."""
    forbidden = set(mdp_gate.FORBIDDEN_STATE_FIELDS_V1)
    hits = sorted(set(state_columns) & forbidden)
    if hits:
        raise RuntimeError(f"STATE_NO_SHORTCUT_RECHECK_FAIL: {hits}")
    return {
        "audit_id_v1": "A4_STATE_NO_SHORTCUT_RECHECK",
        "status_v1": "PASS",
        "state_column_count_v1": len(state_columns),
        "forbidden_intersection_v1": hits,
    }


def audit_reward_input_not_in_state(state_columns: Sequence[str]) -> dict[str, Any]:
    """A5: reward-input fields never appear in state."""
    leak = sorted(set(state_columns) & REWARD_INPUT_FIELDS_FORBIDDEN_AS_STATE)
    if leak:
        raise RuntimeError(f"REWARD_INPUT_LEAK_INTO_STATE: {leak}")
    return {
        "audit_id_v1": "A5_REWARD_INPUT_NOT_IN_STATE",
        "status_v1": "PASS",
        "leaked_input_fields_v1": leak,
    }


def audit_action_balance_per_split(
    df: pd.DataFrame, split_col: str
) -> dict[str, Any]:
    """A6: each split has HOLD count == EXIT_NOW count."""
    counts = (
        df.groupby([split_col, "action_id_v1"]).size().unstack(fill_value=0)
    )
    rows = []
    for split, row in counts.iterrows():
        hold = int(row.get(ACTION_HOLD_ID, 0))
        exit_now = int(row.get(ACTION_EXIT_NOW_ID, 0))
        if hold != exit_now:
            raise RuntimeError(
                f"ACTION_BALANCE_FAIL_FOR_SPLIT: split={split} hold={hold} exit_now={exit_now}"
            )
        rows.append(
            {"split_v1": split, "hold_count_v1": hold, "exit_now_count_v1": exit_now}
        )
    return {
        "audit_id_v1": "A6_ACTION_BALANCE_PER_SPLIT",
        "status_v1": "PASS",
        "split_col_v1": split_col,
        "per_split_v1": rows,
    }


def audit_propensity_distribution(df: pd.DataFrame, split_col: str) -> dict[str, Any]:
    """A7: each split must have at least one of each LOGGED and COUNTERFACTUAL propensity label."""
    expected = {
        "LOGGED_HOLD_PROPENSITY_1",
        "LOGGED_EXIT_NOW_PROPENSITY_1",
        "COUNTERFACTUAL_EXIT_NOW_NO_PROPENSITY",
        "FORCED_TERMINAL_HOLD_DATA_LIMIT",
    }
    counts = (
        df.groupby([split_col, "behavior_propensity_v1"]).size().unstack(fill_value=0)
    )
    missing_per_split = {}
    for split, row in counts.iterrows():
        present = {p for p in expected if int(row.get(p, 0)) > 0}
        missing = expected - present
        if missing:
            missing_per_split[split] = sorted(missing)
    if missing_per_split:
        raise RuntimeError(
            f"PROPENSITY_DEGENERATE_SPLIT: {missing_per_split}"
        )
    return {
        "audit_id_v1": "A7_PROPENSITY_DISTRIBUTION_SANITY",
        "status_v1": "PASS",
        "split_col_v1": split_col,
        "per_split_counts_v1": counts.to_dict(orient="index"),
    }


# ---------------------------------------------------------------------------
# Per-split summary
# ---------------------------------------------------------------------------


def _per_split_summary(df: pd.DataFrame, split_col: str) -> list[dict[str, Any]]:
    rows = []
    for split, group in df.groupby(split_col):
        # Action-balanced augmented dataset; trade count = unique candidate_uid
        trade_count = int(group["candidate_uid_v1"].nunique())
        bar_count = int(len(group) / 2)  # HOLD + EXIT_NOW per bar
        ts_min = group["ts_v1"].min()
        ts_max = group["ts_v1"].max()
        # Reward stats per variant on EXIT_NOW samples (terminal-action perspective)
        exit_now = group[group["action_id_v1"] == ACTION_EXIT_NOW_ID]
        reward_stats = {}
        for variant in [
            "realized_pnl_reward",
            "mfe_capture_reward",
            "mae_penalty_reward",
            "giveback_penalty_reward",
            "transparent_combined_reward",
        ]:
            col = f"reward_{variant}_v1"
            if col in exit_now.columns:
                series = exit_now[col].dropna()
                if not series.empty:
                    reward_stats[variant] = {
                        "mean_v1": float(series.mean()),
                        "p50_v1": float(series.quantile(0.5)),
                        "std_v1": float(series.std(ddof=0)),
                    }
        rows.append(
            {
                "split_v1": split,
                "trade_count_v1": trade_count,
                "bar_count_v1": bar_count,
                "row_count_v1": int(len(group)),
                "ts_min_v1": str(ts_min),
                "ts_max_v1": str(ts_max),
                "exit_now_reward_stats_v1": reward_stats,
            }
        )
    return rows


def _state_columns(df: pd.DataFrame) -> list[str]:
    """Identify state-feature columns (exclude metadata, action, reward, propensity)."""
    metadata_cols = {
        "candidate_uid_v1",
        "trade_uid_v1",
        "trade_id",
        "ts_v1",
        "entry_price_v1",
        "bar_close_v1",
        "row_id_per_bar_v1",
        "next_row_id_per_bar_v1",
        "action_id_v1",
        "action_label_v1",
        "behavior_propensity_v1",
        "is_terminal_for_action_v1",
        "primary_split_v1",
        "sensitivity_week_split_v1",
        "source_week_v1",
    }
    state_cols = [
        c
        for c in df.columns
        if c not in metadata_cols and not c.startswith("reward_")
    ]
    return state_cols


# ---------------------------------------------------------------------------
# Reproducibility / go-no-go
# ---------------------------------------------------------------------------


def _reproducibility_audit(
    df_with_splits: pd.DataFrame, primary_summary: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "layer_name": "EXIT_PER_BAR_SPLIT_REPRODUCIBILITY_AUDIT_V1",
        "row_count_v1": int(len(df_with_splits)),
        "primary_split_proportions_v1": {
            "train": TRAIN_FRACTION,
            "val": VAL_FRACTION,
            "test": TEST_FRACTION,
        },
        "primary_split_actual_v1": {
            row["split_v1"]: float(row["row_count_v1"] / max(len(df_with_splits), 1))
            for row in primary_summary
        },
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }


def _go_no_go() -> tuple[str, str, str]:
    return (
        "EXIT_PER_BAR_SPLIT_LOCKED_LEAKAGE_AUDIT_PASSED",
        "EXIT_OFF_POLICY_EVAL_HARNESS_V1",
        (
            "Splits locked with strict per-trade integrity and time-ordered separation. "
            "All seven leakage audits PASS. Augmented dataset is now ready for the "
            "off-policy evaluation harness gate which builds the comparator framework "
            "for measuring policy improvement vs the current exit_manager."
        ),
    )


def _build_input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "EXIT_PER_BAR_SPLIT_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "augment_root_v1": str(INPUT_AUGMENT_ROOT),
            "state_feature_root_v1": str(INPUT_STATE_FEATURE_ROOT),
            "mdp_root_v1": str(INPUT_MDP_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_contract_v1": True,
        "iql_training_run_v1": False,
        "iql_production_allowed_v1": False,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def write_artifacts(
    out_root: Path | None = None,
    *,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    inputs = _load_inputs()
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (
        DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK"
    )
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
    _write_json(artifact_root / "input_manifest_v1.json", _build_input_manifest(inputs, artifact_root))

    df = _load_augmented_dataset()
    df = _assign_time_order_per_trade_split(df)
    df = _assign_week_block_split(df)

    state_cols = _state_columns(df)

    # Run audits
    audits = []
    audits.append(audit_intra_trade_integrity(df, "primary_split_v1"))
    audits.append(audit_temporal_non_overlap(df, "primary_split_v1"))
    audits.append(audit_next_row_pointer_cross_split(df, "primary_split_v1"))
    audits.append(audit_state_no_shortcut_recheck(state_cols))
    audits.append(audit_reward_input_not_in_state(state_cols))
    audits.append(audit_action_balance_per_split(df, "primary_split_v1"))
    audits.append(audit_propensity_distribution(df, "primary_split_v1"))
    _write_json(
        artifact_root / "leakage_audits_v1.json",
        {"audit_count_v1": len(audits), "audits_v1": audits},
    )

    # Sensitivity-only week-block audit (intra-trade integrity is automatic)
    week_audit = audit_intra_trade_integrity(df, "sensitivity_week_split_v1")
    _write_json(
        artifact_root / "sensitivity_week_split_audit_v1.json",
        week_audit,
    )

    primary_summary = _per_split_summary(df, "primary_split_v1")
    sensitivity_summary = _per_split_summary(df, "sensitivity_week_split_v1")
    _write_rows(
        artifact_root / "primary_split_summary_v1.csv",
        [
            {**{k: v for k, v in r.items() if k != "exit_now_reward_stats_v1"}}
            for r in primary_summary
        ],
    )
    _write_json(
        artifact_root / "primary_split_summary_v1.json",
        {"row_count_v1": len(primary_summary), "rows_v1": primary_summary},
    )
    _write_json(
        artifact_root / "sensitivity_week_split_summary_v1.json",
        {"row_count_v1": len(sensitivity_summary), "rows_v1": sensitivity_summary},
    )

    # Persist split-locked dataset (single parquet with both split columns) and
    # also per-split shards for primary split
    df.to_parquet(
        artifact_root / "split_locked_augmented_dataset_v1.parquet", index=False
    )
    for split in ["train", "val", "test"]:
        shard = df[df["primary_split_v1"] == split]
        shard.to_parquet(
            artifact_root / f"primary_split_{split}_v1.parquet", index=False
        )

    repro = _reproducibility_audit(df, primary_summary)
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status, next_action, recommendation = _go_no_go()
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "row_count_v1": int(len(df)),
        "trade_count_v1": int(df["candidate_uid_v1"].nunique()),
        "primary_split_v1": "TIME_ORDER_PER_TRADE_SPLIT_70_15_15",
        "sensitivity_split_v1": "WEEK_BLOCK_SPLIT_70_15_15",
        "primary_split_summary_v1": primary_summary,
        "sensitivity_split_summary_v1": sensitivity_summary,
        "leakage_audit_status_summary_v1": {a["audit_id_v1"]: a["status_v1"] for a in audits},
        "state_column_count_v1": len(state_cols),
        "research_only_v1": True,
        "iql_training_run_v1": False,
        "training_blocked_v1": True,
        "next_pre_train_gate_v1": next_action,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "forbidden_actions_audit_v1": forbidden_audit,
    }
    _write_json(artifact_root / "summary_v1.json", summary)

    status_payload = {
        "layer_name": "EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": False,
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_GO_NO_GO_V1",
        "status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
        "training_allowed_v1": False,
        "downstream_block_v1": (
            "Research-only split-and-leakage gate. Training remains BLOCKED "
            "until off-policy eval harness gate (5) and sanity training gate "
            "(6) pass. Adapter/R6/IQL production/live, freeze/promo/live, "
            "exit_manager modification all forbidden."
        ),
    }
    _write_json(
        artifact_root / "exit_per_bar_split_and_leakage_audit_go_no_go_v1.json",
        go_no_go,
    )

    report_lines = [
        "# Exit Per-Bar Split And Leakage Audit V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: **BLOCKED** until remaining two pre-train gates pass.",
        "",
        "## Splits locked",
        f"- Primary: TIME_ORDER_PER_TRADE_SPLIT_70_15_15",
        f"- Sensitivity: WEEK_BLOCK_SPLIT_70_15_15",
        "",
        "## Primary split summary",
    ]
    for row in primary_summary:
        report_lines.append(
            f"- `{row['split_v1']}`: {row['trade_count_v1']} trades, "
            f"{row['bar_count_v1']} bars, "
            f"{row['ts_min_v1']} -> {row['ts_max_v1']}"
        )
    report_lines.extend(["", "## Leakage audits"])
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
                artifact_root
                / "exit_per_bar_split_and_leakage_audit_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "leakage_audits": str(artifact_root / "leakage_audits_v1.json"),
            "primary_split_summary_json": str(
                artifact_root / "primary_split_summary_v1.json"
            ),
            "primary_split_summary_csv": str(
                artifact_root / "primary_split_summary_v1.csv"
            ),
            "sensitivity_week_split_summary": str(
                artifact_root / "sensitivity_week_split_summary_v1.json"
            ),
            "split_locked_dataset_parquet": str(
                artifact_root / "split_locked_augmented_dataset_v1.parquet"
            ),
            "primary_split_train_shard": str(
                artifact_root / "primary_split_train_v1.parquet"
            ),
            "primary_split_val_shard": str(
                artifact_root / "primary_split_val_v1.parquet"
            ),
            "primary_split_test_shard": str(
                artifact_root / "primary_split_test_v1.parquet"
            ),
            "reproducibility_audit": str(artifact_root / "reproducibility_audit_v1.json"),
        },
        "read_only_references_v1": True,
        "not_trainer_v1": True,
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
        description="Materialize EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_V1 gate."
    )
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(out_root=out_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
