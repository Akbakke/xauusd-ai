#!/usr/bin/env python3
"""LIVE smart_seq520 entry adapter — serving-wave gap 4 (vedtak SMART_JOINT_POLICY_PROMOTION_20260708).

Loads the CONTRACT-RESOLVED ACTIVE v10_entry smart_seq520 bundle (cand#4) through
the one-truth offline loader (gx1.models.entry_v10.entry_v10_bundle.
load_entry_v10_ctx_bundle — strict load + direction/path calibration installed
into the forward), forwards it per M5 close on the live smart520 state
(Smart520StateBuilder, gap 2) + live multi-TF windows, and applies the PINNED
operating point read from PROJECT_STATE_artifacts.json (v10_entry.operating_point
— session gate US/OVERLAP + edge_score threshold; ONE truth, never re-declared
here or in the launcher).

Extend-don't-fork note (CLAUDE.md rule 7): the existing live wrapper
v12_v10_live.V10LiveInference implements the RETIRED legacy 41-dim
MASTER_TRANSFORMER_LOCK contract with a hand-built model constructor; the smart
bundle's one-truth load path is load_entry_v10_ctx_bundle (calibration +
specialist fusion + parked-head handling), which the offline evaluator
(evaluate_entry_candidate_selective_edge_v1._predict_bundle) also uses — this
adapter mirrors THAT forward exactly, so serve == the promoted evidence path.

edge_score / side (one-truth mirror of _predict_bundle, evaluate_entry_candidate_
selective_edge_v1.py:716-718):
    probs      = softmax(direction_logits)        # calibrated inside the model
    edge_score = max(p_long, p_short) - p_flat
    side       = LONG if p_long >= p_short else SHORT

Exit-bound snapshot: cand#4 heads -> v10_snapshot keys EXACTLY as the joint-replay
driver proved offline (reports/joint_smart_policy_replay_20260708/scripts/
replay_driver.py build_snapshot):
    direction_probs=[p_long,p_short,p_flat], path_quality=path_quality_pred,
    mfe_first_n=mfe_first_n_pred (raw), tradable_prob, bad_path_prob (carried,
    NOT consumed by the ACTIVE exit state), tf_agreement_pred/path_quality_std/
    position_size_pred = 0.0 (NOT consumed), atr_bps = live cv3 atr_bps at the
    prediction bar T. hold_horizon_bars_pred is DELIBERATELY ABSENT -> TradeState
    keeps the -1 sentinel -> the HOLD_HORIZON_EXPIRED Strategy-F rule stays INERT.
    That delta is live-equivalent BY CONSTRUCTION: cand#4's hold_horizon head is
    BLOCKED (bundle metadata blocked_heads) and the a1/deferral reference replays
    were snapshot-inert on it too — do NOT "fix" this by wiring a substitute value.
"""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gx1.execution.v12_smart520_state_live import (
    SEQ_LEN_SMART520,
    SIGNAL_DIM_SMART520,
    Smart520StateContract,
    Smart520StateBuilder,
    append_multi_tf_incremental,
    build_multi_tf_from_cv3,
)

LOG = logging.getLogger("v12_smart_entry_live")

SESSION_NAMES = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}
SIDE_ACTION = {0: "TAKE_LONG_NOW", 1: "TAKE_SHORT_NOW"}

SMART_PARITY_GATE_LATEST = Path(
    "/home/andre2/GX1_DATA/reports/smart520_serve_parity_v1/SMART520_SERVE_PARITY_latest.json"
)
SMART_DIRECTION_AUDIT_LATEST = Path(
    "/home/andre2/GX1_DATA/reports/smart_direction_live_like_pocket_audit_v1/"
    "SMART_DIRECTION_LIVE_LIKE_POCKET_AUDIT_latest.json"
)
SMART_PARITY_GATE_MAX_AGE_HOURS = float(os.environ.get("GX1_SMART_PARITY_GATE_MAX_AGE_HOURS", "18"))
SMART_PARITY_GATE_MAX_CUTOFF_LAG_HOURS = float(
    os.environ.get("GX1_SMART_PARITY_GATE_MAX_CUTOFF_LAG_HOURS", "18")
)
SMART_DIRECTION_AUDIT_MAX_AGE_HOURS = float(os.environ.get("GX1_SMART_DIRECTION_AUDIT_MAX_AGE_HOURS", "18"))

# Fail-closed context-staleness cap for LIVE decisions (serving-wave gap 3): when the
# last COMPLETED smart-context snapshot lags the decision bar by MORE than this many
# cv3 M5 bars, entry decisions are SKIPPED (journaled smart_ctx_stale_refresh_pending)
# until the background refresh lands — never decide on rotten context. Steady state is
# age<=1: the ~2-min refresh finishes well inside one M5 cycle.
SMART_CTX_MAX_STALENESS_M5 = int(os.environ.get("GX1_SMART_CTX_MAX_STALENESS_M5", "0"))

# Kill-switch for the (self-test-proven) incremental MTF splice at age>=1;
# 0 falls back to the raw snapshot bundle (staleness stays journaled via
# context_age_m5_bars). See SMART520_MTF_SPLICE_TFS in v12_smart520_state_live.
SMART_CTX_MTF_INCREMENTAL = os.environ.get("GX1_SMART_CTX_MTF_INCREMENTAL", "1") == "1"
SMART_EXPECTED_UTILITY_BAD_PATH_PENALTY_BPS = float(
    os.environ.get("GX1_ENTRY_EXPECTED_UTILITY_BAD_PATH_PENALTY_BPS", "15.0")
)
SMART_EXPECTED_UTILITY_UNCERTAINTY_PENALTY_BPS = float(
    os.environ.get("GX1_ENTRY_EXPECTED_UTILITY_UNCERTAINTY_PENALTY_BPS", "5.0")
)
SMART_EXPECTED_UTILITY_RAIL_PENALTY_BPS = float(
    os.environ.get("GX1_ENTRY_EXPECTED_UTILITY_RAIL_PENALTY_BPS", "25.0")
)
SMART_EXPECTED_UTILITY_INVALID_SIDE_PENALTY_BPS = float(
    os.environ.get("GX1_ENTRY_EXPECTED_UTILITY_INVALID_SIDE_PENALTY_BPS", "35.0")
)
SMART_EXPECTED_UTILITY_THRESHOLD_BPS = float(
    os.environ.get("GX1_ENTRY_EXPECTED_UTILITY_THRESHOLD_BPS", "0.0")
)


class SmartContextStaleError(RuntimeError):
    """Raised by predict_live_bar when the context snapshot is older than
    SMART_CTX_MAX_STALENESS_M5 bars behind the decision bar — the pipeline
    journals it as a SKIP (fail-closed) and retries on the next poll."""

    def __init__(self, age: int, cap: int, ctx_cutoff: pd.Timestamp, end_ts: pd.Timestamp):
        super().__init__(
            f"[SMART_ENTRY] context snapshot {age} M5 bars behind decision bar {end_ts} "
            f"(cutoff {ctx_cutoff}, cap {cap}) — refusing to decide on stale context"
        )
        self.age = int(age)
        self.cap = int(cap)
        self.ctx_cutoff = ctx_cutoff
        self.end_ts = end_ts


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _np1d(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy().reshape(-1)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _softmax_np(values: np.ndarray | None) -> np.ndarray | None:
    if values is None or len(values) == 0:
        return None
    arr = values.astype(np.float64, copy=False)
    arr = arr - np.nanmax(arr)
    exp = np.exp(arr)
    denom = float(np.nansum(exp))
    if denom <= 0.0 or not np.isfinite(denom):
        return None
    return (exp / denom).astype(np.float32)


def _sigmoid_float(value: float) -> float:
    value = float(np.clip(value, -80.0, 80.0))
    return float(1.0 / (1.0 + np.exp(-value)))


def _feature_value(row: np.ndarray, names: list[str], candidates: list[str]) -> float | None:
    for name in candidates:
        if name in names:
            idx = int(names.index(name))
            if 0 <= idx < len(row):
                val = float(row[idx])
                return val if np.isfinite(val) else None
    return None


def _feature_max(row: np.ndarray, names: list[str], candidates: list[str]) -> float | None:
    values: list[float] = []
    for name in candidates:
        if name in names:
            idx = int(names.index(name))
            if 0 <= idx < len(row):
                val = float(row[idx])
                if np.isfinite(val):
                    values.append(val)
    return float(max(values)) if values else None


def _mean_optional(*values: float | None) -> float | None:
    clean = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not clean:
        return None
    return float(np.mean(clean))


def _optional_diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    if not np.isfinite(float(left)) or not np.isfinite(float(right)):
        return None
    return float(left) - float(right)


@dataclass(frozen=True)
class SmartCtxSnapshot:
    """One COMPLETED smart-context build — swapped in as a single atomic reference
    (the loader's 2026-06-01 async-refresh pattern) so a decision that grabbed the
    snapshot can never observe a half-refreshed context. Immutable by convention:
    the background refresh builds a NEW snapshot and replaces the reference."""
    multi_tf: dict
    frame_overrides: pd.DataFrame       # bucket ctx_cat + HTF/REGIME_V4 override cols
    cv3_cutoff: pd.Timestamp
    built_utc: pd.Timestamp
    build_seconds: float


def _smart_gate_git_state() -> tuple[str, bool]:
    repo = Path(__file__).resolve().parents[2]
    commit = "unknown"
    dirty = True
    try:
        commit_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_proc.returncode == 0:
            commit = commit_proc.stdout.strip()
        dirty_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if dirty_proc.returncode == 0:
            dirty = bool(dirty_proc.stdout.strip())
    except Exception:
        dirty = True
    return commit, dirty


def assert_smart_serving_gate() -> dict:
    """ONE-TRUTH launch gate for the smart serving path (launcher + runner):
    (1) the TRAIN==SERVE parity gate artifact must be decision=PASS and must
        have been produced for the CONTRACT-ACTIVE v10_entry bundle;
    (2) the directional live-like pocket audit must be decision=PASS for the
        CONTRACT-ACTIVE v10_entry bundle;
    (3) the contract must be smart_seq520_candidate with a complete
        operating_point.
    Raises RuntimeError on any violation; returns the gate report on success.
    """
    import json
    from gx1_guards.artifacts import load_decision_entry
    if not SMART_PARITY_GATE_LATEST.is_file():
        raise RuntimeError(
            f"[SMART_GATE] parity gate artifact missing: {SMART_PARITY_GATE_LATEST} — run "
            f"gx1.scripts.verify_smart520_serve_parity_v1 (capped) first"
        )
    rep = json.loads(SMART_PARITY_GATE_LATEST.read_text())
    entry = load_decision_entry("v10_entry")
    problems: list[str] = []
    if rep.get("decision") != "PASS":
        problems.append(f"parity decision={rep.get('decision')!r} failures={list(rep.get('failures') or [])[:3]}")
    current_commit, worktree_dirty = _smart_gate_git_state()
    parity_commit = str(rep.get("git_commit") or "").strip()
    if not parity_commit:
        problems.append("parity report missing git_commit")
    elif current_commit != parity_commit:
        problems.append(f"parity git_commit {parity_commit} != current git_commit {current_commit}")
    if worktree_dirty:
        problems.append("smart serving git worktree is dirty; rerun parity on the exact source before launch")
    now_utc = pd.Timestamp.now(tz="UTC")
    created_utc = pd.to_datetime(rep.get("created_utc"), utc=True, errors="coerce")
    if pd.isna(created_utc):
        problems.append(f"parity created_utc invalid/missing: {rep.get('created_utc')!r}")
    elif SMART_PARITY_GATE_MAX_AGE_HOURS > 0:
        age_hours = (now_utc - created_utc).total_seconds() / 3600.0
        if age_hours > SMART_PARITY_GATE_MAX_AGE_HOURS:
            problems.append(
                f"parity report stale: age_hours={age_hours:.2f} "
                f"> cap={SMART_PARITY_GATE_MAX_AGE_HOURS:.2f}"
            )
    cutoff_utc = pd.to_datetime(rep.get("live_prebuilt_cutoff"), utc=True, errors="coerce")
    if pd.isna(cutoff_utc):
        problems.append(f"parity live_prebuilt_cutoff invalid/missing: {rep.get('live_prebuilt_cutoff')!r}")
    elif SMART_PARITY_GATE_MAX_CUTOFF_LAG_HOURS > 0:
        cutoff_lag_hours = (now_utc - cutoff_utc).total_seconds() / 3600.0
        if cutoff_lag_hours > SMART_PARITY_GATE_MAX_CUTOFF_LAG_HOURS:
            problems.append(
                f"parity prebuilt cutoff stale: cutoff_lag_hours={cutoff_lag_hours:.2f} "
                f"> cap={SMART_PARITY_GATE_MAX_CUTOFF_LAG_HOURS:.2f}"
            )
    if str(rep.get("bundle_dir")) != str(entry["path"]):
        problems.append(f"parity bundle {rep.get('bundle_dir')} != contract-ACTIVE {entry['path']}")
    bundle_meta_path = Path(str(entry["path"])) / "bundle_metadata.json"
    bundle_state_contract = {}
    if not bundle_meta_path.is_file():
        problems.append(f"contract-ACTIVE bundle metadata missing: {bundle_meta_path}")
    else:
        try:
            bundle_meta = json.loads(bundle_meta_path.read_text(encoding="utf-8"))
            raw_contract = bundle_meta.get("smart520_state_contract")
            bundle_state_contract = raw_contract if isinstance(raw_contract, dict) else {}
        except Exception as exc:
            problems.append(f"contract-ACTIVE bundle metadata unreadable: {bundle_meta_path}: {exc}")
    parity_state_contract = rep.get("smart520_state_contract")
    if not isinstance(parity_state_contract, dict):
        problems.append("parity report missing smart520_state_contract")
    else:
        for key in (
            "frame_anchor_utc",
            "model_range_start_utc",
            "rank_reference_end_utc",
            "rank_reference_npz",
            "rank_reference_npz_sha256",
        ):
            parity_value = str(parity_state_contract.get(key) or "").strip()
            bundle_value = str(bundle_state_contract.get(key) or "").strip()
            if not parity_value:
                problems.append(f"parity smart520_state_contract missing {key}")
            if bundle_value and parity_value and parity_value != bundle_value:
                problems.append(
                    f"parity smart520_state_contract.{key} {parity_value} != bundle metadata {bundle_value}"
                )
            if parity_value and not bundle_value:
                problems.append(f"bundle smart520_state_contract missing {key}")
        rank_ref_low = str(parity_state_contract.get("rank_reference_npz") or "").lower()
        for stale_marker in ("julyext", "smart_candidate_20260630", "utilityrepair", "20260710"):
            if stale_marker in rank_ref_low:
                problems.append(
                    f"parity smart520_state_contract rank_reference_npz references stale marker "
                    f"{stale_marker!r}: {parity_state_contract.get('rank_reference_npz')}"
                )
        rank_ref = Path(str(parity_state_contract.get("rank_reference_npz") or "")).expanduser()
        expected_sha = str(parity_state_contract.get("rank_reference_npz_sha256") or "").strip().lower()
        if str(parity_state_contract.get("rank_reference_npz") or "").strip() and not rank_ref.is_file():
            problems.append(f"parity smart520_state_contract rank_reference_npz missing: {rank_ref}")
        if rank_ref.is_file() and expected_sha:
            actual_sha = _sha256_file(rank_ref)
            if actual_sha != expected_sha:
                problems.append(
                    "parity smart520_state_contract rank_reference_npz_sha256 mismatch: "
                    f"metadata={expected_sha} actual={actual_sha} path={rank_ref}"
                )
            sidecar = rank_ref.with_suffix(rank_ref.suffix + ".json")
            if not sidecar.is_file():
                problems.append(f"parity smart520_state_contract rank reference sidecar missing: {sidecar}")
            else:
                try:
                    sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
                    sidecar_sha = str(sidecar_data.get("out_npz_sha256") or "").strip().lower()
                    if sidecar_sha != expected_sha:
                        problems.append(
                            "parity smart520_state_contract sidecar out_npz_sha256 mismatch: "
                            f"sidecar={sidecar_sha!r} metadata={expected_sha!r}"
                        )
                except Exception as exc:
                    problems.append(f"parity smart520_state_contract rank reference sidecar unreadable: {sidecar}: {exc}")
        parsed_contract_ts = {
            key: pd.to_datetime(parity_state_contract.get(key), utc=True, errors="coerce")
            for key in ("frame_anchor_utc", "model_range_start_utc", "rank_reference_end_utc")
        }
        if all(not pd.isna(ts) for ts in parsed_contract_ts.values()):
            if parsed_contract_ts["frame_anchor_utc"] < parsed_contract_ts["model_range_start_utc"]:
                problems.append("parity smart520_state_contract frame_anchor_utc precedes model_range_start_utc")
            if parsed_contract_ts["rank_reference_end_utc"] < parsed_contract_ts["model_range_start_utc"]:
                problems.append("parity smart520_state_contract rank_reference_end_utc precedes model_range_start_utc")
            if parsed_contract_ts["frame_anchor_utc"] > parsed_contract_ts["rank_reference_end_utc"]:
                problems.append("parity smart520_state_contract frame_anchor_utc exceeds rank_reference_end_utc")
    parity_dataset = str(rep.get("dataset_dir") or "").strip()
    parity_dataset_low = parity_dataset.lower()
    if not parity_dataset:
        problems.append("parity report missing dataset_dir")
    elif "xau" not in parity_dataset_low or "eur" in parity_dataset_low:
        problems.append(f"parity dataset_dir must be XAU-only, got {parity_dataset}")
    for stale_marker in ("utilityrepair", "20260710", "smart_candidate_20260630", "julyext"):
        if stale_marker in parity_dataset_low:
            problems.append(
                f"parity dataset_dir references stale XAU repair marker {stale_marker!r}: {parity_dataset}"
            )
    if not SMART_DIRECTION_AUDIT_LATEST.is_file():
        problems.append(
            f"direction pocket audit missing: {SMART_DIRECTION_AUDIT_LATEST} — run "
            "gx1.scripts.audit_smart_direction_live_like_pockets_v1 for the contract-ACTIVE bundle"
        )
    else:
        direction_audit = json.loads(SMART_DIRECTION_AUDIT_LATEST.read_text(encoding="utf-8"))
        if direction_audit.get("decision") != "PASS":
            problems.append(
                f"direction pocket audit decision={direction_audit.get('decision')!r} "
                f"failures={list(direction_audit.get('failures') or [])[:3]}"
            )
        direction_created_utc = pd.to_datetime(direction_audit.get("created_utc"), utc=True, errors="coerce")
        if pd.isna(direction_created_utc):
            problems.append(f"direction pocket audit created_utc invalid/missing: {direction_audit.get('created_utc')!r}")
        elif SMART_DIRECTION_AUDIT_MAX_AGE_HOURS > 0:
            direction_age_hours = (now_utc - direction_created_utc).total_seconds() / 3600.0
            if direction_age_hours > SMART_DIRECTION_AUDIT_MAX_AGE_HOURS:
                problems.append(
                    f"direction pocket audit stale: age_hours={direction_age_hours:.2f} "
                    f"> cap={SMART_DIRECTION_AUDIT_MAX_AGE_HOURS:.2f}"
                )
        if str(direction_audit.get("required_selection_score_mode") or "").strip().lower() != "expected_utility":
            problems.append("direction pocket audit must require expected_utility selection mode")
        observed_modes_raw = direction_audit.get("observed_selection_score_modes")
        observed_modes = (
            [str(x).strip().lower() for x in observed_modes_raw]
            if isinstance(observed_modes_raw, list)
            else []
        )
        if not observed_modes or any(mode != "expected_utility" for mode in observed_modes):
            problems.append(f"direction pocket audit observed_selection_score_modes invalid: {observed_modes_raw!r}")
        for audit_field in ("predictions_parquet", "dataset_dir"):
            audit_path = str(direction_audit.get(audit_field) or "").strip()
            audit_low = audit_path.lower()
            if not audit_path:
                problems.append(f"direction pocket audit missing {audit_field}")
            elif "xau" not in audit_low or "eur" in audit_low:
                problems.append(f"direction pocket audit {audit_field} must be XAU-only, got {audit_path}")
            for stale_marker in ("utilityrepair", "20260710", "smart_candidate_20260630", "julyext"):
                if stale_marker in audit_low:
                    problems.append(
                        f"direction pocket audit {audit_field} references stale XAU repair marker "
                        f"{stale_marker!r}: {audit_path}"
                    )
        if float(direction_audit.get("max_bad_side_rate", 1.0)) > 0.35:
            problems.append(
                f"direction pocket audit max_bad_side_rate={direction_audit.get('max_bad_side_rate')} "
                "> required 0.35"
            )
        if int(direction_audit.get("min_selected_rows", 10**9)) > 30:
            problems.append(
                f"direction pocket audit min_selected_rows={direction_audit.get('min_selected_rows')} "
                "> required 30"
            )
        required_direction_repair_pockets = {
            "rising_channel_support_touch",
            "support_retest_continuation",
            "rising_channel_support_continuation",
            "countertrend_short_trap",
            "short_high_mae_low_mfe_early_failure",
            "falling_channel_resistance_touch",
            "resistance_retest_continuation",
            "falling_channel_resistance_continuation",
            "countertrend_long_trap",
            "long_high_mae_low_mfe_early_failure",
        }
        audit_pockets = direction_audit.get("pockets")
        if not isinstance(audit_pockets, dict):
            problems.append("direction pocket audit lacks pockets dict")
        else:
            missing_pockets = sorted(required_direction_repair_pockets - set(audit_pockets))
            if missing_pockets:
                problems.append(
                    "direction pocket audit lacks required XAU direction-repair pockets: "
                    + ",".join(missing_pockets)
                )
            max_bad_side_rate = float(direction_audit.get("max_bad_side_rate", 0.35))
            min_selected_rows = int(direction_audit.get("min_selected_rows", 30))
            short_bad_pockets = {
                "rising_channel_support_touch",
                "support_retest_continuation",
                "rising_channel_support_continuation",
                "countertrend_short_trap",
                "short_high_mae_low_mfe_early_failure",
            }
            long_bad_pockets = {
                "falling_channel_resistance_touch",
                "resistance_retest_continuation",
                "falling_channel_resistance_continuation",
                "countertrend_long_trap",
                "long_high_mae_low_mfe_early_failure",
            }
            utility_pockets = (
                required_direction_repair_pockets
                - {"short_high_mae_low_mfe_early_failure", "long_high_mae_low_mfe_early_failure"}
            )
            for pocket_name in sorted(required_direction_repair_pockets & set(audit_pockets)):
                row = audit_pockets.get(pocket_name)
                if not isinstance(row, dict):
                    problems.append(f"direction pocket audit {pocket_name} is not a metrics dict")
                    continue
                try:
                    rows = int(row.get("rows"))
                    selected_rows = int(row.get("selected_rows"))
                except Exception:
                    problems.append(f"direction pocket audit {pocket_name} lacks integer rows/selected_rows")
                    continue
                if rows < min_selected_rows:
                    problems.append(
                        f"direction pocket audit {pocket_name} rows={rows} < required {min_selected_rows}"
                    )
                if selected_rows < min_selected_rows:
                    problems.append(
                        f"direction pocket audit {pocket_name} selected_rows={selected_rows} < required {min_selected_rows}"
                    )
                if pocket_name in short_bad_pockets:
                    short_rate = float(row.get("selected_side_short_rate", 1.0))
                    if short_rate > max_bad_side_rate:
                        problems.append(
                            f"direction pocket audit {pocket_name} selected SHORT rate {short_rate:.3f} "
                            f"> required {max_bad_side_rate:.3f}"
                        )
                if pocket_name in long_bad_pockets:
                    long_rate = float(row.get("selected_side_long_rate", 1.0))
                    if long_rate > max_bad_side_rate:
                        problems.append(
                            f"direction pocket audit {pocket_name} selected LONG rate {long_rate:.3f} "
                            f"> required {max_bad_side_rate:.3f}"
                        )
                if pocket_name in utility_pockets:
                    mean_pnl = row.get("selected_mean_proxy_pnl_bps")
                    if mean_pnl is None or float(mean_pnl) <= 0.0:
                        problems.append(
                            f"direction pocket audit {pocket_name} selected_mean_proxy_pnl_bps={mean_pnl} "
                            "> required 0"
                        )
        if str(direction_audit.get("bundle_dir")) != str(entry["path"]):
            problems.append(
                f"direction pocket audit bundle {direction_audit.get('bundle_dir')} "
                f"!= contract-ACTIVE {entry['path']}"
            )
    if str(entry.get("contract_mode")) != "smart_seq520_candidate":
        problems.append(f"contract_mode={entry.get('contract_mode')!r}")
    op = entry.get("operating_point")
    if not isinstance(op, dict) or "edge_score_threshold" not in op or "sessions" not in op:
        problems.append("v10_entry.operating_point missing/incomplete")
    elif str(op.get("selection_score") or "").strip().lower() != "expected_utility":
        problems.append("v10_entry.operating_point.selection_score must be expected_utility")
    elif "expected_utility_threshold_bps" not in op:
        problems.append("v10_entry.operating_point missing expected_utility_threshold_bps")
    if SMART_CTX_MAX_STALENESS_M5 != 0:
        problems.append(
            "GX1_SMART_CTX_MAX_STALENESS_M5 must be 0 for expected-utility XAU repair serving; "
            f"got {SMART_CTX_MAX_STALENESS_M5}"
        )
    if problems:
        raise RuntimeError("[SMART_GATE] LAUNCH BLOCKED: " + " | ".join(problems))
    return rep


@dataclass
class SmartEntryLiveInference:
    bundle_dir: Path
    operating_point: dict[str, Any]
    device: str = "cpu"
    _model: Any = field(default=None)
    _meta: dict = field(default_factory=dict)
    _builder: Smart520StateBuilder | None = field(default=None)
    _state_contract: Smart520StateContract | None = field(default=None)
    _per_tf_seq_lens: dict[str, int] = field(default_factory=dict)
    _multi_tf_shift: dict = field(default_factory=dict, repr=False)
    _multi_tf_target_availability_shift: pd.Timedelta = field(
        default_factory=lambda: pd.Timedelta(minutes=5),
        repr=False,
    )
    # LAST COMPLETED context snapshot (one atomic reference — loader async pattern)
    # + the in-flight background refresh thread (serving-wave gap 3). The per-M1
    # EXIT path never touches either — no lock exists to starve it.
    _ctx: SmartCtxSnapshot | None = field(default=None, repr=False)
    _ctx_refresh_thread: threading.Thread | None = field(default=None, repr=False)
    # per-decision-bucket cache of the prepared anchored frame states
    _last_state_bucket: pd.Timestamp | None = field(default=None)

    # ── loading ──────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, bundle_dir: Path | None = None, device: str = "cpu") -> "SmartEntryLiveInference":
        from gx1_guards.artifacts import load_decision_entry
        entry = load_decision_entry("v10_entry")
        contract_bundle = Path(entry["path"])
        if bundle_dir is None:
            bundle_dir = contract_bundle
        else:
            bundle_dir = Path(bundle_dir)
            if bundle_dir.resolve() != contract_bundle.resolve():
                raise RuntimeError(
                    f"[SMART_ENTRY] explicit bundle_dir {bundle_dir} != contract-ACTIVE "
                    f"{contract_bundle} — rule 8: serve resolves ONLY through the contract"
                )
        mode = str(entry.get("contract_mode") or "")
        if mode != "smart_seq520_candidate":
            raise RuntimeError(
                f"[SMART_ENTRY] contract v10_entry.contract_mode={mode!r} — this adapter "
                f"serves smart_seq520_candidate only"
            )
        op = entry.get("operating_point")
        if not isinstance(op, dict):
            raise RuntimeError("[SMART_ENTRY] contract v10_entry.operating_point missing — fail-closed")
        for req in ("edge_score_threshold", "sessions", "selection_score", "expected_utility_threshold_bps"):
            if req not in op:
                raise RuntimeError(f"[SMART_ENTRY] operating_point missing '{req}' — fail-closed")
        if str(op.get("selection_score") or "").strip().lower() != "expected_utility":
            raise RuntimeError("[SMART_ENTRY] operating_point.selection_score must be expected_utility — fail-closed")

        from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
        bundle = load_entry_v10_ctx_bundle(bundle_dir=bundle_dir, device=device, xgb_models=None)
        model = bundle.transformer_model
        model.eval()
        meta = dict(bundle.metadata)
        if meta.get("neutral_xgb_bridge") is not True:
            raise RuntimeError("[SMART_ENTRY] bundle must declare neutral_xgb_bridge=true — refusing XGB-anchored entry")
        if str(meta.get("xgb_bridge_source") or "") != "neutral_uniform_proba":
            raise RuntimeError(
                "[SMART_ENTRY] bundle must declare xgb_bridge_source=neutral_uniform_proba — "
                f"got {meta.get('xgb_bridge_source')!r}"
            )
        state_contract = Smart520StateContract.from_metadata(
            meta.get("smart520_state_contract"),
            require_xau_direction_repair=True,
        )
        if int(meta.get("seq_input_dim") or 0) != SIGNAL_DIM_SMART520:
            raise RuntimeError(
                f"[SMART_ENTRY] bundle seq_input_dim={meta.get('seq_input_dim')} != {SIGNAL_DIM_SMART520}"
            )
        if int(meta.get("seq_len") or 0) != SEQ_LEN_SMART520:
            raise RuntimeError(f"[SMART_ENTRY] bundle seq_len={meta.get('seq_len')} != {SEQ_LEN_SMART520}")
        direction_calibration = meta.get("direction_calibration")
        if not isinstance(direction_calibration, dict) or direction_calibration.get("enabled") is not True:
            raise RuntimeError(
                "[SMART_ENTRY] bundle lacks enabled direction_calibration — the promoted cand#4 is the "
                "CALIBRATED bundle; refusing an uncalibrated load"
            )
        path_calibration = meta.get("path_calibration")
        if not isinstance(path_calibration, dict) or path_calibration.get("enabled") is not True:
            raise RuntimeError(
                "[SMART_ENTRY] bundle lacks enabled path_calibration — live/replay path heads "
                "must be calibrated before serving"
            )
        mtf = meta.get("multi_tf") or {}
        if not bool(mtf.get("enabled")) or not bool(mtf.get("v2_mode")):
            raise RuntimeError("[SMART_ENTRY] bundle must be multi-TF v2 — refusing")
        mtf_shift_minutes = float(mtf.get("target_availability_shift_minutes", 5.0) or 0.0)
        if abs(mtf_shift_minutes - 5.0) > 1e-9:
            raise RuntimeError(
                "[SMART_ENTRY] bundle multi_tf.target_availability_shift_minutes must be 5.0 "
                f"for closed-bar XAU repair serving, got {mtf_shift_minutes!r}"
            )
        per_tf = {
            "M5": int(mtf.get("m5_seq_len", 96)),
            "M15": int(mtf.get("m15_seq_len", 96)),
            "H1": int(mtf.get("h1_seq_len", 96)),
            "H4": int(mtf.get("h4_seq_len", 96)),
            "D1": int(mtf.get("d1_seq_len", 96)),
        }
        names = [str(x) for x in (meta.get("ordered_signal_names") or [])]
        builder = Smart520StateBuilder(ordered_signal_names=names, state_contract=state_contract)
        LOG.info(
            "[SMART_ENTRY] loaded contract-ACTIVE %s (mode=%s, thr=%.17g, sessions=%s, anchor=%s)",
            bundle_dir.name, mode, float(op["edge_score_threshold"]),
            list(op.get("sessions") or []), state_contract.frame_anchor_utc,
        )
        return cls(
            bundle_dir=bundle_dir, operating_point=dict(op), device=device,
            _model=model, _meta=meta, _builder=builder, _state_contract=state_contract, _per_tf_seq_lens=per_tf,
            _multi_tf_target_availability_shift=pd.Timedelta(minutes=mtf_shift_minutes),
        )

    # ── smart context (in-memory snapshot, refreshed on cv3 cutoff advance) ──
    # The build (~2 min: float32 MTF over full cv3 + frozen-rank buckets + full-
    # frame HTF/REGIME_V4 overrides) ran SYNCHRONOUSLY in the runner loop pre
    # gap-3 — every cv3 cutoff advance starved the per-M1 exit decisions for
    # ~2 min. Now it follows the loader's async-refresh pattern
    # (v12_state_from_prebuilt 2026-06-01): background thread builds a NEW
    # SmartCtxSnapshot on a LOCAL cv3 reference, then swaps ONE attribute
    # (GIL-atomic); decisions read the last completed snapshot and journal
    # context_age_m5_bars. No lock anywhere — the exit path cannot be starved.

    def _build_ctx_snapshot(self, cv3: pd.DataFrame) -> SmartCtxSnapshot:
        """The FULL context build (unchanged math — same one-truth functions the
        blocking path always used). Runs on local state only; safe in a thread."""
        from gx1.execution.v12_smart520_state_live import (
            compute_bucket_ctx_cat_full_frame,
            compute_htf_ctx_full_frame,
        )
        if self._state_contract is None:
            raise RuntimeError("[SMART_ENTRY] smart520 state contract not loaded")
        t0 = time.perf_counter()
        cutoff = cv3.index[-1]
        multi_tf = build_multi_tf_from_cv3(cv3)
        # full-frame overrides: ctx_cat buckets (offline frame-global-rank
        # convention) + the 5 long-lookback HTF ctx cols (fresh full-frame
        # recompute; B28's incremental M1-lane stamping is one M5 bar behind
        # the offline convention — parity gate finding 2026-07-08)
        overrides = pd.concat(
            [
                compute_bucket_ctx_cat_full_frame(cv3, self._state_contract),
                compute_htf_ctx_full_frame(cv3, self._state_contract),
            ],
            axis=1,
        )
        return SmartCtxSnapshot(
            multi_tf=multi_tf, frame_overrides=overrides,
            cv3_cutoff=cutoff, built_utc=pd.Timestamp.utcnow(),
            build_seconds=time.perf_counter() - t0,
        )

    def _install_ctx_snapshot(self, snap: SmartCtxSnapshot) -> None:
        """Single-reference swap (GIL-atomic). The builder mirror exists only for
        direct Smart520StateBuilder callers; the live decision path passes the
        snapshot's bundle explicitly so it never races the mirror write."""
        self._ctx = snap
        if self._builder is not None:
            self._builder.multi_tf = snap.multi_tf

    def refresh_multi_tf(self, cv3: pd.DataFrame) -> None:
        """BLOCKING context (re)build when cv3's cutoff advanced — the startup /
        parity-gate / offline-driver path (semantics unchanged from pre-gap-3).
        The live runner path uses maybe_schedule_ctx_refresh + predict_live_bar
        instead and never blocks on this."""
        cutoff = cv3.index[-1]
        ctx = self._ctx
        if ctx is not None and ctx.cv3_cutoff == cutoff:
            return
        from gx1.features.htf_features import MULTI_TF_SHIFT
        LOG.info("[SMART_ENTRY] building smart-context snapshot from cv3 (cutoff=%s, blocking)…", cutoff)
        self._multi_tf_shift = dict(MULTI_TF_SHIFT)
        snap = self._build_ctx_snapshot(cv3)
        self._install_ctx_snapshot(snap)
        LOG.info("[SMART_ENTRY] smart-context snapshot ready (cutoff=%s, %.1fs)",
                 cutoff, snap.build_seconds)

    def maybe_schedule_ctx_refresh(self, cv3: pd.DataFrame) -> bool:
        """NON-BLOCKING: schedule a background context rebuild when cv3's cutoff
        advanced past the snapshot's and no refresh is in flight (the loader's
        refresh_if_changed pattern). Returns True only on the scheduling cycle."""
        ctx = self._ctx
        if ctx is None:
            raise RuntimeError(
                "[SMART_ENTRY] no context snapshot — the initial (blocking) "
                "refresh_multi_tf() at startup is mandatory before live decisions"
            )
        if cv3.index[-1] <= ctx.cv3_cutoff:
            return False
        t = self._ctx_refresh_thread
        if t is not None and t.is_alive():
            return False
        t = threading.Thread(
            target=self._async_ctx_refresh, args=(cv3,), daemon=True,
            name="smart_ctx_async_refresh",
        )
        self._ctx_refresh_thread = t
        t.start()
        return True

    def _async_ctx_refresh(self, cv3: pd.DataFrame) -> None:
        """Background-thread worker: full context build on the cv3 reference
        grabbed at schedule time (the loader swaps — never mutates — its frames,
        so this read is race-free), then one atomic snapshot swap. Fail-SAFE:
        on error the previous snapshot stays live and the staleness cap
        (SMART_CTX_MAX_STALENESS_M5) turns a persistent failure into journaled
        entry SKIPs — exits are never affected."""
        try:
            old = self._ctx
            snap = self._build_ctx_snapshot(cv3)
            self._install_ctx_snapshot(snap)
            LOG.info("[smart-ctx-refresh] snapshot cutoff %s → %s (took %.1fs, decisions never blocked)",
                     old.cv3_cutoff if old is not None else None,
                     snap.cv3_cutoff, snap.build_seconds)
        except Exception as exc:  # noqa: BLE001 — fail-safe: keep prior snapshot
            LOG.error(f"[smart-ctx-refresh] FAILED: {exc} — keeping previous snapshot "
                      f"(staleness cap will SKIP entries if this persists)")

    @staticmethod
    def context_age_m5_bars(cv3: pd.DataFrame, end_ts: pd.Timestamp,
                            ctx: SmartCtxSnapshot) -> int:
        """cv3 M5 bars in (ctx.cv3_cutoff, end_ts] — 0 ⇒ the snapshot covers the
        decision bar (may be negative for historical end_ts, e.g. the parity gate)."""
        idx = cv3.index
        return int(idx.searchsorted(end_ts, side="right")
                   - idx.searchsorted(ctx.cv3_cutoff, side="right"))

    def _effective_context(
        self, cv3: pd.DataFrame, ctx: SmartCtxSnapshot, end_ts: pd.Timestamp,
    ) -> tuple[dict, pd.DataFrame, int, bool]:
        """The snapshot context extended to end_ts (age > 0 = gap bars exist):
          * override tables — CHEAP (~0.6s, gap-3 probe) FULL-frame recompute on
            the current cv3 via the same one-truth functions the snapshot build
            used: causal + frozen-rank digitize, so overlapping rows are
            bit-identical and the gap bars are EXACT by construction (no ffill,
            no staleness).
          * MTF cache — the heavy part (~94s full): self-test-proven incremental
            tail splice (append_multi_tf_incremental) for M5/M15/H1; H4/D1 keep
            snapshot rows (forming-bar staleness only, journaled via
            context_age_m5_bars, capped by SMART_CTX_MAX_STALENESS_M5).
        Returns (multi_tf, frame_overrides, age, mtf_spliced)."""
        age = self.context_age_m5_bars(cv3, end_ts, ctx)
        if age <= 0:
            return ctx.multi_tf, ctx.frame_overrides, age, False
        from gx1.execution.v12_smart520_state_live import (
            compute_bucket_ctx_cat_full_frame,
            compute_htf_ctx_full_frame,
        )
        if self._state_contract is None:
            raise RuntimeError("[SMART_ENTRY] smart520 state contract not loaded")
        overrides = pd.concat(
            [
                compute_bucket_ctx_cat_full_frame(cv3, self._state_contract),
                compute_htf_ctx_full_frame(cv3, self._state_contract),
            ],
            axis=1,
        )
        multi_tf, spliced = ctx.multi_tf, False
        if SMART_CTX_MTF_INCREMENTAL:
            multi_tf, spliced = append_multi_tf_incremental(cv3, ctx.multi_tf)
        return multi_tf, overrides, age, spliced

    def _prepare_anchored_frame(
        self, loader, cv3: pd.DataFrame, end_ts: pd.Timestamp,
        overrides: pd.DataFrame, multi_tf: dict,
    ) -> pd.DataFrame:
        """Shared anchored-window build + prepare (ONE truth for the blocking
        gate path and the live async path)."""
        if self._state_contract is None:
            raise RuntimeError("[SMART_ENTRY] smart520 state contract not loaded")
        anchor = self._state_contract.frame_anchor_utc
        cv3_idx = cv3.index
        n_from_anchor = int(cv3_idx.searchsorted(end_ts, side="right")
                            - cv3_idx.searchsorted(anchor, side="left"))
        if n_from_anchor < SEQ_LEN_SMART520:
            raise RuntimeError(f"[SMART_ENTRY] anchored frame too short: {n_from_anchor} bars")
        joined = loader.get_window(end_ts, n_bars=n_from_anchor)
        if joined.empty or joined.index[0] < anchor:
            raise RuntimeError(
                f"[SMART_ENTRY] anchored window build failed: rows={len(joined)} "
                f"start={joined.index[0] if len(joined) else None}"
            )
        return self._builder.prepare_frame(joined, bucket_ctx_cat=overrides, multi_tf=multi_tf)

    def build_anchored_frame(
        self, loader, end_ts: pd.Timestamp, ctx: SmartCtxSnapshot | None = None,
    ) -> pd.DataFrame:
        """ONE-TRUTH anchored state frame [smart520_state_contract.frame_anchor_utc .. end_ts]
        from the live prebuilt loader (joined cv3+BASE28), prepared with all
        smart520 recomputes. Shared by the parity gate and the live pipeline.
        ctx=None (gate/startup path): BLOCKING refresh first — behavior and
        values identical to the pre-gap-3 synchronous implementation."""
        if self._builder is None:
            raise RuntimeError("[SMART_ENTRY] not loaded")
        if ctx is None:
            self.refresh_multi_tf(loader._cv3)
            ctx = self._ctx
        cv3 = loader._cv3
        multi_tf, overrides, _age, _spliced = self._effective_context(cv3, ctx, end_ts)
        return self._prepare_anchored_frame(loader, cv3, end_ts, overrides, multi_tf)

    def _multi_tf_window_tensors(
        self, ts: pd.Timestamp, multi_tf: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-TF windows at-or-before ts with the BUNDLE's per-TF seq lens —
        the exact offline dataset path (EntryV10CtxDataset._get_multi_tf_window:
        get_last_n_at_or_before(feats, ts + 5min, n=per_tf,
        tf_shift=MULTI_TF_SHIFT)).
        `multi_tf=None` uses the current snapshot (gate/offline callers)."""
        if multi_tf is None:
            ctx = self._ctx
            if ctx is None:
                raise RuntimeError("[SMART_ENTRY] multi-TF not built — call refresh_multi_tf() first")
            multi_tf = ctx.multi_tf
        from gx1.features.htf_features import get_last_n_at_or_before
        out: dict[str, torch.Tensor] = {}
        availability_ts = pd.Timestamp(ts) + self._multi_tf_target_availability_shift
        for tf, feats in multi_tf.items():
            n = int(self._per_tf_seq_lens.get(tf, SEQ_LEN_SMART520))
            arr = get_last_n_at_or_before(
                feats,
                availability_ts,
                n=n,
                tf_shift=self._multi_tf_shift[tf],
            )
            out[f"seq_{tf.lower()}"] = torch.from_numpy(
                arr.astype(np.float32, copy=False)
            ).unsqueeze(0).to(self.device)
        return out

    # ── forward ───────────────────────────────────────────────────────────────

    def forward_states(
        self, states: dict[str, Any], multi_tf: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Forward pre-built smart520 states (from Smart520StateBuilder) through
        the calibrated model. Mirrors evaluate_entry_candidate_selective_edge_v1
        _predict_bundle head-for-head. Returns one dict per state row.
        `multi_tf=None` uses the current snapshot (gate/offline callers); the
        live path passes the SAME bundle the states were built with."""
        if self._model is None:
            raise RuntimeError("[SMART_ENTRY] not loaded")
        results: list[dict[str, Any]] = []
        n = states["seq"].shape[0]
        with torch.no_grad():
            for k in range(n):
                ts = pd.Timestamp(states["times"][k])
                seq_t = torch.from_numpy(states["seq"][k]).unsqueeze(0).to(self.device)
                snap_t = torch.from_numpy(states["snap"][k]).unsqueeze(0).to(self.device)
                ctx_cont_t = torch.from_numpy(states["ctx_cont"][k]).unsqueeze(0).to(self.device)
                ctx_cat_t = torch.from_numpy(states["ctx_cat"][k]).unsqueeze(0).to(self.device)
                mtf_kwargs = self._multi_tf_window_tensors(ts, multi_tf=multi_tf)
                out = self._model(seq_t, snap_t, ctx_cat=ctx_cat_t, ctx_cont=ctx_cont_t, **mtf_kwargs)
                for key, value in out.items():
                    if torch.is_tensor(value) and not bool(torch.isfinite(value).all().item()):
                        raise RuntimeError(f"[SMART_ENTRY] non-finite model output '{key}' at {ts}")
                probs = torch.softmax(out["direction_logits"], dim=-1).cpu().float().numpy()[0]
                p_long, p_short, p_flat = float(probs[0]), float(probs[1]), float(probs[2])
                edge_score = max(p_long, p_short) - p_flat
                anchor_logits = _np1d(out.get("anchor_logits"))
                delta_logits = _np1d(out.get("delta_logits"))
                mtf_logits = _np1d(out.get("mtf_dir_logits"))
                anchor_gate = _np1d(out.get("anchor_gate"))
                anchor_probs = _softmax_np(anchor_logits)
                mtf_probs = _softmax_np(mtf_logits)
                trade_logit = _np1d(out.get("trade_logit"))
                side_logits = _np1d(out.get("side_logits"))
                side_utility = _np1d(out.get("side_utility"))
                side_bad_path_logit = _np1d(out.get("side_bad_path_logit"))
                side_mae = _np1d(out.get("side_mae"))
                side_validity_logit = _np1d(out.get("side_validity_logit"))
                trendline_rail_logits = _np1d(out.get("trendline_rail_logits"))
                hier_meta = self._meta.get("hierarchical_entry_heads") or {}
                utility_scale_bps = float(hier_meta.get("side_utility_scale_bps", 1.0) or 1.0)
                mae_scale_bps = float(hier_meta.get("side_mae_scale_bps", 1.0) or 1.0)
                side_probs = _softmax_np(side_logits)
                if trendline_rail_logits is not None and len(trendline_rail_logits) >= 4:
                    trendline_rail_probs = [_sigmoid_float(float(x)) for x in trendline_rail_logits]
                    trendline_short_early_failure_prob = (
                        float(trendline_rail_probs[4]) if len(trendline_rail_probs) >= 6 else None
                    )
                    trendline_long_early_failure_prob = (
                        float(trendline_rail_probs[5]) if len(trendline_rail_probs) >= 6 else None
                    )
                    anti_short_parts = [trendline_rail_probs[0], trendline_rail_probs[2]]
                    anti_long_parts = [trendline_rail_probs[1], trendline_rail_probs[3]]
                    if trendline_short_early_failure_prob is not None:
                        anti_short_parts.append(trendline_short_early_failure_prob)
                    if trendline_long_early_failure_prob is not None:
                        anti_long_parts.append(trendline_long_early_failure_prob)
                    rail_anti_short_prob = float(max(anti_short_parts))
                    rail_anti_long_prob = float(max(anti_long_parts))
                else:
                    trendline_rail_probs = None
                    trendline_short_early_failure_prob = None
                    trendline_long_early_failure_prob = None
                    rail_anti_short_prob = 0.0
                    rail_anti_long_prob = 0.0
                rail_long_penalty = float(rail_anti_long_prob * SMART_EXPECTED_UTILITY_RAIL_PENALTY_BPS)
                rail_short_penalty = float(rail_anti_short_prob * SMART_EXPECTED_UTILITY_RAIL_PENALTY_BPS)
                if trade_logit is not None and len(trade_logit):
                    p_trade_hier = _sigmoid_float(float(trade_logit[0]))
                    p_flat_hier = float(1.0 - p_trade_hier)
                else:
                    p_trade_hier = float(max(0.0, min(1.0, p_long + p_short)))
                    p_flat_hier = float(max(0.0, min(1.0, p_flat)))
                if side_probs is not None and len(side_probs) >= 2:
                    p_long_given_trade = float(side_probs[0])
                    p_short_given_trade = float(side_probs[1])
                    side_uncertainty = float(1.0 - max(p_long_given_trade, p_short_given_trade))
                else:
                    denom = max(1e-9, p_long + p_short)
                    p_long_given_trade = float(p_long / denom)
                    p_short_given_trade = float(p_short / denom)
                    side_uncertainty = float(1.0 - max(p_long_given_trade, p_short_given_trade))
                if side_bad_path_logit is not None and len(side_bad_path_logit) >= 2:
                    long_bad_path_prob = _sigmoid_float(float(side_bad_path_logit[0]))
                    short_bad_path_prob = _sigmoid_float(float(side_bad_path_logit[1]))
                else:
                    classic_bad = float(torch.sigmoid(out["bad_path_logit"]).cpu().float().numpy().reshape(-1)[0])
                    long_bad_path_prob = classic_bad
                    short_bad_path_prob = classic_bad
                if side_validity_logit is not None and len(side_validity_logit) >= 2:
                    long_validity_prob = _sigmoid_float(float(side_validity_logit[0]))
                    short_validity_prob = _sigmoid_float(float(side_validity_logit[1]))
                else:
                    long_validity_prob = 1.0
                    short_validity_prob = 1.0
                invalid_long_penalty = float(
                    (1.0 - long_validity_prob) * SMART_EXPECTED_UTILITY_INVALID_SIDE_PENALTY_BPS
                )
                invalid_short_penalty = float(
                    (1.0 - short_validity_prob) * SMART_EXPECTED_UTILITY_INVALID_SIDE_PENALTY_BPS
                )
                if side_utility is not None and len(side_utility) >= 2:
                    long_path_utility_pred = float(side_utility[0]) * utility_scale_bps
                    short_path_utility_pred = float(side_utility[1]) * utility_scale_bps
                    expected_utility_long = (
                        p_trade_hier * p_long_given_trade * long_path_utility_pred
                        - long_bad_path_prob * SMART_EXPECTED_UTILITY_BAD_PATH_PENALTY_BPS
                        - side_uncertainty * SMART_EXPECTED_UTILITY_UNCERTAINTY_PENALTY_BPS
                        - rail_long_penalty
                        - invalid_long_penalty
                    )
                    expected_utility_short = (
                        p_trade_hier * p_short_given_trade * short_path_utility_pred
                        - short_bad_path_prob * SMART_EXPECTED_UTILITY_BAD_PATH_PENALTY_BPS
                        - side_uncertainty * SMART_EXPECTED_UTILITY_UNCERTAINTY_PENALTY_BPS
                        - rail_short_penalty
                        - invalid_short_penalty
                    )
                    expected_utility_side = 0 if expected_utility_long >= expected_utility_short else 1
                else:
                    long_path_utility_pred = None
                    short_path_utility_pred = None
                    expected_utility_long = None
                    expected_utility_short = None
                    expected_utility_side = int(0 if p_long >= p_short else 1)
                if side_mae is not None and len(side_mae) >= 2:
                    long_expected_mae = float(max(0.0, side_mae[0] * mae_scale_bps))
                    short_expected_mae = float(max(0.0, side_mae[1] * mae_scale_bps))
                else:
                    long_expected_mae = None
                    short_expected_mae = None
                signal_names = [str(x) for x in (self._meta.get("ordered_signal_names") or [])]
                snap_row = np.asarray(states["snap"][k], dtype=np.float32).reshape(-1)
                geometry_support_evidence = _feature_max(
                    snap_row,
                    signal_names,
                    [
                        "chart.geometry_support_line_proximity_stack",
                        "chart.sr_memory_support_level_proximity_stack",
                        "chart.sr_memory_support_respect_pressure_long",
                        "chart.sr_memory_support_reclaim_pressure_long",
                        "chart.sr_memory_liquidity_low_level_rejection_long",
                        "chart.geometry_fib_support_confluence_long_pressure",
                        "chart.geometry_rising_support_rail_long_pressure",
                    ],
                )
                geometry_resistance_evidence = _feature_max(
                    snap_row,
                    signal_names,
                    [
                        "chart.geometry_resistance_line_proximity_stack",
                        "chart.sr_memory_resistance_level_proximity_stack",
                        "chart.sr_memory_resistance_respect_pressure_short",
                        "chart.sr_memory_resistance_reclaim_pressure_short",
                        "chart.sr_memory_liquidity_high_level_rejection_short",
                        "chart.geometry_fib_resistance_confluence_short_pressure",
                        "chart.geometry_falling_resistance_rail_short_pressure",
                    ],
                )
                geometry_channel_edge = _feature_value(
                    snap_row,
                    signal_names,
                    ["chart.geometry_channel_edge_pressure"],
                )
                geometry_rising_support_rail_long = _feature_value(
                    snap_row,
                    signal_names,
                    ["chart.geometry_rising_support_rail_long_pressure"],
                )
                geometry_rising_support_rail_short_trap = _feature_value(
                    snap_row,
                    signal_names,
                    ["chart.geometry_rising_support_rail_short_trap_pressure"],
                )
                geometry_falling_resistance_rail_short = _feature_value(
                    snap_row,
                    signal_names,
                    ["chart.geometry_falling_resistance_rail_short_pressure"],
                )
                geometry_falling_resistance_rail_long_trap = _feature_value(
                    snap_row,
                    signal_names,
                    ["chart.geometry_falling_resistance_rail_long_trap_pressure"],
                )
                trendline_rail_long_evidence = _mean_optional(
                    geometry_rising_support_rail_long,
                    geometry_falling_resistance_rail_long_trap,
                )
                trendline_rail_short_evidence = _mean_optional(
                    geometry_falling_resistance_rail_short,
                    geometry_rising_support_rail_short_trap,
                )
                trendline_rail_long_minus_short = _optional_diff(
                    trendline_rail_long_evidence,
                    trendline_rail_short_evidence,
                )
                mtf_trend_evidence = _feature_value(
                    snap_row,
                    signal_names,
                    ["trend.mtf_confluence_trend_direction_score", "trend.ema_stack_alignment_score"],
                )
                res = {
                    "time": ts,
                    "p_long": p_long, "p_short": p_short, "p_flat": p_flat,
                    "edge_score": float(edge_score),
                    "legacy_trade_side": 0 if p_long >= p_short else 1,
                    "trade_side": 0 if p_long >= p_short else 1,
                    "session_id": int(states["ctx_cat"][k][0]),
                    "path_quality_pred": float(out["path_quality"].cpu().float().numpy().reshape(-1)[0]),
                    "bad_path_prob": float(torch.sigmoid(out["bad_path_logit"]).cpu().float().numpy().reshape(-1)[0]),
                    "tradable_prob": float(torch.sigmoid(out["tradable_logit"]).cpu().float().numpy().reshape(-1)[0]),
                    "mfe_first_n_pred": float(out["mfe_first_n"].cpu().float().numpy().reshape(-1)[0]),
                    "p_trade": p_trade_hier,
                    "p_flat_hier": p_flat_hier,
                    "p_long_given_trade": p_long_given_trade,
                    "p_short_given_trade": p_short_given_trade,
                    "side_uncertainty": side_uncertainty,
                    "long_path_utility_pred_bps": long_path_utility_pred,
                    "short_path_utility_pred_bps": short_path_utility_pred,
                    "long_bad_path_prob": long_bad_path_prob,
                    "short_bad_path_prob": short_bad_path_prob,
                    "side_validity_logit": side_validity_logit.tolist() if side_validity_logit is not None else None,
                    "long_validity_prob": long_validity_prob,
                    "short_validity_prob": short_validity_prob,
                    "long_expected_mae_bps": long_expected_mae,
                    "short_expected_mae_bps": short_expected_mae,
                    "expected_utility_long_bps": expected_utility_long,
                    "expected_utility_short_bps": expected_utility_short,
                    "expected_utility_side": int(expected_utility_side),
                    "expected_utility_bad_path_penalty_bps": SMART_EXPECTED_UTILITY_BAD_PATH_PENALTY_BPS,
                    "expected_utility_uncertainty_penalty_bps": SMART_EXPECTED_UTILITY_UNCERTAINTY_PENALTY_BPS,
                    "expected_utility_rail_penalty_bps": SMART_EXPECTED_UTILITY_RAIL_PENALTY_BPS,
                    "expected_utility_invalid_side_penalty_bps": SMART_EXPECTED_UTILITY_INVALID_SIDE_PENALTY_BPS,
                    "expected_utility_long_rail_penalty_bps": rail_long_penalty,
                    "expected_utility_short_rail_penalty_bps": rail_short_penalty,
                    "expected_utility_long_invalid_side_penalty_bps": invalid_long_penalty,
                    "expected_utility_short_invalid_side_penalty_bps": invalid_short_penalty,
                    "anchor_logits": anchor_logits.tolist() if anchor_logits is not None else None,
                    "anchor_probs": anchor_probs.tolist() if anchor_probs is not None else None,
                    "delta_logits": delta_logits.tolist() if delta_logits is not None else None,
                    "mtf_dir_logits": mtf_logits.tolist() if mtf_logits is not None else None,
                    "mtf_dir_probs": mtf_probs.tolist() if mtf_probs is not None else None,
                    "anchor_gate": anchor_gate.tolist() if anchor_gate is not None else None,
                    "geometry_support_evidence": geometry_support_evidence,
                    "geometry_resistance_evidence": geometry_resistance_evidence,
                    "geometry_channel_edge_pressure": geometry_channel_edge,
                    "geometry_rising_support_rail_long_pressure": geometry_rising_support_rail_long,
                    "geometry_rising_support_rail_short_trap_pressure": geometry_rising_support_rail_short_trap,
                    "geometry_falling_resistance_rail_short_pressure": geometry_falling_resistance_rail_short,
                    "geometry_falling_resistance_rail_long_trap_pressure": geometry_falling_resistance_rail_long_trap,
                    "trendline_rail_logits": trendline_rail_logits.tolist() if trendline_rail_logits is not None else None,
                    "trendline_rail_probs": trendline_rail_probs,
                    "trendline_rail_short_early_failure_prob": trendline_short_early_failure_prob,
                    "trendline_rail_long_early_failure_prob": trendline_long_early_failure_prob,
                    "trendline_rail_anti_short_prob": rail_anti_short_prob,
                    "trendline_rail_anti_long_prob": rail_anti_long_prob,
                    "trendline_rail_long_evidence": trendline_rail_long_evidence,
                    "trendline_rail_short_evidence": trendline_rail_short_evidence,
                    "trendline_rail_long_minus_short": trendline_rail_long_minus_short,
                    "mtf_trend_evidence": mtf_trend_evidence,
                    "calibration_version": self._meta.get("direction_calibration", {}).get("version"),
                    "direction_calibration_enabled": bool((self._meta.get("direction_calibration") or {}).get("enabled", False)),
                    "direction_calibration_temperature": (self._meta.get("direction_calibration") or {}).get("temperature"),
                    "direction_calibration_bias": (self._meta.get("direction_calibration") or {}).get("bias"),
                    "path_calibration_enabled": bool((self._meta.get("path_calibration") or {}).get("enabled", False)),
                    "path_calibration": self._meta.get("path_calibration"),
                    "anchored_entry_enabled": self._meta.get("anchored_entry_enabled"),
                    "anchor_source": self._meta.get("anchor_source"),
                }
                results.append(res)
        return results

    # ── live per-M5 forward (async-context path — serving-wave gap 3) ────────

    def predict_live_bar(self, loader, end_ts: pd.Timestamp) -> dict[str, Any]:
        """LIVE per-M5 decision forward: uses the LAST COMPLETED context snapshot
        — NEVER blocks on the ~2-min context refresh (which now runs in a
        background thread, scheduled here on cv3 cutoff advance). One atomic
        snapshot grab keeps state build + model forward internally consistent.

        Fail-closed: raises SmartContextStaleError when the snapshot lags the
        decision bar by more than SMART_CTX_MAX_STALENESS_M5 cv3 bars (the
        pipeline journals the SKIP and retries next poll). Journals staleness on
        every result: context_age_m5_bars / context_cutoff_ts /
        context_refresh_in_flight / context_mtf_incremental.
        """
        if self._builder is None or self._model is None:
            raise RuntimeError("[SMART_ENTRY] not loaded")
        cv3 = loader._cv3
        self.maybe_schedule_ctx_refresh(cv3)
        ctx = self._ctx   # ONE atomic grab — never re-read during this decision
        if ctx is None:
            raise RuntimeError("[SMART_ENTRY] no context snapshot — startup refresh missing")
        age = self.context_age_m5_bars(cv3, end_ts, ctx)
        if age > SMART_CTX_MAX_STALENESS_M5:
            raise SmartContextStaleError(
                age=age, cap=SMART_CTX_MAX_STALENESS_M5,
                ctx_cutoff=ctx.cv3_cutoff, end_ts=end_ts,
            )
        multi_tf, overrides, age, spliced = self._effective_context(cv3, ctx, end_ts)
        frame = self._prepare_anchored_frame(loader, cv3, end_ts, overrides, multi_tf)
        states = self._builder.build_states(frame, [end_ts])
        head = self.forward_states(states, multi_tf=multi_tf)[0]
        t = self._ctx_refresh_thread
        head["context_age_m5_bars"] = int(max(age, 0))
        head["context_cutoff_ts"] = str(ctx.cv3_cutoff)
        head["context_refresh_in_flight"] = bool(t is not None and t.is_alive())
        head["context_mtf_incremental"] = bool(spliced)
        return head

    # ── decision (operating point from the contract — ONE truth) ─────────────

    def decide(self, head_out: dict[str, Any], atr_bps: float) -> dict[str, Any]:
        """Apply the pinned operating point to one forward result. Emits the
        runner-facing decision dict incl. the exit-bound _v10_snapshot."""
        thr = float(self.operating_point["edge_score_threshold"])
        selection_mode = str(self.operating_point.get("selection_score", "edge_score")).strip().lower()
        requested_selection_mode = selection_mode
        sessions = {str(s) for s in (self.operating_point.get("sessions") or [])}
        session = SESSION_NAMES.get(int(head_out["session_id"]), f"UNKNOWN_{head_out['session_id']}")
        edge = float(head_out["edge_score"])
        expected_utility_long = head_out.get("expected_utility_long_bps")
        expected_utility_short = head_out.get("expected_utility_short_bps")
        if requested_selection_mode in {"expected_utility", "expected_utility_side", "utility"}:
            selection_mode = "expected_utility"
            selection_threshold = float(
                self.operating_point.get(
                    "expected_utility_threshold_bps",
                    SMART_EXPECTED_UTILITY_THRESHOLD_BPS,
                )
            )
            if expected_utility_long is None or expected_utility_short is None:
                raise RuntimeError(
                    "[SMART_ENTRY] expected_utility selection requires utility heads; "
                    f"long={expected_utility_long!r} short={expected_utility_short!r}"
                )
            eu_long = float(expected_utility_long)
            eu_short = float(expected_utility_short)
            if not (np.isfinite(eu_long) and np.isfinite(eu_short)):
                raise RuntimeError(
                    "[SMART_ENTRY] expected_utility selection received non-finite utility heads: "
                    f"long={eu_long!r} short={eu_short!r}"
                )
            side_idx = 0 if eu_long >= eu_short else 1
            supplied_side = head_out.get("expected_utility_side")
            if supplied_side is not None and int(supplied_side) != int(side_idx):
                raise RuntimeError(
                    "[SMART_ENTRY] expected_utility_side mismatch: "
                    f"supplied={supplied_side} recomputed={side_idx} "
                    f"long={eu_long:.6g} short={eu_short:.6g}"
                )
            selection_score = float(max(eu_long, eu_short))
            score_ok = selection_score >= selection_threshold
            below_reason = "expected_utility_below_threshold"
        else:
            selection_mode = "edge_score"
            side_idx = int(head_out["trade_side"])
            selection_score = edge
            selection_threshold = thr
            score_ok = edge >= thr
            below_reason = "edge_below_threshold"
        take = (session in sessions) and score_ok
        action = SIDE_ACTION[side_idx] if take else "SKIP"
        skip_reason = None
        if not take:
            skip_reason = "session_gate" if session not in sessions else below_reason

        # Exit-bound snapshot — replay-driver-proven mapping (module docstring).
        # hold_horizon_bars_pred DELIBERATELY ABSENT (blocked head -> -1 sentinel
        # -> HOLD_HORIZON_EXPIRED inert; live-equivalent to the joint replay).
        snapshot = {
            "decision_ts": str(head_out["time"]),
            "direction_probs": [head_out["p_long"], head_out["p_short"], head_out["p_flat"]],
            "path_quality": head_out["path_quality_pred"],
            "mfe_first_n": head_out["mfe_first_n_pred"],
            "tradable_prob": head_out["tradable_prob"],
            "bad_path_prob": head_out["bad_path_prob"],
            "p_trade": head_out.get("p_trade"),
            "p_flat_hier": head_out.get("p_flat_hier"),
            "p_long_given_trade": head_out.get("p_long_given_trade"),
            "p_short_given_trade": head_out.get("p_short_given_trade"),
            "legacy_trade_side": int(head_out["trade_side"]),
            "expected_utility_side": head_out.get("expected_utility_side"),
            "selected_side": int(side_idx),
            "expected_utility_long_bps": expected_utility_long,
            "expected_utility_short_bps": expected_utility_short,
            "long_path_utility_pred_bps": head_out.get("long_path_utility_pred_bps"),
            "short_path_utility_pred_bps": head_out.get("short_path_utility_pred_bps"),
            "expected_utility_bad_path_penalty_bps": head_out.get("expected_utility_bad_path_penalty_bps"),
            "expected_utility_uncertainty_penalty_bps": head_out.get("expected_utility_uncertainty_penalty_bps"),
            "expected_utility_rail_penalty_bps": head_out.get("expected_utility_rail_penalty_bps"),
            "expected_utility_long_rail_penalty_bps": head_out.get("expected_utility_long_rail_penalty_bps"),
            "expected_utility_short_rail_penalty_bps": head_out.get("expected_utility_short_rail_penalty_bps"),
            "expected_utility_invalid_side_penalty_bps": head_out.get("expected_utility_invalid_side_penalty_bps"),
            "expected_utility_long_invalid_side_penalty_bps": head_out.get("expected_utility_long_invalid_side_penalty_bps"),
            "expected_utility_short_invalid_side_penalty_bps": head_out.get("expected_utility_short_invalid_side_penalty_bps"),
            "long_bad_path_prob": head_out.get("long_bad_path_prob"),
            "short_bad_path_prob": head_out.get("short_bad_path_prob"),
            "side_validity_logit": head_out.get("side_validity_logit"),
            "long_validity_prob": head_out.get("long_validity_prob"),
            "short_validity_prob": head_out.get("short_validity_prob"),
            "long_expected_mae_bps": head_out.get("long_expected_mae_bps"),
            "short_expected_mae_bps": head_out.get("short_expected_mae_bps"),
            "anchor_logits": head_out.get("anchor_logits"),
            "delta_logits": head_out.get("delta_logits"),
            "mtf_dir_logits": head_out.get("mtf_dir_logits"),
            "anchor_gate": head_out.get("anchor_gate"),
            "geometry_support_evidence": head_out.get("geometry_support_evidence"),
            "geometry_resistance_evidence": head_out.get("geometry_resistance_evidence"),
            "geometry_channel_edge_pressure": head_out.get("geometry_channel_edge_pressure"),
            "geometry_rising_support_rail_long_pressure": head_out.get("geometry_rising_support_rail_long_pressure"),
            "geometry_rising_support_rail_short_trap_pressure": head_out.get("geometry_rising_support_rail_short_trap_pressure"),
            "geometry_falling_resistance_rail_short_pressure": head_out.get("geometry_falling_resistance_rail_short_pressure"),
            "geometry_falling_resistance_rail_long_trap_pressure": head_out.get("geometry_falling_resistance_rail_long_trap_pressure"),
            "trendline_rail_logits": head_out.get("trendline_rail_logits"),
            "trendline_rail_probs": head_out.get("trendline_rail_probs"),
            "trendline_rail_short_early_failure_prob": head_out.get("trendline_rail_short_early_failure_prob"),
            "trendline_rail_long_early_failure_prob": head_out.get("trendline_rail_long_early_failure_prob"),
            "trendline_rail_anti_short_prob": head_out.get("trendline_rail_anti_short_prob"),
            "trendline_rail_anti_long_prob": head_out.get("trendline_rail_anti_long_prob"),
            "trendline_rail_long_evidence": head_out.get("trendline_rail_long_evidence"),
            "trendline_rail_short_evidence": head_out.get("trendline_rail_short_evidence"),
            "trendline_rail_long_minus_short": head_out.get("trendline_rail_long_minus_short"),
            "mtf_trend_evidence": head_out.get("mtf_trend_evidence"),
            "direction_calibration_enabled": head_out.get("direction_calibration_enabled"),
            "direction_calibration_temperature": head_out.get("direction_calibration_temperature"),
            "direction_calibration_bias": head_out.get("direction_calibration_bias"),
            "path_calibration_enabled": head_out.get("path_calibration_enabled"),
            "path_calibration": head_out.get("path_calibration"),
            "anchored_entry_enabled": head_out.get("anchored_entry_enabled"),
            "anchor_source": head_out.get("anchor_source"),
            "tf_agreement_pred": 0.0,
            "path_quality_std": 0.0,
            "position_size_pred": 0.0,
            "atr_bps": float(atr_bps),
        }
        out = {
            "action": action,
            "action_id": {"SKIP": 0, "TAKE_LONG_NOW": 1, "TAKE_SHORT_NOW": 2}[action],
            "edge_score": edge,
            "edge_score_threshold": thr,
            "selection_score_mode": selection_mode,
            "selection_score": selection_score,
            "selection_score_threshold": selection_threshold,
            "session": session,
            "smart_skip_reason": skip_reason,
            "p_long": head_out["p_long"],
            "p_short": head_out["p_short"],
            "p_flat": head_out["p_flat"],
            "p_trade": head_out.get("p_trade"),
            "p_flat_hier": head_out.get("p_flat_hier"),
            "p_long_given_trade": head_out.get("p_long_given_trade"),
            "p_short_given_trade": head_out.get("p_short_given_trade"),
            "legacy_trade_side": int(head_out["trade_side"]),
            "expected_utility_side": head_out.get("expected_utility_side"),
            "selected_side": int(side_idx),
            "expected_utility_long_bps": expected_utility_long,
            "expected_utility_short_bps": expected_utility_short,
            "long_path_utility_pred_bps": head_out.get("long_path_utility_pred_bps"),
            "short_path_utility_pred_bps": head_out.get("short_path_utility_pred_bps"),
            "expected_utility_bad_path_penalty_bps": head_out.get("expected_utility_bad_path_penalty_bps"),
            "expected_utility_uncertainty_penalty_bps": head_out.get("expected_utility_uncertainty_penalty_bps"),
            "expected_utility_rail_penalty_bps": head_out.get("expected_utility_rail_penalty_bps"),
            "expected_utility_long_rail_penalty_bps": head_out.get("expected_utility_long_rail_penalty_bps"),
            "expected_utility_short_rail_penalty_bps": head_out.get("expected_utility_short_rail_penalty_bps"),
            "expected_utility_invalid_side_penalty_bps": head_out.get("expected_utility_invalid_side_penalty_bps"),
            "expected_utility_long_invalid_side_penalty_bps": head_out.get("expected_utility_long_invalid_side_penalty_bps"),
            "expected_utility_short_invalid_side_penalty_bps": head_out.get("expected_utility_short_invalid_side_penalty_bps"),
            "long_bad_path_prob": head_out.get("long_bad_path_prob"),
            "short_bad_path_prob": head_out.get("short_bad_path_prob"),
            "side_validity_logit": head_out.get("side_validity_logit"),
            "long_validity_prob": head_out.get("long_validity_prob"),
            "short_validity_prob": head_out.get("short_validity_prob"),
            "long_expected_mae_bps": head_out.get("long_expected_mae_bps"),
            "short_expected_mae_bps": head_out.get("short_expected_mae_bps"),
            "anchor_logits": head_out.get("anchor_logits"),
            "anchor_probs": head_out.get("anchor_probs"),
            "delta_logits": head_out.get("delta_logits"),
            "mtf_dir_logits": head_out.get("mtf_dir_logits"),
            "mtf_dir_probs": head_out.get("mtf_dir_probs"),
            "anchor_gate": head_out.get("anchor_gate"),
            "geometry_support_evidence": head_out.get("geometry_support_evidence"),
            "geometry_resistance_evidence": head_out.get("geometry_resistance_evidence"),
            "geometry_channel_edge_pressure": head_out.get("geometry_channel_edge_pressure"),
            "geometry_rising_support_rail_long_pressure": head_out.get("geometry_rising_support_rail_long_pressure"),
            "geometry_rising_support_rail_short_trap_pressure": head_out.get("geometry_rising_support_rail_short_trap_pressure"),
            "geometry_falling_resistance_rail_short_pressure": head_out.get("geometry_falling_resistance_rail_short_pressure"),
            "geometry_falling_resistance_rail_long_trap_pressure": head_out.get("geometry_falling_resistance_rail_long_trap_pressure"),
            "trendline_rail_short_early_failure_prob": head_out.get("trendline_rail_short_early_failure_prob"),
            "trendline_rail_long_early_failure_prob": head_out.get("trendline_rail_long_early_failure_prob"),
            "trendline_rail_long_evidence": head_out.get("trendline_rail_long_evidence"),
            "trendline_rail_short_evidence": head_out.get("trendline_rail_short_evidence"),
            "trendline_rail_long_minus_short": head_out.get("trendline_rail_long_minus_short"),
            "mtf_trend_evidence": head_out.get("mtf_trend_evidence"),
            "calibration_version": head_out.get("calibration_version"),
            "direction_calibration_enabled": head_out.get("direction_calibration_enabled"),
            "direction_calibration_temperature": head_out.get("direction_calibration_temperature"),
            "direction_calibration_bias": head_out.get("direction_calibration_bias"),
            "path_calibration_enabled": head_out.get("path_calibration_enabled"),
            "path_calibration": head_out.get("path_calibration"),
            "anchored_entry_enabled": head_out.get("anchored_entry_enabled"),
            "anchor_source": head_out.get("anchor_source"),
            "v10_path_quality_pred": head_out["path_quality_pred"],
            "v10_mfe_pred_at_entry": head_out["mfe_first_n_pred"],
            "v10_tradable_prob": head_out["tradable_prob"],
            "v10_bad_path_prob": head_out["bad_path_prob"],
            "decision_ts": str(head_out["time"]),
            "_v10_snapshot": snapshot,
            "policy": "smart_seq520_candidate_v1",
            "stub": False,
        }
        # async-context staleness journal (serving-wave gap 3) — present only on
        # the live predict_live_bar path; the parity gate forwards heads directly.
        for k in ("context_age_m5_bars", "context_cutoff_ts",
                  "context_refresh_in_flight", "context_mtf_incremental"):
            if k in head_out:
                out[k] = head_out[k]
        return out
