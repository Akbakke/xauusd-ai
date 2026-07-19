#!/usr/bin/env python3
"""V12 daily trade review for the strict model-native Entry journal.

For each completed (and still-open) trade in the trade journal, reconstructs the
full narrative: immutable model evidence → bar trajectory → exit.
Outputs:

  - REVIEW.md        : day-level markdown with summary tables + per-trade rows
  - trades_metrics.csv : one row per trade (for batch analysis / retraining feed)
  - trades/<id>.md   : per-trade detail (narrative + key moments)

Designed to be cron-friendly: idempotent rebuild of today's review every N min.

Inputs:
  - /home/andre2/GX1_DATA/reports/v12_paper_runs/trade_journal/trades/*.json
  - /home/andre2/GX1_DATA/reports/v12_paper_runs/trade_journal/
    trade_journal_index_model_native_v1.csv

There is deliberately no compatibility read from ``entry_score``, SMART/XGB
overlays, Entry-IQL Q values, an old index, or recovered open-trade state.  A
trade whose immutable model-native evidence is absent or malformed fails the
review closed.

Run:
  python3 gx1/execution/v12_daily_trade_review.py [--date 20260708]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS,
    MODEL_NATIVE_RUNTIME_POLICY,
    ModelNativeRuntimeEvidenceError,
    require_model_native_entry_time,
    require_model_native_runtime_evidence,
)
from gx1.contracts.entry_model_native_sizing_authority_v1 import (
    require_model_native_sizing_application_record,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)

LOG = logging.getLogger("v12_review")

JOURNAL_DIR = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")
TRADE_JSON_DIR = JOURNAL_DIR / "trade_journal" / "trades"
INDEX_CSV = (
    JOURNAL_DIR
    / "trade_journal"
    / "trade_journal_index_model_native_v1.csv"
)
REVIEW_BASE = JOURNAL_DIR / "daily_reviews"


class ModelNativeTradeReviewError(RuntimeError):
    """The journal cannot prove a model-native Entry decision."""


def _fail(field: str, detail: str = "missing or invalid") -> None:
    raise ModelNativeTradeReviewError(
        f"[DAILY_REVIEW_MODEL_NATIVE_EVIDENCE_INVALID] {field}: {detail}"
    )


def _finite_scalar(values: dict[str, Any], key: str) -> float:
    if key not in values:
        _fail(key, "missing")
    value = values[key]
    if isinstance(value, bool):
        _fail(key, "boolean is not numeric evidence")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelNativeTradeReviewError(
            f"[DAILY_REVIEW_MODEL_NATIVE_EVIDENCE_INVALID] {key}: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        _fail(key, f"non-finite value {value!r}")
    return parsed


def _finite_vector(values: dict[str, Any], key: str, size: int) -> tuple[float, ...]:
    if key not in values or isinstance(values[key], (str, bytes, dict)):
        _fail(key, f"expected finite vector[{size}]")
    try:
        raw = list(values[key])
    except TypeError as exc:
        raise ModelNativeTradeReviewError(
            f"[DAILY_REVIEW_MODEL_NATIVE_EVIDENCE_INVALID] {key}: expected vector[{size}]"
        ) from exc
    if len(raw) != size:
        _fail(key, f"size={len(raw)} expected={size}")
    parsed: list[float] = []
    for value in raw:
        if isinstance(value, bool):
            _fail(key, "boolean element")
        try:
            item = float(value)
        except (TypeError, ValueError) as exc:
            raise ModelNativeTradeReviewError(
                f"[DAILY_REVIEW_MODEL_NATIVE_EVIDENCE_INVALID] {key}: non-numeric element"
            ) from exc
        if not math.isfinite(item):
            _fail(key, "non-finite element")
        parsed.append(item)
    return tuple(parsed)


def _require_utc_timestamp(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        _fail(key, "missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ModelNativeTradeReviewError(
            f"[DAILY_REVIEW_MODEL_NATIVE_EVIDENCE_INVALID] {key}: invalid ISO timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        _fail(key, "must be timezone-aware UTC")
    return value


def _require_model_evidence(entry: dict[str, Any]) -> dict[str, Any]:
    evidence = entry.get("model_evidence")
    if not isinstance(evidence, dict) or not evidence:
        _fail("entry_snapshot.model_evidence", "missing or empty")
    try:
        validated = require_model_native_runtime_evidence(
            evidence,
            context="DAILY_REVIEW",
        )
    except ModelNativeRuntimeEvidenceError as exc:
        raise ModelNativeTradeReviewError(
            str(exc)
        ) from exc
    if not MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS.issubset(validated):
        _fail(
            "entry_snapshot.model_evidence timing",
            "complete executable timing evidence is required",
        )
    return validated


def _require_entry_snapshot(trade_json: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    entry = trade_json.get("entry_snapshot")
    if not isinstance(entry, dict):
        _fail("entry_snapshot", "missing")
    evidence = _require_model_evidence(entry)

    for key in ("entry_time", "instrument", "side", "session", "model_policy"):
        value = entry.get(key)
        if not isinstance(value, str) or not value.strip():
            _fail(f"entry_snapshot.{key}", "missing")
    _require_utc_timestamp(entry, "entry_time")
    try:
        require_model_native_entry_time(
            evidence,
            entry["entry_time"],
            context="DAILY_REVIEW_ENTRY",
        )
    except ModelNativeRuntimeEvidenceError as exc:
        raise ModelNativeTradeReviewError(str(exc)) from exc
    if (
        entry["model_policy"] != MODEL_NATIVE_RUNTIME_POLICY
        or entry["model_policy"] != evidence["model_policy"]
    ):
        _fail(
            "entry_snapshot.model_policy",
            f"{entry['model_policy']!r} != snapshot {evidence['model_policy']!r} ",
        )
    if entry["session"] != evidence["session"]:
        _fail(
            "entry_snapshot.session",
            f"{entry['session']!r} != snapshot {evidence['session']!r}",
        )
    expected_side = str(evidence["model_direction"]).lower()
    if entry["side"] != expected_side or expected_side not in ("long", "short"):
        _fail(
            "entry_snapshot.side",
            f"{entry['side']!r} != model direction {expected_side!r}",
        )

    market = {
        key: _finite_scalar(entry, key)
        for key in (
            "entry_price",
            "entry_bid",
            "entry_ask",
            "entry_spread_bps",
            "atr_bps",
        )
    }
    if (
        market["entry_price"] <= 0.0
        or market["entry_bid"] <= 0.0
        or market["entry_ask"] < market["entry_bid"]
        or market["entry_spread_bps"] < 0.0
        or market["atr_bps"] <= 0.0
    ):
        _fail("entry_snapshot market evidence", f"range violation: {market}")
    if not math.isclose(
        market["atr_bps"],
        float(evidence["atr_bps"]),
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        _fail("entry_snapshot.atr_bps", "model evidence parity mismatch")

    try:
        sizing_application = require_model_native_sizing_application_record(
            entry.get("sizing_application"),
            context="V12_DAILY_TRADE_REVIEW",
        )
    except RuntimeError as exc:
        _fail("entry_snapshot sizing authority", str(exc))
    sizing_parity = {
        "capacity_units": entry.get("capacity_units"),
        "reference_pre_round_units": entry.get("reference_pre_round_units"),
        "pre_round_units": entry.get("pre_round_units"),
        "units": entry.get("units"),
        "applied_size_multiplier": entry.get("applied_size_multiplier"),
        "model_direction": evidence["model_direction"],
        "position_size_logit": evidence["position_size_logit"],
        "sizing_authority_contract": evidence["sizing_authority_contract"],
    }
    mismatched_sizing = sorted(
        key
        for key, value in sizing_parity.items()
        if sizing_application.get(key) != value
    )
    if mismatched_sizing:
        _fail(
            "entry_snapshot sizing parity",
            ",".join(mismatched_sizing),
        )
    checks = entry.get("execution_checks")
    if (
        not isinstance(checks, list)
        or not checks
        or any(not isinstance(item, str) or not item.strip() for item in checks)
    ):
        _fail("entry_snapshot.execution_checks", "missing")
    return entry, evidence


# ── trade analysis ─────────────────────────────────────────────────────────

def _bar_metrics(bars: list[dict]) -> dict[str, Any]:
    """Derive trade-level metrics from per-bar decision trace."""
    if not bars:
        return {}

    n = len(bars)
    pnls = [b["current_pnl_bps"] for b in bars]
    mfes = [b["cum_mfe_bps"] for b in bars]
    maes = [b["cum_mae_bps"] for b in bars]
    v3_probs = [b.get("v3_should_exit_prob") or 0.0 for b in bars]

    max_mfe = max(mfes)
    max_mae = min(maes)
    final_pnl = pnls[-1]
    mfe_peak_bar = next((i for i, m in enumerate(mfes) if m == max_mfe), 0)
    mae_worst_bar = next((i for i, m in enumerate(maes) if m == max_mae), 0)

    # V3 alarms
    v3_max_prob = max(v3_probs) if v3_probs else 0.0
    v3_first_alarm_bar = next((i for i, p in enumerate(v3_probs) if p > 0.5), -1)

    # Exit-IQL held while the V3 exit model alarmed.
    exit_iql_held_through_v3 = sum(
        1 for b in bars
        if (b.get("v3_should_exit_prob") or 0) > 0.5 and b.get("iql_action") != "EXIT_NOW"
    )

    # MFE giveback ratio: (max_mfe - final_pnl) / max_mfe
    giveback_bps = max_mfe - final_pnl if max_mfe > 0 else 0.0
    giveback_pct = giveback_bps / max_mfe if max_mfe > 0 else 0.0

    # Jojo count: each time pnl crosses 0 (long-only proxy for swings)
    jojo = sum(
        1 for i in range(1, n)
        if (pnls[i - 1] >= 0) != (pnls[i] >= 0)
    )

    return {
        "n_bars": n,
        "max_mfe_bps": max_mfe,
        "max_mae_bps": max_mae,
        "final_pnl_bps": final_pnl,
        "mfe_peak_bar": mfe_peak_bar,
        "mae_worst_bar": mae_worst_bar,
        "v3_max_prob": v3_max_prob,
        "v3_first_alarm_bar": v3_first_alarm_bar,
        "exit_iql_held_through_v3_count": exit_iql_held_through_v3,
        "mfe_giveback_bps": giveback_bps,
        "mfe_giveback_pct": giveback_pct,
        "pnl_zero_crossings": jojo,
    }


def _classify_pattern(metrics: dict, exit_summary: dict | None) -> list[str]:
    """Tag a trade with diagnostic patterns for easier filtering."""
    tags = []
    final = metrics.get("final_pnl_bps", 0.0)
    mfe = metrics.get("max_mfe_bps", 0.0)
    mae = metrics.get("max_mae_bps", 0.0)
    giveback = metrics.get("mfe_giveback_pct", 0.0)

    if exit_summary is None:
        tags.append("STILL_OPEN")
    elif final > 0:
        tags.append("WINNER")
    else:
        tags.append("LOSER")

    if mfe >= 20 and giveback >= 0.5:
        tags.append("MFE_GIVEBACK_50PCT")
    if mfe >= 15 and final <= 0:
        tags.append("MFE_HIT_BUT_NEGATIVE_EXIT")
    if mae <= -20 and final > 0:
        tags.append("RECOVERY_FROM_DEEP_MAE")
    if mae <= -20 and final <= mae * 0.8:
        tags.append("RAN_TO_STOP")
    if metrics.get("exit_iql_held_through_v3_count", 0) >= 5:
        tags.append("EXIT_IQL_HELD_THROUGH_V3_ALARM")
    if metrics.get("pnl_zero_crossings", 0) >= 4:
        tags.append("JOJO_4PLUS")
    return tags


# ── trade JSON → summary row ───────────────────────────────────────────────

def trade_summary_row(trade_json: dict) -> dict[str, Any]:
    """Flatten one proven model-native trade into a metrics row.

    Required Entry evidence is never defaulted. Any missing or inconsistent
    field raises :class:`ModelNativeTradeReviewError` before a row is emitted.
    """
    entry, evidence = _require_entry_snapshot(trade_json)
    exit_s = trade_json.get("exit_summary")
    bars = trade_json.get("v12_bar_decisions") or []

    metrics = _bar_metrics(bars)
    tags = _classify_pattern(metrics, exit_s)
    direction_logits = _finite_vector(evidence, "direction_logits", 3)
    direction_probs = _finite_vector(evidence, "direction_probs", 3)
    public_logits = _finite_vector(
        evidence, "public_trade_flat_decision_logits", 2
    )
    public_probs = _finite_vector(
        evidence, "public_trade_flat_decision_probs", 2
    )
    mtf_logits = _finite_vector(evidence, "mtf_dir_logits", 3)
    mtf_probs = _finite_vector(evidence, "mtf_dir_probs", 3)
    side_utility = _finite_vector(evidence, "side_utility", 2)
    side_bad_logits = _finite_vector(evidence, "side_bad_path_logit", 2)
    side_validity_logits = _finite_vector(evidence, "side_validity_logit", 2)
    side_mae = _finite_vector(evidence, "side_mae", 2)
    specialist_gate = _finite_vector(
        evidence,
        "specialist_gate",
        len(MODEL_NATIVE_TRAINING_SPECIALISTS),
    )

    row: dict[str, Any] = {
        "trade_id": trade_json.get("trade_id") or trade_json.get("trade_key") or "",
        "side": entry["side"],
        "entry_time": entry["entry_time"],
        "entry_price": _finite_scalar(entry, "entry_price"),
        "entry_spread_bps": _finite_scalar(entry, "entry_spread_bps"),
        "atr_bps": _finite_scalar(entry, "atr_bps"),
        "session": entry["session"],
        "model_policy": entry["model_policy"],
        "runtime_evidence_schema_version": evidence[
            "runtime_evidence_schema_version"
        ],
        "session_id": evidence["session_id"],
        "execution_checks": json.dumps(
            entry["execution_checks"], separators=(",", ":")
        ),
        "capacity_units": int(_finite_scalar(entry, "capacity_units")),
        "reference_pre_round_units": _finite_scalar(
            entry, "reference_pre_round_units"
        ),
        "pre_round_units": _finite_scalar(entry, "pre_round_units"),
        "units": int(_finite_scalar(entry, "units")),
        "applied_size_multiplier": _finite_scalar(
            entry, "applied_size_multiplier"
        ),
        "decision_ts": evidence["decision_ts"],
        "model_direction": evidence["model_direction"],
        "model_direction_index": evidence["model_direction_index"],
        "direction_logit_long": direction_logits[0],
        "direction_logit_short": direction_logits[1],
        "direction_logit_flat": direction_logits[2],
        "direction_p_long": direction_probs[0],
        "direction_p_short": direction_probs[1],
        "direction_p_flat": direction_probs[2],
        "public_trade_flat_decision": evidence["public_trade_flat_decision"],
        "public_trade_flat_decision_index": evidence[
            "public_trade_flat_decision_index"
        ],
        "public_logit_trade": public_logits[0],
        "public_logit_flat": public_logits[1],
        "p_trade": public_probs[0],
        "p_flat_hier": public_probs[1],
        "p_long_given_trade": _finite_scalar(evidence, "p_long_given_trade"),
        "p_short_given_trade": _finite_scalar(evidence, "p_short_given_trade"),
        "path_quality": _finite_scalar(evidence, "path_quality"),
        "path_quality_log_var": _finite_scalar(
            evidence, "path_quality_log_var"
        ),
        "path_quality_std": _finite_scalar(evidence, "path_quality_std"),
        "mfe_first_n": _finite_scalar(evidence, "mfe_first_n"),
        "tradable_prob": _finite_scalar(evidence, "tradable_prob"),
        "bad_path_prob": _finite_scalar(evidence, "bad_path_prob"),
        "clean_edge_prob": _finite_scalar(evidence, "clean_edge_prob"),
        "survival_prob": _finite_scalar(evidence, "survival_prob"),
        "tf_agreement_logit": _finite_scalar(evidence, "tf_agreement_logit"),
        "tf_agreement_pred": _finite_scalar(evidence, "tf_agreement_pred"),
        "position_size_logit": _finite_scalar(evidence, "position_size_logit"),
        "position_size_pred": _finite_scalar(evidence, "position_size_pred"),
        "side_utility_long_raw": side_utility[0],
        "side_utility_short_raw": side_utility[1],
        "side_bad_path_logit_long": side_bad_logits[0],
        "side_bad_path_logit_short": side_bad_logits[1],
        "long_bad_path_prob": _finite_scalar(evidence, "long_bad_path_prob"),
        "short_bad_path_prob": _finite_scalar(evidence, "short_bad_path_prob"),
        "side_validity_logit_long": side_validity_logits[0],
        "side_validity_logit_short": side_validity_logits[1],
        "long_validity_prob": _finite_scalar(evidence, "long_validity_prob"),
        "short_validity_prob": _finite_scalar(evidence, "short_validity_prob"),
        "side_mae_long_raw": side_mae[0],
        "side_mae_short_raw": side_mae[1],
        "mtf_logit_long": mtf_logits[0],
        "mtf_logit_short": mtf_logits[1],
        "mtf_logit_flat": mtf_logits[2],
        "mtf_p_long": mtf_probs[0],
        "mtf_p_short": mtf_probs[1],
        "mtf_p_flat": mtf_probs[2],
        "mtf_trend_evidence": _finite_scalar(evidence, "mtf_trend_evidence"),
        "specialist_names": json.dumps(
            list(MODEL_NATIVE_TRAINING_SPECIALISTS), separators=(",", ":")
        ),
        "specialist_gate": json.dumps(specialist_gate, separators=(",", ":")),
        "calibration_version": evidence["calibration_version"],
        # Outcome
        "exit_time": (exit_s or {}).get("exit_time", ""),
        "exit_price": (exit_s or {}).get("exit_price", ""),
        "exit_reason": (exit_s or {}).get("exit_reason", "STILL_OPEN"),
        "realized_pnl_bps": (exit_s or {}).get("realized_pnl_bps", metrics.get("final_pnl_bps", 0.0)),
        # Bar-trace derived
        "n_bars": metrics.get("n_bars", 0),
        "max_mfe_bps": metrics.get("max_mfe_bps", 0.0),
        "max_mae_bps": metrics.get("max_mae_bps", 0.0),
        "mfe_peak_bar": metrics.get("mfe_peak_bar", -1),
        "mae_worst_bar": metrics.get("mae_worst_bar", -1),
        "mfe_giveback_bps": metrics.get("mfe_giveback_bps", 0.0),
        "mfe_giveback_pct": metrics.get("mfe_giveback_pct", 0.0),
        "v3_max_prob": metrics.get("v3_max_prob", 0.0),
        "v3_first_alarm_bar": metrics.get("v3_first_alarm_bar", -1),
        "exit_iql_held_through_v3_count": metrics.get(
            "exit_iql_held_through_v3_count", 0
        ),
        "pnl_zero_crossings": metrics.get("pnl_zero_crossings", 0),
        "tags": ",".join(tags),
    }
    for specialist, weight in zip(
        MODEL_NATIVE_TRAINING_SPECIALISTS,
        specialist_gate,
        strict=True,
    ):
        row[f"specialist_gate_{specialist}"] = weight
    return row


# ── per-trade detail markdown ──────────────────────────────────────────────

def render_trade_detail(trade_json: dict, summary: dict, out_path: Path) -> None:
    entry, evidence = _require_entry_snapshot(trade_json)
    exit_s = trade_json.get("exit_summary")
    bars = trade_json.get("v12_bar_decisions") or []
    tid = trade_json.get("trade_id") or trade_json.get("trade_key") or "?"
    direction_logits = _finite_vector(evidence, "direction_logits", 3)
    direction_probs = _finite_vector(evidence, "direction_probs", 3)
    public_logits = _finite_vector(
        evidence, "public_trade_flat_decision_logits", 2
    )
    public_probs = _finite_vector(
        evidence, "public_trade_flat_decision_probs", 2
    )
    specialist_gate = _finite_vector(
        evidence,
        "specialist_gate",
        len(MODEL_NATIVE_TRAINING_SPECIALISTS),
    )
    mtf_logits = _finite_vector(evidence, "mtf_dir_logits", 3)
    mtf_probs = _finite_vector(evidence, "mtf_dir_probs", 3)
    side_utility = _finite_vector(evidence, "side_utility", 2)
    rail_logits = _finite_vector(evidence, "trendline_rail_logits", 6)
    rail_probs = _finite_vector(evidence, "trendline_rail_probs", 6)

    lines: list[str] = []
    lines.append(f"# Trade #{tid}\n")
    lines.append(f"**Tags:** `{summary['tags']}`\n")

    lines.append("## Entry\n")
    lines.append(f"- **Time:** {entry['entry_time']}")
    lines.append(
        f"- **Side:** {entry['side']}  ·  **Price:** "
        f"{float(entry['entry_price']):.2f}  ·  **Spread:** "
        f"{float(entry['entry_spread_bps']):.2f} bps  ·  **ATR:** "
        f"{float(entry['atr_bps']):.2f} bps"
    )
    lines.append(
        f"- **Decision M5 bucket:** {evidence['decision_ts']}  ·  "
        f"**Session:** {entry['session']}  ·  **Policy:** {entry['model_policy']}"
    )
    lines.append(
        "- **Execution checks:** " + ", ".join(entry["execution_checks"])
    )
    lines.append("")

    lines.append("### Model-native direction (sole Entry authority)")
    lines.append(
        f"- **Decision:** {evidence['model_direction']} "
        f"(index {evidence['model_direction_index']})"
    )
    lines.append(
        "- calibrated logits LONG/SHORT/FLAT: "
        f"{direction_logits[0]:+.4f} / {direction_logits[1]:+.4f} / "
        f"{direction_logits[2]:+.4f}"
    )
    lines.append(
        "- calibrated probabilities LONG/SHORT/FLAT: "
        f"{direction_probs[0]:.4f} / {direction_probs[1]:.4f} / "
        f"{direction_probs[2]:.4f}"
    )
    lines.append("")

    lines.append("### Public TRADE/FLAT hierarchy")
    lines.append(
        f"- **Decision:** {evidence['public_trade_flat_decision']} "
        f"(index {evidence['public_trade_flat_decision_index']})"
    )
    lines.append(
        f"- logits TRADE/FLAT: {public_logits[0]:+.4f} / {public_logits[1]:+.4f}"
    )
    lines.append(
        f"- probabilities TRADE/FLAT: {public_probs[0]:.4f} / "
        f"{public_probs[1]:.4f}"
    )
    lines.append(
        "- conditional side probabilities LONG|TRADE / SHORT|TRADE: "
        f"{float(evidence['p_long_given_trade']):.4f} / "
        f"{float(evidence['p_short_given_trade']):.4f}"
    )
    lines.append("")

    lines.append("### Learned path, quality and agreement evidence")
    lines.append(
        f"- path_quality: {float(evidence['path_quality']):+.4f}  ·  "
        f"std: {float(evidence['path_quality_std']):.4f}  ·  "
        f"log_var: {float(evidence['path_quality_log_var']):+.4f}"
    )
    lines.append(
        f"- MFE first-N: {float(evidence['mfe_first_n']):+.3f}  ·  "
        f"tradable: {float(evidence['tradable_prob']):.4f}  ·  "
        f"bad-path: {float(evidence['bad_path_prob']):.4f}"
    )
    lines.append(
        f"- clean-edge: {float(evidence['clean_edge_prob']):.4f}  ·  "
        f"survival: {float(evidence['survival_prob']):.4f}  ·  "
        f"TF agreement: {float(evidence['tf_agreement_pred']):.4f} "
        f"(logit {float(evidence['tf_agreement_logit']):+.4f})"
    )
    lines.append("")

    lines.append("### Side utility diagnostics (evidence only)")
    lines.append(
        f"- raw utility LONG/SHORT: {side_utility[0]:+.4f} / "
        f"{side_utility[1]:+.4f}"
    )
    lines.append(
        "- bad-path probability LONG/SHORT: "
        f"{float(evidence['long_bad_path_prob']):.4f} / "
        f"{float(evidence['short_bad_path_prob']):.4f}"
    )
    lines.append(
        "- validity probability LONG/SHORT: "
        f"{float(evidence['long_validity_prob']):.4f} / "
        f"{float(evidence['short_validity_prob']):.4f}"
    )
    side_mae = _finite_vector(evidence, "side_mae", 2)
    lines.append(
        f"- raw MAE LONG/SHORT: {side_mae[0]:+.4f} / {side_mae[1]:+.4f}"
    )
    lines.append("")

    lines.append("### Specialist fusion")
    lines.append("| specialist | learned gate weight |")
    lines.append("|---|---:|")
    for specialist, weight in zip(
        MODEL_NATIVE_TRAINING_SPECIALISTS,
        specialist_gate,
        strict=True,
    ):
        lines.append(f"| {specialist} | {weight:.6f} |")
    lines.append("")

    lines.append("### MTF trend and geometry evidence")
    lines.append(
        "- MTF logits LONG/SHORT/FLAT: "
        f"{mtf_logits[0]:+.4f} / {mtf_logits[1]:+.4f} / {mtf_logits[2]:+.4f}"
    )
    lines.append(
        "- MTF probabilities LONG/SHORT/FLAT: "
        f"{mtf_probs[0]:.4f} / {mtf_probs[1]:.4f} / {mtf_probs[2]:.4f}  ·  "
        f"trend evidence: {float(evidence['mtf_trend_evidence']):+.4f}"
    )
    lines.append(
        "- channel-edge / rising-support L / rising-support short-trap: "
        f"{float(evidence['geometry_channel_edge_pressure']):+.4f} / "
        f"{float(evidence['geometry_rising_support_rail_long_pressure']):+.4f} / "
        f"{float(evidence['geometry_rising_support_rail_short_trap_pressure']):+.4f}"
    )
    lines.append(
        "- falling-resistance S / falling-resistance long-trap: "
        f"{float(evidence['geometry_falling_resistance_rail_short_pressure']):+.4f} / "
        f"{float(evidence['geometry_falling_resistance_rail_long_trap_pressure']):+.4f}"
    )
    lines.append(
        "- learned rail logits: " + " / ".join(f"{value:+.4f}" for value in rail_logits)
    )
    lines.append(
        "- learned rail probabilities: "
        + " / ".join(f"{value:.4f}" for value in rail_probs)
    )
    lines.append("")

    lines.append("### Learned sizing evidence vs applied exposure")
    lines.append(
        f"- learned position_size_pred: **{float(evidence['position_size_pred']):.4f}** "
        f"(logit {float(evidence['position_size_logit']):+.4f})"
    )
    lines.append(
        f"- applied_size_multiplier: **{float(entry['applied_size_multiplier']):.4f}** "
        f"· capacity units: {int(entry['capacity_units'])} "
        f"· reference units (pre-round): {float(entry['reference_pre_round_units']):.4f} "
        f"· learned units (pre-round): {float(entry['pre_round_units']):.4f} "
        f"· applied integer units: {int(entry['units'])}"
    )
    lines.append(
        "- The adopted TRAIN/VAL calibration maps the model logit monotonically to "
        "a broker-capacity fraction; the journaled integer units are its exact output."
    )
    lines.append(
        "- dynamic sizing authorized: **true** only under the SHA-bound newest "
        "learned_calibrated adoption and recomputed TEST/OOS proof recorded above."
    )
    lines.append("")

    if bars:
        lines.append("## In-trade trajectory\n")
        lines.append("| bar | time | bid | pnl | mfe | mae | dd_from_peak | v3_prob | v3_consec | exit_action | exit_source |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        # Sample at most ~30 bars across the trade for readability
        n = len(bars)
        step = max(1, n // 30)
        sampled_idx = list(range(0, n, step))
        # always include MFE peak + MAE worst bars
        if summary["mfe_peak_bar"] not in sampled_idx and summary["mfe_peak_bar"] >= 0:
            sampled_idx.append(summary["mfe_peak_bar"])
        if summary["mae_worst_bar"] not in sampled_idx and summary["mae_worst_bar"] >= 0:
            sampled_idx.append(summary["mae_worst_bar"])
        if summary["v3_first_alarm_bar"] not in sampled_idx and summary["v3_first_alarm_bar"] >= 0:
            sampled_idx.append(summary["v3_first_alarm_bar"])
        if n - 1 not in sampled_idx:
            sampled_idx.append(n - 1)
        for i in sorted(set(sampled_idx)):
            b = bars[i]
            mark = ""
            if i == summary["mfe_peak_bar"]:
                mark = " 🟢MFE"
            if i == summary["mae_worst_bar"]:
                mark += " 🔴MAE"
            if i == summary["v3_first_alarm_bar"]:
                mark += " ⚠️V3"
            ts = b.get("timestamp", "?")[11:16]
            dd = (b.get("cum_mfe_bps", 0) - b.get("current_pnl_bps", 0))
            lines.append(
                f"| {b.get('bars_in_trade', i)}{mark} | {ts} | {b.get('bid', 0):.2f} | "
                f"{b.get('current_pnl_bps', 0):+.2f} | {b.get('cum_mfe_bps', 0):.2f} | "
                f"{b.get('cum_mae_bps', 0):.2f} | {dd:.2f} | "
                f"{b.get('v3_should_exit_prob', 0):.3f} | {b.get('v3_consecutive_exits', 0)} | "
                f"{b.get('iql_action', '?')} | {b.get('iql_decision_source', '?')} |"
            )
        lines.append("")

    lines.append("## Exit\n")
    if exit_s:
        lines.append(f"- **Time:** {exit_s.get('exit_time')}")
        lines.append(f"- **Price:** {exit_s.get('exit_price'):.2f}  ·  **Reason:** {exit_s.get('exit_reason')}")
        lines.append(f"- **Realized PnL:** {exit_s.get('realized_pnl_bps'):+.2f} bps")
        lines.append(f"- **Max MFE / MAE:** {exit_s.get('max_mfe_bps'):.2f} / {exit_s.get('max_mae_bps'):.2f} bps")
        lines.append(f"- **Intratrade drawdown from MFE peak:** {exit_s.get('intratrade_drawdown_bps'):.2f} bps")
    else:
        lines.append(f"- **STILL OPEN** (bars_in_trade={summary['n_bars']})")
        lines.append(f"- Current unrealized: **{summary['realized_pnl_bps']:+.2f} bps**")
    lines.append("")

    lines.append("## Verdikt\n")
    tags = summary["tags"].split(",")
    if "MFE_HIT_BUT_NEGATIVE_EXIT" in tags:
        lines.append("- ⚠️ **MFE peak ble ikke kapitalisert** — modellen så peak ≥15 bps men endte i minus.")
    if "MFE_GIVEBACK_50PCT" in tags:
        lines.append(f"- ⚠️ **Giveback ≥50%** — MFE peak {summary['max_mfe_bps']:.1f} bps → final {summary['realized_pnl_bps']:+.1f} bps (giveback {summary['mfe_giveback_pct']*100:.0f}%).")
    if "RECOVERY_FROM_DEEP_MAE" in tags:
        lines.append(f"- ✅ **Recovery fra deep MAE** — bunn {summary['max_mae_bps']:.1f} bps → final {summary['realized_pnl_bps']:+.1f}. Exit-IQL holdt rett.")
    if "EXIT_IQL_HELD_THROUGH_V3_ALARM" in tags:
        lines.append(
            f"- 🟡 **Exit-IQL ignored V3** — V3 sa exit i "
            f"{summary['exit_iql_held_through_v3_count']} bars; Exit-IQL holdt."
        )
    if "JOJO_4PLUS" in tags:
        lines.append(f"- ⚠️ **{summary['pnl_zero_crossings']} svingninger** rundt break-even — roller-coaster mønster.")
    if "RAN_TO_STOP" in tags:
        lines.append("- 🔴 **Løp til stop-like deep MAE** — Exit-IQL holdt for lenge i tap, eller markedet snudde aldri.")
    if not any(t in tags for t in ["MFE_GIVEBACK_50PCT", "MFE_HIT_BUT_NEGATIVE_EXIT", "RECOVERY_FROM_DEEP_MAE",
                                    "EXIT_IQL_HELD_THROUGH_V3_ALARM", "JOJO_4PLUS", "RAN_TO_STOP"]):
        lines.append("- (Ingen patologiske mønstre flagget)")
    lines.append("")

    out_path.write_text("\n".join(lines))


# ── day-level review markdown ──────────────────────────────────────────────

def render_day_review(rows: list[dict], date_str: str, out_dir: Path) -> None:
    closed = [r for r in rows if r["exit_reason"] != "STILL_OPEN"]
    still_open = [r for r in rows if r["exit_reason"] == "STILL_OPEN"]
    winners = [r for r in closed if (r["realized_pnl_bps"] or 0) > 0]
    losers = [r for r in closed if (r["realized_pnl_bps"] or 0) <= 0]

    total_realized = sum(float(r["realized_pnl_bps"] or 0) for r in closed)
    total_unrealized = sum(float(r["realized_pnl_bps"] or 0) for r in still_open)

    # Cluster detection: trades opened within 5 min, same side
    sorted_rows = sorted(rows, key=lambda r: r["entry_time"] or "")
    clusters: list[list[dict]] = []
    for r in sorted_rows:
        if not r["entry_time"]:
            continue
        if clusters and clusters[-1]:
            last = clusters[-1][-1]
            last_t = datetime.fromisoformat(last["entry_time"].replace("Z", "+00:00"))
            this_t = datetime.fromisoformat(r["entry_time"].replace("Z", "+00:00"))
            if r["side"] == last["side"] and (this_t - last_t).total_seconds() < 300:
                clusters[-1].append(r)
                continue
        clusters.append([r])
    multi_clusters = [c for c in clusters if len(c) >= 2]

    lines = []
    lines.append(f"# V12 Daily Trade Review — {date_str}\n")
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n")

    lines.append("## Day summary\n")
    lines.append(f"- **Trades opened today:** {len(rows)}  ·  **closed:** {len(closed)}  ·  **still open:** {len(still_open)}")
    if closed:
        wr = 100.0 * len(winners) / len(closed)
        lines.append(f"- **Win rate (closed):** {wr:.1f}%  ({len(winners)}W / {len(losers)}L)")
    lines.append(f"- **Realized PnL (sum bps):** {total_realized:+.2f}")
    lines.append(f"- **Unrealized PnL (sum bps):** {total_unrealized:+.2f}")
    if rows:
        mfes = [float(r["max_mfe_bps"] or 0) for r in rows]
        maes = [float(r["max_mae_bps"] or 0) for r in rows]
        def med(xs: list[float]) -> float:
            return sorted(xs)[len(xs) // 2] if xs else 0.0

        lines.append(f"- **Median MFE peak:** {med(mfes):.1f} bps  ·  **Median MAE worst:** {med(maes):.1f} bps")
        lines.append(f"- **Trades with MFE peak ≥20 bps:** {sum(1 for m in mfes if m >= 20)}")
        lines.append(f"- **Trades with MAE ≤-20 bps:** {sum(1 for m in maes if m <= -20)}")
    lines.append("")

    # Clusters
    if multi_clusters:
        lines.append("## Concentration / clustering\n")
        for i, c in enumerate(multi_clusters, 1):
            sides = c[0]["side"]
            tstart = c[0]["entry_time"][:19]
            tend = c[-1]["entry_time"][:19]
            ids = ", ".join(str(r["trade_id"]) for r in c)
            total_pnl = sum(float(r["realized_pnl_bps"] or 0) for r in c)
            lines.append(f"- **Cluster {i}:** {len(c)}× {sides.upper()} from {tstart} to {tend}  →  combined PnL {total_pnl:+.2f} bps  ({ids})")
        lines.append("\n*(Same-side trades within 5 min of each other = 1 V12 signal × N entries, ikke uavhengige.)*\n")

    # Per-trade table
    lines.append("## Per-trade summary\n")
    lines.append("| id | model direction | public | p(L/S/F) | size learned/applied | bars | MFE | MAE | realized | tags |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        entry_t = (r["entry_time"] or "")[11:16] if r["entry_time"] else "?"
        rea = r["realized_pnl_bps"]
        rea_str = f"{float(rea):+.1f}" if rea != "" and rea is not None else "?"
        tag_short = r["tags"].replace(",", " · ")[:60]
        lines.append(
            f"| {r['trade_id']} ({entry_t}) | {r['model_direction']} | "
            f"{r['public_trade_flat_decision']} | "
            f"{r['direction_p_long']:.2f}/{r['direction_p_short']:.2f}/"
            f"{r['direction_p_flat']:.2f} | {r['position_size_pred']:.3f}/"
            f"{r['applied_size_multiplier']:.3f} | {r['n_bars']} | "
            f"{float(r['max_mfe_bps'] or 0):.1f} | {float(r['max_mae_bps'] or 0):.1f} | "
            f"{rea_str} | {tag_short} |"
        )
    lines.append("")

    # Pattern findings
    lines.append("## Pattern findings\n")
    flagged = {
        "MFE_GIVEBACK_50PCT": [],
        "MFE_HIT_BUT_NEGATIVE_EXIT": [],
        "RECOVERY_FROM_DEEP_MAE": [],
        "EXIT_IQL_HELD_THROUGH_V3_ALARM": [],
        "JOJO_4PLUS": [],
        "RAN_TO_STOP": [],
    }
    for r in rows:
        for t in r["tags"].split(","):
            if t in flagged:
                flagged[t].append(r["trade_id"])
    for tag, ids in flagged.items():
        if ids:
            lines.append(f"- **{tag}** ({len(ids)}): {', '.join(str(i) for i in ids)}")
    if not any(flagged.values()):
        lines.append("- (Ingen patologiske mønstre i dagens trades)")
    lines.append("")

    lines.append("## Per-trade detail\n")
    for r in rows:
        lines.append(f"- [Trade #{r['trade_id']}](trades/{r['trade_id']}.md) — {r['side']} {r['n_bars']}b  pnl={float(r['realized_pnl_bps'] or 0):+.1f}")
    lines.append("")

    lines.append("---\n")
    lines.append(
        "_Datakilder: `trade_journal/trades/*.json`, "
        f"`{INDEX_CSV.name}`._\n"
    )
    lines.append(
        "_Entry-feltene er validerte model-native bevis; Exit-feltene er "
        "observerte bane- og exitdata._\n"
    )

    (out_dir / "REVIEW.md").write_text("\n".join(lines))


# ── main ───────────────────────────────────────────────────────────────────

def _generate_review(date_str: str, out_dir: Path) -> int:
    LOG.info(f"Reading trade journals from {TRADE_JSON_DIR}")
    rows: list[dict[str, Any]] = []
    reviewed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    files = sorted(TRADE_JSON_DIR.glob("*.json"))
    iso_date = _to_iso_date(date_str)
    for jf in files:
        try:
            data = json.loads(jf.read_text())
        except Exception as e:
            raise ModelNativeTradeReviewError(
                f"[DAILY_REVIEW_JOURNAL_PARSE_FAILED] {jf.name}"
            ) from e
        if not isinstance(data, dict):
            raise ModelNativeTradeReviewError(
                f"[DAILY_REVIEW_JOURNAL_INVALID] {jf.name}: expected object"
            )
        entry = data.get("entry_snapshot")
        if not isinstance(entry, dict):
            raise ModelNativeTradeReviewError(
                f"[DAILY_REVIEW_ENTRY_SNAPSHOT_MISSING] {jf.name}"
            )
        entry_t = entry.get("entry_time")
        if not isinstance(entry_t, str) or not entry_t.strip():
            raise ModelNativeTradeReviewError(
                f"[DAILY_REVIEW_ENTRY_TIME_MISSING] {jf.name}"
            )
        if not entry_t.startswith(iso_date):
            continue
        try:
            s = trade_summary_row(data)
        except ModelNativeTradeReviewError as exc:
            raise ModelNativeTradeReviewError(
                f"[DAILY_REVIEW_TRADE_BLOCKED] {jf.name}: {exc}"
            ) from exc
        rows.append(s)
        reviewed.append((data, s))

    rows.sort(key=lambda r: r["entry_time"] or "")
    reviewed.sort(key=lambda pair: pair[1]["entry_time"])
    LOG.info(f"  {len(rows)} trades from {date_str}")

    trades_out = out_dir / "trades"
    trades_out.mkdir(parents=True, exist_ok=True)
    for data, summary in reviewed:
        trade_id = data.get("trade_id") or data.get("trade_key")
        render_trade_detail(data, summary, trades_out / f"{trade_id}.md")

    # Write CSV
    csv_path = out_dir / "trades_metrics.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        LOG.info(f"  wrote {csv_path}")
    else:
        csv_path.unlink(missing_ok=True)

    # Day review
    render_day_review(rows, date_str, out_dir)
    LOG.info(f"  wrote {out_dir / 'REVIEW.md'}")
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="V12 daily trade review")
    p.add_argument("--date", type=str, default=None,
                   help="YYYYMMDD (default: today UTC)")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (default: GX1_DATA/.../daily_reviews/<date>/)")
    p.add_argument("--loop", action="store_true",
                   help="Regenerate review every --interval seconds. "
                        "If --date is omitted, follows the current UTC date "
                        "(auto-rotates at 00:00 UTC).")
    p.add_argument("--interval", type=int, default=120,
                   help="Seconds between rebuilds in --loop mode (default: 120)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")

    if args.loop:
        import time
        while True:
            date_str = args.date or datetime.now(timezone.utc).strftime("%Y%m%d")
            out_dir = Path(args.out_dir) if args.out_dir else REVIEW_BASE / date_str
            try:
                _generate_review(date_str, out_dir)
            except Exception as exc:
                LOG.error(f"review generation failed: {exc}")
            time.sleep(args.interval)
    else:
        date_str = args.date or datetime.now(timezone.utc).strftime("%Y%m%d")
        out_dir = Path(args.out_dir) if args.out_dir else REVIEW_BASE / date_str
        _generate_review(date_str, out_dir)
    return 0


def _to_iso_date(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


if __name__ == "__main__":
    raise SystemExit(main())
