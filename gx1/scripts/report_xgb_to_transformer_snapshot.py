#!/usr/bin/env python3
"""
XGB → Transformer Snapshot Report

PURPOSE: Generate a precise SSoT report of what XGB provides to Transformer:
    - Channel names, injection points, counts, statistics
    - Missing/constant rate analysis
    - Edge dependency check (per-channel ablation summary if requested)

Usage:
    python3 gx1/scripts/report_xgb_to_transformer_snapshot.py <run_root>
    python3 gx1/scripts/report_xgb_to_transformer_snapshot.py <run_root> --write-json --write-md
    python3 gx1/scripts/report_xgb_to_transformer_snapshot.py <run_root> --top-k 20

Output:
    XGB_TO_TRANSFORMER_SNAPSHOT.json
    XGB_TO_TRANSFORMER_SNAPSHOT.md
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from gx1.contracts.signal_bridge_v1 import (
    ORDERED_FIELDS as SIGNAL_BRIDGE_FIELDS,
    SEQ_SIGNAL_DIM as SIGNAL_BRIDGE_SEQ_DIM,
    SNAP_SIGNAL_DIM as SIGNAL_BRIDGE_SNAP_DIM,
    SIGNAL_BRIDGE_ID,
    ORDERED_CTX_CONT_NAMES_EXTENDED,
    ORDERED_CTX_CAT_NAMES_EXTENDED,
)


# ============================================================================
# SCORE LANDSCAPE AUDIT (EVAL LOG)
# ============================================================================

_MARGIN_BUCKETS = [
    ("lt_-0.05", float("-inf"), -0.05),
    ("-0.05_to_-0.02", -0.05, -0.02),
    ("-0.02_to_0", -0.02, 0.0),
    ("0_to_0.02", 0.0, 0.02),
    ("0.02_to_0.05", 0.02, 0.05),
    ("gt_0.05", 0.05, float("inf")),
]


def _winner_from_probs(p_long: float, p_short: float, p_flat: float) -> str:
    if p_long >= p_short and p_long >= p_flat:
        return "LONG"
    if p_short >= p_long and p_short >= p_flat:
        return "SHORT"
    return "FLAT"


def _bucket_margin(val: float) -> str:
    for name, lo, hi in _MARGIN_BUCKETS:
        if lo == float("-inf") and val < hi:
            return name
        if hi == float("inf") and val > lo:
            return name
        if lo <= val < hi:
            return name
    return "unknown"


def _init_margin_hist() -> Dict[str, int]:
    return {name: 0 for name, _, _ in _MARGIN_BUCKETS}


def _find_eval_logs(run_root: Path) -> List[Path]:
    candidates = []
    # Standard replay location
    candidates.extend(sorted((run_root / "replay" / "chunk_0" / "logs").glob("eval_log_*.jsonl")))
    # Fallback: allow passing replay dir directly
    candidates.extend(sorted((run_root / "chunk_0" / "logs").glob("eval_log_*.jsonl")))
    # Final fallback: any eval_log under run_root (bounded search)
    if not candidates:
        candidates.extend(sorted(run_root.glob("**/eval_log_*.jsonl")))
    # De-dup
    seen = set()
    unique = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _score_landscape_audit(eval_logs: List[Path]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "rows": 0,
        "missing_rows": 0,
        "winner_counts": {"xgb": {"LONG": 0, "SHORT": 0, "FLAT": 0}, "entry": {"LONG": 0, "SHORT": 0, "FLAT": 0}},
        "margin_hist": {
            "xgb_margin_long_short": _init_margin_hist(),
            "entry_margin_long_short": _init_margin_hist(),
        },
        "transitions": {
            "XGB_LONG__ENTRY_SHORT": 0,
            "XGB_SHORT__ENTRY_LONG": 0,
            "XGB_LONG__ENTRY_LONG": 0,
            "XGB_SHORT__ENTRY_SHORT": 0,
        },
        "sessions": {},
        "entry_long_bias_audit": {
            "rows": 0,
            "missing_rows": 0,
            "xgb_short_margin_ge_0.02": {"count": 0, "rate": 0.0},
            "xgb_short_margin_ge_0.05": {"count": 0, "rate": 0.0},
            "entry_long_minus_short_stats": {},
            "xgb_probs_stats": {},
            "entry_probs_stats": {},
            "entry_minus_xgb_stats": {},
            "session_split": {},
            "ctx_cont6_summary": {"available": False, "per_index": {}},
            "ctx_cat6_counts": {"available": False, "per_index": {}},
        },
    }

    session_keys = ["EU", "OVERLAP", "US"]
    for sess in session_keys:
        out["sessions"][sess] = {
            "rows": 0,
            "missing_rows": 0,
            "winner_counts": {"xgb": {"LONG": 0, "SHORT": 0, "FLAT": 0}, "entry": {"LONG": 0, "SHORT": 0, "FLAT": 0}},
            "margin_hist": {
                "xgb_margin_long_short": _init_margin_hist(),
                "entry_margin_long_short": _init_margin_hist(),
            },
            "transitions": {
                "XGB_LONG__ENTRY_SHORT": 0,
                "XGB_SHORT__ENTRY_LONG": 0,
                "XGB_LONG__ENTRY_LONG": 0,
                "XGB_SHORT__ENTRY_SHORT": 0,
            },
        }

    def _apply_stats(target: Dict[str, Any], xgb_p_long: float, xgb_p_short: float, xgb_p_flat: float,
                     entry_p_long: float, entry_p_short: float, entry_p_flat: float) -> None:
        target["rows"] += 1
        xgb_winner = _winner_from_probs(xgb_p_long, xgb_p_short, xgb_p_flat)
        entry_winner = _winner_from_probs(entry_p_long, entry_p_short, entry_p_flat)
        target["winner_counts"]["xgb"][xgb_winner] += 1
        target["winner_counts"]["entry"][entry_winner] += 1

        xgb_margin = xgb_p_long - xgb_p_short
        entry_margin = entry_p_long - entry_p_short
        target["margin_hist"]["xgb_margin_long_short"][_bucket_margin(xgb_margin)] += 1
        target["margin_hist"]["entry_margin_long_short"][_bucket_margin(entry_margin)] += 1

        if xgb_winner in ("LONG", "SHORT") and entry_winner in ("LONG", "SHORT"):
            key = f"XGB_{xgb_winner}__ENTRY_{entry_winner}"
            if key in target["transitions"]:
                target["transitions"][key] += 1

    bias = out["entry_long_bias_audit"]
    bias_xgb_long = []
    bias_xgb_short = []
    bias_xgb_flat = []
    bias_entry_long = []
    bias_entry_short = []
    bias_entry_flat = []
    bias_entry_minus_xgb_long = []
    bias_entry_minus_xgb_short = []
    bias_entry_minus_xgb_flat = []
    bias_entry_long_minus_short = []
    bias_ctx_cont = [[] for _ in range(6)]
    bias_ctx_cat_counts: Dict[int, Dict[int, int]] = {i: {} for i in range(6)}

    def _bias_add_ctx(ctx_cont, ctx_cat) -> None:
        if isinstance(ctx_cont, list) and len(ctx_cont) >= 6:
            bias["ctx_cont6_summary"]["available"] = True
            for i in range(6):
                try:
                    val = float(ctx_cont[i])
                except Exception:
                    continue
                bias_ctx_cont[i].append(val)
        if isinstance(ctx_cat, list) and len(ctx_cat) >= 6:
            bias["ctx_cat6_counts"]["available"] = True
            for i in range(6):
                try:
                    cat_val = int(ctx_cat[i])
                except Exception:
                    continue
                bias_ctx_cat_counts[i][cat_val] = bias_ctx_cat_counts[i].get(cat_val, 0) + 1

    for eval_path in eval_logs:
        with eval_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    out["missing_rows"] += 1
                    continue

                xgb_p_long = rec.get("xgb_p_long")
                xgb_p_short = rec.get("xgb_p_short")
                xgb_p_flat = rec.get("xgb_p_flat")
                entry_p_long = rec.get("entry_p_long")
                entry_p_short = rec.get("entry_p_short")
                entry_p_flat = rec.get("entry_p_flat")
                session = rec.get("session")

                if None in (xgb_p_long, xgb_p_short, xgb_p_flat, entry_p_long, entry_p_short, entry_p_flat):
                    out["missing_rows"] += 1
                    if session in out["sessions"]:
                        out["sessions"][session]["missing_rows"] += 1
                    continue

                _apply_stats(out, xgb_p_long, xgb_p_short, xgb_p_flat, entry_p_long, entry_p_short, entry_p_flat)

                if session in out["sessions"]:
                    _apply_stats(out["sessions"][session], xgb_p_long, xgb_p_short, xgb_p_flat,
                                 entry_p_long, entry_p_short, entry_p_flat)
                else:
                    # Count rows even if session is not in requested split
                    out["rows"] += 0

                # Entry long bias audit: only XGB_SHORT -> ENTRY_LONG cases
                xgb_winner = _winner_from_probs(xgb_p_long, xgb_p_short, xgb_p_flat)
                entry_winner = _winner_from_probs(entry_p_long, entry_p_short, entry_p_flat)
                if xgb_winner == "SHORT" and entry_winner == "LONG":
                    bias["rows"] += 1
                    bias_xgb_long.append(float(xgb_p_long))
                    bias_xgb_short.append(float(xgb_p_short))
                    bias_xgb_flat.append(float(xgb_p_flat))
                    bias_entry_long.append(float(entry_p_long))
                    bias_entry_short.append(float(entry_p_short))
                    bias_entry_flat.append(float(entry_p_flat))
                    bias_entry_minus_xgb_long.append(float(entry_p_long) - float(xgb_p_long))
                    bias_entry_minus_xgb_short.append(float(entry_p_short) - float(xgb_p_short))
                    bias_entry_minus_xgb_flat.append(float(entry_p_flat) - float(xgb_p_flat))
                    bias_entry_long_minus_short.append(float(entry_p_long) - float(entry_p_short))

                    xgb_short_margin = float(xgb_p_short) - float(xgb_p_long)
                    if xgb_short_margin >= 0.02:
                        bias["xgb_short_margin_ge_0.02"]["count"] += 1
                    if xgb_short_margin >= 0.05:
                        bias["xgb_short_margin_ge_0.05"]["count"] += 1

                    if session:
                        bias["session_split"][session] = bias["session_split"].get(session, 0) + 1

                    _bias_add_ctx(rec.get("ctx_cont"), rec.get("ctx_cat"))

    # Finalize bias stats
    total_bias = bias["rows"]
    if total_bias > 0:
        bias["xgb_short_margin_ge_0.02"]["rate"] = bias["xgb_short_margin_ge_0.02"]["count"] / total_bias
        bias["xgb_short_margin_ge_0.05"]["rate"] = bias["xgb_short_margin_ge_0.05"]["count"] / total_bias

    bias["entry_long_minus_short_stats"] = _summary_stats(bias_entry_long_minus_short)
    bias["xgb_probs_stats"] = {
        "xgb_p_long": _summary_stats(bias_xgb_long),
        "xgb_p_short": _summary_stats(bias_xgb_short),
        "xgb_p_flat": _summary_stats(bias_xgb_flat),
    }
    bias["entry_probs_stats"] = {
        "entry_p_long": _summary_stats(bias_entry_long),
        "entry_p_short": _summary_stats(bias_entry_short),
        "entry_p_flat": _summary_stats(bias_entry_flat),
    }
    bias["entry_minus_xgb_stats"] = {
        "entry_minus_xgb_long": _summary_stats(bias_entry_minus_xgb_long),
        "entry_minus_xgb_short": _summary_stats(bias_entry_minus_xgb_short),
        "entry_minus_xgb_flat": _summary_stats(bias_entry_minus_xgb_flat),
    }
    if bias["ctx_cont6_summary"]["available"]:
        bias["ctx_cont6_summary"]["per_index"] = {
            str(i): _summary_stats(vals) for i, vals in enumerate(bias_ctx_cont)
        }
    if bias["ctx_cat6_counts"]["available"]:
        bias["ctx_cat6_counts"]["per_index"] = {
            str(i): {str(k): v for k, v in sorted(counts.items(), key=lambda kv: kv[0])}
            for i, counts in bias_ctx_cat_counts.items()
        }

    return out


def _format_score_landscape_block(audit: Dict[str, Any]) -> str:
    lines = []
    lines.append("[SCORE_LANDSCAPE_AUDIT]")
    lines.append(f"rows={audit['rows']} missing_rows={audit['missing_rows']}")
    lines.append("winner_counts.xgb=" + json.dumps(audit["winner_counts"]["xgb"], sort_keys=True))
    lines.append("winner_counts.entry=" + json.dumps(audit["winner_counts"]["entry"], sort_keys=True))
    lines.append("margin_hist.xgb_margin_long_short=" + json.dumps(audit["margin_hist"]["xgb_margin_long_short"], sort_keys=True))
    lines.append("margin_hist.entry_margin_long_short=" + json.dumps(audit["margin_hist"]["entry_margin_long_short"], sort_keys=True))
    lines.append("transitions=" + json.dumps(audit["transitions"], sort_keys=True))
    lines.append("session_splits=" + json.dumps({
        k: {
            "rows": v["rows"],
            "missing_rows": v["missing_rows"],
            "winner_counts": v["winner_counts"],
            "margin_hist": v["margin_hist"],
            "transitions": v["transitions"],
        } for k, v in audit["sessions"].items()
    }, sort_keys=True))
    return "\n".join(lines)


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return float(vals[0])
    idx = (len(vals) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    if lo == hi:
        return float(vals[lo])
    frac = idx - lo
    return float(vals[lo] + (vals[hi] - vals[lo]) * frac)


def _summary_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    total = float(sum(values))
    count = len(values)
    return {
        "count": count,
        "mean": total / count,
        "median": _percentile(values, 0.50),
        "p10": _percentile(values, 0.10),
        "p90": _percentile(values, 0.90),
    }


def _format_entry_long_bias_block(audit: Dict[str, Any]) -> str:
    lines = []
    lines.append("[ENTRY_LONG_BIAS_AUDIT]")
    lines.append(f"rows={audit.get('rows', 0)} missing_rows={audit.get('missing_rows', 0)}")
    lines.append("xgb_short_margin_ge_0.02=" + json.dumps(audit.get("xgb_short_margin_ge_0.02", {}), sort_keys=True))
    lines.append("xgb_short_margin_ge_0.05=" + json.dumps(audit.get("xgb_short_margin_ge_0.05", {}), sort_keys=True))
    lines.append("entry_long_minus_short_stats=" + json.dumps(audit.get("entry_long_minus_short_stats", {}), sort_keys=True))
    lines.append("xgb_probs_stats=" + json.dumps(audit.get("xgb_probs_stats", {}), sort_keys=True))
    lines.append("entry_probs_stats=" + json.dumps(audit.get("entry_probs_stats", {}), sort_keys=True))
    lines.append("entry_minus_xgb_stats=" + json.dumps(audit.get("entry_minus_xgb_stats", {}), sort_keys=True))
    lines.append("session_split=" + json.dumps(audit.get("session_split", {}), sort_keys=True))
    lines.append("ctx_cont6_summary=" + json.dumps(audit.get("ctx_cont6_summary", {}), sort_keys=True))
    lines.append("ctx_cat6_counts=" + json.dumps(audit.get("ctx_cat6_counts", {}), sort_keys=True))
    return "\n".join(lines)


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as e:
        log.warning("Failed to load JSON: %s (%s)", path, e)
        return None


def _find_entry_features_used(run_root: Path) -> Optional[Path]:
    candidates = [
        run_root / "replay" / "chunk_0" / "ENTRY_FEATURES_USED.json",
        run_root / "chunk_0" / "ENTRY_FEATURES_USED.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_entry_features_telemetry(run_root: Path) -> Optional[Path]:
    candidates = [
        run_root / "replay" / "chunk_0" / "ENTRY_FEATURES_TELEMETRY.json",
        run_root / "chunk_0" / "ENTRY_FEATURES_TELEMETRY.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_model_used_capsule(run_root: Path) -> Optional[Path]:
    candidates = [
        run_root / "replay" / "chunk_0" / "MODEL_USED_CAPSULE.json",
        run_root / "chunk_0" / "MODEL_USED_CAPSULE.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _runtime_contract_audit(run_root: Path, eval_logs: List[Path], sample_limit: int = 5) -> Dict[str, Any]:
    entry_features_used_path = _find_entry_features_used(run_root)
    entry_features_used = _load_json_if_exists(entry_features_used_path) if entry_features_used_path else None

    entry_features_telemetry_path = _find_entry_features_telemetry(run_root)
    entry_features_telemetry = _load_json_if_exists(entry_features_telemetry_path) if entry_features_telemetry_path else None

    model_used_capsule_path = _find_model_used_capsule(run_root)
    model_used_capsule = _load_json_if_exists(model_used_capsule_path) if model_used_capsule_path else None

    bundle_meta = None
    bundle_dir = None
    if model_used_capsule and model_used_capsule.get("bundle_dir"):
        bundle_dir = Path(model_used_capsule["bundle_dir"])
        bundle_meta = _load_json_if_exists(bundle_dir / "bundle_metadata.json")

    runtime_signal_order = None
    if entry_features_used:
        runtime_signal_order = entry_features_used.get("xgb_seq_channels", {}).get("names") or None
    if not runtime_signal_order:
        runtime_signal_order = list(SIGNAL_BRIDGE_FIELDS)

    sample_rows = []
    for eval_path in eval_logs:
        with eval_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(sample_rows) >= sample_limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                xgb_p_long = rec.get("xgb_p_long")
                xgb_p_short = rec.get("xgb_p_short")
                xgb_p_flat = rec.get("xgb_p_flat")
                entry_p_long = rec.get("entry_p_long")
                entry_p_short = rec.get("entry_p_short")
                entry_p_flat = rec.get("entry_p_flat")
                if None in (xgb_p_long, xgb_p_short, xgb_p_flat, entry_p_long, entry_p_short, entry_p_flat):
                    continue
                if _winner_from_probs(xgb_p_long, xgb_p_short, xgb_p_flat) != "SHORT":
                    continue
                if _winner_from_probs(entry_p_long, entry_p_short, entry_p_flat) != "LONG":
                    continue

                signal7_vals = {
                    "p_long": rec.get("xgb_p_long"),
                    "p_short": rec.get("xgb_p_short"),
                    "p_flat": rec.get("xgb_p_flat"),
                    "p_hat": rec.get("xgb_p_hat"),
                    "uncertainty_score": rec.get("xgb_uncertainty_score"),
                    "margin_top1_top2": rec.get("xgb_margin_top1_top2"),
                    "entropy": rec.get("xgb_entropy"),
                }
                ordered_vals = [signal7_vals.get(name) for name in runtime_signal_order]

                sample_rows.append({
                    "ts_utc": rec.get("ts_utc"),
                    "session": rec.get("session"),
                    "xgb_winner": "SHORT",
                    "entry_winner": "LONG",
                    "signal7_order": runtime_signal_order,
                    "signal7_values": ordered_vals,
                    "xgb_probs": {"p_long": xgb_p_long, "p_short": xgb_p_short, "p_flat": xgb_p_flat},
                    "entry_probs": {"p_long": entry_p_long, "p_short": entry_p_short, "p_flat": entry_p_flat},
                    "xgb_short_minus_long": float(xgb_p_short) - float(xgb_p_long),
                    "entry_long_minus_short": float(entry_p_long) - float(entry_p_short),
                })
        if len(sample_rows) >= sample_limit:
            break

    transformer_input_sample = None
    normalized_values_available = False
    if entry_features_telemetry:
        transformer_inputs = entry_features_telemetry.get("transformer_inputs") or []
        if transformer_inputs:
            transformer_input_sample = transformer_inputs[0]
        if entry_features_telemetry.get("mask_telemetry"):
            normalized_values_available = True

    feature_names = {}
    if entry_features_used:
        feature_names = {
            "seq_features": entry_features_used.get("seq_features", {}).get("names", []),
            "snap_features": entry_features_used.get("snap_features", {}).get("names", []),
            "xgb_seq_channels": entry_features_used.get("xgb_seq_channels", {}).get("names", []),
            "xgb_snap_channels": entry_features_used.get("xgb_snap_channels", {}).get("names", []),
        }

    contract_match = {
        "signal_bridge_id_match": bool(bundle_meta and bundle_meta.get("signal_bridge_id") == SIGNAL_BRIDGE_ID),
        "ctx_cont_dim_match": bool(bundle_meta and int(bundle_meta.get("ctx_cont_dim", -1)) == len(ORDERED_CTX_CONT_NAMES_EXTENDED)),
        "ctx_cat_dim_match": bool(bundle_meta and int(bundle_meta.get("ctx_cat_dim", -1)) == len(ORDERED_CTX_CAT_NAMES_EXTENDED)),
        "seq_dim_match": bool(bundle_meta and int(bundle_meta.get("seq_input_dim", -1)) == SIGNAL_BRIDGE_SEQ_DIM),
        "snap_dim_match": bool(bundle_meta and int(bundle_meta.get("snap_input_dim", -1)) == SIGNAL_BRIDGE_SNAP_DIM),
        "runtime_signal_order_match": (runtime_signal_order == list(SIGNAL_BRIDGE_FIELDS)),
    }
    contract_match["overall_match"] = all(contract_match.values())

    return {
        "run_root": str(run_root),
        "bundle_dir": str(bundle_dir) if bundle_dir else None,
        "bundle_metadata": {
            "num_classes": bundle_meta.get("num_classes") if bundle_meta else None,
            "class_order": bundle_meta.get("class_order") if bundle_meta else None,
            "signal_bridge_id": bundle_meta.get("signal_bridge_id") if bundle_meta else None,
            "ctx_cont_dim": bundle_meta.get("ctx_cont_dim") if bundle_meta else None,
            "ctx_cat_dim": bundle_meta.get("ctx_cat_dim") if bundle_meta else None,
            "seq_input_dim": bundle_meta.get("seq_input_dim") if bundle_meta else None,
            "snap_input_dim": bundle_meta.get("snap_input_dim") if bundle_meta else None,
        },
        "feature_names": feature_names,
        "signal_bridge_fields": list(SIGNAL_BRIDGE_FIELDS),
        "runtime_signal_order": runtime_signal_order,
        "normalized_values_available": normalized_values_available,
        "transformer_input_sample": transformer_input_sample,
        "sample_limit": sample_limit,
        "samples": sample_rows,
        "contract_match": contract_match,
    }


def _format_runtime_contract_block(audit: Dict[str, Any]) -> str:
    lines = []
    lines.append("[ENTRY_RUNTIME_CONTRACT_AUDIT]")
    lines.append(f"bundle_dir={audit.get('bundle_dir')}")
    lines.append("contract_match=" + json.dumps(audit.get("contract_match", {}), sort_keys=True))
    lines.append("feature_names=" + json.dumps(audit.get("feature_names", {}), sort_keys=True))
    lines.append(f"samples={len(audit.get('samples', []))}")
    return "\n".join(lines)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file safely."""
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Failed to load JSON from {path}: {e}")
        return None


def find_telemetry_files(run_root: Path) -> Dict[str, Path]:
    """Find all telemetry files in run root."""
    files = {}
    
    # Priority 1: Master file in run root
    master = run_root / "ENTRY_FEATURES_USED_MASTER.json"
    if master.exists():
        files["master"] = master
    
    # Priority 2: Chunk files
    chunk_dirs = sorted(run_root.glob("chunk_*"))
    for chunk_dir in chunk_dirs:
        # ENTRY_FEATURES_USED.json
        ef = chunk_dir / "ENTRY_FEATURES_USED.json"
        if ef.exists():
            files[f"chunk_{chunk_dir.name}_entry_features"] = ef
        
        # ENTRY_FEATURES_TELEMETRY.json
        tel = chunk_dir / "ENTRY_FEATURES_TELEMETRY.json"
        if tel.exists():
            files[f"chunk_{chunk_dir.name}_telemetry"] = tel
        
        # chunk_footer.json
        footer = chunk_dir / "chunk_footer.json"
        if footer.exists():
            files[f"chunk_{chunk_dir.name}_footer"] = footer
    
    return files


def load_aggregated_telemetry(run_root: Path) -> Dict[str, Any]:
    """Load and aggregate telemetry from all sources."""
    aggregated = {
        "sources_found": [],
        "transformer_forward_calls": 0,
        "model_attempts": {},
        "model_forwards": {},
        "xgb_pre_predict_count": 0,
        "xgb_post_predict_count": 0,
        "n_xgb_channels_in_transformer_input": 0,
        "xgb_channel_names": [],
        "xgb_seq_channel_names": [],
        "xgb_snap_channel_names": [],
        "xgb_used_as": "none",
        "post_predict_called": False,
        "veto_applied_count": 0,
        "entry_routing": {},
        "seq_feature_names": [],
        "snap_feature_names": [],
        "toggles": {},
    }
    
    files = find_telemetry_files(run_root)
    aggregated["sources_found"] = list(files.keys())
    
    # Load from master if exists
    if "master" in files:
        master_data = load_json_safe(files["master"])
        if master_data:
            _merge_telemetry(aggregated, master_data)
    
    # Load from chunks
    for key, path in files.items():
        if "entry_features" in key and "master" not in key:
            data = load_json_safe(path)
            if data:
                _merge_telemetry(aggregated, data)
    
    return aggregated


def _merge_telemetry(aggregated: Dict[str, Any], data: Dict[str, Any]) -> None:
    """Merge telemetry data into aggregated dict."""
    # Scalar sums
    aggregated["transformer_forward_calls"] += data.get("transformer_forward_calls", 0)
    
    # XGB flow
    xgb_flow = data.get("xgb_flow", {})
    aggregated["xgb_pre_predict_count"] += xgb_flow.get("xgb_pre_predict_count", 0)
    aggregated["xgb_post_predict_count"] += xgb_flow.get("xgb_post_predict_count", 0)
    aggregated["veto_applied_count"] += xgb_flow.get("veto_applied_count", 0)
    
    # Take first non-zero/non-empty values for constants
    if aggregated["n_xgb_channels_in_transformer_input"] == 0:
        aggregated["n_xgb_channels_in_transformer_input"] = xgb_flow.get("n_xgb_channels_in_transformer_input", 0)
    
    if aggregated["xgb_used_as"] == "none":
        aggregated["xgb_used_as"] = xgb_flow.get("xgb_used_as", "none")
    
    if not aggregated["post_predict_called"]:
        aggregated["post_predict_called"] = xgb_flow.get("post_predict_called", False)
    
    # Channel names (take first non-empty)
    if not aggregated["xgb_channel_names"]:
        xgb_seq = data.get("xgb_seq_channels", {})
        xgb_snap = data.get("xgb_snap_channels", {})
        aggregated["xgb_seq_channel_names"] = xgb_seq.get("names", [])
        aggregated["xgb_snap_channel_names"] = xgb_snap.get("names", [])
        aggregated["xgb_channel_names"] = aggregated["xgb_seq_channel_names"] + aggregated["xgb_snap_channel_names"]
    
    # Feature names
    if not aggregated["seq_feature_names"]:
        seq_features = data.get("seq_features", {})
        aggregated["seq_feature_names"] = seq_features.get("names", [])
    
    if not aggregated["snap_feature_names"]:
        snap_features = data.get("snap_features", {})
        aggregated["snap_feature_names"] = snap_features.get("names", [])
    
    # Model attempts/forwards
    model_entry = data.get("model_entry", {})
    for model, count in model_entry.get("model_attempt_calls", {}).items():
        aggregated["model_attempts"][model] = aggregated["model_attempts"].get(model, 0) + count
    for model, count in model_entry.get("model_forward_calls", {}).items():
        aggregated["model_forwards"][model] = aggregated["model_forwards"].get(model, 0) + count
    
    # Entry routing
    entry_routing = data.get("entry_routing_aggregate", {})
    for model, count in entry_routing.get("selected_model_counts", {}).items():
        aggregated["entry_routing"][model] = aggregated["entry_routing"].get(model, 0) + count
    
    # Toggles
    toggles = data.get("toggles", {})
    if toggles:
        aggregated["toggles"].update(toggles)


# ============================================================================
# ANALYSIS
# ============================================================================

@dataclass
class XGBChannelAnalysis:
    """Analysis of a single XGB channel."""
    name: str
    injection_point: str  # "seq", "snap", or "both"
    present_in_telemetry: bool = True
    sample_count: int = 0
    missing_rate: float = 0.0
    constant_rate: float = 0.0
    std: float = 0.0
    mean: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    is_useful: bool = True
    useful_flag: str = ""
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "injection_point": self.injection_point,
            "present_in_telemetry": self.present_in_telemetry,
            "sample_count": self.sample_count,
            "missing_rate": self.missing_rate,
            "constant_rate": self.constant_rate,
            "std": self.std,
            "mean": self.mean,
            "min": self.min_val,
            "max": self.max_val,
            "is_useful": self.is_useful,
            "useful_flag": self.useful_flag,
            "notes": self.notes,
        }


def analyze_xgb_channels(aggregated: Dict[str, Any]) -> List[XGBChannelAnalysis]:
    """Analyze each XGB channel for usefulness."""
    analyses = []
    
    seq_channels = set(aggregated["xgb_seq_channel_names"])
    snap_channels = set(aggregated["xgb_snap_channel_names"])
    all_channels = seq_channels | snap_channels
    
    for channel_name in sorted(all_channels):
        # Determine injection point
        in_seq = channel_name in seq_channels
        in_snap = channel_name in snap_channels
        if in_seq and in_snap:
            injection = "both"
        elif in_seq:
            injection = "seq"
        else:
            injection = "snap"
        
        analysis = XGBChannelAnalysis(
            name=channel_name,
            injection_point=injection,
            sample_count=aggregated["xgb_pre_predict_count"],
        )
        
        # Without detailed per-sample telemetry, we can only mark as "PRESENT"
        # and flag based on whether it's actually used
        if aggregated["xgb_pre_predict_count"] > 0:
            analysis.useful_flag = "PRESENT_IN_PIPELINE"
            analysis.notes.append(f"Used in {aggregated['xgb_pre_predict_count']:,} XGB pre-predict calls")
        else:
            analysis.is_useful = False
            analysis.useful_flag = "NOT_USED"
            analysis.notes.append("XGB pre-predict not called")
        
        analyses.append(analysis)
    
    return analyses


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_snapshot_report(
    run_root: Path,
    aggregated: Dict[str, Any],
    channel_analyses: List[XGBChannelAnalysis],
    output_dir: Optional[Path] = None,
    write_json: bool = True,
    write_md: bool = True,
) -> Dict[str, Any]:
    """Generate the XGB → Transformer snapshot report."""
    
    timestamp = datetime.now().isoformat()
    
    report = {
        "report_type": "XGB_TO_TRANSFORMER_SNAPSHOT",
        "timestamp": timestamp,
        "run_root": str(run_root),
        "sources_found": aggregated["sources_found"],
        
        # A) Pipeline evidence
        "pipeline_evidence": {
            "transformer_forward_calls": aggregated["transformer_forward_calls"],
            "model_attempts": aggregated["model_attempts"],
            "model_forwards": aggregated["model_forwards"],
            "entry_routing": aggregated["entry_routing"],
            "xgb_pre_predict_count": aggregated["xgb_pre_predict_count"],
            "n_xgb_channels_in_transformer_input": aggregated["n_xgb_channels_in_transformer_input"],
            "xgb_channel_names": aggregated["xgb_channel_names"],
            "xgb_used_as": aggregated["xgb_used_as"],
            # POST fields (to be removed in DEL 3)
            "post_predict_called": aggregated["post_predict_called"],
            "xgb_post_predict_count": aggregated["xgb_post_predict_count"],
            "veto_applied_count": aggregated["veto_applied_count"],
        },
        
        # B) Injection points
        "injection_points": {
            "xgb_seq_channel_names": aggregated["xgb_seq_channel_names"],
            "xgb_snap_channel_names": aggregated["xgb_snap_channel_names"],
            "n_seq_xgb_channels": len(aggregated["xgb_seq_channel_names"]),
            "n_snap_xgb_channels": len(aggregated["xgb_snap_channel_names"]),
            "seq_feature_names_sample": aggregated["seq_feature_names"][:10] if aggregated["seq_feature_names"] else [],
            "snap_feature_names_sample": aggregated["snap_feature_names"][:10] if aggregated["snap_feature_names"] else [],
        },
        
        # C) Channel analysis
        "channel_analyses": [a.to_dict() for a in channel_analyses],
        
        # D) Toggles
        "toggles": aggregated["toggles"],
    }
    
    # Determine output directory
    if output_dir is None:
        output_dir = run_root
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write JSON
    if write_json:
        json_path = output_dir / "XGB_TO_TRANSFORMER_SNAPSHOT.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Written JSON: {json_path}")
    
    # Write Markdown
    if write_md:
        md_path = output_dir / "XGB_TO_TRANSFORMER_SNAPSHOT.md"
        md_content = generate_markdown(report, channel_analyses)
        with open(md_path, "w") as f:
            f.write(md_content)
        log.info(f"Written Markdown: {md_path}")
    
    return report


def generate_markdown(report: Dict[str, Any], channel_analyses: List[XGBChannelAnalysis]) -> str:
    """Generate markdown report."""
    pe = report["pipeline_evidence"]
    ip = report["injection_points"]
    
    md = f"""# XGB → Transformer Snapshot Report

**Generated:** {report['timestamp']}  
**Run Root:** `{report['run_root']}`

---

## A) Pipeline Evidence

| Metric | Value |
|--------|-------|
| `transformer_forward_calls` | {pe['transformer_forward_calls']:,} |
| `xgb_pre_predict_count` | {pe['xgb_pre_predict_count']:,} |
| `n_xgb_channels_in_transformer_input` | {pe['n_xgb_channels_in_transformer_input']} |
| `xgb_used_as` | **{pe['xgb_used_as']}** |
| `post_predict_called` | {pe['post_predict_called']} |
| `xgb_post_predict_count` | {pe['xgb_post_predict_count']:,} |
| `veto_applied_count` | {pe['veto_applied_count']} |

### Entry Routing

| Model | Count |
|-------|-------|
"""
    for model, count in pe.get("entry_routing", {}).items():
        md += f"| {model} | {count:,} |\n"
    
    md += f"""
---

## B) Injection Points

XGB channels are injected into Transformer input at two points:

| Point | Count | Channel Names |
|-------|-------|---------------|
| **Sequence (seq)** | {ip['n_seq_xgb_channels']} | {', '.join(ip['xgb_seq_channel_names']) or 'None'} |
| **Snapshot (snap)** | {ip['n_snap_xgb_channels']} | {', '.join(ip['xgb_snap_channel_names']) or 'None'} |

### Full Channel List

"""
    for name in pe.get("xgb_channel_names", []):
        md += f"- `{name}`\n"
    
    md += f"""
---

## C) Channel Analysis (Usefulness)

| Channel | Injection | Sample Count | Status | Notes |
|---------|-----------|--------------|--------|-------|
"""
    for a in channel_analyses:
        notes_str = "; ".join(a.notes[:2]) if a.notes else "-"
        status = "✅ " + a.useful_flag if a.is_useful else "❌ " + a.useful_flag
        md += f"| `{a.name}` | {a.injection_point} | {a.sample_count:,} | {status} | {notes_str} |\n"
    
    md += f"""
---

## D) Summary

| Category | Value |
|----------|-------|
| Total XGB channels in Transformer | **{pe['n_xgb_channels_in_transformer_input']}** |
| XGB channels in seq | {ip['n_seq_xgb_channels']} |
| XGB channels in snap | {ip['n_snap_xgb_channels']} |
| XGB usage mode | **{pe['xgb_used_as']}** |
| XGB pre-predict calls | {pe['xgb_pre_predict_count']:,} |
| XGB post-predict calls | {pe['xgb_post_predict_count']:,} |
| POST active | **{'YES ⚠️' if pe['post_predict_called'] or pe['xgb_post_predict_count'] > 0 else 'NO ✅'}** |

---

## E) Recommendations

"""
    if pe["xgb_post_predict_count"] > 0 or pe["post_predict_called"]:
        md += """### ⚠️ XGB POST is still active

XGB post-predict (calibration/veto) is still being called. This should be removed as per DEL 3.

"""
    else:
        md += """### ✅ XGB POST is inactive

XGB post-predict (calibration/veto) is not being called. Ready for removal.

"""
    
    md += """### XGB → Transformer Channel Status

All XGB channels are currently **PRESENT_IN_PIPELINE**. Detailed per-sample statistics require additional instrumentation.

To get per-channel ablation results, run:
```bash
python3 gx1/scripts/run_xgb_flow_ablation_qsmoke.py --arm test1_channels ...
```

"""
    return md


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate XGB → Transformer Snapshot Report")
    parser.add_argument("run_root", type=Path, help="Path to run root directory")
    parser.add_argument("--top-k", type=int, default=20, help="Top K channels to show")
    parser.add_argument("--write-json", action="store_true", default=True, help="Write JSON report")
    parser.add_argument("--write-md", action="store_true", default=True, help="Write Markdown report")
    parser.add_argument("--require-telemetry", action="store_true", default=True, help="Require telemetry files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: run_root)")
    parser.add_argument("--score-landscape", action="store_true", default=False, help="Generate score landscape audit from eval_log")
    parser.add_argument("--score-landscape-only", action="store_true", default=False, help="Only run score landscape audit (skip telemetry report)")
    
    args = parser.parse_args()
    
    run_root = args.run_root.resolve()
    if not run_root.exists():
        log.error(f"Run root does not exist: {run_root}")
        return 1
    
    output_dir = args.output_dir or run_root

    if not args.score_landscape_only:
        log.info(f"Loading telemetry from: {run_root}")

        # Load aggregated telemetry
        aggregated = load_aggregated_telemetry(run_root)

        if args.require_telemetry and not aggregated["sources_found"]:
            log.error(f"No telemetry files found in {run_root}")
            return 1

        log.info(f"Found {len(aggregated['sources_found'])} telemetry sources")

        # Analyze XGB channels
        channel_analyses = analyze_xgb_channels(aggregated)
        log.info(f"Analyzed {len(channel_analyses)} XGB channels")

        # Generate report
        report = generate_snapshot_report(
            run_root,
            aggregated,
            channel_analyses,
            output_dir=output_dir,
            write_json=args.write_json,
            write_md=args.write_md,
        )

        # Print summary
        print("\n" + "=" * 80)
        print("XGB → TRANSFORMER SNAPSHOT SUMMARY")
        print("=" * 80)
        pe = report["pipeline_evidence"]
        print(f"  transformer_forward_calls:        {pe['transformer_forward_calls']:,}")
        print(f"  xgb_pre_predict_count:            {pe['xgb_pre_predict_count']:,}")
        print(f"  n_xgb_channels_in_transformer:    {pe['n_xgb_channels_in_transformer_input']}")
        print(f"  xgb_used_as:                      {pe['xgb_used_as']}")
        print(f"  xgb_post_predict_count:           {pe['xgb_post_predict_count']:,}")
        print(f"  post_predict_called:              {pe['post_predict_called']}")
        print(f"  veto_applied_count:               {pe['veto_applied_count']}")
        print("")
        print("XGB Channels:")
        for name in pe.get("xgb_channel_names", []):
            print(f"  - {name}")
        print("=" * 80)

    if args.score_landscape or args.score_landscape_only:
        eval_logs = _find_eval_logs(run_root)
        if not eval_logs:
            log.error(f"No eval_log_*.jsonl found under {run_root}")
            return 1
        audit = _score_landscape_audit(eval_logs)
        def _bound(val: float) -> Optional[float]:
            if val == float("inf") or val == float("-inf"):
                return None
            return val

        audit_payload = {
            "run_root": str(run_root),
            "eval_logs": [str(p) for p in eval_logs],
            "bucket_definitions": [{"name": n, "min": _bound(lo), "max": _bound(hi)} for n, lo, hi in _MARGIN_BUCKETS],
            **audit,
        }
        output_path = output_dir / "score_landscape_audit.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(audit_payload, handle, indent=2, sort_keys=True)
        log.info(f"Wrote score landscape audit: {output_path}")
        log.info(_format_score_landscape_block(audit))

        entry_bias = audit.get("entry_long_bias_audit", {})
        entry_bias_payload = {
            "run_root": str(run_root),
            "eval_logs": [str(p) for p in eval_logs],
            **entry_bias,
        }
        entry_bias_path = output_dir / "entry_long_bias_audit.json"
        with entry_bias_path.open("w", encoding="utf-8") as handle:
            json.dump(entry_bias_payload, handle, indent=2, sort_keys=True)
        log.info(f"Wrote entry long bias audit: {entry_bias_path}")
        log.info(_format_entry_long_bias_block(entry_bias))

        runtime_audit = _runtime_contract_audit(run_root, eval_logs, sample_limit=5)
        runtime_audit_path = output_dir / "entry_runtime_contract_audit.json"
        with runtime_audit_path.open("w", encoding="utf-8") as handle:
            json.dump(runtime_audit, handle, indent=2, sort_keys=True)
        log.info(f"Wrote entry runtime contract audit: {runtime_audit_path}")
        log.info(_format_runtime_contract_block(runtime_audit))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
