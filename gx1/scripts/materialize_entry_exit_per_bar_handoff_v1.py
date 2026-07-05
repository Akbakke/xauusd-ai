#!/usr/bin/env python3
"""Materialize active Entry-bound per-bar Exit handoff substrate.

This is a data-substrate materializer only. It reads the active Entry-IQL
student trade log and canonical M5 bid/ask bars, emits per-bar HOLD/EXIT_NOW
state rows, and keeps Exit training/IQL/shadow/live closed.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gx1.scripts.audit_entry_exit_handoff_readiness_v1 import REQUIRED_EXIT_SUBSTRATE_FIELDS
from gx1.scripts.materialize_entry_specialist_challenger_extension_manifest_v1 import SMART_LAYER_FEATURES
from gx1.scripts.verify_entry_foundation_state_v1 import FOUNDATION_DATASET_DIR, REPORTS_ROOT


DEFAULT_IQL_TRADE_LOG = (
    REPORTS_ROOT / "entry_iql_student_trade_log_20260628_v1/entry_iql_student_trade_log.csv"
)
DEFAULT_IQL_SLICE_AUDIT_JSON = (
    REPORTS_ROOT / "entry_iql_replay_slice_audit_20260628_v1/ENTRY_IQL_REPLAY_SLICE_AUDIT_latest.json"
)
DEFAULT_IQL_COMPARISON_JSON = (
    REPORTS_ROOT / "entry_iql_replay_comparison_20260628_v1/ENTRY_IQL_REPLAY_COMPARISON_latest.json"
)
DEFAULT_M5_PRICE_PARQUET = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/xauusd_m5_CANONICAL_V3_2020_2026.parquet"
)
DEFAULT_SUPPLEMENTAL_M1_GLOB = "/home/andre2/GX1_DATA/reports/v12_live_data/xauusd_m1_*.parquet"
DEFAULT_FOUNDATION_DATASET_DIR = FOUNDATION_DATASET_DIR
DEFAULT_ENTRY_M5_PREBUILT = Path(
    "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet"
)
DEFAULT_ENTRY_MTF_CACHE_DIR = Path("/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/MULTI_TF_V2_CACHE")
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_exit_per_bar_handoff_20260630_v1"
ATR_PERIOD_BARS = 14
ENTRY_ALIGNMENT_SNAP_FEATURES = (
    "chart.foundation_hh_state",
    "chart.foundation_hl_state",
    "chart.foundation_lh_state",
    "chart.foundation_ll_state",
    "chart.foundation_bos_up_age_bars",
    "chart.foundation_bos_down_age_bars",
    "chart.foundation_choch_age_bars",
    "chart.foundation_sweep_low_reclaim_up_proxy",
    "chart.foundation_sweep_high_reclaim_down_proxy",
    "chart.foundation_false_breakout_high_followthrough_down_proxy",
    "chart.foundation_false_breakout_low_followthrough_up_proxy",
    "chart.foundation_compression_state",
    "chart.foundation_expansion_state",
    "chart.foundation_compression_release_trigger",
    "chart.foundation_compression_release_up",
    "chart.foundation_compression_release_down",
    "chart.foundation_impulse_direction",
    "chart.foundation_impulse_age_proxy",
    "chart.foundation_impulse_pullback_alignment",
)
ENTRY_ALIGNMENT_CTX_CONT_FEATURES = (
    "H1_range_compression_ratio",
    "M15_range_compression_ratio",
    "D1_atr_percentile_252",
    "micro_momentum_3",
    "micro_momentum_5",
    "micro_acceleration",
    "retracement_from_last_impulse",
    "_v1h1_ema_diff",
    "_v1h4_ema_diff",
    "d1_ema_slope_20_canon_v2",
    "m15_regime_class_id_v2",
    "h1_regime_class_id_v2",
    "h4_regime_class_id_v2",
    "d1_regime_class_id_v2",
    "m15_trend_age_bars_norm_v2",
    "h1_trend_age_bars_norm_v2",
    "h4_trend_age_bars_norm_v2",
    "d1_trend_age_bars_norm_v2",
    "regime_tf_agreement_v3",
    "smc_bos_pressure_last48",
    "smc_sweep_bull_pressure_last48",
    "sr_support_minus_resistance_prox",
    "liquidity_hi_nearest_abs_atr",
    "liquidity_lo_nearest_abs_atr",
)
ENTRY_SMART_ALIGNMENT_SNAP_FEATURES = tuple(
    feature
    for _label, (_version, features, _builder, _path) in SMART_LAYER_FEATURES.items()
    for feature in features
)
SPECIALIST_GATE_OUTPUT_FIELDS = {
    "structure_swing_encoder": "entry_structure_swing_gate_weight",
    "smc_liquidity_encoder": "entry_smc_liquidity_gate_weight",
    "trend_ema_encoder": "entry_trend_ema_gate_weight",
    "vol_compression_encoder": "entry_vol_compression_gate_weight",
    "momentum_flow_encoder": "entry_momentum_flow_gate_weight",
    "session_regime_encoder": "entry_session_regime_gate_weight",
    "chart_geometry_encoder": "entry_chart_geometry_gate_weight",
    "price_action_candle_encoder": "entry_price_action_candle_gate_weight",
}
FOUNDATION_SPECIALIST_GATE_SET = (
    "structure_swing_encoder",
    "smc_liquidity_encoder",
    "trend_ema_encoder",
    "vol_compression_encoder",
    "momentum_flow_encoder",
    "session_regime_encoder",
)
CHALLENGER_SEQ215_SPECIALIST_GATE_SET = (
    *FOUNDATION_SPECIALIST_GATE_SET,
    "chart_geometry_encoder",
    "price_action_candle_encoder",
)
# smart_seq520_candidate trains the SAME 8 specialists as challenger_seq215
# (the 305 smart-layer features route INTO those specialists; the gate set is
# identical). Exit-parity wave 2026-07-04 (goal-doc step 7): the handoff must
# accept the smart contract so Exit training can consume smart-entry replay
# traces — and in smart mode the smart-layer snap fields become REQUIRED.
SMART_SEQ520_SPECIALIST_GATE_SET = CHALLENGER_SEQ215_SPECIALIST_GATE_SET

PRICE_COLUMNS = (
    "time",
    "open",
    "high",
    "low",
    "close",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
    "atr_bps",
    "spread_bps",
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _check(name: str, condition: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(condition), "details": details or {}}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _normal_path(value: Any) -> str:
    raw = str(value or "").strip()
    return str(Path(raw).expanduser().resolve()) if raw else ""


def _safe_feature_field(prefix: str, name: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    while "__" in clean:
        clean = clean.replace("__", "_")
    return f"{prefix}_{clean.strip('_')}"


ENTRY_SMART_ALIGNMENT_STATE_FEATURES = tuple(
    _safe_feature_field("entry_snap", field) for field in ENTRY_SMART_ALIGNMENT_SNAP_FEATURES
)


def _split_manifest(dataset_dir: Path, split: str) -> dict[str, Any]:
    matches = sorted(dataset_dir.glob(f"*_{split}.manifest.json"))
    # Rule 8 (2026-07-05 sweep): two stems in one dataset dir must be a hard
    # error, never an alphabetical silent pick.
    if len(matches) > 1:
        raise RuntimeError(
            f"[HANDOFF_MANIFEST_AMBIGUOUS] {len(matches)} *_{split}.manifest.json under {dataset_dir}: {matches}"
        )
    return _read_json_or_empty(matches[0]) if matches else {}


def _feature_index_contract(dataset_dir: Path, *, entry_contract_mode: str = "") -> dict[str, Any]:
    manifest = _split_manifest(dataset_dir, "train")
    manifest_matches = sorted(dataset_dir.glob("*_train.manifest.json"))
    # Entry contract mode (exit-parity wave 2026-07-04, goal-doc step 7):
    # derived from the dataset manifest_variant, overridable by explicit arg —
    # a mismatch between the two is a hard error, never a silent pick.
    manifest_mode = str(manifest.get("manifest_variant") or "").strip()
    if entry_contract_mode and manifest_mode and entry_contract_mode != manifest_mode:
        raise RuntimeError(
            f"[ENTRY_EXIT_HANDOFF_CONTRACT_MISMATCH] --entry-contract-mode={entry_contract_mode} "
            f"but dataset manifest_variant={manifest_mode} ({dataset_dir})"
        )
    effective_mode = entry_contract_mode or manifest_mode or "foundation_seq146"
    # Width-agnostic smart detection: the generator writes
    # f"smart_seq{width}_candidate" (materialize_entry_specialist_challenger_
    # extension_manifest_v1.py) — match the family by prefix/suffix so a smart
    # dataset at any width still makes the smart snap fields REQUIRED (never
    # silently demote them to optional on a non-520 width).
    smart_required = effective_mode.startswith("smart_seq") and effective_mode.endswith("_candidate")
    signal_bridge = (
        ((manifest.get("extra") or {}).get("signal_bridge") or {})
        if isinstance(manifest.get("extra"), dict)
        else {}
    )
    ctx_contract = (
        ((manifest.get("extra") or {}).get("ctx_contract") or {})
        if isinstance(manifest.get("extra"), dict)
        else {}
    )
    snap_fields = list(signal_bridge.get("snap_fields") or [])
    ctx_cont_names = list(ctx_contract.get("ctx_cont_names") or [])
    fields: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    optional_fields: list[dict[str, Any]] = []
    optional_missing: list[dict[str, str]] = []
    for source, names, requested, prefix in (
        ("snap", snap_fields, ENTRY_ALIGNMENT_SNAP_FEATURES, "entry_snap"),
        ("ctx_cont", ctx_cont_names, ENTRY_ALIGNMENT_CTX_CONT_FEATURES, "entry_ctx"),
    ):
        index = {name: idx for idx, name in enumerate(names)}
        for name in requested:
            if name not in index:
                missing.append({"source": source, "name": name})
                continue
            fields.append(
                {
                    "source": source,
                    "name": name,
                    "index": int(index[name]),
                    "output_field": _safe_feature_field(prefix, name),
                }
            )
    snap_index = {name: idx for idx, name in enumerate(snap_fields)}
    for name in ENTRY_SMART_ALIGNMENT_SNAP_FEATURES:
        if name not in snap_index:
            # smart mode: smart-layer fields are REQUIRED (goal-doc step 7 —
            # "No Exit training if the Entry reason-for-trade is missing from
            # Exit state"); foundation/seq215: optional as before.
            if smart_required:
                missing.append({"source": "snap", "name": name, "smart_layer": True})
            else:
                optional_missing.append({"source": "snap", "name": name})
            continue
        row = {
            "source": "snap",
            "name": name,
            "index": int(snap_index[name]),
            "output_field": _safe_feature_field("entry_snap", name),
            "optional_smart_layer": not smart_required,
            "smart_layer": True,
        }
        fields.append(row)
        optional_fields.append(row)
    return {
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_matches[0]) if manifest_matches else "",
        "entry_contract_mode": effective_mode,
        "smart_snap_features_required": bool(smart_required),
        "snap_field_count": len(snap_fields),
        "ctx_cont_field_count": len(ctx_cont_names),
        "requested_snap_features": list(ENTRY_ALIGNMENT_SNAP_FEATURES),
        "requested_ctx_cont_features": list(ENTRY_ALIGNMENT_CTX_CONT_FEATURES),
        "optional_smart_snap_features": list(ENTRY_SMART_ALIGNMENT_SNAP_FEATURES),
        "optional_smart_snap_feature_count": len(ENTRY_SMART_ALIGNMENT_SNAP_FEATURES),
        "present_optional_smart_snap_feature_count": len(optional_fields),
        "missing_optional_smart_snap_feature_count": len(optional_missing),
        "missing_optional_smart_snap_features": optional_missing,
        "optional_smart_output_fields": [str(field["output_field"]) for field in optional_fields],
        "fields": fields,
        "missing_requested_features": missing,
        "ready": bool(fields and not missing),
    }


def _split_overlaps(manifest: dict[str, Any], split: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    ranges = manifest.get("ts_min_max_by_split") if isinstance(manifest.get("ts_min_max_by_split"), dict) else {}
    row = ranges.get(split) if isinstance(ranges.get(split), dict) else {}
    raw_min = row.get("min") or row.get("ts_min")
    raw_max = row.get("max") or row.get("ts_max")
    if not raw_min or not raw_max:
        return split == "test"
    ts_min = pd.to_datetime(raw_min, utc=True, errors="coerce")
    ts_max = pd.to_datetime(raw_max, utc=True, errors="coerce")
    if pd.isna(ts_min) or pd.isna(ts_max):
        return split == "test"
    return bool(ts_min <= end and ts_max >= start)


def _foundation_split_parquets(dataset_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    train_manifest = _split_manifest(dataset_dir, "train")
    out: list[Path] = []
    for split in ("train", "val", "test"):
        if not _split_overlaps(train_manifest, split, start, end):
            continue
        out.extend(sorted(dataset_dir.glob(f"*_{split}.parquet")))
    return out


def _value_from_array(value: Any, index: int) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value[index])
    except Exception:
        return float("nan")


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else float("nan")
    except Exception:
        return float("nan")


def _load_entry_alignment_snapshots(
    dataset_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    entry_contract_mode: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    contract = _feature_index_contract(dataset_dir, entry_contract_mode=entry_contract_mode)
    fields = list(contract.get("fields") or [])
    parquets = _foundation_split_parquets(dataset_dir, start, end)
    frames: list[pd.DataFrame] = []
    for path in parquets:
        part = pd.read_parquet(path, columns=["time", "snap", "ctx_cont"])
        part["time"] = pd.to_datetime(part["time"], utc=True, errors="coerce")
        part = part.loc[(part["time"] >= start) & (part["time"] <= end)].copy()
        if part.empty:
            continue
        extracted: dict[str, Any] = {"time": part["time"].to_numpy()}
        for field in fields:
            source = str(field["source"])
            output_field = str(field["output_field"])
            index = int(field["index"])
            extracted[output_field] = part[source].map(lambda value, idx=index: _value_from_array(value, idx)).to_numpy()
        frames.append(pd.DataFrame(extracted))
    snapshots = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["time"])
    if not snapshots.empty:
        snapshots = snapshots.dropna(subset=["time"]).drop_duplicates("time", keep="last")
    diagnostics = {
        "foundation_dataset_dir": str(dataset_dir),
        "contract": contract,
        "source_parquets": [str(path) for path in parquets],
        "snapshot_rows": int(len(snapshots)),
        "output_fields": [str(field["output_field"]) for field in fields],
    }
    return snapshots, diagnostics


def _timestamp_ns_set(values: pd.Series) -> set[int]:
    parsed = pd.to_datetime(values, utc=True, errors="coerce").dropna().drop_duplicates()
    return {int(ts.value) for ts in parsed}


def _materialize_entry_gate_subset_parquet(
    *,
    dataset_dir: Path,
    entry_times: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    out_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    target_ns = _timestamp_ns_set(entry_times)
    out_path = out_dir / "entry_specialist_gate_foundation_subset.parquet"
    parquets = _foundation_split_parquets(dataset_dir, start, end)
    selected_tables: list[pa.Table] = []
    observed_ns: set[int] = set()
    for path in parquets:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=512):
            table = pa.Table.from_batches([batch])
            times = pd.to_datetime(table.column("time").to_pandas(), utc=True, errors="coerce")
            mask = np.array(
                [False if pd.isna(ts) else int(ts.value) in target_ns for ts in times],
                dtype=bool,
            )
            if not bool(mask.any()):
                continue
            selected = table.filter(pa.array(mask))
            selected_tables.append(selected)
            observed_ns.update(int(ts.value) for ts in times.loc[mask].dropna())
    if selected_tables:
        subset = pa.concat_tables(selected_tables)
        pq.write_table(subset, out_path)
        subset_rows = int(subset.num_rows)
    else:
        pq.write_table(pa.table({"time": pa.array([], type=pa.timestamp("ns", tz="UTC"))}), out_path)
        subset_rows = 0
    missing_ns = sorted(target_ns - observed_ns)
    diagnostics = {
        "subset_parquet": str(out_path),
        "source_parquets": [str(path) for path in parquets],
        "requested_entry_time_count": int(len(target_ns)),
        "observed_entry_time_count": int(len(observed_ns)),
        "subset_rows": subset_rows,
        "missing_entry_time_count": int(len(missing_ns)),
        "missing_entry_time_examples": [
            pd.Timestamp(ns, unit="ns", tz="UTC").isoformat() for ns in missing_ns[:20]
        ],
        "ready": bool(target_ns and not missing_ns and subset_rows >= len(target_ns)),
    }
    return out_path, diagnostics


def _specialist_names_from_bundle(bundle: Any) -> list[str]:
    names = list(getattr(bundle.transformer_model, "_specialist_names", []) or [])
    if names:
        return [str(name) for name in names]
    cfg = (bundle.metadata.get("specialist_fusion") or {}) if isinstance(bundle.metadata, dict) else {}
    indices = cfg.get("input_indices") if isinstance(cfg.get("input_indices"), dict) else {}
    return sorted(str(name) for name in indices.keys())


def _supported_specialist_gate_set(names: list[str]) -> bool:
    observed = set(str(name) for name in names)
    return observed in (
        set(FOUNDATION_SPECIALIST_GATE_SET),
        set(CHALLENGER_SEQ215_SPECIALIST_GATE_SET),
        set(SMART_SEQ520_SPECIALIST_GATE_SET),
    )


def _gate_output_fields_for_names(names: list[str]) -> list[str]:
    return [SPECIALIST_GATE_OUTPUT_FIELDS[str(name)] for name in names]


def _extract_entry_specialist_gate_outputs(
    *,
    dataset_dir: Path,
    entry_times: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    out_dir: Path,
    bundle_dir: str,
    m5_prebuilt_path: Path,
    multi_tf_cache_dir: Path,
    batch_size: int,
    required: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_fields = _gate_output_fields_for_names(list(FOUNDATION_SPECIALIST_GATE_SET))
    diagnostics: dict[str, Any] = {
        "required": bool(required),
        "bundle_dir": str(bundle_dir or ""),
        "m5_prebuilt_path": str(m5_prebuilt_path),
        "multi_tf_cache_dir": str(multi_tf_cache_dir),
        "output_fields": output_fields,
        "supported_specialist_sets": {
            "foundation_seq146": list(FOUNDATION_SPECIALIST_GATE_SET),
            "seq215_challenger": list(CHALLENGER_SEQ215_SPECIALIST_GATE_SET),
        },
        "ready": False,
        "failures": [],
    }
    if not required:
        diagnostics["ready"] = True
        diagnostics["output_fields"] = []
        diagnostics["skipped_reason"] = "entry specialist gate extraction not required for this run"
        return pd.DataFrame(columns=["time"]), diagnostics
    raw_bundle_dir = str(bundle_dir or "").strip()
    bundle_path = Path(raw_bundle_dir).expanduser().resolve() if raw_bundle_dir else None
    if bundle_path is None or not bundle_path.is_dir():
        diagnostics["failures"].append(f"candidate bundle dir missing: {raw_bundle_dir or '<empty>'}")
        return pd.DataFrame(columns=["time", *output_fields]), diagnostics
    if not (multi_tf_cache_dir / "manifest.json").is_file():
        diagnostics["failures"].append(f"multi-TF cache manifest missing: {multi_tf_cache_dir / 'manifest.json'}")
        return pd.DataFrame(columns=["time", *output_fields]), diagnostics
    try:
        from torch.utils.data import DataLoader
        import torch

        from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
        from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
            EntryV10CtxDataset,
            _multi_tf_kwargs_from_batch,
        )
        from gx1.scripts.audit_entry_foundation_smoke_bundle_v1 import _bundle_dataset_kwargs

        os.environ["GX1_V10_MULTI_TF_V2_CACHE_DIR"] = str(multi_tf_cache_dir)
        bundle = load_entry_v10_ctx_bundle(bundle_dir=bundle_path, device="cpu", is_replay=True)
        model = bundle.transformer_model.eval()
        specialist_names = _specialist_names_from_bundle(bundle)
        diagnostics["specialist_names"] = specialist_names
        if not _supported_specialist_gate_set(specialist_names):
            diagnostics["failures"].append(
                "specialist set mismatch expected one of "
                f"{[sorted(FOUNDATION_SPECIALIST_GATE_SET), sorted(CHALLENGER_SEQ215_SPECIALIST_GATE_SET)]} "
                f"observed={sorted(specialist_names)}"
            )
            return pd.DataFrame(columns=["time", *output_fields]), diagnostics
        output_fields = _gate_output_fields_for_names(specialist_names)
        diagnostics["output_fields"] = output_fields
        subset_path, subset_diagnostics = _materialize_entry_gate_subset_parquet(
            dataset_dir=dataset_dir,
            entry_times=entry_times,
            start=start,
            end=end,
            out_dir=out_dir,
        )
        diagnostics["subset"] = subset_diagnostics
        if not bool(subset_diagnostics.get("ready")):
            diagnostics["failures"].append("foundation subset missing one or more trade entry_time rows")
            return pd.DataFrame(columns=["time", *output_fields]), diagnostics
        dataset_kwargs = _bundle_dataset_kwargs(bundle.metadata, m5_prebuilt_path)
        dataset = EntryV10CtxDataset(
            parquet_path=subset_path,
            seq_len=int(bundle.transformer_config["seq_len"]),
            allow_constant_labels=True,
            **dataset_kwargs,
        )
        loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
        dataset_times = pd.to_datetime(dataset.df.iloc[dataset.indices]["time"], utc=True, errors="coerce").reset_index(drop=True)
        rows: list[dict[str, Any]] = []
        gates: list[np.ndarray] = []
        offset = 0
        with torch.no_grad():
            for batch in loader:
                seq_x = batch["seq_x"].to(bundle.device)
                snap_x = batch["snap_x"].to(bundle.device)
                ctx_cat = batch["ctx_cat"].to(bundle.device)
                ctx_cont = batch["ctx_cont"].to(bundle.device)
                out = model(
                    seq_x,
                    snap_x,
                    ctx_cat=ctx_cat,
                    ctx_cont=ctx_cont,
                    **_multi_tf_kwargs_from_batch(batch, bundle.device),
                )
                gate_tensor = out.get("specialist_gate")
                if gate_tensor is None:
                    diagnostics["failures"].append("model emitted no specialist_gate")
                    return pd.DataFrame(columns=["time", *output_fields]), diagnostics
                gate = gate_tensor.detach().cpu().float().numpy()
                gates.append(gate)
                for local_idx in range(gate.shape[0]):
                    row: dict[str, Any] = {"time": dataset_times.iloc[offset + local_idx]}
                    for gate_idx, name in enumerate(specialist_names):
                        row[SPECIALIST_GATE_OUTPUT_FIELDS[name]] = float(gate[local_idx, gate_idx])
                    rows.append(row)
                offset += int(gate.shape[0])
        gate_matrix = np.concatenate(gates, axis=0) if gates else np.zeros((0, len(specialist_names)), dtype=np.float32)
        gate_frame = pd.DataFrame(rows).drop_duplicates("time", keep="last")
        finite = bool(np.isfinite(gate_matrix).all()) if gate_matrix.size else False
        row_sum_error = (
            float(np.max(np.abs(gate_matrix.sum(axis=1) - 1.0)))
            if gate_matrix.size
            else None
        )
        diagnostics.update(
            {
                "rows": int(len(gate_frame)),
                "finite": finite,
                "row_sum_max_abs_error": row_sum_error,
                "mean_weight": {
                    str(name): float(gate_matrix[:, idx].mean()) if gate_matrix.size else None
                    for idx, name in enumerate(specialist_names)
                },
            }
        )
        if not finite:
            diagnostics["failures"].append("specialist_gate has non-finite values")
        if row_sum_error is None or row_sum_error > 1e-4:
            diagnostics["failures"].append(f"specialist_gate row sums drift: {row_sum_error}")
        if len(gate_frame) < len(_timestamp_ns_set(entry_times)):
            diagnostics["failures"].append("specialist_gate rows do not cover all requested entry times")
        diagnostics["ready"] = not diagnostics["failures"]
        return gate_frame, diagnostics
    except Exception as exc:
        diagnostics["failures"].append(f"{type(exc).__name__}: {exc}")
        return pd.DataFrame(columns=["time", *output_fields]), diagnostics


def _candidate_bundle_dir(comparison: dict[str, Any]) -> str:
    for source in (
        comparison.get("evidence_identity"),
        (comparison.get("comparison") or {}).get("evidence_identity") if isinstance(comparison.get("comparison"), dict) else {},
    ):
        if isinstance(source, dict):
            raw = source.get("candidate_bundle_dir") or source.get("replay_identity_candidate_bundle_dir")
            if raw:
                return str(raw)
    return ""


def _aggregate_m1_to_m5(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    required = [
        col
        for col in (
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "bid_open",
            "bid_high",
            "bid_low",
            "bid_close",
            "ask_open",
            "ask_high",
            "ask_low",
            "ask_close",
        )
        if col in available
    ]
    if "time" not in required:
        return pd.DataFrame()
    frame = pd.read_parquet(path, columns=required)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame[(frame["time"] >= start) & (frame["time"] <= end)].copy()
    if frame.empty:
        return frame
    frame["time"] = frame["time"].dt.floor("5min")
    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "bid_open": "first",
        "bid_high": "max",
        "bid_low": "min",
        "bid_close": "last",
        "ask_open": "first",
        "ask_high": "max",
        "ask_low": "min",
        "ask_close": "last",
    }
    agg = {key: value for key, value in agg.items() if key in frame.columns}
    out = frame.sort_values("time").groupby("time", as_index=False).agg(agg)
    if {"ask_close", "bid_close", "close"}.issubset(out.columns):
        out["spread_bps"] = ((out["ask_close"] - out["bid_close"]) / out["close"].clip(lower=1e-9)) * 10000.0
    out["atr_bps"] = np.nan
    return out


def _mid_or_fallback(frame: pd.DataFrame, bid_col: str, ask_col: str, fallback_col: str) -> pd.Series:
    values = pd.Series(np.nan, index=frame.index, dtype="float64")
    if bid_col in frame.columns and ask_col in frame.columns:
        bid = pd.to_numeric(frame[bid_col], errors="coerce")
        ask = pd.to_numeric(frame[ask_col], errors="coerce")
        values = (bid + ask) / 2.0
    if fallback_col in frame.columns:
        fallback = pd.to_numeric(frame[fallback_col], errors="coerce")
        values = values.where(values.notna(), fallback)
    return values


def _deterministic_atr_bps(frame: pd.DataFrame, *, period_bars: int = ATR_PERIOD_BARS) -> pd.Series:
    """Train==serve (2026-07-05 sweep fix): fill missing atr_bps with the SAME
    family the live serve produces — one-truth htf_features._atr (min_periods=n)
    on plain OHLC, atr/close*1e4 (v12_ctx_augment_live._add_spread_atr_bps).
    The prior local formula (mid-of-bid/ask TR, min_periods=1) produced an ATR
    the serve path never emits. Warmup rows (<n bars) stay NaN and are counted
    in the manifest, never silently substituted."""
    from gx1.features.htf_features import _atr as _one_truth_atr

    def _plain(col: str, bid: str, ask: str) -> pd.Series:
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce")
        return _mid_or_fallback(frame, bid, ask, col)

    high = _plain("high", "bid_high", "ask_high")
    low = _plain("low", "bid_low", "ask_low")
    close = _plain("close", "bid_close", "ask_close")
    atr = _one_truth_atr(high, low, close, int(period_bars))
    close_abs = close.abs().replace(0.0, np.nan)
    atr_bps = (atr / close_abs) * 10000.0
    return atr_bps.where(np.isfinite(atr_bps) & (atr_bps >= 0.0), np.nan)


def _glob_paths(pattern: str) -> list[Path]:
    if not pattern:
        return []
    return [Path(raw).expanduser().resolve() for raw in sorted(glob.glob(str(Path(pattern).expanduser())))]


def _load_prices(
    path: Path,
    supplemental_m1_glob: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    bar_grid: str = "m5",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if bar_grid == "m1" and path.is_dir():
        # M1 grid (exit-parity wave 2026-07-04): read the partitioned canonical
        # M1 bid/ask tape raw — never aggregate; exit is M1-native by rule.
        # Glob only *.parquet: the dataset dir also holds MANIFEST.json, which
        # pd.read_parquet(dir) would try (and fail) to read as parquet.
        _parts = sorted(path.glob("year=*/**/*.parquet")) or sorted(path.glob("**/*.parquet"))
        if not _parts:
            raise RuntimeError(f"[M1_TAPE_EMPTY] no parquet parts under {path}")
        prices = pd.concat([pd.read_parquet(_pp) for _pp in _parts], ignore_index=True)
        prices["time"] = pd.to_datetime(prices["time"], utc=True)
        prices = prices[(prices["time"] >= start) & (prices["time"] <= end)].copy()
        if "spread_bps" not in prices.columns and {"ask_close", "bid_close"}.issubset(prices.columns):
            mid = (pd.to_numeric(prices["ask_close"], errors="coerce") + pd.to_numeric(prices["bid_close"], errors="coerce")) / 2.0
            prices["spread_bps"] = (
                (pd.to_numeric(prices["ask_close"], errors="coerce") - pd.to_numeric(prices["bid_close"], errors="coerce")) / mid * 10000.0
            )
        prices["price_source"] = "canonical_m1"
        prices["price_source_path"] = str(path)
        available = set(prices.columns)
    else:
        available = set(pq.ParquetFile(path).schema_arrow.names)
        cols = [col for col in PRICE_COLUMNS if col in available]
        if not cols:
            cols = list(PRICE_COLUMNS)
        prices = pd.read_parquet(path, columns=cols)
        prices["time"] = pd.to_datetime(prices["time"], utc=True)
        prices = prices[(prices["time"] >= start) & (prices["time"] <= end)].copy()
        prices["price_source"] = "canonical_m5"
        prices["price_source_path"] = str(path)
    supplemental_frames: list[pd.DataFrame] = []
    supplemental_paths = _glob_paths(supplemental_m1_glob)
    for supplemental_path in supplemental_paths:
        frame = _aggregate_m1_to_m5(supplemental_path, start, end)
        if not frame.empty:
            frame["price_source"] = "supplemental_live_m1_to_m5"
            frame["price_source_path"] = str(supplemental_path)
            supplemental_frames.append(frame)
    if supplemental_frames:
        prices = pd.concat([prices, *supplemental_frames], ignore_index=True, sort=False)
        prices["_price_source_priority"] = np.where(prices["price_source"] == "canonical_m5", 0, 1)
        prices = (
            prices.sort_values(["time", "_price_source_priority"])
            .drop_duplicates("time", keep="first")
            .drop(columns=["_price_source_priority"])
        )
    prices = prices.sort_values("time").reset_index(drop=True)
    atr_source_column_present = "atr_bps" in available
    if "atr_bps" not in prices.columns:
        prices["atr_bps"] = np.nan
    prices["atr_bps"] = pd.to_numeric(prices["atr_bps"], errors="coerce")
    atr_missing_before = int(prices["atr_bps"].isna().sum())
    deterministic_atr = _deterministic_atr_bps(prices)
    fill_mask = prices["atr_bps"].isna() & deterministic_atr.notna()
    if bool(fill_mask.any()):
        prices.loc[fill_mask, "atr_bps"] = deterministic_atr.loc[fill_mask]
    atr_missing_after = int(prices["atr_bps"].isna().sum())
    supplemental_rows = (
        prices[prices["price_source"] == "supplemental_live_m1_to_m5"]
        if "price_source" in prices
        else pd.DataFrame()
    )
    supplemental_used_paths = sorted(str(path) for path in supplemental_rows.get("price_source_path", pd.Series(dtype=str)).dropna().unique())
    supplemental_input_sha256 = {used_path: _sha256_file(Path(used_path)) for used_path in supplemental_used_paths}
    diagnostics = {
        "canonical_rows": int((prices["price_source"] == "canonical_m5").sum()) if "price_source" in prices else int(len(prices)),
        "supplemental_rows_used": int((prices["price_source"] == "supplemental_live_m1_to_m5").sum()) if "price_source" in prices else 0,
        "supplemental_paths_considered": [str(candidate) for candidate in supplemental_paths],
        "supplemental_paths_used": supplemental_used_paths,
        "supplemental_input_sha256": supplemental_input_sha256,
        "atr_bps_fill_method": "preserve_source_else_mid_bid_ask_true_range_rolling_14_closed_m5_bars",
        "atr_bps_period_bars": ATR_PERIOD_BARS,
        "atr_bps_source_column_present": bool(atr_source_column_present),
        "atr_bps_null_rows_before_fill": atr_missing_before,
        "atr_bps_filled_rows": int(fill_mask.sum()),
        "atr_bps_null_rows_after_fill": atr_missing_after,
    }
    return prices, diagnostics


def _side_entry_price(trade: pd.Series) -> float:
    return float(trade["entry_price"])


def _side_exit_close(row: pd.Series, side: str) -> float:
    return float(row["bid_close"] if side == "LONG" else row["ask_close"])


def _pnl_bps(price: float, entry: float, side: str) -> float:
    if side == "LONG":
        return ((price - entry) / entry) * 10000.0
    return ((entry - price) / entry) * 10000.0


def _build_trade_rows(
    *,
    trade_idx: int,
    trade: pd.Series,
    bars: pd.DataFrame,
    candidate_bundle_dir: str,
    replay_identity_hash: str,
    entry_alignment_fields: tuple[str, ...],
    bar_grid_minutes: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    side = str(trade["side"]).upper()
    entry = _side_entry_price(trade)
    # held_bars is denominated in M5 bars (the trade-log sim tape); rescale to
    # the requested bar grid (m1 -> 5x bars per trade).
    expected = int(round(int(trade.get("held_bars") or 0) * 5 / max(1, int(bar_grid_minutes)))) + 1
    entry_trade_id = f"{trade.get('policy_id')}:{trade_idx}:{trade.get('entry_time')}:{side}"
    rows: list[dict[str, Any]] = []
    missing = max(0, expected - len(bars))
    bar_times = pd.to_datetime(bars["time"], utc=True, errors="coerce") if "time" in bars.columns else pd.Series(dtype="datetime64[ns, UTC]")
    invalid_bar_timestamp_count = int(bar_times.isna().sum()) if len(bar_times) else 0
    bar_time_diffs = bar_times.diff().dropna() if len(bar_times) else pd.Series(dtype="timedelta64[ns]")
    non_contiguous_5min_gap_count = int((bar_time_diffs != pd.Timedelta(minutes=int(bar_grid_minutes))).sum())
    first_bar_matches_entry = bool(len(bar_times) and bar_times.iloc[0] == pd.Timestamp(trade["entry_time"]))
    last_bar_matches_exit = bool(len(bar_times) and bar_times.iloc[-1] == pd.Timestamp(trade["exit_time"]))
    if side == "LONG":
        favorable = pd.to_numeric(bars["bid_high"], errors="coerce").cummax()
        adverse = pd.to_numeric(bars["bid_low"], errors="coerce").cummin()
        running_mfe = ((favorable - entry) / entry) * 10000.0
        running_mae = ((entry - adverse) / entry) * 10000.0
    else:
        favorable = pd.to_numeric(bars["ask_low"], errors="coerce").cummin()
        adverse = pd.to_numeric(bars["ask_high"], errors="coerce").cummax()
        running_mfe = ((entry - favorable) / entry) * 10000.0
        running_mae = ((adverse - entry) / entry) * 10000.0

    for bar_index, (_, bar) in enumerate(bars.iterrows()):
        exit_close = _side_exit_close(bar, side)
        running_pnl = _pnl_bps(exit_close, entry, side)
        mfe = float(max(0.0, running_mfe.iloc[bar_index]))
        mae = float(max(0.0, running_mae.iloc[bar_index]))
        row = {
            "entry_trade_id": entry_trade_id,
            "bar_ts": bar["time"].isoformat(),
            "bar_index": int(bar_index),
            "side": side,
            "action_set": "HOLD,EXIT_NOW",
            "running_pnl_bps": float(running_pnl),
            "running_mfe_bps": mfe,
            "running_mae_bps": mae,
            "running_giveback_bps": float(max(0.0, mfe - running_pnl)),
            "bars_held": int(bar_index),
            "session": str(trade.get("session") or trade.get("state_session") or "UNKNOWN").upper(),
            "vol_regime": str(trade.get("vol_regime") or trade.get("state_vol_regime") or "UNKNOWN"),
            "spread_bps": float(bar.get("spread_bps", np.nan))
            if pd.notna(bar.get("spread_bps", np.nan))
            else float(((bar["ask_close"] - bar["bid_close"]) / max(1e-9, bar["close"])) * 10000.0),
            "atr_bps": float(bar.get("atr_bps", np.nan)) if pd.notna(bar.get("atr_bps", np.nan)) else np.nan,
            "bar_price_source": str(bar.get("price_source") or ""),
            "bar_price_source_path": str(bar.get("price_source_path") or ""),
            "entry_score": float(trade.get("score", np.nan)),
            "entry_p_long": float(trade.get("p_long", np.nan)),
            "entry_p_short": float(trade.get("p_short", np.nan)),
            "entry_p_flat": float(trade.get("p_flat", np.nan)),
            "entry_path_quality_pred": float(trade.get("path_quality_pred", np.nan)),
            "entry_bad_path_prob": float(trade.get("bad_path_prob", np.nan)),
            "entry_candidate_bundle_dir": candidate_bundle_dir,
            "entry_iql_policy_id": str(trade.get("policy_id") or ""),
            "entry_replay_identity_hash": replay_identity_hash,
            "entry_time": pd.Timestamp(trade["entry_time"]).isoformat(),
            "exit_time": pd.Timestamp(trade["exit_time"]).isoformat(),
            "realized_net_pnl_bps": float(trade.get("net_pnl_bps", np.nan)),
            "realized_gross_pnl_bps": float(trade.get("gross_pnl_bps", np.nan)),
            "realized_mfe_bps": float(trade.get("mfe_bps", np.nan)),
            "realized_mae_bps": float(trade.get("mae_bps", np.nan)),
            "realized_exit_reason": str(trade.get("exit_reason") or ""),
            "is_realized_exit_bar": bool(bar_index == max(0, len(bars) - 1)),
        }
        for field in entry_alignment_fields:
            row[field] = _float_or_nan(trade.get(field, np.nan))
        rows.append(row)
    diag = {
        "entry_trade_id": entry_trade_id,
        "expected_bar_count": expected,
        "observed_bar_count": int(len(bars)),
        "missing_bar_count": int(missing),
        "invalid_bar_timestamp_count": invalid_bar_timestamp_count,
        "non_contiguous_5min_gap_count": non_contiguous_5min_gap_count,
        "first_bar_matches_entry": first_bar_matches_entry,
        "last_bar_matches_exit": last_bar_matches_exit,
        "coverage_ready": bool(
            int(missing) == 0
            and invalid_bar_timestamp_count == 0
            and non_contiguous_5min_gap_count == 0
            and first_bar_matches_entry
            and last_bar_matches_exit
        ),
        "entry_time": pd.Timestamp(trade["entry_time"]).isoformat(),
        "exit_time": pd.Timestamp(trade["exit_time"]).isoformat(),
    }
    return rows, diag


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Exit Per-Bar Handoff",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Dataset rows: `{report['dataset_rows']}`",
        f"- Source trade count: `{report['source_trade_count']}`",
        f"- Included trade count: `{report['included_trade_count']}`",
        f"- Excluded trade count: `{report['excluded_trade_count']}`",
        f"- Complete trades: `{report['complete_trade_count']}`",
        f"- Exit training allowed: `{report['exit_training_allowed']}`",
        f"- Exit IQL allowed: `{report['exit_iql_allowed']}`",
        f"- Dataset: `{report['dataset_csv']}`",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        for failure in report["failures"]:
            lines.append(f"- `{failure['check']}`")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    trade_log_path = Path(args.iql_trade_log).expanduser().resolve()
    price_path = Path(args.m5_price_parquet).expanduser().resolve()
    comparison_path = Path(args.iql_comparison_json).expanduser().resolve()
    slice_audit_path = Path(args.iql_slice_audit_json).expanduser().resolve()
    comparison = _read_json_or_empty(comparison_path)
    slice_audit = _read_json_or_empty(slice_audit_path)
    trades = pd.read_csv(trade_log_path)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
    start = trades["entry_time"].min()
    end = trades["exit_time"].max()
    foundation_dataset_dir = (
        Path(getattr(args, "foundation_dataset_dir", DEFAULT_FOUNDATION_DATASET_DIR))
        .expanduser()
        .resolve()
    )
    entry_alignment_snapshots, entry_alignment_diagnostics = _load_entry_alignment_snapshots(
        foundation_dataset_dir,
        start,
        end,
        entry_contract_mode=str(getattr(args, "entry_contract_mode", "") or ""),
    )
    entry_snapshot_alignment_fields = tuple(str(field) for field in entry_alignment_diagnostics.get("output_fields") or [])
    if not entry_alignment_snapshots.empty:
        trades = trades.merge(
            entry_alignment_snapshots.rename(columns={"time": "entry_time"}),
            on="entry_time",
            how="left",
            validate="many_to_one",
        )
    else:
        for field in entry_snapshot_alignment_fields:
            trades[field] = np.nan
    candidate_bundle_dir = _candidate_bundle_dir(comparison)
    entry_specialist_gate_outputs, entry_specialist_gate_diagnostics = _extract_entry_specialist_gate_outputs(
        dataset_dir=foundation_dataset_dir,
        entry_times=trades["entry_time"],
        start=start,
        end=end,
        out_dir=out_dir,
        bundle_dir=candidate_bundle_dir,
        m5_prebuilt_path=Path(getattr(args, "entry_m5_prebuilt_path", DEFAULT_ENTRY_M5_PREBUILT)).expanduser().resolve(),
        multi_tf_cache_dir=Path(getattr(args, "entry_multi_tf_cache_dir", DEFAULT_ENTRY_MTF_CACHE_DIR)).expanduser().resolve(),
        batch_size=int(getattr(args, "entry_specialist_gate_batch_size", 32)),
        required=bool(getattr(args, "require_entry_specialist_gates", True)),
    )
    entry_specialist_gate_fields = tuple(str(field) for field in entry_specialist_gate_diagnostics.get("output_fields") or [])
    if not entry_specialist_gate_outputs.empty:
        trades = trades.merge(
            entry_specialist_gate_outputs.rename(columns={"time": "entry_time"}),
            on="entry_time",
            how="left",
            validate="many_to_one",
        )
    else:
        for field in entry_specialist_gate_fields:
            trades[field] = np.nan
    # Gate-weight carry-forward e2e validation (exit-parity wave 2026-07-04):
    # softmax gate weights must arrive complete per trade and sum to ~1.0 —
    # NaN-filled or partial rows would train Exit on corrupt specialist
    # consensus. Recorded loud; counted rows surface in the report.
    gate_weight_incomplete_rows = 0
    gate_weight_badsum_rows = 0
    if entry_specialist_gate_fields:
        _gw = trades[list(entry_specialist_gate_fields)]
        _has_any = _gw.notna().any(axis=1)
        _has_all = _gw.notna().all(axis=1)
        gate_weight_incomplete_rows = int((_has_any & ~_has_all).sum())
        _sums = _gw[_has_all].sum(axis=1)
        gate_weight_badsum_rows = int(((_sums - 1.0).abs() > 1e-3).sum())
        if gate_weight_incomplete_rows or gate_weight_badsum_rows:
            print(
                f"[ENTRY_EXIT_HANDOFF_GATE_WEIGHTS] incomplete_rows={gate_weight_incomplete_rows} "
                f"badsum_rows={gate_weight_badsum_rows} of {len(trades)} trades",
                flush=True,
            )
    entry_alignment_fields = tuple(
        dict.fromkeys([*entry_snapshot_alignment_fields, *entry_specialist_gate_fields])
    )
    bar_grid = str(getattr(args, "bar_grid", "m5") or "m5")
    bar_grid_minutes = 1 if bar_grid == "m1" else 5
    prices, price_diagnostics = _load_prices(
        price_path, str(args.supplemental_m1_glob), start, end, bar_grid=bar_grid
    )
    replay_identity_hash = hashlib.sha256(
        (
            _sha256_file(trade_log_path)
            + _sha256_file(comparison_path)
            + _sha256_file(slice_audit_path)
        ).encode("utf-8")
    ).hexdigest()

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    excluded_trades: list[dict[str, Any]] = []
    included_source_indices: list[int] = []
    for trade_idx, trade in trades.iterrows():
        mask = (prices["time"] >= trade["entry_time"]) & (prices["time"] <= trade["exit_time"])
        trade_bars = prices.loc[mask].copy()
        trade_rows, diag = _build_trade_rows(
            trade_idx=int(trade_idx),
            trade=trade,
            bars=trade_bars,
            candidate_bundle_dir=candidate_bundle_dir,
            replay_identity_hash=replay_identity_hash,
            entry_alignment_fields=entry_alignment_fields,
            bar_grid_minutes=bar_grid_minutes,
        )
        diag["source_trade_index"] = int(trade_idx)
        diagnostics.append(diag)
        if bool(diag.get("coverage_ready")):
            included_source_indices.append(int(trade_idx))
            rows.extend(trade_rows)
        else:
            reasons: list[str] = []
            if int(diag["missing_bar_count"]) > 0:
                reasons.append("missing_per_bar_price_coverage")
            if int(diag["invalid_bar_timestamp_count"]) > 0:
                reasons.append("invalid_per_bar_price_timestamp")
            if int(diag["non_contiguous_5min_gap_count"]) > 0:
                reasons.append("non_contiguous_5min_per_bar_price_coverage")
            if not bool(diag["first_bar_matches_entry"]):
                reasons.append("first_bar_missing_at_entry_time")
            if not bool(diag["last_bar_matches_exit"]):
                reasons.append("last_bar_missing_at_exit_time")
            excluded_trades.append(
                {
                    **diag,
                    "source_trade_index": int(trade_idx),
                    "side": str(trade.get("side") or ""),
                    "reason": ",".join(reasons) if reasons else "per_bar_price_coverage_not_ready",
                }
            )

    dataset = pd.DataFrame(rows)
    diagnostics_df = pd.DataFrame(diagnostics)
    exclusions_df = pd.DataFrame(excluded_trades)
    missing_fields = [field for field in REQUIRED_EXIT_SUBSTRATE_FIELDS if field not in set(dataset.columns)]
    complete_trade_count = int(diagnostics_df["coverage_ready"].astype(bool).sum()) if not diagnostics_df.empty else 0
    source_trade_count = int(len(trades))
    included_trade_count = complete_trade_count
    excluded_trade_count = int(len(excluded_trades))
    covered_trade_ratio = float(included_trade_count / source_trade_count) if source_trade_count else 0.0
    min_covered_trade_ratio = float(getattr(args, "min_covered_trade_ratio", 0.95))
    min_covered_trades = int(getattr(args, "min_covered_trades", 100))
    included_alignment = (
        trades.loc[included_source_indices, list(entry_alignment_fields)]
        if entry_alignment_fields and included_source_indices
        else pd.DataFrame()
    )
    entry_alignment_missing_field_counts = (
        {field: int(included_alignment[field].isna().sum()) for field in entry_alignment_fields}
        if not included_alignment.empty
        else {field: 0 for field in entry_alignment_fields}
    )
    entry_alignment_missing_included_trade_count = (
        int(included_alignment.isna().any(axis=1).sum())
        if not included_alignment.empty
        else (included_trade_count if not entry_alignment_fields else 0)
    )
    entry_alignment_missing_trade_examples = (
        trades.loc[included_source_indices, ["entry_time", *list(entry_alignment_fields)]]
        .loc[lambda frame: frame[list(entry_alignment_fields)].isna().any(axis=1)]
        .head(20)
        .assign(entry_time=lambda frame: frame["entry_time"].astype(str))
        .to_dict(orient="records")
        if entry_alignment_fields and included_source_indices
        else []
    )
    dataset_csv = out_dir / "entry_exit_per_bar_handoff.csv"
    diagnostics_csv = out_dir / "entry_exit_per_bar_handoff_trade_coverage.csv"
    exclusions_csv = out_dir / "entry_exit_per_bar_handoff_gap_exclusions.csv"
    dataset.to_csv(dataset_csv, index=False)
    diagnostics_df.to_csv(diagnostics_csv, index=False)
    exclusions_df.to_csv(exclusions_csv, index=False)
    checks = [
        _check("IQL comparison ready", str(comparison.get("decision")) == "READY_FOR_PROMOTION_REVIEW_VEDTAK", {"decision": comparison.get("decision")}),
        _check("IQL slice audit PASS", str(slice_audit.get("decision")) == "PASS", {"decision": slice_audit.get("decision")}),
        _check("IQL trade log rows are present", not trades.empty, {"rows": int(len(trades)), "path": str(trade_log_path)}),
        _check("canonical M5 price rows cover requested date range", not prices.empty, {"rows": int(len(prices)), "start": str(start), "end": str(end), "path": str(price_path)}),
        _check("per-bar dataset rows were produced", not dataset.empty, {"rows": int(len(dataset))}),
        _check(
            "included trades have complete per-bar coverage",
            included_trade_count == complete_trade_count and included_trade_count > 0,
            {
                "included_trade_count": included_trade_count,
                "complete_trade_count": complete_trade_count,
                "source_trade_count": source_trade_count,
            },
        ),
        _check(
            "per-bar coverage ratio meets review floor",
            covered_trade_ratio >= min_covered_trade_ratio and included_trade_count >= min_covered_trades,
            {
                "covered_trade_ratio": covered_trade_ratio,
                "min_covered_trade_ratio": min_covered_trade_ratio,
                "included_trade_count": included_trade_count,
                "min_covered_trades": min_covered_trades,
                "excluded_trade_count": excluded_trade_count,
                "missing": excluded_trades[:20],
            },
        ),
        _check(
            "gap exclusions are explicit and excluded from substrate",
            excluded_trade_count == source_trade_count - included_trade_count and exclusions_csv.exists(),
            {
                "excluded_trade_count": excluded_trade_count,
                "exclusions_csv": str(exclusions_csv),
                "excluded_reasons": sorted(set(exclusions_df["reason"].astype(str))) if not exclusions_df.empty else [],
            },
        ),
        _check("per-bar dataset has required Exit substrate fields", not missing_fields, {"missing_fields": missing_fields, "required_fields": list(REQUIRED_EXIT_SUBSTRATE_FIELDS)}),
        _check(
            "Entry foundation snapshot alignment contract is ready",
            bool((entry_alignment_diagnostics.get("contract") or {}).get("ready")),
            entry_alignment_diagnostics.get("contract") or {},
        ),
        _check(
            "Entry specialist gate outputs are extracted from candidate bundle",
            bool(entry_specialist_gate_diagnostics.get("ready")),
            entry_specialist_gate_diagnostics,
        ),
        _check(
            # Exit-parity wave 2026-07-04 (weld finding 3): gate weights must be
            # row-complete and sum to ~1.0 per trade — never NaN-filled silently.
            "Entry specialist gate weights are row-complete and normalized",
            gate_weight_incomplete_rows == 0 and gate_weight_badsum_rows == 0,
            {
                "gate_weight_incomplete_rows": int(gate_weight_incomplete_rows),
                "gate_weight_badsum_rows": int(gate_weight_badsum_rows),
                "trade_rows": int(len(trades)),
            },
        ),
        _check(
            "Entry alignment fields cover included trades",
            bool(entry_alignment_fields)
            and entry_alignment_missing_included_trade_count == 0
            and set(entry_alignment_fields).issubset(set(dataset.columns)),
            {
                "foundation_dataset_dir": str(foundation_dataset_dir),
                "snapshot_rows": entry_alignment_diagnostics.get("snapshot_rows"),
                "entry_alignment_field_count": len(entry_alignment_fields),
                "included_trade_count": included_trade_count,
                "missing_included_trade_count": entry_alignment_missing_included_trade_count,
                "missing_field_counts": entry_alignment_missing_field_counts,
                "missing_trade_examples": entry_alignment_missing_trade_examples,
            },
        ),
        _check("materializer never trains, replays, builds adapters, promotes, shadows, or starts live", True, {"trainer_started": False, "replay_started": False, "adapter_built": False, "exit_training_allowed": False, "exit_iql_allowed": False, "promotion_shadow_live_allowed": False}),
    ]
    failures = [{"check": check["name"], "details": check.get("details") or {}} for check in checks if not check["ok"]]
    ready = not failures
    decision = "PASS_WITH_EXPLICIT_GAP_EXCLUSIONS" if ready and excluded_trade_count else ("PASS" if ready else "FAIL")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"ENTRY_EXIT_PER_BAR_HANDOFF_{timestamp}.json"
    md_path = out_dir / f"ENTRY_EXIT_PER_BAR_HANDOFF_{timestamp}.md"
    report = {
        "schema_version": "entry_exit_per_bar_handoff_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "iql_trade_log": str(trade_log_path),
        "iql_trade_log_sha256": _sha256_file(trade_log_path),
        "m5_price_parquet": str(price_path),
        "m5_price_parquet_sha256": (
            _sha256_file(price_path)
            if price_path.is_file()
            else _sha256_file(price_path / "MANIFEST.json")
            if (price_path / "MANIFEST.json").is_file()
            else ""
        ),
        "bar_grid": bar_grid,
        "supplemental_m1_glob": str(args.supplemental_m1_glob),
        "price_diagnostics": price_diagnostics,
        "entry_alignment_diagnostics": entry_alignment_diagnostics,
        "entry_snapshot_alignment_output_fields": list(entry_snapshot_alignment_fields),
        "entry_specialist_gate_diagnostics": entry_specialist_gate_diagnostics,
        "entry_specialist_gate_output_fields": list(entry_specialist_gate_fields),
        "entry_alignment_output_fields": list(entry_alignment_fields),
        "entry_alignment_snapshot_rows": int(entry_alignment_diagnostics.get("snapshot_rows") or 0),
        "entry_alignment_missing_included_trade_count": entry_alignment_missing_included_trade_count,
        "entry_alignment_missing_field_counts": entry_alignment_missing_field_counts,
        "entry_alignment_missing_trade_examples": entry_alignment_missing_trade_examples,
        "iql_comparison_json": str(comparison_path),
        "iql_slice_audit_json": str(slice_audit_path),
        "candidate_bundle_dir": candidate_bundle_dir,
        "entry_replay_identity_hash": replay_identity_hash,
        "dataset_csv": str(dataset_csv),
        "trade_coverage_csv": str(diagnostics_csv),
        "gap_exclusions_csv": str(exclusions_csv),
        "gap_exclusions_csv_sha256": _sha256_file(exclusions_csv),
        "dataset_rows": int(len(dataset)),
        "trade_count": source_trade_count,
        "source_trade_count": source_trade_count,
        "included_trade_count": included_trade_count,
        "excluded_trade_count": excluded_trade_count,
        "complete_trade_count": complete_trade_count,
        "covered_trade_ratio": covered_trade_ratio,
        "min_covered_trade_ratio": min_covered_trade_ratio,
        "min_covered_trades": min_covered_trades,
        "gap_exclusion_policy": "exclude source trades with missing or non-contiguous per-bar price coverage; never synthesize bars",
        "required_exit_substrate_fields": list(REQUIRED_EXIT_SUBSTRATE_FIELDS),
        "checks": checks,
        "failures": failures,
        "exit_training_allowed": False,
        "exit_iql_allowed": False,
        "trainer_started": False,
        "replay_started": False,
        "adapter_built": False,
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "rerun entry-exit-handoff and audit per-bar reconstruction before any Exit training"
            if ready
            else "repair per-bar handoff coverage before Exit handoff can pass"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_EXIT_PER_BAR_HANDOFF_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_EXIT_PER_BAR_HANDOFF_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "dataset_rows": report["dataset_rows"],
                    "trade_count": report["trade_count"],
                    "complete_trade_count": report["complete_trade_count"],
                    "failures": failures,
                    "json_path": str(json_path),
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
    if args.fail_on_not_ready and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iql-trade-log", default=str(DEFAULT_IQL_TRADE_LOG))
    ap.add_argument("--m5-price-parquet", default=str(DEFAULT_M5_PRICE_PARQUET))
    ap.add_argument(
        "--bar-grid",
        choices=("m5", "m1"),
        default="m5",
        help=(
            "Per-bar walk grid. m5 = prior behaviour (default). m1 = native M1 exit grid: "
            "pass the partitioned canonical M1 bid/ask tape dir as --m5-price-parquet; "
            "held_bars (M5-denominated) is rescaled 5x; contiguity checks run at 1min. "
            "Exit is M1-native by constitution — this UPGRADES the substrate, never coarsens."
        ),
    )
    ap.add_argument("--supplemental-m1-glob", default=DEFAULT_SUPPLEMENTAL_M1_GLOB)
    ap.add_argument("--iql-comparison-json", default=str(DEFAULT_IQL_COMPARISON_JSON))
    ap.add_argument("--iql-slice-audit-json", default=str(DEFAULT_IQL_SLICE_AUDIT_JSON))
    ap.add_argument("--foundation-dataset-dir", default=str(DEFAULT_FOUNDATION_DATASET_DIR))
    ap.add_argument("--entry-m5-prebuilt-path", default=str(DEFAULT_ENTRY_M5_PREBUILT))
    ap.add_argument("--entry-multi-tf-cache-dir", default=str(DEFAULT_ENTRY_MTF_CACHE_DIR))
    ap.add_argument("--entry-specialist-gate-batch-size", type=int, default=32)
    ap.add_argument("--require-entry-specialist-gates", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--entry-contract-mode",
        choices=("", "foundation_seq146", "seq215_challenger", "smart_seq520_candidate"),
        default="",
        help=(
            "Explicit entry contract mode; must match the dataset manifest_variant "
            "(mismatch = hard error). smart_seq520_candidate makes the smart-layer "
            "snap fields REQUIRED in the alignment contract. Empty = derive from "
            "the dataset manifest."
        ),
    )
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--min-covered-trade-ratio", type=float, default=0.95)
    ap.add_argument("--min-covered-trades", type=int, default=100)
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
