"""Preflight the tabular no-XGB live-shadow wiring.

This script does not start the live runner and cannot place orders. It validates
the manifest-resolved shadow scorer, checks the live-style feature vector path,
and writes an explicit env file for a later shadow-only launch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.audit.entry_transformer_feature_audit import _stack_list_column
from gx1.runtime.entry_tabular_no_xgb_candidate import (
    EntryTabularNoXGBShadow,
    XGB_SIGNAL_FIELD_COUNT,
    build_feature_matrix,
    feature_vector_from_live_inputs,
    json_default,
    score_probabilities,
)
from gx1.scripts.evaluate_entry_selective_edge_v1 import _split_files


ACK = "20260627_ENTRY_NO_XGB_LIVE_SHADOW"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def _live_matrix_from_parquet(parquet_path: Path, feature_names: list[str], max_rows: int) -> tuple[np.ndarray, int]:
    df = pd.read_parquet(parquet_path, columns=["snap", "ctx_cont", "ctx_cat"])
    total_rows = len(df)
    if max_rows > 0 and len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)
    snap = _stack_list_column(df["snap"], np.float32)
    ctx_cont = _stack_list_column(df["ctx_cont"], np.float32)
    ctx_cat = _stack_list_column(df["ctx_cat"], np.int64)
    rows = [
        feature_vector_from_live_inputs(
            snap_x=snap[i],
            ctx_cont=ctx_cont[i],
            ctx_cat=ctx_cat[i],
            expected_feature_names=feature_names,
        )[0]
        for i in range(len(df))
    ]
    x = np.vstack(rows).astype(np.float32, copy=False) if rows else np.zeros((0, len(feature_names)), dtype=np.float32)
    if snap.shape[1] <= XGB_SIGNAL_FIELD_COUNT:
        raise RuntimeError("snap dim too small for no-XGB shadow")
    return x, total_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_manifest = Path(args.candidate_manifest).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold = float(args.score_threshold)
    shadow = EntryTabularNoXGBShadow.load(manifest_path=candidate_manifest, score_threshold=threshold)
    split_file = _split_files(dataset_dir, [args.split])[args.split]
    live_x, total_rows = _live_matrix_from_parquet(split_file, shadow.feature_names, int(args.max_rows))
    research_x, research_names = build_feature_matrix(split_file, expected_feature_names=shadow.feature_names)
    if int(args.max_rows) > 0 and len(research_x) > int(args.max_rows):
        research_x = research_x[-int(args.max_rows):]
    if research_names != shadow.feature_names:
        raise RuntimeError("research feature names differ from manifest")
    if live_x.shape != research_x.shape:
        raise RuntimeError(f"live/research shape mismatch: {live_x.shape} vs {research_x.shape}")
    max_abs_diff = float(np.max(np.abs(live_x.astype(np.float64) - research_x.astype(np.float64)))) if live_x.size else 0.0
    if max_abs_diff != 0.0:
        raise RuntimeError(f"live-style vector path differs from research path: max_abs_diff={max_abs_diff}")

    probs = shadow.model.predict_proba(live_x)
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    side, chosen_prob, score = score_probabilities(probs)
    would_take = score >= threshold
    long_take = would_take & (side == 0)
    short_take = would_take & (side == 1)

    env_path = out_dir / "entry_tabular_no_xgb_live_shadow.env"
    env_path.write_text(
        "\n".join([
            "# Source only for shadow-only observation. Does not promote or pin.",
            f"export GX1_ENTRY_TABULAR_NO_XGB_SHADOW_ACK={ACK}",
            f"export GX1_ENTRY_TABULAR_NO_XGB_SHADOW_MANIFEST={candidate_manifest}",
            f"export GX1_ENTRY_TABULAR_NO_XGB_SHADOW_THRESHOLD={threshold:.17g}",
            "",
        ]),
        encoding="utf-8",
    )
    report = {
        "schema_version": "entry_tabular_no_xgb_live_shadow_wiring_preflight_v1",
        "status": "PASS",
        "decision": "READY_FOR_MANUAL_LIVE_SHADOW_WIRING_NOT_LIVE_PIN",
        "candidate_id": shadow.candidate_id,
        "candidate_manifest": str(candidate_manifest),
        "dataset_dir": str(dataset_dir),
        "split": str(args.split),
        "split_file": str(split_file),
        "rows_total": int(total_rows),
        "rows_checked": int(len(live_x)),
        "n_features": int(live_x.shape[1]),
        "feature_contract_hash": shadow.feature_hash,
        "live_research_max_abs_diff": max_abs_diff,
        "score_threshold": threshold,
        "would_take_rows": int(would_take.sum()),
        "would_take_rate": float(would_take.mean()) if len(would_take) else 0.0,
        "would_take_long_rows": int(long_take.sum()),
        "would_take_short_rows": int(short_take.sum()),
        "mean_score": float(np.mean(score)) if len(score) else None,
        "score_p95": float(np.percentile(score, 95)) if len(score) else None,
        "score_p99": float(np.percentile(score, 99)) if len(score) else None,
        "env_file": str(env_path),
        "live_order_placement": "NOT_STARTED_NOT_ENABLED",
        "next_required_gate": "manual_review_then_start_shadow_only_runner_with_env_file",
    }
    _write_json(out_dir / "live_shadow_wiring_preflight.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, default=json_default))
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-manifest", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--score-threshold", type=float, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-rows", type=int, default=0, help="0 means all rows")
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
