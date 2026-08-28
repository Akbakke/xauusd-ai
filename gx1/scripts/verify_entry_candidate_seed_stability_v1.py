#!/usr/bin/env python3
"""Verify the pre-registered five-seed stability gate for one candidate recipe.

This is deliberately an evidence consumer, not a trainer or model selector.
It accepts exactly five explicit VAL selective-edge reports, proves that they
share the same frozen substrate and recipe contract, then classifies the
raw-Q behaviour of each seed.  The output cannot authorize promotion, shadow,
live trading, or a production edge claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_FLAT_INDEX,
    MODEL_DIRECTION_LONG_INDEX,
    MODEL_DIRECTION_SHORT_INDEX,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    atomic_write_text,
    sha256_file,
)


SCHEMA_VERSION = "entry_candidate_seed_stability_v1"
EVENT_PREFIX = "ENTRY_CANDIDATE_SEED_STABILITY"
REQUIRED_SEED_COUNT = 5
VAL_EVIDENCE_STAGE = "pre_calibration"
REQUIRED_COVERAGES = (1.00, 0.50, 0.25, 0.10, 0.05, 0.02, 0.01)
COLLAPSE_RATE = 0.95


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _read_regular_json(path_value: str, *, label: str) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value).expanduser()
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
        or "latest" in path.name.lower()
    ):
        raise RuntimeError(f"SEED_STABILITY_{label}_PATH_INVALID: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"SEED_STABILITY_{label}_JSON_INVALID: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"SEED_STABILITY_{label}_ROOT_INVALID: {path}")
    return path, payload


def _classify_raw_q_regime(pred_direction: np.ndarray) -> dict[str, Any]:
    values = np.asarray(pred_direction, dtype=np.int64)
    if values.ndim != 1 or not len(values):
        raise RuntimeError("SEED_STABILITY_DIRECTION_EMPTY")
    allowed = {
        MODEL_DIRECTION_LONG_INDEX,
        MODEL_DIRECTION_SHORT_INDEX,
        MODEL_DIRECTION_FLAT_INDEX,
    }
    if not set(values).issubset(allowed):
        raise RuntimeError("SEED_STABILITY_DIRECTION_INVALID")
    rates = {
        "long": float(np.mean(values == MODEL_DIRECTION_LONG_INDEX)),
        "short": float(np.mean(values == MODEL_DIRECTION_SHORT_INDEX)),
        "flat": float(np.mean(values == MODEL_DIRECTION_FLAT_INDEX)),
    }
    if rates["flat"] >= COLLAPSE_RATE:
        regime = "flat_drift"
    elif rates["long"] >= COLLAPSE_RATE:
        regime = "long_side_collapse"
    elif rates["short"] >= COLLAPSE_RATE:
        regime = "short_side_collapse"
    else:
        regime = "mixed_raw_q_actions"
    return {"regime": regime, "rates": rates, "rows": int(len(values))}


def _same_recipe_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the seed-invariant train/source identity from one bundle.

    A five-seed experiment may vary only stochastic seed and resulting model
    state.  Comparing just dataset/schema leaves room for a different epoch
    budget, optimizer geometry, cache, source checkout or prefreeze seal to
    masquerade as seed stability.
    """

    required = (
        "git_commit",
        "execution_tier",
        "seq_len",
        "dropout",
        "batch_size",
        "epochs",
        "lr",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "grad_clip_norm",
        "weight_decay",
        "model_architecture_schema_version",
        "model_output_schema_version",
        "model_native_signal_contract",
        "model_native_training_objective",
        "aux_head_target_contract",
        "m1_feature_surface_binding",
        "sequence_source_reconstruction",
        "prefreeze_test_seal_lineage",
        "input_normalization",
        "multi_tf",
        "specialist_fusion",
        "context_specialist_routing",
        "run_lineage",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise RuntimeError(
            f"SEED_STABILITY_BUNDLE_RECIPE_IDENTITY_MISSING: {missing}"
        )
    lineage = metadata["run_lineage"]
    if not isinstance(lineage, Mapping):
        raise RuntimeError("SEED_STABILITY_BUNDLE_RUN_LINEAGE_INVALID")
    # A training run identifier differs by construction between the five
    # independently run seeds. Everything else must be invariant.
    lineage_without_run = {
        str(key): value
        for key, value in lineage.items()
        if str(key) != "training_run_id"
    }
    if not lineage_without_run:
        raise RuntimeError("SEED_STABILITY_BUNDLE_RUN_LINEAGE_INVALID")
    identity = {
        key: metadata[key]
        for key in required
        if key != "run_lineage"
    }
    identity["run_lineage_without_training_run_id"] = lineage_without_run
    return identity


def _seed_report_evidence(path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    if (
        report.get("decision") != "PASS"
        or report.get("failures") != []
        or report.get("evidence_stage") != VAL_EVIDENCE_STAGE
        or report.get("outcome_economics") != "gross_spread_inclusive_research_only"
        or report.get("production_authority_ready") is not False
        or report.get("edge_claim_allowed") is not False
    ):
        raise RuntimeError("SEED_STABILITY_SELECTIVE_REPORT_NOT_CLEAN_RESEARCH_EVIDENCE")
    preregistered = report.get("preregistered_selective_edge")
    if not isinstance(preregistered, Mapping) or tuple(
        float(value) for value in preregistered.get("coverage_grid") or ()
    ) != REQUIRED_COVERAGES:
        raise RuntimeError("SEED_STABILITY_PREREGISTRATION_CONTRACT_INVALID")
    metadata_path, metadata = _read_regular_json(
        str(report.get("bundle_metadata_path") or ""), label="BUNDLE_METADATA"
    )
    metadata_sha = sha256_file(metadata_path)
    if str(report.get("bundle_metadata_sha256") or "").lower() != metadata_sha:
        raise RuntimeError("SEED_STABILITY_BUNDLE_METADATA_SHA256_INVALID")
    seed = metadata.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RuntimeError("SEED_STABILITY_BUNDLE_SEED_INVALID")
    prediction_path = Path(str(report.get("predictions_path") or ""))
    if (
        not prediction_path.is_absolute()
        or prediction_path.is_symlink()
        or not prediction_path.is_file()
        or prediction_path.resolve() != prediction_path
    ):
        raise RuntimeError("SEED_STABILITY_PREDICTION_PATH_INVALID")
    declared = report.get("prediction_evidence")
    if not isinstance(declared, Mapping) or declared.get("path") != str(prediction_path):
        raise RuntimeError("SEED_STABILITY_PREDICTION_BINDING_INVALID")
    observed_sha = sha256_file(prediction_path)
    if observed_sha != declared.get("sha256"):
        raise RuntimeError("SEED_STABILITY_PREDICTION_SHA256_INVALID")
    prediction = pd.read_parquet(prediction_path, columns=["split", "pred_direction"])
    if tuple(sorted(str(value) for value in prediction["split"].unique())) != ("val",):
        raise RuntimeError("SEED_STABILITY_PREDICTION_SPLIT_INVALID")
    direction = pd.to_numeric(
        prediction["pred_direction"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(direction).all() or not np.array_equal(direction, np.rint(direction)):
        raise RuntimeError("SEED_STABILITY_PREDICTION_DIRECTION_INVALID")
    return {
        "report": {"path": str(path), "sha256": sha256_file(path)},
        "seed": int(seed),
        "bundle": {"path": str(metadata_path), "sha256": metadata_sha},
        "same_recipe_identity": _same_recipe_identity(metadata),
        "prediction": {"path": str(prediction_path), "sha256": observed_sha},
        "dataset_dir": str(report.get("dataset_dir") or ""),
        "model_native_signal_contract": report.get("model_native_signal_contract"),
        "dataset_signal_contract": report.get("dataset_signal_contract"),
        "direction_decision_contract": report.get("direction_decision_contract"),
        "coverage_grid": list(preregistered["coverage_grid"]),
        "preregistered_hypothesis_decision": preregistered.get("decision"),
        **_classify_raw_q_regime(direction.astype(np.int64)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_reports = list(args.selective_edge_report or [])
    if len(raw_reports) != REQUIRED_SEED_COUNT:
        raise RuntimeError(
            f"SEED_STABILITY_REQUIRES_EXACTLY_{REQUIRED_SEED_COUNT}_REPORTS"
        )
    paths: list[Path] = []
    reports: list[dict[str, Any]] = []
    for raw in raw_reports:
        path, report = _read_regular_json(raw, label="SELECTIVE_REPORT")
        paths.append(path)
        reports.append(report)
    if len(set(paths)) != REQUIRED_SEED_COUNT:
        raise RuntimeError("SEED_STABILITY_REPORTS_MUST_BE_UNIQUE")
    evidence = [_seed_report_evidence(path, report) for path, report in zip(paths, reports)]
    seeds = [int(row["seed"]) for row in evidence]
    if len(set(seeds)) != REQUIRED_SEED_COUNT:
        raise RuntimeError("SEED_STABILITY_SEEDS_MUST_BE_UNIQUE")
    for key in (
        "dataset_dir",
        "model_native_signal_contract",
        "dataset_signal_contract",
        "direction_decision_contract",
        "coverage_grid",
        "same_recipe_identity",
    ):
        values = [_canonical_sha256(row[key]) for row in evidence]
        if len(set(values)) != 1:
            raise RuntimeError(f"SEED_STABILITY_SHARED_{key.upper()}_MISMATCH")
    regimes = [str(row["regime"]) for row in evidence]
    qualitative_agreement = len(set(regimes)) == 1
    stable_mixed_actions = qualitative_agreement and regimes == ["mixed_raw_q_actions"] * REQUIRED_SEED_COUNT
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if stable_mixed_actions else "FAIL",
        "seed_count": REQUIRED_SEED_COUNT,
        "seeds": sorted(seeds),
        "collapse_rate": COLLAPSE_RATE,
        "qualitative_agreement": qualitative_agreement,
        "stable_mixed_actions": stable_mixed_actions,
        "per_seed": sorted(evidence, key=lambda row: int(row["seed"])),
        "production_authority_ready": False,
        "edge_claim_allowed": False,
        "promotion_shadow_live_allowed": False,
        "failures": ([] if stable_mixed_actions else [
            "five seeds do not all exhibit mixed raw-Q actions under one exact substrate"
        ]),
    }
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not out_dir.is_absolute():
        raise RuntimeError("SEED_STABILITY_OUT_DIR_INVALID")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = out_dir / f"{EVENT_PREFIX}_{timestamp}.json"
    if path.exists():
        raise RuntimeError("SEED_STABILITY_EVENT_EXISTS")
    report["json_path"] = str(path)
    atomic_write_text(path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    if report["decision"] != "PASS":
        raise SystemExit(1)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selective-edge-report", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
