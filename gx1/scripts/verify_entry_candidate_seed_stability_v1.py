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
from io import StringIO
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import (
    require_pretest_technical_recipe_metadata,
)
from gx1.contracts.entry_model_native_train_launch_v1 import (
    require_training_recipe_source_provenance_metadata,
)
from gx1.contracts.entry_model_native_training_run_lineage_v1 import (
    require_training_run_lineage,
)
from gx1.contracts.immutable_event_authority_v1 import (
    next_immutable_event_created_utc,
    write_immutable_json_event,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_FLAT_INDEX,
    MODEL_DIRECTION_LONG_INDEX,
    MODEL_DIRECTION_SHORT_INDEX,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    sha256_file,
)
from gx1.scripts.evaluate_entry_candidate_selective_edge_v1 import (
    EVALUATION_COVERAGES,
    EVALUATION_MODEL_NAME,
    HAC_OBSERVATION_ORDER,
    PREREGISTERED_SELECTIVE_EDGE_SCHEMA_VERSION,
    RESEARCH_LONG_OUTCOME_COLUMN,
    RESEARCH_SHORT_OUTCOME_COLUMN,
    _preregistered_hypothesis,
    build_metric_rows,
)


SCHEMA_VERSION = "entry_candidate_seed_stability_v1"
EVENT_PREFIX = "ENTRY_CANDIDATE_SEED_STABILITY"
REQUIRED_SEED_COUNT = 5
VAL_EVIDENCE_STAGE = "pre_calibration"
REQUIRED_COVERAGES = EVALUATION_COVERAGES
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


def _regular_file(path_value: str, *, label: str) -> Path:
    path = Path(path_value).expanduser()
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
        or any("latest" in part.lower() for part in path.parts)
    ):
        raise RuntimeError(f"SEED_STABILITY_{label}_PATH_INVALID: {path}")
    return path


def _read_regular_json(path_value: str, *, label: str) -> tuple[Path, dict[str, Any]]:
    path = _regular_file(path_value, label=label)
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


def _validated_seed_recipe(
    metadata: Mapping[str, Any],
    *,
    metadata_path: Path,
    report: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Bind a candidate to its durable pre-TEST recipe, without reopening TEST.

    Source bindings describe the executed checkout, not this consumer's later
    checkout. The launch owner validated their bytes there; here we validate
    the exact recipe bytes and their metadata/lock provenance, retaining every
    source and input binding. Unsupported recipe schemas fail closed at the
    existing pre-TEST metadata owner rather than taking a compatibility lane.
    """

    provenance = require_training_recipe_source_provenance_metadata(
        metadata.get("recipe_source_provenance"), context="SEED_STABILITY"
    )
    lineage = require_training_run_lineage(metadata.get("run_lineage"))
    if lineage["training_profile"] != "candidate":
        raise RuntimeError("SEED_STABILITY_REQUIRES_CANDIDATE_POPULATIONS")
    if (
        metadata_path.name != "bundle_metadata.json"
        or report.get("bundle_dir") != str(metadata_path.parent)
    ):
        raise RuntimeError("SEED_STABILITY_BUNDLE_DIRECTORY_MISMATCH")
    lock_path, lock = _read_regular_json(
        str(metadata_path.parent / "MASTER_TRANSFORMER_LOCK.json"),
        label="BUNDLE_LOCK",
    )
    lock_provenance = require_training_recipe_source_provenance_metadata(
        lock.get("recipe_source_provenance"), context="SEED_STABILITY_LOCK"
    )
    if (
        lock_provenance != provenance
        or lock.get("run_lineage") != lineage
    ):
        raise RuntimeError("SEED_STABILITY_BUNDLE_META_LOCK_SPLIT_BRAIN")
    recipe_path, recipe = _read_regular_json(
        provenance["recipe_audit_path"], label="RECIPE"
    )
    if sha256_file(recipe_path) != provenance["recipe_audit_sha256"]:
        raise RuntimeError("SEED_STABILITY_RECIPE_SHA256_INVALID")
    recipe = require_pretest_technical_recipe_metadata(
        recipe,
        expected_profile="candidate",
        expected_run_id=lineage["training_run_id"],
        expected_dataset_run_id=lineage["dataset_run_id"],
        expected_dataset_dir=report.get("dataset_dir"),
        expected_out_bundle_dir=metadata_path.parent,
    )
    if report.get("dataset_dir") != recipe["dataset_dir"]:
        raise RuntimeError("SEED_STABILITY_RECIPE_DATASET_MISMATCH")
    for key in ("source_commit", "source_bindings", "source_bindings_sha256"):
        if recipe[key] != provenance[key]:
            raise RuntimeError(f"SEED_STABILITY_RECIPE_{key.upper()}_MISMATCH")
    cli = recipe["trainer_cli"]
    metadata_cli_fields = {
        "seed": "seed",
        "execution_tier": "execution_tier",
        "seq_len": "seq_len",
        "dropout": "dropout",
        "batch_size": "batch_size",
        "epochs": "epochs",
        "lr": "learning_rate",
        "early_stopping_patience": "early_stop_patience",
        "early_stopping_min_delta": "early_stop_min_delta",
        "grad_clip_norm": "grad_clip_norm",
        "weight_decay": "weight_decay",
    }
    for field, cli_key in metadata_cli_fields.items():
        if field not in metadata or metadata[field] != cli[cli_key]:
            raise RuntimeError(f"SEED_STABILITY_RECIPE_BUNDLE_CLI_MISMATCH: {field}")
    if (
        lineage["execution_tier"] != cli["execution_tier"]
        or lineage["requested_subsample_rows"] != cli["subsample_rows"]
    ):
        raise RuntimeError("SEED_STABILITY_RECIPE_POPULATION_MISMATCH")
    for section, fields in (
        ("multi_tf", {
            "multi_tf_num_layers": "multi_tf_num_layers",
            "multi_tf_scale": "multi_tf_scale",
            **{f"{timeframe}_seq_len": f"per_tf_seq_len_{timeframe}"
               for timeframe in ("m5", "m15", "h1", "h4", "d1")},
        }),
        ("specialist_fusion", {
            "num_layers": "specialist_num_layers",
            "fusion_scale": "specialist_fusion_scale",
            "cross_family_fusion_scale": "cross_family_fusion_scale",
        }),
    ):
        values = metadata.get(section)
        if not isinstance(values, Mapping) or any(
            field not in values or values[field] != cli[cli_key]
            for field, cli_key in fields.items()
        ):
            raise RuntimeError(f"SEED_STABILITY_RECIPE_BUNDLE_CLI_MISMATCH: {section}")
    for split in ("train", "val"):
        binding = recipe["artifact_bindings"][f"{split}_parquet"]
        if (
            metadata.get(f"{split}_data") != binding["path"]
            or metadata.get(f"{split}_data_sha256") != binding["sha256"]
        ):
            raise RuntimeError(f"SEED_STABILITY_RECIPE_BUNDLE_DATA_MISMATCH: {split}")
    if metadata.get("prefreeze_test_seal_lineage") != recipe["test_guard_lineage"]:
        raise RuntimeError("SEED_STABILITY_RECIPE_BUNDLE_GUARD_MISMATCH")
    return recipe, provenance, {"path": str(lock_path), "sha256": sha256_file(lock_path)}


def _same_recipe_identity(
    metadata: Mapping[str, Any], *, recipe: Mapping[str, Any]
) -> dict[str, Any]:
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
    recipe_semantics = {
        key: value
        for key, value in recipe.items()
        if key not in {"created_utc", "run_id", "out_bundle_dir", "trainer_cli_sha256"}
    }
    recipe_semantics["trainer_cli"] = {
        key: value for key, value in recipe["trainer_cli"].items() if key != "seed"
    }
    identity["validated_recipe_semantics"] = recipe_semantics
    return identity


def _experiment_primary_flags(
    report: Mapping[str, Any], prediction: pd.DataFrame
) -> dict[str, Any]:
    """Recompute the current chronological metrics; never trust stored flags."""

    metrics_path = _regular_file(str(report.get("metrics_path") or ""), label="METRICS")
    metrics_sha = sha256_file(metrics_path)
    if metrics_sha != report.get("metrics_sha256"):
        raise RuntimeError("SEED_STABILITY_METRICS_SHA256_INVALID")
    recomputed_metrics = pd.DataFrame(
        build_metric_rows(prediction, top_fracs=list(EVALUATION_COVERAGES))
    )
    observed = pd.read_csv(metrics_path)
    expected = pd.read_csv(StringIO(recomputed_metrics.to_csv(index=False)))
    if not observed.equals(expected):
        raise RuntimeError("SEED_STABILITY_METRICS_RECOMPUTATION_MISMATCH")
    recomputed = _preregistered_hypothesis(
        recomputed_metrics, evidence_stage=VAL_EVIDENCE_STAGE, val_reference=None
    )
    if report.get("preregistered_selective_edge") != recomputed:
        raise RuntimeError("SEED_STABILITY_PREREGISTRATION_RECOMPUTATION_MISMATCH")
    return {"path": str(metrics_path), "sha256": metrics_sha, "recomputed": recomputed}


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
    if (
        not isinstance(preregistered, Mapping)
        or preregistered.get("schema_version") != PREREGISTERED_SELECTIVE_EDGE_SCHEMA_VERSION
        or preregistered.get("standard_error_observation_order") != HAC_OBSERVATION_ORDER
        or tuple(float(value) for value in preregistered.get("coverage_grid") or ())
        != REQUIRED_COVERAGES
    ):
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
    recipe, provenance, lock = _validated_seed_recipe(
        metadata, metadata_path=metadata_path, report=report
    )
    prediction_path = _regular_file(
        str(report.get("predictions_path") or ""), label="PREDICTION"
    )
    declared = report.get("prediction_evidence")
    if not isinstance(declared, Mapping) or declared.get("path") != str(prediction_path):
        raise RuntimeError("SEED_STABILITY_PREDICTION_BINDING_INVALID")
    observed_sha = sha256_file(prediction_path)
    if observed_sha != declared.get("sha256"):
        raise RuntimeError("SEED_STABILITY_PREDICTION_SHA256_INVALID")
    prediction = pd.read_parquet(prediction_path, columns=[
        "split", "model", "time", "pred_direction", "selection_score",
        "selection_score_mode", "edge_score",
        RESEARCH_LONG_OUTCOME_COLUMN, RESEARCH_SHORT_OUTCOME_COLUMN,
    ])
    if tuple(sorted(str(value) for value in prediction["split"].unique())) != ("val",):
        raise RuntimeError("SEED_STABILITY_PREDICTION_SPLIT_INVALID")
    if tuple(prediction["model"].unique()) != (EVALUATION_MODEL_NAME,):
        raise RuntimeError("SEED_STABILITY_PREDICTION_MODEL_INVALID")
    direction = pd.to_numeric(
        prediction["pred_direction"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(direction).all() or not np.array_equal(direction, np.rint(direction)):
        raise RuntimeError("SEED_STABILITY_PREDICTION_DIRECTION_INVALID")
    regime = _classify_raw_q_regime(direction.astype(np.int64))
    metrics = _experiment_primary_flags(report, prediction)
    return {
        "report": {"path": str(path), "sha256": sha256_file(path)},
        "seed": int(seed),
        "bundle": {"path": str(metadata_path), "sha256": metadata_sha},
        "bundle_lock": lock,
        "training_run_id": recipe["run_id"],
        "recipe_source_provenance": provenance,
        "same_recipe_identity": _same_recipe_identity(metadata, recipe=recipe),
        "prediction": {"path": str(prediction_path), "sha256": observed_sha},
        "metrics": metrics,
        "dataset_dir": str(report.get("dataset_dir") or ""),
        "model_native_signal_contract": report.get("model_native_signal_contract"),
        "dataset_signal_contract": report.get("dataset_signal_contract"),
        "direction_decision_contract": report.get("direction_decision_contract"),
        "coverage_grid": list(preregistered["coverage_grid"]),
        "preregistered_hypothesis_decision": preregistered.get("decision"),
        **regime,
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
    if len({row["training_run_id"] for row in evidence}) != REQUIRED_SEED_COUNT:
        raise RuntimeError("SEED_STABILITY_TRAINING_RUNS_MUST_BE_UNIQUE")
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
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        raise RuntimeError("SEED_STABILITY_OUT_DIR_INVALID")
    out_dir = out_dir.resolve()
    created_utc = next_immutable_event_created_utc(out_dir, EVENT_PREFIX)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created_utc.isoformat(),
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
    _, report = write_immutable_json_event(out_dir, EVENT_PREFIX, report)
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
