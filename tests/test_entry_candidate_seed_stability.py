from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_aux_targets_v3 import model_native_aux_target_contract_metadata
from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import (
    TRAINER_CLI_KEYS,
    canonical_json_sha256,
    require_pretest_technical_recipe_metadata,
)
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SEQ_LEN
from gx1.contracts.entry_model_native_train_launch_v1 import (
    TRAINING_RECIPE_SOURCE_PROVENANCE_SCHEMA,
    artifact_binding,
)
from gx1.contracts.entry_model_native_training_objective_v1 import training_objective_contract_metadata
from gx1.contracts.entry_model_native_training_run_lineage_v1 import (
    FULL_POPULATION_ALGORITHM,
    build_training_run_lineage,
    population_selection_descriptor,
)
from gx1.contracts import immutable_event_authority_v1 as event_owner
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_FLAT_INDEX,
    MODEL_DIRECTION_LONG_INDEX,
    MODEL_DIRECTION_SELECTION_MODE,
    MODEL_DIRECTION_SHORT_INDEX,
    model_direction_decision_contract_metadata,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    MODEL_ARCHITECTURE_SCHEMA_VERSION,
    MODEL_OUTPUT_SCHEMA_VERSION,
)
from gx1.scripts import evaluate_entry_candidate_selective_edge_v1 as evaluator
from gx1.scripts import verify_entry_candidate_seed_stability_v1 as verifier
from tests.test_entry_candidate_readiness import _signal_contract
from tests.test_entry_model_native_input_normalization import _contract as normalization_fixture
from tests.test_entry_model_native_pretest_technical_recipe import _recipe as recipe_fixture


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh(row: dict, *, refresh_recipe: bool = True, refresh_lock: bool = True) -> None:
    recipe = row["recipe"]
    if refresh_recipe:
        for field in ("trainer_cli", "artifact_bindings", "source_bindings"):
            recipe[f"{field}_sha256"] = canonical_json_sha256(recipe[field])
        _write_json(row["recipe_path"], recipe)
        row["metadata"]["recipe_source_provenance"] = {
            "schema_version": TRAINING_RECIPE_SOURCE_PROVENANCE_SCHEMA,
            "recipe_audit_path": str(row["recipe_path"]),
            "recipe_audit_sha256": _sha(row["recipe_path"]),
            **{key: copy.deepcopy(recipe[key]) for key in (
                "source_commit", "source_bindings", "source_bindings_sha256"
            )},
        }
    if refresh_lock:
        row["lock"] = {
            "version": "entry_v10_ctx_lock_v3",
            "recipe_source_provenance": copy.deepcopy(row["metadata"]["recipe_source_provenance"]),
            "run_lineage": copy.deepcopy(row["metadata"]["run_lineage"]),
        }
    _write_json(row["metadata_path"].parent / "MASTER_TRANSFORMER_LOCK.json", row["lock"])
    _write_json(row["metadata_path"], row["metadata"])
    row["report"]["bundle_metadata_sha256"] = _sha(row["metadata_path"])
    _write_json(row["report_path"], row["report"])


def _refresh_metrics(row: dict) -> None:
    prediction = pd.read_parquet(row["prediction_path"])
    metrics = pd.DataFrame(evaluator.build_metric_rows(
        prediction, top_fracs=list(evaluator.EVALUATION_COVERAGES)
    ))
    metrics_path = Path(row["report"]["metrics_path"])
    metrics.to_csv(metrics_path, index=False)
    row["report"]["metrics_sha256"] = _sha(metrics_path)
    row["report"]["prediction_evidence"]["sha256"] = _sha(row["prediction_path"])
    row["report"]["preregistered_selective_edge"] = evaluator._preregistered_hypothesis(
        metrics, evidence_stage=verifier.VAL_EVIDENCE_STAGE, val_reference=None
    )
    _write_json(row["report_path"], row["report"])


@pytest.fixture
def experiment(tmp_path: Path) -> list[dict]:
    """Schema-valid recipe/lineage mechanics, not trained-model quality evidence."""

    tmp_path = tmp_path.resolve()
    template = recipe_fixture(tmp_path)
    template["profile"] = "candidate"
    template["trainer_cli"].update({
        "execution_tier": "canonical", "epochs": 30, "early_stop_patience": 5,
        "subsample_rows": 0, "train_time_window": None, "seq_len": MODEL_NATIVE_SEQ_LEN,
    })
    template["source_bindings"] = {
        "python:tests/test_entry_candidate_seed_stability.py": artifact_binding(Path(__file__).resolve())
    }
    normalization, _, _ = normalization_fixture()
    train_rows = normalization["lineage"]["entry_train_decision_row_count"]
    direction_pattern = (
        MODEL_DIRECTION_LONG_INDEX, MODEL_DIRECTION_SHORT_INDEX, MODEL_DIRECTION_FLAT_INDEX
    )
    pattern_repeats = evaluator.CIRCULAR_SHIFT_NULL_DRAWS // len(direction_pattern) + 1
    val_rows = len(direction_pattern) * pattern_repeats
    train_selection = population_selection_descriptor(
        population_rows=train_rows, selected_indices=np.arange(train_rows),
        algorithm=FULL_POPULATION_ALGORITHM,
    )
    val_selection = population_selection_descriptor(
        population_rows=val_rows, selected_indices=np.arange(val_rows),
        algorithm=FULL_POPULATION_ALGORITHM,
    )
    signal = _signal_contract()
    rows = []
    for seed in (11, 12, 13, 14, 15):
        recipe = copy.deepcopy(template)
        recipe["created_utc"] = f"2026-09-06T12:00:{seed}+00:00"
        recipe["run_id"] = f"SEED_CANDIDATE_RUN_{seed}"
        recipe["out_bundle_dir"] = str(tmp_path / f"bundle_{seed}")
        recipe["trainer_cli"]["seed"] = seed
        cli = recipe["trainer_cli"]
        metadata_path = Path(recipe["out_bundle_dir"]) / "bundle_metadata.json"
        metadata = {
            "schema_version": "entry_v10_ctx_bundle_metadata_v3",
            "created_at_utc": recipe["created_utc"],
            "git_commit": recipe["source_commit"],
            "state_dict_sha256": hashlib.sha256(f"untrained-fixture-{seed}".encode()).hexdigest(),
            "best_epoch": seed,
            "best_val_loss": float(seed),
            **{key: cli[key] for key in (
                "seed", "execution_tier", "seq_len", "dropout", "batch_size",
                "epochs", "grad_clip_norm", "weight_decay",
            )},
            "lr": cli["learning_rate"],
            "early_stopping_patience": cli["early_stop_patience"],
            "early_stopping_min_delta": cli["early_stop_min_delta"],
            "model_architecture_schema_version": MODEL_ARCHITECTURE_SCHEMA_VERSION,
            "model_output_schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
            "model_native_signal_contract": signal,
            "model_native_training_objective": training_objective_contract_metadata(),
            "aux_head_target_contract": model_native_aux_target_contract_metadata(),
            "m1_feature_surface_binding": {"dataset_run_id": recipe["dataset_run_id"], "sha256": "a" * 64},
            "sequence_source_reconstruction": {
                "schema_version": "entry_model_native_sequence_source_reconstruction_v1",
                "authority": "data_reconstruction_only",
                "candidate": False, "test": False, "promotion": False, "paper": False, "live": False,
                "splits": {split: recipe["artifact_bindings"][f"{split}_sequence_source_reconstruction"]
                           for split in ("train", "val")},
            },
            "prefreeze_test_seal_lineage": copy.deepcopy(recipe["test_guard_lineage"]),
            "input_normalization": copy.deepcopy(normalization),
            "multi_tf": {
                "multi_tf_num_layers": cli["multi_tf_num_layers"], "multi_tf_scale": cli["multi_tf_scale"],
                **{f"{timeframe}_seq_len": cli[f"per_tf_seq_len_{timeframe}"]
                   for timeframe in ("m5", "m15", "h1", "h4", "d1")},
                "shared_cache_manifest_path": recipe["artifact_bindings"]["multi_tf_cache_manifest"]["path"],
                "shared_cache_manifest_sha256": recipe["artifact_bindings"]["multi_tf_cache_manifest"]["sha256"],
            },
            "specialist_fusion": {
                "num_layers": cli["specialist_num_layers"], "fusion_scale": cli["specialist_fusion_scale"],
                "cross_family_fusion_scale": cli["cross_family_fusion_scale"],
                "audit_json": recipe["artifact_bindings"]["specialist_audit"]["path"],
            },
            "context_specialist_routing": {"temporal_alias_policy": {"aliases": []}},
            "run_lineage": build_training_run_lineage(
                training_run_id=recipe["run_id"], dataset_run_id=recipe["dataset_run_id"],
                training_profile="candidate", execution_tier="canonical", requested_subsample_rows=0,
                physical_train_rows=train_rows, train_selection=train_selection,
                physical_val_rows=val_rows, val_selection=val_selection,
            ),
        }
        for split in ("train", "val"):
            binding = recipe["artifact_bindings"][f"{split}_parquet"]
            metadata[f"{split}_data"] = binding["path"]
            metadata[f"{split}_data_sha256"] = binding["sha256"]
        prediction_path = tmp_path / f"predictions_{seed}.parquet"
        pd.DataFrame({
            "split": ["val"] * val_rows, "model": [evaluator.EVALUATION_MODEL_NAME] * val_rows,
            "time": pd.date_range("2026-06-01", periods=val_rows, freq="5min", tz="UTC"),
            "pred_direction": direction_pattern * pattern_repeats,
            "selection_score": np.arange(val_rows, dtype=float),
            "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE, "edge_score": np.ones(val_rows),
            evaluator.RESEARCH_LONG_OUTCOME_COLUMN: np.arange(val_rows, dtype=float),
            evaluator.RESEARCH_SHORT_OUTCOME_COLUMN: -np.arange(val_rows, dtype=float),
        }).to_parquet(prediction_path, index=False)
        report_path = tmp_path / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_20260906T1200{seed}000000Z.json"
        report = {
            "decision": "PASS", "failures": [], "evidence_stage": verifier.VAL_EVIDENCE_STAGE,
            "outcome_economics": evaluator.RESEARCH_OUTCOME_ECONOMICS,
            "production_authority_ready": False, "edge_claim_allowed": False,
            "bundle_dir": str(metadata_path.parent), "bundle_metadata_path": str(metadata_path),
            "predictions_path": str(prediction_path), "prediction_evidence": {"path": str(prediction_path)},
            "metrics_path": str(tmp_path / f"metrics_{seed}.csv"),
            "dataset_dir": recipe["dataset_dir"], "model_native_signal_contract": signal,
            "dataset_signal_contract": {"contract": signal},
            "direction_decision_contract": model_direction_decision_contract_metadata(),
        }
        row = {
            "recipe": recipe, "recipe_path": tmp_path / f"recipe_{seed}.json",
            "metadata": metadata, "metadata_path": metadata_path, "report": report,
            "report_path": report_path, "prediction_path": prediction_path,
        }
        _refresh(row)
        _refresh_metrics(row)
        rows.append(row)
    return rows


def _run(experiment: list[dict], tmp_path: Path) -> dict:
    return verifier.run(SimpleNamespace(
        selective_edge_report=[str(row["report_path"]) for row in experiment],
        out_dir=str(tmp_path.resolve() / "events"), quiet=True,
    ))


def test_seed_regime_classification_rejects_flat_and_side_collapse() -> None:
    for direction, expected in ((2, "flat_drift"), (0, "long_side_collapse"), (1, "short_side_collapse")):
        assert verifier._classify_raw_q_regime(np.array([direction] * 100))["regime"] == expected
    assert verifier._classify_raw_q_regime(np.array([0, 1, 2]))["regime"] == "mixed_raw_q_actions"


def test_five_distinct_valid_seed_recipes_publish_one_valid_event(experiment, tmp_path) -> None:
    for row in experiment:
        require_pretest_technical_recipe_metadata(row["recipe"], expected_profile="candidate")
        prediction_rows = len(pd.read_parquet(row["prediction_path"], columns=["pred_direction"]))
        lineage = row["metadata"]["run_lineage"]
        assert prediction_rows > evaluator.CIRCULAR_SHIFT_NULL_DRAWS
        assert prediction_rows == lineage["physical_val_rows"] == lineage["effective_val_rows"]
        assert lineage["physical_train_rows"] == (
            row["metadata"]["input_normalization"]["lineage"]["entry_train_decision_row_count"]
        )
    assert len({_sha(row["recipe_path"]) for row in experiment}) == 5
    assert len({row["recipe"]["trainer_cli_sha256"] for row in experiment}) == 5
    before = copy.deepcopy(experiment)
    result = _run(experiment, tmp_path)
    assert experiment == before
    assert result["decision"] == "PASS"
    assert result["seeds"] == [11, 12, 13, 14, 15]
    assert len({row["recipe_source_provenance"]["recipe_audit_sha256"] for row in result["per_seed"]}) == 5
    assert len({canonical_json_sha256(row["same_recipe_identity"]) for row in result["per_seed"]}) == 1
    assert event_owner.require_newest_immutable_event(Path(result["json_path"]), verifier.EVENT_PREFIX)
    assert json.loads(Path(result["json_path"]).read_text()) == result
    assert result["production_authority_ready"] is False
    assert result["edge_claim_allowed"] is False


@pytest.mark.parametrize("key", sorted(TRAINER_CLI_KEYS - {"seed"}))
def test_every_nonseed_cli_field_remains_in_semantic_identity(experiment, key) -> None:
    row = experiment[0]
    baseline = verifier._same_recipe_identity(row["metadata"], recipe=row["recipe"])
    changed = copy.deepcopy(row["recipe"])
    changed["trainer_cli"][key] = {"deliberately_different": key}
    assert verifier._same_recipe_identity(row["metadata"], recipe=changed) != baseline


@pytest.mark.parametrize("mutation,expected", [
    ("count", "REQUIRES_EXACTLY_5_REPORTS"), ("report", "REPORTS_MUST_BE_UNIQUE"),
    ("seed", "SEEDS_MUST_BE_UNIQUE"), ("run", "TRAINING_RUNS_MUST_BE_UNIQUE"),
])
def test_seed_experiment_members_must_be_independent(experiment, tmp_path, mutation, expected) -> None:
    if mutation == "count":
        experiment.pop()
    elif mutation == "report":
        experiment[-1] = experiment[0]
    elif mutation == "seed":
        row = experiment[-1]
        row["recipe"]["trainer_cli"]["seed"] = experiment[0]["metadata"]["seed"]
        row["metadata"]["seed"] = row["recipe"]["trainer_cli"]["seed"]
        _refresh(row)
    else:
        row = experiment[-1]
        row["recipe"]["run_id"] = experiment[0]["recipe"]["run_id"]
        row["metadata"]["run_lineage"]["training_run_id"] = row["recipe"]["run_id"]
        _refresh(row)
    with pytest.raises(RuntimeError, match=expected):
        _run(experiment, tmp_path)


@pytest.mark.parametrize("mutation,expected", [
    ("bytes", "PREDICTION_SHA256_INVALID"), ("split", "PREDICTION_SPLIT_INVALID"),
    ("model", "PREDICTION_MODEL_INVALID"), ("fractional", "PREDICTION_DIRECTION_INVALID"),
    ("invalid", "DIRECTION_INVALID"),
])
def test_prediction_identity_remains_fail_closed(experiment, tmp_path, mutation, expected) -> None:
    row = experiment[-1]
    prediction = pd.read_parquet(row["prediction_path"])
    if mutation == "split":
        prediction["split"] = "test"
    elif mutation == "model":
        prediction["model"] = "other"
    else:
        prediction["pred_direction"] = 0.5 if mutation == "fractional" else 5
    prediction.to_parquet(row["prediction_path"], index=False)
    if mutation != "bytes":
        row["report"]["prediction_evidence"]["sha256"] = _sha(row["prediction_path"])
        _write_json(row["report_path"], row["report"])
    with pytest.raises(RuntimeError, match=expected):
        _run(experiment, tmp_path)


@pytest.mark.parametrize("cli_key,metadata_key,value", [
    ("batch_size", "batch_size", 9), ("epochs", "epochs", 31),
    ("learning_rate", "lr", 0.0004), ("weight_decay", "weight_decay", 0.00002),
    ("early_stop_patience", "early_stopping_patience", 6),
    ("minimum_epochs_before_stop", None, 2), ("save_top_k", None, 2),
    ("grad_accum_steps", None, 2),
])
def test_seed_gate_rejects_validly_rehashed_recipe_drift(experiment, tmp_path, cli_key, metadata_key, value) -> None:
    row = experiment[-1]
    row["recipe"]["trainer_cli"][cli_key] = value
    if metadata_key is not None:
        row["metadata"][metadata_key] = value
    _refresh(row)
    with pytest.raises(RuntimeError, match="SHARED_SAME_RECIPE_IDENTITY_MISMATCH"):
        _run(experiment, tmp_path)


@pytest.mark.parametrize("mutation", [
    "source", "source_stat", "source_path", "commit", "artifact", "artifact_path",
    "normalization", "cache", "population", "guard", "signal", "reconstruction",
])
def test_seed_gate_preserves_nonseed_provenance(experiment, tmp_path, mutation) -> None:
    row = experiment[-1]
    recipe = row["recipe"]
    metadata = row["metadata"]
    source = next(iter(recipe["source_bindings"].values()))
    if mutation == "source":
        source["sha256"] = "f" * 64
    elif mutation == "source_stat":
        source["inode"] += 1
    elif mutation == "source_path":
        source["path"] += ".different"
    elif mutation == "commit":
        recipe["source_commit"] = "c" * 40
    elif mutation == "artifact":
        recipe["artifact_bindings"]["target_audit"]["sha256"] = "f" * 64
    elif mutation == "artifact_path":
        recipe["artifact_bindings"]["target_audit"]["path"] += ".different"
    elif mutation == "normalization":
        metadata["input_normalization"]["lineage"]["train_parquet_sha256"] = "f" * 64
    elif mutation == "cache":
        metadata["multi_tf"]["shared_cache_manifest_sha256"] = "f" * 64
    elif mutation == "population":
        metadata["run_lineage"]["population_sampling"]["train"]["selection_sha256"] = "f" * 64
    elif mutation == "guard":
        recipe["test_guard_lineage"]["guard_event"]["sha256"] = "f" * 64
        metadata["prefreeze_test_seal_lineage"] = copy.deepcopy(recipe["test_guard_lineage"])
    elif mutation == "signal":
        metadata["model_native_signal_contract"] = {**metadata["model_native_signal_contract"], "changed": True}
    else:
        metadata["sequence_source_reconstruction"]["splits"]["train"]["sha256"] = "f" * 64
    _refresh(row)
    with pytest.raises(RuntimeError, match="SHARED_SAME_RECIPE_IDENTITY_MISMATCH"):
        _run(experiment, tmp_path)


@pytest.mark.parametrize("mutation,expected", [
    ("recipe_bytes", "RECIPE_SHA256_INVALID"), ("recipe_hash", "RECIPE_SHA256_INVALID"),
    ("cli_hash", "trainer CLI contract hash mismatch"), ("source_hash", "BINDINGS_SHA_INVALID"),
    ("artifact_hash", "artifact binding hash mismatch"), ("seed", "BUNDLE_CLI_MISMATCH"),
    ("lineage", "run ID mismatch"), ("dataset", "dataset dir mismatch"),
    ("data", "RECIPE_BUNDLE_DATA_MISMATCH"),
    ("output", "output dir mismatch"), ("source_binding", "SOURCE_BINDINGS_MISMATCH"),
    ("lock", "META_LOCK_SPLIT_BRAIN"), ("missing_provenance", "PROVENANCE_MISSING"),
    ("missing_recipe", "RECIPE_PATH_INVALID"), ("symlink", "RECIPE_PATH_INVALID"),
    ("unknown_schema", "safety boundary invalid"), ("unknown_cli", "CLI contract keys invalid"),
    ("smoke", "REQUIRES_CANDIDATE_POPULATIONS"),
])
def test_per_seed_provenance_fails_closed(experiment, tmp_path, mutation, expected) -> None:
    row = experiment[-1]
    provenance = row["metadata"]["recipe_source_provenance"]
    if mutation == "recipe_bytes":
        row["recipe_path"].write_text(row["recipe_path"].read_text() + "\n")
    elif mutation == "recipe_hash":
        provenance["recipe_audit_sha256"] = "f" * 64
    elif mutation in {"cli_hash", "artifact_hash"}:
        field = "trainer_cli_sha256" if mutation == "cli_hash" else "artifact_bindings_sha256"
        row["recipe"][field] = "f" * 64
        _write_json(row["recipe_path"], row["recipe"])
        provenance["recipe_audit_sha256"] = _sha(row["recipe_path"])
    elif mutation == "source_hash":
        provenance["source_bindings_sha256"] = "f" * 64
    elif mutation == "seed":
        row["metadata"]["seed"] += 100
    elif mutation == "lineage":
        row["metadata"]["run_lineage"]["training_run_id"] = "UNRELATED_TRAINING_RUN"
    elif mutation == "dataset":
        row["report"]["dataset_dir"] += "_other"
    elif mutation == "data":
        row["metadata"]["train_data_sha256"] = "f" * 64
    elif mutation in {"output", "unknown_schema", "unknown_cli"}:
        if mutation == "output":
            row["recipe"]["out_bundle_dir"] += "_other"
        elif mutation == "unknown_schema":
            row["recipe"]["schema_version"] = "unsupported_recipe"
        else:
            row["recipe"]["trainer_cli"]["unowned_optimizer"] = "other"
        _refresh(row)
    elif mutation == "source_binding":
        next(iter(provenance["source_bindings"].values()))["sha256"] = "f" * 64
        provenance["source_bindings_sha256"] = canonical_json_sha256(provenance["source_bindings"])
    elif mutation == "lock":
        row["lock"]["recipe_source_provenance"]["recipe_audit_sha256"] = "f" * 64
    elif mutation == "missing_provenance":
        row["metadata"].pop("recipe_source_provenance")
    elif mutation == "missing_recipe":
        provenance["recipe_audit_path"] += ".missing"
    elif mutation == "symlink":
        alias = tmp_path.resolve() / "recipe_alias.json"
        alias.symlink_to(row["recipe_path"])
        provenance["recipe_audit_path"] = str(alias)
    else:
        lineage = row["metadata"]["run_lineage"]
        lineage.update(
            training_profile="smoke",
            requested_subsample_rows=max(lineage["physical_train_rows"], lineage["physical_val_rows"]),
        )
    _refresh(row, refresh_recipe=False, refresh_lock=mutation not in {"lock", "missing_provenance"})
    with pytest.raises(RuntimeError, match=expected):
        _run(experiment, tmp_path)


@pytest.mark.parametrize("mutation,expected", [
    ("schema", "PREREGISTRATION_CONTRACT_INVALID"), ("order", "PREREGISTRATION_CONTRACT_INVALID"),
    ("metrics_bytes", "METRICS_SHA256_INVALID"), ("metric_order", "METRICS_RECOMPUTATION_MISMATCH"),
    ("metric_value", "METRICS_RECOMPUTATION_MISMATCH"), ("grid", "METRICS_RECOMPUTATION_MISMATCH"),
    ("flags", "PREREGISTRATION_RECOMPUTATION_MISMATCH"),
])
def test_old_or_forged_selective_evidence_is_rejected(experiment, tmp_path, mutation, expected) -> None:
    row = experiment[-1]
    report = row["report"]
    metrics_path = Path(report["metrics_path"])
    if mutation == "schema":
        report["preregistered_selective_edge"]["schema_version"] = "xau_selective_edge_preregistered_v1"
    elif mutation == "order":
        report["preregistered_selective_edge"]["standard_error_observation_order"] = "score_order"
    elif mutation == "metrics_bytes":
        metrics_path.write_text(metrics_path.read_text() + "\n")
    elif mutation == "flags":
        hypothesis = report["preregistered_selective_edge"]
        hypothesis["decision"] = "FAIL" if hypothesis["decision"] == "PASS" else "PASS"
    else:
        metrics = pd.read_csv(metrics_path)
        if mutation == "metric_order":
            metrics["standard_error_observation_order"] = "score_order"
        elif mutation == "metric_value":
            metrics.loc[0, "advantage_standard_error_bps"] += 1.0
        else:
            metrics.loc[0, "coverage_fraction"] = 0.9
        metrics.to_csv(metrics_path, index=False)
        report["metrics_sha256"] = _sha(metrics_path)
    _write_json(row["report_path"], report)
    with pytest.raises(RuntimeError, match=expected):
        _run(experiment, tmp_path)


def test_event_collision_does_not_replace_existing_evidence(experiment, tmp_path, monkeypatch) -> None:
    instant = datetime(2026, 9, 6, 12, 0, 0, 123456, tzinfo=timezone.utc)
    monkeypatch.setattr(verifier, "next_immutable_event_created_utc", lambda *_args: instant)
    result = _run(experiment, tmp_path)
    path = Path(result["json_path"])
    original = path.read_bytes()
    with pytest.raises(event_owner.ImmutableEventAuthorityError, match="already exists"):
        _run(experiment, tmp_path)
    assert path.read_bytes() == original


def test_later_collapse_remains_newest_under_clock_rollback(experiment, tmp_path, monkeypatch) -> None:
    class Clock(datetime):
        instant = datetime(2026, 9, 6, 12, 0, 0, 123456, tzinfo=timezone.utc)

        @classmethod
        def now(cls, _timezone=None):
            return cls.instant

    monkeypatch.setattr(event_owner, "datetime", Clock)
    green = _run(experiment, tmp_path)
    Clock.instant -= timedelta(seconds=30)
    row = experiment[-1]
    prediction = pd.read_parquet(row["prediction_path"])
    prediction["pred_direction"] = 2
    prediction.to_parquet(row["prediction_path"], index=False)
    _refresh_metrics(row)
    with pytest.raises(SystemExit):
        _run(experiment, tmp_path)
    newest = event_owner.select_latest_immutable_event(tmp_path / "events", verifier.EVENT_PREFIX)
    assert newest != Path(green["json_path"])
    assert json.loads(newest.read_text())["decision"] == "FAIL"
