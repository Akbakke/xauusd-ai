from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager

import copy
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_decision_token_v1 import (
    entry_decision_token_projection_metadata,
)
from gx1.contracts.entry_exit_feature_usefulness_v1 import (
    DECISION,
    POLICY,
    canonical_json_sha256,
    feature_usefulness_layout,
    require_feature_usefulness_report,
)
from gx1.contracts.entry_fitted_q_v1 import (
    ENTRY_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION,
    ENTRY_FITTED_Q_SCHEMA_VERSION,
    entry_fitted_q_contract,
)
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS,
    MTF_SEMANTIC_CATEGORICAL_DOMAINS,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DOMAINS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
    MODEL_NATIVE_STATIC_CONTRACT_SHA256,
    ordered_model_native_signal_fields,
)
from gx1.contracts.unified_exit_episode_pack_v1 import (
    UNIFIED_EXIT_EPISODE_SIDE_COUNT,
    UNIFIED_EXIT_EPISODE_STATE_COUNT,
    unified_exit_episode_pack_contract,
)
from gx1.contracts.unified_exit_fitted_q_v1 import (
    UNIFIED_EXIT_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION,
    UNIFIED_EXIT_FITTED_Q_SCHEMA_VERSION,
    unified_exit_fitted_q_contract,
)
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM,
    UNIFIED_EXIT_PATH_FEATURE_ORDER,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    require_multi_tf_specialist_routing_v4,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4
from gx1.scripts import entry_candidate_prediction_evidence_v1 as prediction_evidence
from gx1.scripts import audit_entry_exit_feature_usefulness_v1 as usefulness_audit
from gx1.scripts.audit_entry_exit_feature_usefulness_v1 import (
    _build_exit_side_pair_plan,
    _fitted_q_loss_and_unique_target_margin,
    _paired_summary,
    _require_exit_fitted_q_state_binding,
    audit_task_feature_usefulness,
    build_native_exit_structure_plans,
    build_feature_usefulness_report,
    build_structure_preserving_donor_plan,
    write_immutable_feature_usefulness_report,
)


def _native_public_arguments():
    return {
        "--execute": None, "--device": "cpu", "--bundle-dir": "/fixture/bundle",
        "--bundle-metadata-sha256": "1" * 64, "--session-contract-sha256": "2" * 64,
        "--active-pointer-sha256": "3" * 64, "--selected-checkpoint-sha256": "4" * 64,
        "--recipe-audit-json": "/fixture/recipe.json", "--recipe-audit-sha256": "5" * 64,
        "--batch-size": "8", "--max-baseline-bytes": "1024", "--max-episode-bytes": "2048",
        "--max-forward-calls": "10000000", "--out-json": "/fixture/reports/usefulness.json",
    }


def _native_public_argv(arguments):
    return [token for flag, value in arguments.items() for token in (
        (flag,) if value is None else (flag, value)
    )]


@pytest.mark.parametrize("flag", tuple(_native_public_arguments()))
def test_native_public_cli_requires_every_explicit_argument(monkeypatch, flag):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid CLI must fail before execution")
    monkeypatch.setattr(usefulness_audit, "_execute_native_feature_usefulness", forbidden)
    arguments = _native_public_arguments()
    del arguments[flag]
    with pytest.raises(SystemExit) as error:
        usefulness_audit.main(_native_public_argv(arguments))
    assert error.value.code == 2


@pytest.mark.parametrize("flag,value", (
    ("--device", "cuda"), ("--device", "auto"), ("--batch-size", "0"),
    ("--max-baseline-bytes", "-1"), ("--max-episode-bytes", "0"),
    ("--max-forward-calls", "0"), ("--recipe-audit-sha256", "A" * 64),
    ("--bundle-metadata-sha256", "bad"), ("--splits", "test"),
    ("--subsample-rows", "1"), ("--feature-mask-json", "/fixture/mask.json"),
    ("--validate-json", "/fixture/report.json"), ("--dev", "cpu"),
))
def test_native_public_cli_rejects_implicit_or_partial_modes(monkeypatch, flag, value):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid CLI must fail before execution")
    monkeypatch.setattr(usefulness_audit, "_execute_native_feature_usefulness", forbidden)
    arguments = _native_public_arguments()
    arguments[flag] = value
    with pytest.raises(SystemExit) as error:
        usefulness_audit.main(_native_public_argv(arguments))
    assert error.value.code == 2


@pytest.mark.parametrize("flag", tuple(_native_public_arguments()))
def test_native_public_cli_rejects_duplicate_arguments(monkeypatch, flag):
    def forbidden(*args, **kwargs):
        pytest.fail("duplicate CLI must fail before execution")
    monkeypatch.setattr(usefulness_audit, "_execute_native_feature_usefulness", forbidden)
    arguments = _native_public_arguments()
    argv = _native_public_argv(arguments)
    argv.append(flag if arguments[flag] is None else flag + "=" + arguments[flag])
    with pytest.raises(SystemExit) as error:
        usefulness_audit.main(argv)
    assert error.value.code == 2


def test_native_public_cli_dispatches_exact_paths_budgets_and_cpu(monkeypatch, capsys):
    observed = []
    def execute(args, *, repo):
        observed.append((vars(args), repo))
        return args.out_json
    monkeypatch.setattr(usefulness_audit, "_execute_native_feature_usefulness", execute)
    assert usefulness_audit.main(_native_public_argv(_native_public_arguments())) == 0
    arguments, repo = observed[0]
    assert repo == Path(usefulness_audit.__file__).resolve().parents[2]
    assert arguments["device"] == "cpu" and arguments["execute"] is True
    assert arguments["bundle_dir"] == Path("/fixture/bundle")
    assert arguments["max_forward_calls"] == 10000000 and arguments["batch_size"] == 8
    assert json.loads(capsys.readouterr().out)["device"] == "cpu"


def test_native_public_cli_preserves_validation_without_execution(tmp_path, usefulness_report, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("validation-only mode cannot execute")
    monkeypatch.setattr(usefulness_audit, "_execute_native_feature_usefulness", forbidden)
    path = write_immutable_feature_usefulness_report(tmp_path / "report.json", usefulness_report)
    assert usefulness_audit.main(["--validate-json", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"decision": DECISION, "path": str(path)}
    with pytest.raises(SystemExit) as error:
        usefulness_audit.main(["--validate-json", str(path), "--device", "cpu"])
    assert error.value.code == 2


@pytest.fixture
def native_cpu_numerics():
    import torch

    before = (
        torch.get_num_threads(), torch.get_default_dtype(), torch.get_rng_state(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(), np.random.get_state(),
    )
    yield torch
    torch.set_num_threads(before[0])
    torch.set_default_dtype(before[1])
    torch.set_rng_state(before[2])
    torch.use_deterministic_algorithms(before[3], warn_only=before[4])
    np.random.set_state(before[5])


def test_native_cpu_numerics_are_one_thread_fp32_and_never_probe_cuda(native_cpu_numerics, monkeypatch):
    torch = native_cpu_numerics
    def forbidden(*args, **kwargs):
        pytest.fail("CPU preparation must not call a CUDA API")
    for name in ("is_available", "manual_seed_all", "current_device", "set_per_process_memory_fraction"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    usefulness_audit._set_native_cpu_numerics(1337)
    expected = torch.rand(4), np.random.random(4)
    usefulness_audit._set_native_cpu_numerics(1337)
    assert torch.equal(expected[0], torch.rand(4))
    np.testing.assert_array_equal(expected[1], np.random.random(4))
    assert torch.get_num_threads() == 1 and torch.get_default_dtype() == torch.float32
    assert torch.are_deterministic_algorithms_enabled()


@pytest.mark.parametrize("fault", ("threads", "dtype", "determinism", "warn_only", "autocast"))
def test_native_cpu_numerics_drift_rejects_without_repair(native_cpu_numerics, fault):
    torch = native_cpu_numerics
    usefulness_audit._set_native_cpu_numerics(1337)
    if fault == "threads":
        torch.set_num_threads(2)
    elif fault == "dtype":
        torch.set_default_dtype(torch.float64)
    elif fault == "determinism":
        torch.use_deterministic_algorithms(False)
    elif fault == "warn_only":
        torch.use_deterministic_algorithms(True, warn_only=True)
    with torch.autocast("cpu", enabled=fault == "autocast"):
        with pytest.raises(RuntimeError, match="NUMERICS_CHANGED"):
            usefulness_audit._require_native_cpu_numerics()


@pytest.fixture
def native_public_io(tmp_path, monkeypatch, usefulness_report, native_cpu_numerics):
    """Real committed-file/atomic-output owners; model and population admission mocked."""

    from gx1.contracts import gx1_capped_execution_v1 as capped
    from gx1.contracts import entry_model_native_pretest_technical_recipe_v1 as recipe_owner
    from gx1.contracts.entry_model_native_bundle_commit_v1 import CORE_ARTIFACTS, write_bundle_commit_manifest
    from gx1.scripts import entry_exit_feature_usefulness_native_v1 as native

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "PROJECT_STATE_xau_direction_launch.json").write_text("{}")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    state_bytes = b"synthetic immutable bytes, not a loadable model"
    (bundle / "model_state_dict.pt").write_bytes(state_bytes)
    digest = hashlib.sha256(state_bytes).hexdigest()
    metadata = {"state_dict_sha256": digest}
    (bundle / "bundle_metadata.json").write_text(json.dumps(metadata))
    (bundle / "MASTER_TRANSFORMER_LOCK.json").write_text(json.dumps({"model_sha256": digest}))
    committed = write_bundle_commit_manifest(
        bundle_dir=bundle, artifact_names=CORE_ARTIFACTS, bundle_kind="trained",
        created_at_utc="2026-01-01T00:00:00Z",
    )
    recipe = {"run_id": "synthetic", "dataset_run_id": "synthetic-data", "dataset_dir": "/unopened/data",
              "trainer_cli": {"seed": 1337}}
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    reports = tmp_path / "reports"
    reports.mkdir()
    args = SimpleNamespace(**{
        flag[2:].replace("-", "_"): value for flag, value in _native_public_arguments().items()
    })
    args.bundle_dir, args.recipe_audit_json, args.out_json = bundle, recipe_path, reports / "report.json"
    args.recipe_audit_sha256 = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    args.bundle_metadata_sha256 = hashlib.sha256((bundle / "bundle_metadata.json").read_bytes()).hexdigest()
    args.batch_size, args.max_baseline_bytes, args.max_episode_bytes, args.max_forward_calls = 8, 1024, 2048, 10000000
    events = []
    selected = SimpleNamespace(bindings={"bundle_commit_sha256": committed["commit_sha256"]})
    def cap():
        events.append("cap")
        return {"class": "audit"}
    def source(**kwargs):
        events.append("source")
        assert kwargs["recipe_audit_sha256"] == args.recipe_audit_sha256
        assert kwargs["out_bundle_dir"] == bundle
        return {}
    def load(**kwargs):
        events.append("load")
        assert kwargs["session_out_bundle_dir"] == bundle
        assert kwargs["repo"] == repo
        return selected
    @contextmanager
    def inputs(**kwargs):
        events.append("open")
        try:
            yield selected
        finally:
            events.append("close")
    def audit(**kwargs):
        events.append("audit")
        assert str(kwargs["device"]) == "cpu" and kwargs["batch_rows"] == 8
        assert kwargs["max_forward_calls"] == args.max_forward_calls
        assert not args.out_json.exists()
        return copy.deepcopy(usefulness_report)
    monkeypatch.setattr(capped, "require_capped_cpu_audit_execution", cap)
    monkeypatch.setattr(recipe_owner, "require_pretest_technical_recipe_metadata", lambda value, **kwargs: value)
    monkeypatch.setattr(native, "require_training_recipe_source_provenance", source)
    monkeypatch.setattr(native, "load_selected_native_exit_pair", load)
    monkeypatch.setattr(native, "require_selected_native_pair_unchanged", lambda **kwargs: events.append("pair_check"))
    monkeypatch.setattr(usefulness_audit, "open_native_val_inputs", inputs)
    monkeypatch.setattr(usefulness_audit, "audit_native_feature_usefulness", audit)
    monkeypatch.setattr(usefulness_audit, "require_native_val_inputs_unchanged", lambda value: events.append("input_check"))
    return SimpleNamespace(args=args, repo=repo, events=events, audit=audit, native=native, capped=capped, selected=selected)


def test_native_public_execution_checks_then_closes_before_atomic_publication(native_public_io, monkeypatch):
    fixture = native_public_io
    writer = usefulness_audit.write_immutable_feature_usefulness_report
    def publish(path, report):
        assert fixture.events == ["cap", "source", "load", "open", "audit", "input_check", "close", "pair_check", "cap"]
        return writer(path, report)
    monkeypatch.setattr(usefulness_audit, "write_immutable_feature_usefulness_report", publish)
    assert usefulness_audit._execute_native_feature_usefulness(fixture.args, repo=fixture.repo) == fixture.args.out_json
    require_feature_usefulness_report(json.loads(fixture.args.out_json.read_text()))


def test_native_public_execution_cleanup_failure_precedes_publication(native_public_io, monkeypatch):
    fixture = native_public_io
    @contextmanager
    def failed_cleanup(**kwargs):
        yield fixture.selected
        raise RuntimeError("synthetic scratch cleanup failure")
    monkeypatch.setattr(usefulness_audit, "open_native_val_inputs", failed_cleanup)
    with pytest.raises(RuntimeError, match="scratch cleanup failure"):
        usefulness_audit._execute_native_feature_usefulness(fixture.args, repo=fixture.repo)
    assert not fixture.args.out_json.exists()


def test_native_public_execution_rechecks_actual_containment_before_publication(native_public_io, monkeypatch):
    fixture = native_public_io
    calls = []
    def cap():
        calls.append("cap")
        if len(calls) == 2:
            raise RuntimeError("synthetic changed cgroup")
    monkeypatch.setattr(fixture.capped, "require_capped_cpu_audit_execution", cap)
    with pytest.raises(RuntimeError, match="changed cgroup"):
        usefulness_audit._execute_native_feature_usefulness(fixture.args, repo=fixture.repo)
    assert len(calls) == 2 and "close" in fixture.events
    assert not fixture.args.out_json.exists()


def test_native_public_execution_rejects_different_selected_bundle(native_public_io):
    fixture = native_public_io
    fixture.selected.bindings["bundle_commit_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="SELECTED_BUNDLE_CHANGED"):
        usefulness_audit._execute_native_feature_usefulness(fixture.args, repo=fixture.repo)
    assert "open" not in fixture.events and not fixture.args.out_json.exists()


def test_native_public_execution_never_publishes_inside_candidate_session(native_public_io):
    fixture = native_public_io
    directory = fixture.args.bundle_dir.parent / (
        fixture.native.trainer._CANDIDATE_TRAINING_SESSION_DIR_PREFIX + fixture.args.bundle_dir.name
    ) / "empty-child"
    directory.mkdir(parents=True)
    fixture.args.out_json = directory / "report.json"
    with pytest.raises(RuntimeError, match="OUTPUT_INSIDE_SESSION"):
        usefulness_audit._execute_native_feature_usefulness(fixture.args, repo=fixture.repo)
    assert "load" not in fixture.events and not fixture.args.out_json.exists()


@pytest.mark.parametrize("fault", ("state", "metadata", "lock", "commit", "inventory", "pair", "input", "budget", "hold", "numerics", "output"))
def test_native_public_execution_never_publishes_after_drift(native_public_io, monkeypatch, fault):
    fixture = native_public_io
    from gx1.contracts.entry_model_native_bundle_commit_v1 import MANIFEST_NAME
    def broken(*args, **kwargs):
        raise RuntimeError("synthetic boundary rejection")
    def audit(**kwargs):
        report = fixture.audit(**kwargs)
        if fault in {"state", "metadata", "lock", "commit", "inventory"}:
            name = {"state": "model_state_dict.pt", "metadata": "bundle_metadata.json", "lock": "MASTER_TRANSFORMER_LOCK.json",
                    "commit": MANIFEST_NAME, "inventory": "extra.json"}[fault]
            path = fixture.args.bundle_dir / name
            path.write_bytes((path.read_bytes() if path.exists() else b"") + b"tampered")
        elif fault == "pair":
            monkeypatch.setattr(fixture.native, "require_selected_native_pair_unchanged", broken)
        elif fault == "input":
            monkeypatch.setattr(usefulness_audit, "require_native_val_inputs_unchanged", broken)
        elif fault == "budget":
            raise RuntimeError("FEATURE_USEFULNESS_NATIVE_FORWARD_BUDGET_EXCEEDED")
        elif fault == "hold":
            (fixture.repo / "PROJECT_STATE_xau_direction_launch.json").write_text('{"pretraining_review_hold": null}')
        elif fault == "numerics":
            import torch
            torch.use_deterministic_algorithms(False)
        else:
            fixture.args.out_json.write_text("competing immutable output")
        return report
    monkeypatch.setattr(usefulness_audit, "audit_native_feature_usefulness", audit)
    with pytest.raises(RuntimeError):
        usefulness_audit._execute_native_feature_usefulness(fixture.args, repo=fixture.repo)
    assert "close" in fixture.events
    if fault == "output":
        assert fixture.args.out_json.read_text() == "competing immutable output"
    else:
        assert not fixture.args.out_json.exists()


@pytest.mark.parametrize("fault", ("cap", "hold", "source", "relative", "ancestor_symlink", "nonempty_output"))
def test_native_public_execution_preflight_rejects_before_model_load(native_public_io, monkeypatch, tmp_path, fault):
    fixture = native_public_io
    def broken(*args, **kwargs):
        raise RuntimeError("synthetic preflight refusal")
    if fault == "cap":
        monkeypatch.setattr(fixture.capped, "require_capped_cpu_audit_execution", broken)
    elif fault == "hold":
        (fixture.repo / "PROJECT_STATE_xau_direction_launch.json").write_text('{"pretraining_review_hold": null}')
    elif fault == "source":
        monkeypatch.setattr(fixture.native, "require_training_recipe_source_provenance", broken)
    elif fault == "relative":
        fixture.args.bundle_dir = Path("relative")
    elif fault == "ancestor_symlink":
        link = tmp_path / "link"
        link.symlink_to(fixture.args.bundle_dir.parent, target_is_directory=True)
        fixture.args.bundle_dir = link / fixture.args.bundle_dir.name
    else:
        (fixture.args.out_json.parent / "unrelated.json").write_text("keep")
    with pytest.raises(RuntimeError):
        usefulness_audit._execute_native_feature_usefulness(fixture.args, repo=fixture.repo)
    assert "load" not in fixture.events and not fixture.args.out_json.exists()


@pytest.fixture
def native_val_reader_io(tmp_path, monkeypatch):
    """Mechanical files with mocked source admission/cache/corpus/Dataset IO.

    Real recipe, signal, split-window and reconstruction metadata owners run.
    No trained pair, Parquet population or native lifecycle is proved here.
    """

    from gx1.contracts.entry_exit_feature_base_v1 import (
        ENTRY_DECISION_BAR_SECONDS, ENTRY_MTF_CONTEXT_TIMEFRAMES,
        EXIT_DECISION_BAR_SECONDS, EXIT_MTF_CONTEXT_TIMEFRAMES,
    )
    from gx1.contracts.entry_exit_production_architecture_v1 import PRODUCTION_MTF_PER_TF_WINDOW_BARS
    from gx1.contracts.entry_model_native_signal_v1 import model_native_signal_contract_metadata
    from gx1.contracts.entry_model_native_train_launch_v1 import (
        TRAINING_RECIPE_SOURCE_PROVENANCE_SCHEMA, artifact_binding,
    )
    from gx1.contracts.entry_sequence_source_reconstruction_v1 import (
        AUTHORITY, REQUIRED_CHECKS, SCHEMA_VERSION as RECONSTRUCTION_SCHEMA,
    )
    from gx1.contracts import unified_exit_lifecycle_v1 as lifecycle_owner
    from gx1.features import htf_features as htf_owner
    from gx1.models.entry_v10 import entry_v10_input_normalization as cache_owner
    from gx1.scripts import entry_exit_feature_usefulness_native_v1 as native
    from gx1.scripts import evaluate_entry_candidate_selective_edge_v1 as evaluation
    from tests.test_entry_model_native_pretest_technical_recipe import _recipe

    root = tmp_path.resolve()
    recipe = _recipe(root)
    recipe["profile"] = "candidate"
    recipe["trainer_cli"].update(
        execution_tier="canonical", subsample_rows=0, num_workers=0, train_time_window=None,
        **{f"per_tf_seq_len_{name.lower()}": length for name, length in PRODUCTION_MTF_PER_TF_WINDOW_BARS},
    )
    recipe["trainer_cli_sha256"] = canonical_json_sha256(recipe["trainer_cli"])
    artifacts = recipe["artifact_bindings"]
    artifacts["unified_exit_lifecycle_manifest"]["path"] = str(
        Path(artifacts["unified_exit_lifecycle_manifest"]["path"]).with_name("UNIFIED_EXIT_LIFECYCLE_MANIFEST.json")
    )
    def write_bound(name, payload):
        path = Path(artifacts[name]["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
        artifacts[name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    val_path = write_bound("val_parquet", b"mechanical IO fixture, not Parquet")
    m5_path = write_bound("m5_prebuilt", b"model M5 source fixture")
    cache_manifest = write_bound("multi_tf_cache_manifest", {"fixture": "not a native cache"})
    lifecycle_path = write_bound("unified_exit_lifecycle_manifest", {
        "m1_authority": {"authority_mode": "pretest_quote_complete_native_v1"},
        "splits": {"train": {"must_not_open": True}, "val": {}},
    })
    cache_source = root / "cache_m5_source.parquet"
    cache_source.write_bytes(b"distinct cache source fixture")
    surface_path = root / "dataset" / "m5_feature_surface.parquet"
    surface_path.write_bytes(b"distinct sequence source fixture")
    surface_manifest = surface_path.with_suffix(".manifest.json")
    surface_manifest.write_text('{"fixture":"feature metadata"}')
    def file_sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    cache_binding = {
        "cache_dir": str(cache_manifest.parent), "manifest_path": str(cache_manifest),
        "manifest_sha256": file_sha(cache_manifest), "cache_identity_sha256": "a" * 64,
        "m5_prebuilt_source": str(cache_source), "m5_prebuilt_source_sha256": file_sha(cache_source),
    }
    signal = model_native_signal_contract_metadata([
        *MODEL_NATIVE_MANDATORY_SELECTED_FIELDS, *MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS
    ])
    seq_len = recipe["trainer_cli"]["seq_len"]
    times = pd.date_range("2026-05-31T23:55:00Z", periods=4, freq="5min").as_unit("ns")
    surface = {
        "dataset_run_id": recipe["dataset_run_id"], "inline_split_recomputation": False,
        "manifest_path": str(surface_manifest), "manifest_sha256": file_sha(surface_manifest),
        "pair_generation_id": "fixture_generation", "path": str(surface_path),
        "rows": seq_len + len(times) - 1, "schema_version": "mechanical_feature_surface",
        "sha256": file_sha(surface_path), "signal_manifest_sha256": "b" * 64,
        "time_alignment": "exact_entry_m5_source_timeline",
    }
    manifest_path = write_bound("val_manifest", {
        "output_data_path": str(val_path),
        "splits": {"val": {"start": "2026-05-31T00:00:00Z", "end": "2026-07-01T00:00:00Z"}},
        "ts_min_max_by_split": {"val": {"ts_min": times[0].isoformat(), "ts_max": times[-1].isoformat()}},
        "extra": {
            "entry_run_id": recipe["dataset_run_id"], "rows": len(times),
            "emission_start_utc": "2026-05-31T00:00:00Z",
            "emission_end_utc": "2026-06-30T23:55:00Z",
            "feature_computation_end_utc": "2026-06-30T23:55:00Z",
            "model_native_signal_contract": signal,
            "multi_tf_cache_binding": cache_binding,
            "source_frame": {"parquet_path": str(m5_path), "parquet_sha256": file_sha(m5_path)},
            "signal_bridge": {"seq_structure_extension_v1": {"feature_surface": surface}},
        },
    })
    reconstruction = {
        "schema_version": RECONSTRUCTION_SCHEMA, "decision": "PASS", "created_utc": "fixture",
        "parquet_path": str(val_path), "parquet_sha256": file_sha(val_path),
        "manifest_path": str(manifest_path), "manifest_sha256": file_sha(manifest_path),
        "feature_surface_path": str(surface_path), "feature_surface_sha256": file_sha(surface_path),
        "feature_surface_manifest_path": str(surface_manifest),
        "feature_surface_manifest_sha256": file_sha(surface_manifest),
        "feature_surface_rows": surface["rows"], "rows": len(times),
        "sequence_shape": [len(times), seq_len, len(signal["fields"])],
        "snapshot_shape": [len(times), len(signal["fields"])],
        "checks": REQUIRED_CHECKS, "sequence_source_chain_sha256": "c" * 64, "authority": AUTHORITY,
    }
    reconstruction_path = write_bound("val_sequence_source_reconstruction", reconstruction)
    source_file = root / "reader_source_fixture.py"
    source_file.write_text('"""Not the actual runtime source closure."""')
    recipe["source_bindings"] = {"fixture": artifact_binding(source_file)}
    recipe["source_bindings_sha256"] = canonical_json_sha256(recipe["source_bindings"])
    recipe["artifact_bindings_sha256"] = canonical_json_sha256(artifacts)
    recipe_path = root / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    provenance = {
        "schema_version": TRAINING_RECIPE_SOURCE_PROVENANCE_SCHEMA,
        "recipe_audit_path": str(recipe_path), "recipe_audit_sha256": file_sha(recipe_path),
        **{key: recipe[key] for key in ("source_commit", "source_bindings", "source_bindings_sha256")},
    }
    metadata = {
        "run_lineage": {"training_profile": "candidate", "training_run_id": recipe["run_id"], "dataset_run_id": recipe["dataset_run_id"]},
        "recipe_source_provenance": provenance, "seq_len": seq_len,
        "model_native_signal_contract": signal, "ordered_signal_names": signal["fields"],
        "input_normalization": {"contract_sha256": "e" * 64},
        "sequence_source_reconstruction": {"splits": {"val": reconstruction}},
        "unified_exit_training_evidence": {"lifecycle": {
            "root_manifest_sha256": file_sha(lifecycle_path),
            "splits": {"train": {"not_read": True}, "val": {"fixture": True}},
        }},
        "multi_tf": {
            "enabled": True, "closed_bar_target_availability": True,
            "entry_route_timeframes": list(ENTRY_MTF_CONTEXT_TIMEFRAMES),
            "exit_route_timeframes": list(EXIT_MTF_CONTEXT_TIMEFRAMES),
            "entry_target_availability_shift_minutes": ENTRY_DECISION_BAR_SECONDS / 60,
            "exit_target_availability_shift_minutes": EXIT_DECISION_BAR_SECONDS / 60,
            **{f"{name.lower()}_seq_len": length for name, length in PRODUCTION_MTF_PER_TF_WINDOW_BARS},
        },
    }
    metadata_path = Path(recipe["out_bundle_dir"]) / "bundle_metadata.json"
    metadata_path.parent.mkdir()
    metadata_path.write_text(json.dumps(metadata))
    pair_bindings = {
        "recipe_source_provenance": provenance,
        "files": {str(recipe_path): file_sha(recipe_path)},
        "bundle_metadata_path": str(metadata_path), "bundle_metadata_sha256": file_sha(metadata_path),
        "input_normalization_sha256": "e" * 64,
        "dataset_artifact_declarations": {
            "train_parquet": artifacts["train_parquet"], "val_parquet": artifacts["val_parquet"],
            "m5_prebuilt_path": artifacts["m5_prebuilt"],
            "unified_exit_lifecycle_manifest": artifacts["unified_exit_lifecycle_manifest"],
        },
    }
    pair = native.SelectedNativeExitPair(model=object(), target_model=object(), metadata=metadata, bindings=pair_bindings)
    allocations = []
    owner_calls = []
    cache_object = object()
    cache_env = native.trainer._TRAIN_MULTI_TF_CACHE_ENV
    def source_owner(**arguments):
        owner_calls.append("source")
        assert arguments["profile"] == "candidate"
        assert arguments["recipe_audit_sha256"] == file_sha(recipe_path)
        return provenance
    def mtf_owner(**arguments):
        owner_calls.append("mtf")
        assert set(arguments["dataset_contract"]["splits"]) == {"val"}
        assert arguments["mtf_cache_dir"] == cache_manifest.parent
        assert arguments["m5_prebuilt"] == m5_path
        return {"split": "val", "cache_binding": cache_binding}
    def loaded_cache_owner(path, **arguments):
        owner_calls.append("loaded_cache")
        assert path == manifest_path and arguments["cache"] is cache_object
        assert arguments["cache_dir"] == cache_manifest.parent
        return cache_binding
    def cache_loader(path):
        owner_calls.append("cache_load")
        assert path == m5_path
        assert os.environ[cache_env] == str(cache_manifest.parent)
        return cache_object
    def cold_cache_loader(path):
        owner_calls.append("cold_cache_load")
        assert path == cache_manifest.parent
        return cache_object
    class CorpusIO:
        def __init__(self, **arguments):
            assert arguments == {
                "root_manifest_path": lifecycle_path, "entry_parquets": {"val": val_path},
                "entry_manifest_bindings": {"val": artifacts["val_manifest"]},
                "dataset_run_id": recipe["dataset_run_id"], "splits": ("val",),
            }
            assert os.environ[cache_env] == str(cache_manifest.parent)
            self._m1_feature_tempdir = tempfile.TemporaryDirectory(prefix="owned_reader_", dir=root)
            Path(self._m1_feature_tempdir.name, "owned_scratch").write_bytes(b"owned scratch")
            self.splits = {"val": SimpleNamespace(split="val", entry_row_count=len(times), _entry_times=times)}
            self.evidence = {"root_manifest_sha256": file_sha(lifecycle_path), "splits": {"val": {"fixture": True}}}
            allocations.append(self)
        def require_files_unchanged(self):
            owner_calls.append("lifecycle_recheck")
    class DatasetIO:
        def __init__(self, **arguments):
            assert arguments == {
                "parquet_path": val_path, "seq_len": seq_len, "sequence_source_audit_json": reconstruction_path,
                "m5_prebuilt_path": m5_path, "multi_tf_closed_bar": True,
                "per_tf_seq_lens": dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS),
            }
            assert os.environ[cache_env] == str(cache_manifest.parent)
            self.df = pd.DataFrame({"time": times})
            self.indices = np.arange(len(times), dtype=np.int64)
            self.seq_len = seq_len
            self._compact_row_indices = None
            self._sequence_source_reconstructed = True
            self._sequence_roll_reconstructed = False
            self._sequence_source_audit = reconstruction
            self._sequence_source_positions = np.arange(seq_len - 1, seq_len - 1 + len(times), dtype=np.int64)
            self._sequence_source_times_ns = pd.date_range(end=times[-1], periods=surface["rows"], freq="5min").as_unit("ns").asi8
            self._np_seq = None
            self._memmap_tmpdir = None
            self.model_native_signal_contract = signal
            self.signal_names = signal["fields"]
            self._multi_tf_feats = cache_object
            for attribute, key in (
                ("_multi_tf_cache_identity_sha256", "cache_identity_sha256"),
                ("_multi_tf_cache_manifest_sha256", "manifest_sha256"),
                ("_multi_tf_cache_dir", "cache_dir"),
                ("_multi_tf_cache_manifest_path", "manifest_path"),
                ("_multi_tf_cache_m5_source", "m5_prebuilt_source"),
                ("_multi_tf_cache_m5_source_sha256", "m5_prebuilt_source_sha256"),
            ):
                setattr(self, attribute, cache_binding[key])
            allocations.append(self)
        def __len__(self):
            return len(self.indices)
        def bind_unified_exit_lifecycle(self, lifecycle):
            assert lifecycle is allocations[0].splits["val"]
            self.bound_lifecycle = lifecycle
    monkeypatch.setattr(native, "require_training_recipe_source_provenance", source_owner)
    monkeypatch.setattr(evaluation, "_require_evaluation_mtf_source_provenance", mtf_owner)
    monkeypatch.setattr(cache_owner, "require_manifest_bound_multi_tf_v4_cache", loaded_cache_owner)
    monkeypatch.setattr(lifecycle_owner, "UnifiedExitLifecycleCorpus", CorpusIO)
    monkeypatch.setattr(native.trainer, "EntryV10CtxDataset", DatasetIO)
    monkeypatch.setattr(native.trainer, "_prebuild_multi_tf_features_once", cache_loader)
    monkeypatch.setattr(htf_owner, "load_multi_tf_v4_cache", cold_cache_loader)
    return SimpleNamespace(
        arguments={"selected_pair": pair, "recipe_audit_path": recipe_path, "recipe_audit_sha256": file_sha(recipe_path)},
        recipe=recipe, artifacts=artifacts, provenance=provenance, metadata_path=metadata_path,
        file_sha=file_sha, native=native, lifecycle_owner=lifecycle_owner, htf_owner=htf_owner,
        cache_manifest=cache_manifest, cache_env=cache_env, allocations=allocations,
        owner_calls=owner_calls, dataset_factory=DatasetIO, times=times,
        reconstruction=reconstruction, reconstruction_path=reconstruction_path,
    )


def _refresh_native_val_reader_recipe(fixture):
    fixture.recipe["artifact_bindings_sha256"] = canonical_json_sha256(fixture.artifacts)
    fixture.recipe["trainer_cli_sha256"] = canonical_json_sha256(fixture.recipe["trainer_cli"])
    path = fixture.arguments["recipe_audit_path"]
    path.write_text(json.dumps(fixture.recipe))
    digest = fixture.file_sha(path)
    fixture.arguments["recipe_audit_sha256"] = digest
    fixture.provenance["recipe_audit_sha256"] = digest
    pair = fixture.arguments["selected_pair"]
    pair.bindings["files"][str(path)] = digest
    fixture.metadata_path.write_text(json.dumps(pair.metadata))
    pair.bindings["bundle_metadata_sha256"] = fixture.file_sha(fixture.metadata_path)


@pytest.mark.parametrize("previous", (None, "", "/untrusted/ambient/cache"))
def test_native_val_reader_mocked_io_full_population_clocks_and_context_cleanup(native_val_reader_io, monkeypatch, previous):
    fixture = native_val_reader_io
    if previous is None:
        monkeypatch.delenv(fixture.cache_env, raising=False)
    else:
        monkeypatch.setenv(fixture.cache_env, previous)
    originals = {
        path: (path.read_bytes(), path.stat().st_mode, path.stat().st_mtime_ns)
        for path in fixture.metadata_path.parents[1].rglob("*") if path.is_file()
    }
    with usefulness_audit.open_native_val_inputs(**fixture.arguments) as inputs:
        assert inputs.dataset is fixture.allocations[1]
        assert inputs.corpus is fixture.allocations[0]
        scratch = Path(inputs.corpus._m1_feature_tempdir.name)
        assert scratch.is_dir()
        assert inputs.dataset.bound_lifecycle is inputs.corpus.splits["val"]
        assert inputs.provenance["entry_row_count"] == len(fixture.times)
        assert not inputs.dataset.indices.flags.writeable
        assert not inputs.entry_bar_open_time_ns.flags.writeable
        assert not inputs.entry_decision_time_ns.flags.writeable
        np.testing.assert_array_equal(inputs.entry_bar_open_time_ns, fixture.times.asi8)
        np.testing.assert_array_equal(inputs.entry_decision_time_ns, (fixture.times + pd.Timedelta(minutes=5)).asi8)
        assert inputs.entry_decision_time_ns[0] == pd.Timestamp("2026-06-01T00:00:00Z").value
        assert inputs.provenance["source_reconstruction"] == fixture.reconstruction
        assert "normalization_path" not in inputs.provenance
        assert os.environ[fixture.cache_env] == str(fixture.cache_manifest.parent)
    assert fixture.owner_calls == ["source", "mtf", "cache_load", "loaded_cache"]
    assert not scratch.exists()
    assert os.environ.get(fixture.cache_env) == previous
    assert all((path.read_bytes(), path.stat().st_mode, path.stat().st_mtime_ns) == value for path, value in originals.items())
    assert not Path(fixture.artifacts["train_parquet"]["path"]).exists()


def test_native_val_stability_reenters_cold_owners_without_reconstruction(native_val_reader_io):
    fixture = native_val_reader_io
    with usefulness_audit.open_native_val_inputs(**fixture.arguments) as inputs:
        allocations = tuple(fixture.allocations)
        usefulness_audit.require_native_val_inputs_unchanged(inputs)
        assert tuple(fixture.allocations) == allocations
        assert fixture.owner_calls[-3:] == ["lifecycle_recheck", "cold_cache_load", "loaded_cache"]
        with pytest.raises(TypeError):
            inputs.provenance["files"]["/new-input"] = "a" * 64
    assert not Path(fixture.artifacts["train_parquet"]["path"]).exists()


@pytest.mark.parametrize("fault", ["direct", "lifecycle", "cache", "post_check_file", "evidence"])
def test_native_val_stability_rejects_changed_backing_before_return(native_val_reader_io, monkeypatch, fault):
    fixture = native_val_reader_io
    with usefulness_audit.open_native_val_inputs(**fixture.arguments) as inputs:
        selected_path = Path(inputs.provenance["source_feature_surface"]["path"])
        if fault == "direct":
            selected_path.write_bytes(b"changed direct selected input")
        elif fault == "lifecycle":
            def changed_lifecycle():
                raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_TEST_MUTATION")
            monkeypatch.setattr(inputs.corpus, "require_files_unchanged", changed_lifecycle)
        elif fault == "cache":
            def changed_cache(_path):
                raise RuntimeError("HTF_V4_CACHE_TEST_MUTATION")
            monkeypatch.setattr(fixture.htf_owner, "load_multi_tf_v4_cache", changed_cache)
        elif fault == "evidence":
            inputs.corpus.evidence["root_manifest_sha256"] = "9" * 64
        else:
            original = fixture.htf_owner.load_multi_tf_v4_cache
            def changed_after_read(path):
                result = original(path)
                selected_path.write_bytes(b"changed during re-admission")
                return result
            monkeypatch.setattr(fixture.htf_owner, "load_multi_tf_v4_cache", changed_after_read)
        expected = {
            "direct": "NATIVE_EXIT_SELECTED_FILE_SHA_MISMATCH",
            "lifecycle": "UNIFIED_EXIT_LIFECYCLE_TEST_MUTATION",
            "cache": "HTF_V4_CACHE_TEST_MUTATION",
            "post_check_file": "NATIVE_EXIT_SELECTED_FILE_SHA_MISMATCH",
            "evidence": "FEATURE_USEFULNESS_NATIVE_VAL_LIFECYCLE_CHANGED",
        }[fault]
        with pytest.raises(RuntimeError, match=expected):
            usefulness_audit.require_native_val_inputs_unchanged(inputs)
        assert len(fixture.allocations) == 2


@pytest.mark.parametrize("artifact", (
    "val_manifest", "val_parquet", "m5_prebuilt", "multi_tf_cache_manifest",
    "unified_exit_lifecycle_manifest", "val_sequence_source_reconstruction",
))
def test_native_val_reader_mocked_io_hash_failure_precedes_allocation(native_val_reader_io, monkeypatch, artifact):
    fixture = native_val_reader_io
    monkeypatch.setenv(fixture.cache_env, "prior")
    path = Path(fixture.artifacts[artifact]["path"])
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="FILE_SHA_MISMATCH"):
        with usefulness_audit.open_native_val_inputs(**fixture.arguments):
            pytest.fail("tampered artifact yielded a reader")
    assert fixture.allocations == []
    assert os.environ[fixture.cache_env] == "prior"


@pytest.mark.parametrize("fault", (
    "recipe_bytes", "recipe_schema", "smoke", "sampled", "source", "pair_recipe",
    "reconstruction", "bundle_reconstruction", "lifecycle_mode", "mtf_geometry",
))
def test_native_val_reader_mocked_io_recipe_failure_precedes_allocation(native_val_reader_io, monkeypatch, fault):
    fixture = native_val_reader_io
    if fault == "recipe_bytes":
        fixture.arguments["recipe_audit_path"].write_text("{}")
    elif fault == "source":
        def refuse_source(**arguments):
            raise RuntimeError("SOURCE_OWNER_REFUSED")
        monkeypatch.setattr(fixture.native, "require_training_recipe_source_provenance", refuse_source)
    elif fault == "pair_recipe":
        fixture.arguments["recipe_audit_sha256"] = "0" * 64
    else:
        if fault == "recipe_schema":
            fixture.recipe["schema_version"] = "entry_model_native_train_recipe_audit_v1"
        elif fault == "smoke":
            fixture.recipe["profile"] = "smoke"
        elif fault == "sampled":
            fixture.recipe["trainer_cli"]["subsample_rows"] = 1
        elif fault == "reconstruction":
            invalid = {**fixture.reconstruction, "rows": len(fixture.times) - 1}
            fixture.reconstruction_path.write_text(json.dumps(invalid))
            fixture.artifacts["val_sequence_source_reconstruction"]["sha256"] = fixture.file_sha(fixture.reconstruction_path)
        elif fault == "bundle_reconstruction":
            fixture.arguments["selected_pair"].metadata["sequence_source_reconstruction"] = {"splits": {"val": {}}}
        elif fault == "lifecycle_mode":
            path = Path(fixture.artifacts["unified_exit_lifecycle_manifest"]["path"])
            path.write_text(json.dumps({"m1_authority": {"authority_mode": None}, "splits": {"train": {}, "val": {}, "test": {}}}))
            fixture.artifacts["unified_exit_lifecycle_manifest"]["sha256"] = fixture.file_sha(path)
        else:
            fixture.recipe["trainer_cli"]["per_tf_seq_len_h1"] += 1
        _refresh_native_val_reader_recipe(fixture)
    with pytest.raises(RuntimeError):
        with usefulness_audit.open_native_val_inputs(**fixture.arguments):
            pytest.fail("invalid recipe/source yielded a reader")
    assert fixture.allocations == []


@pytest.mark.parametrize("target", ("val_parquet", "recipe"))
def test_native_val_reader_mocked_io_symlinks_never_reach_allocation(native_val_reader_io, target):
    fixture = native_val_reader_io
    path = fixture.arguments["recipe_audit_path"] if target == "recipe" else Path(fixture.artifacts[target]["path"])
    retained = path.with_name(path.name + ".retained")
    path.rename(retained)
    path.symlink_to(retained)
    with pytest.raises(RuntimeError, match="FILE_PATH_INVALID"):
        with usefulness_audit.open_native_val_inputs(**fixture.arguments):
            pytest.fail("indirect artifact yielded a reader")
    assert fixture.allocations == []


def test_native_val_reader_mocked_io_bad_cache_bytes_precede_dataset_allocation(native_val_reader_io, monkeypatch):
    fixture = native_val_reader_io
    monkeypatch.setenv(fixture.cache_env, "prior")
    def cache_refusal(path):
        assert os.environ[fixture.cache_env] == str(fixture.cache_manifest.parent)
        raise RuntimeError("CACHE_OWNER_BYTES_REFUSED")
    monkeypatch.setattr(fixture.native.trainer, "_prebuild_multi_tf_features_once", cache_refusal)
    with pytest.raises(RuntimeError, match="CACHE_OWNER_BYTES_REFUSED"):
        with usefulness_audit.open_native_val_inputs(**fixture.arguments):
            pytest.fail("invalid cache yielded a reader")
    assert fixture.allocations == []
    assert os.environ[fixture.cache_env] == "prior"


@pytest.mark.parametrize("fault", ("subset", "permuted", "dtype", "clock", "source_clock", "source_flag", "cache", "signal"))
def test_native_val_reader_mocked_io_population_mismatch_restores_environment(native_val_reader_io, monkeypatch, fault):
    fixture = native_val_reader_io
    monkeypatch.setenv(fixture.cache_env, "prior")
    def corrupted_dataset(**arguments):
        dataset = fixture.dataset_factory(**arguments)
        if fault == "subset":
            dataset.indices = dataset.indices[:-1]
        elif fault == "permuted":
            dataset.indices = dataset.indices[::-1]
        elif fault == "dtype":
            dataset.indices = dataset.indices.astype(np.float32)
        elif fault == "clock":
            dataset.df.loc[1, "time"] += pd.Timedelta(minutes=1)
        elif fault == "source_clock":
            dataset._sequence_source_positions = dataset._sequence_source_positions.copy()
            dataset._sequence_source_positions[1] -= 1
        elif fault == "source_flag":
            dataset._sequence_source_reconstructed = False
        elif fault == "cache":
            dataset._multi_tf_cache_identity_sha256 = "0" * 64
        else:
            dataset.signal_names = dataset.signal_names[::-1]
        return dataset
    monkeypatch.setattr(fixture.native.trainer, "EntryV10CtxDataset", corrupted_dataset)
    with pytest.raises(RuntimeError, match="FEATURE_USEFULNESS_NATIVE_VAL_"):
        with usefulness_audit.open_native_val_inputs(**fixture.arguments):
            pytest.fail("corrupted full population yielded a reader")
    assert not Path(fixture.allocations[0]._m1_feature_tempdir.name).exists()
    assert os.environ[fixture.cache_env] == "prior"


@pytest.mark.parametrize("failure", ("corpus", "dataset", "consumer"))
def test_native_val_reader_mocked_io_failures_restore_environment_and_owned_scratch(native_val_reader_io, monkeypatch, failure):
    fixture = native_val_reader_io
    monkeypatch.delenv(fixture.cache_env, raising=False)
    def refuse_allocation(**arguments):
        assert os.environ[fixture.cache_env] == str(fixture.cache_manifest.parent)
        raise RuntimeError("MECHANICAL_IO_FAILURE")
    if failure == "corpus":
        monkeypatch.setattr(fixture.lifecycle_owner, "UnifiedExitLifecycleCorpus", refuse_allocation)
    elif failure == "dataset":
        monkeypatch.setattr(fixture.native.trainer, "EntryV10CtxDataset", refuse_allocation)
    with pytest.raises(RuntimeError, match="MECHANICAL_IO_FAILURE"):
        with usefulness_audit.open_native_val_inputs(**fixture.arguments):
            raise RuntimeError("MECHANICAL_IO_FAILURE")
    assert fixture.cache_env not in os.environ
    if fixture.allocations:
        assert not Path(fixture.allocations[0]._m1_feature_tempdir.name).exists()


def test_paired_summary_labels_iid_uncertainty_without_independence_claim() -> None:
    from gx1.contracts.entry_exit_feature_usefulness_v1 import (
        STANDARD_ERROR_METHOD,
        _require_summary,
    )

    values = np.array([1.0, 2.0, 4.0, 8.0])
    summary = _paired_summary(values, domain=b"mechanical_summary")
    assert summary["standard_error_method"] == STANDARD_ERROR_METHOD
    assert summary["standard_error"] == pytest.approx(
        np.sqrt(values.var(ddof=1) / len(values))
    )
    _require_summary(summary, count=len(values), label="TEST")
    for method in ("hac", "episode_independent", None):
        invalid = {**summary, "standard_error_method": method}
        with pytest.raises(RuntimeError, match="STANDARD_ERROR_METHOD_INVALID"):
            _require_summary(invalid, count=len(values), label="TEST")
    legacy = dict(summary)
    legacy.pop("standard_error_method")
    with pytest.raises(RuntimeError, match="SUMMARY_SURFACE_INVALID"):
        _require_summary(legacy, count=len(values), label="TEST")


def _exit_fitted_q_iteration() -> dict[str, object]:
    return {
        "schema_version": UNIFIED_EXIT_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION,
        "iteration_index": 4,
        "target_model_state_sha256": "a" * 64,
        "train_split_sha256": "b" * 64,
        "train_fold_sha256": "c" * 64,
        "source_lineage_sha256": "d" * 64,
        "normalization_sha256": "e" * 64,
        "fitted_q_contract": unified_exit_fitted_q_contract(),
        "target_updated_from_val_or_test": False,
    }


def _entry_fitted_q_iteration() -> dict[str, object]:
    exit_iteration = _exit_fitted_q_iteration()
    return {
        "schema_version": ENTRY_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION,
        "iteration_index": 4,
        "entry_target_model_state_sha256": "a" * 64,
        "exit_target_model_state_sha256": "a" * 64,
        "exit_fitted_q_iteration_state_sha256": canonical_json_sha256(
            exit_iteration
        ),
        "train_split_sha256": "b" * 64,
        "train_fold_sha256": "c" * 64,
        "source_lineage_sha256": "d" * 64,
        "normalization_sha256": "e" * 64,
        "entry_fitted_q_contract": entry_fitted_q_contract(),
        "exit_fitted_q_contract": unified_exit_fitted_q_contract(),
        "target_updated_from_val_or_test": False,
    }


def _signal_names() -> tuple[str, ...]:
    return ordered_model_native_signal_fields(
        [
            *MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
            *MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
        ]
    )


@pytest.fixture
def native_baseline_io(tmp_path, monkeypatch):
    """Synthetic full-dimension episodes with explicit mocked data/model admission.

    No TRAIN/VAL artifact or selected candidate is loaded. The existing pack,
    signal, target and report owners still validate their actual contracts.
    """

    import importlib.util
    import torch
    from gx1.contracts.entry_decision_token_v1 import ENTRY_DECISION_TOKEN_DIM
    from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_MTF_CONTEXT_TIMEFRAMES
    from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SEQ_LEN, model_native_signal_contract_metadata
    from gx1.scripts import entry_exit_feature_usefulness_native_v1 as native

    support_path = Path(__file__).with_name("test_entry_exit_feature_usefulness_native_v1.py")
    support_spec = importlib.util.spec_from_file_location("native_baseline_fixture_support", support_path)
    support = importlib.util.module_from_spec(support_spec)
    support_spec.loader.exec_module(support)
    normalization = support.normalization.__wrapped__()
    times = pd.date_range("2026-06-01", periods=4, freq="5min", tz="UTC")
    episodes = {}
    for pair_index, entry in enumerate((0, 2)):
        pack = support._episode(normalization, entry)
        offset = times[entry].value - int(pack["exit_state_row_time_ns"][0])
        for key in pack:
            if "time_ns" in key:
                pack[key] = pack[key] + offset
        pack["episode_index_by_side"] = [pair_index * 2, pair_index * 2 + 1]
        pack["exit_entry_bid_ask"] = np.array([[2000.0, 2000.5], [2000.0, 2000.5]], dtype="<f8")
        episodes[entry] = support._seal_fixture(pack)
    lifecycle = SimpleNamespace(
        split="val", entry_row_count=4, state_population_sha256="a" * 64,
        state_population_rows=2 * UNIFIED_EXIT_EPISODE_SIDE_COUNT * UNIFIED_EXIT_EPISODE_STATE_COUNT,
        _episode_pointers={(entry, side): (entry + side, 0, 2000.0, 2000.5) for entry in episodes for side in (0, 1)},
    )
    states, _indices = _states(tuple(ENTRY_MTF_CONTEXT_TIMEFRAMES))
    rng = np.random.default_rng(20260906)
    states["seq_signal"] = np.repeat(support._surface(normalization, "signal", 8, rng)[:, None], 3, axis=1)
    states["ctx_cont"] = support._surface(normalization, "ctx_cont", 8, rng)
    for alias in support.model_native_context_temporal_alias_policy(_signal_names())["aliases"]:
        states["seq_signal"][:, -1, alias["signal_index"]] = states["ctx_cont"][:, alias["ctx_cont_index"]]
    states["snap_signal"] = states["seq_signal"][:, -1].copy()
    for timeframe in ENTRY_MTF_CONTEXT_TIMEFRAMES:
        lower = timeframe.lower()
        states[f"seq_{lower}"] = np.repeat(support._surface(normalization, f"mtf_{lower}", 8, rng)[:, None], 2, axis=1)
    calls = []
    class DatasetIO:
        indices = np.arange(4, dtype=np.int64)
        signal_names = _signal_names()
        seq_len = MODEL_NATIVE_SEQ_LEN
        per_tf_seq_lens = support._TF_LENGTHS
        _multi_tf_cache_identity_sha256 = "b" * 64
        _unified_exit_lifecycle = lifecycle
        def __len__(self):
            return 4
        def __getitem__(self, index):
            calls.append(("entry", index))
            result = {"entry_row_index": torch.tensor(index, dtype=torch.int64)}
            for surface, value in states.items():
                key = {"seq_signal": "seq_x", "snap_signal": "snap_x"}.get(surface, surface)
                sample = value[index]
                if surface == "seq_signal":
                    sample = np.repeat(sample[-1:], self.seq_len, axis=0)
                elif surface.startswith("seq_"):
                    sample = np.repeat(sample[-1:], self.per_tf_seq_lens[surface[4:].upper()], axis=0)
                result[key] = torch.from_numpy(sample.copy())
            return result
        def materialize_full_exit_episode(self, entry):
            calls.append(("episode", entry))
            return episodes.get(entry)
    dataset = DatasetIO()
    common = _identity(tmp_path)
    metadata = {
        "run_lineage": {"dataset_run_id": common["dataset_run_id"]},
        "unified_exit_training_evidence": {"selected_fitted_q_iteration_state": _exit_fitted_q_iteration()},
        "selected_entry_fitted_q_iteration_state": _entry_fitted_q_iteration(),
    }
    bindings = {
        "bundle_metadata_path": common["selection_artifacts"]["bundle_metadata"]["path"],
        "online_model_state_sha256": common["model_state_sha256"],
        "input_normalization_sha256": common["normalization_contract_sha256"],
        "dataset_artifact_declarations": {"train_parquet": {"sha256": common["train_split_sha256"]}},
        **{key: common[key] for key in (
            "bundle_metadata_sha256", "target_model_state_sha256", "selected_epoch", "last_epoch",
            "selection_artifacts", "recipe_source_provenance",
        )},
    }
    pair = native.SelectedNativeExitPair(model=object(), target_model=object(), metadata=metadata, bindings=bindings)
    signal = model_native_signal_contract_metadata([*MODEL_NATIVE_MANDATORY_SELECTED_FIELDS, *MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS])
    provenance = {
        "entry_row_count": 4, "files": {},
        "entry_split_window": {"emission_start": times[0] - pd.Timedelta(minutes=5), "end": pd.Timestamp("2026-06-02T00:00:00Z")},
        "dataset_contract": {"contract": signal, "splits": {"val": {
            "manifest_path": common["val_manifest_path"], "manifest_sha256": common["val_manifest_sha256"],
            "parquet_path": common["val_data_path"], "parquet_sha256": common["val_data_sha256"],
        }}},
        "lifecycle": {"root_manifest_path": common["lifecycle_manifest_path"], "root_manifest_sha256": common["lifecycle_manifest_sha256"]},
    }
    inputs = usefulness_audit.NativeVALInputs(
        dataset=dataset, corpus=SimpleNamespace(splits={"val": lifecycle}),
        entry_bar_open_time_ns=(times - pd.Timedelta(minutes=5)).asi8,
        entry_decision_time_ns=times.asi8.copy(), provenance=provenance,
    )
    for value, key, domain in (
        (dataset.indices, "entry_indices_sha256", b"native_val_entry_indices"),
        (inputs.entry_bar_open_time_ns, "entry_bar_open_time_ns_sha256", b"native_val_entry_bar_open"),
        (inputs.entry_decision_time_ns, "entry_decision_time_ns_sha256", b"native_val_entry_decision"),
    ):
        provenance[key] = usefulness_audit._array_sha256(value, domain=domain)
    original_forward = usefulness_audit._native_entry_forward
    original_adapter = native.CompactNativeExitUsefulnessAdapter
    def stable_pair(**arguments):
        assert arguments["selected_pair"] is pair
        calls.append(("pair_check",))
    def stable_inputs(observed):
        assert observed.dataset is dataset and observed.corpus is inputs.corpus
        usefulness_audit._require_native_val_population_unchanged(observed)
        calls.append(("input_check",))
    def entry_forward(*, model, batch, device):
        role = "online" if model is pair.model else "teacher"
        calls.append(("entry_forward", role, len(batch["seq_signal"])))
        return (
            np.tile(np.array([1, 2, 3], dtype=np.float32), (len(batch["seq_signal"]), 1)),
            np.full((len(batch["seq_signal"]), ENTRY_DECISION_TOKEN_DIM), 11 if role == "online" else 22, dtype=np.float32),
        )
    class AdapterIO:
        def __init__(self, **arguments):
            assert arguments["model"] is pair.model
        def predict_spec(self, *, episode, online_entry_token):
            assert torch.all(online_entry_token == 11)
            calls.append(("exit_online", episode["entry_row_index"]))
            return np.full(episode["exit_action_valid_mask"].shape, 0.5, dtype=np.float32)
        def baseline_supervision(self, *, episode, target_model, target_entry_token):
            assert target_model is pair.target_model and torch.all(target_entry_token == 22)
            calls.append(("exit_teacher", episode["entry_row_index"]))
            targets = np.broadcast_to(np.array([3, 4], dtype=np.float32), episode["exit_action_valid_mask"].shape).copy()
            valid = episode["exit_action_valid_mask"].copy()
            valid[..., -1, 0] = False
            equivalent = valid & (targets == np.max(np.where(valid, targets, -np.inf), axis=-1, keepdims=True))
            return native.NativeExitSupervision(
                q_targets_bps=targets, action_valid_mask=valid, action_equivalence_mask=equivalent,
                terminal_mask=episode["exit_terminal_mask"].copy(),
                entry_first_side_values_bps=np.array([31, 41], dtype=np.float32) + episode["entry_row_index"],
                entry_side_valid_mask=np.ones(UNIFIED_EXIT_EPISODE_SIDE_COUNT, dtype=np.bool_),
            )
    monkeypatch.setattr(native, "require_selected_native_pair_unchanged", stable_pair)
    monkeypatch.setattr(usefulness_audit, "require_native_val_inputs_unchanged", stable_inputs)
    monkeypatch.setattr(usefulness_audit, "_native_entry_forward", entry_forward)
    monkeypatch.setattr(native, "CompactNativeExitUsefulnessAdapter", AdapterIO)
    shapes = usefulness_audit._native_baseline_shapes(4, 2)
    budget = sum(int(np.prod(shape)) * np.dtype(dtype).itemsize for shape, dtype in shapes.values()) + 2 * np.dtype("<i8").itemsize
    return SimpleNamespace(
        support=support, normalization=normalization, episodes=episodes, lifecycle=lifecycle,
        inputs=inputs, pair=pair, native=native, calls=calls, adapter=AdapterIO,
        original_forward=original_forward, original_adapter=original_adapter,
        arguments={
            "selected_pair": pair, "inputs": inputs, "repo": tmp_path, "device": "cpu",
            "batch_rows": 2, "max_baseline_bytes": budget,
            "max_episode_bytes": support._budget(tuple(episodes.values())),
        },
    )


@pytest.fixture
def native_intervention_io(native_baseline_io, monkeypatch):
    """Exercise complete orchestration with explicitly synthetic forward stubs."""

    fixture = native_baseline_io
    observed = []
    baselines = []
    original_collect = usefulness_audit.collect_native_usefulness_baseline

    def collect(**arguments):
        result = original_collect(**arguments)
        baselines.append(result)
        return result

    class Adapter(fixture.adapter):
        def predict_spec(self, *, episode, online_entry_token, spec=None, donor_episode=None, donor_online_entry_token=None):
            if spec is None:
                return super().predict_spec(episode=episode, online_entry_token=online_entry_token)
            entry, donor = episode["entry_row_index"], donor_episode["entry_row_index"]
            assert entry != donor and {entry, donor} == {0, 2}
            assert (online_entry_token == 11).all() and (donor_online_entry_token == 11).all()
            observed.append((entry, donor, spec["physical_id"]))
            return np.full(episode["exit_action_valid_mask"].shape, 0.75 + entry * 0.25, dtype=np.float32)

    monkeypatch.setattr(usefulness_audit, "collect_native_usefulness_baseline", collect)
    monkeypatch.setattr(fixture.native, "CompactNativeExitUsefulnessAdapter", Adapter)
    layout = feature_usefulness_layout(_signal_names())
    specs = {
        task: tuple(spec for section in (
            "physical_field_perturbations", "family_tf_routes", "local_family_effects",
            "joint_effects", "exit_episode_effects",
        ) for spec in layout["tasks"][task][section])
        for task in ("entry", "exit")
    }
    fixture.audit_arguments = {
        **fixture.arguments, "max_forward_calls": 2 * (2 + len(specs["entry"])) + 2 * (2 + len(specs["exit"])),
        "created": datetime(2026, 9, 6, tzinfo=timezone.utc),
    }
    fixture.interventions = observed
    fixture.baselines = baselines
    fixture.specs = specs
    return fixture


def test_native_interventions_cover_every_spec_and_publish_no_partial_scope(native_intervention_io):
    fixture = native_intervention_io
    report = usefulness_audit.audit_native_feature_usefulness(**fixture.audit_arguments)
    assert require_feature_usefulness_report(report) == report
    assert report["test_rows_read"] is False and report["test_artifacts_read"] == []
    supervision = report["tasks"]["exit"]["supervision"]
    assert supervision["terminal_row_count"] == 0
    assert supervision["single_valid_action_row_count"] == 2 * UNIFIED_EXIT_EPISODE_SIDE_COUNT
    assert fixture.interventions == [
        (entry, donor, spec["physical_id"])
        for entry, donor in ((0, 2), (2, 0)) for spec in fixture.specs["exit"]
    ]
    assert [call[1] for call in fixture.calls if call[0] == "exit_teacher"] == [0, 2]
    assert len([call for call in fixture.calls if call[:2] == ("entry_forward", "teacher")]) == 2
    assert fixture.calls.count(("pair_check",)) == 3
    assert fixture.calls.count(("input_check",)) == 3
    baseline, = fixture.baselines
    for task in ("entry", "exit"):
        width = len(report["tasks"][task]["class_order"])
        output = baseline.arrays[f"{task}_q_bps"].reshape(-1, width)
        target = baseline.arrays[f"{task}_targets_bps"].reshape(-1, width)
        valid = baseline.arrays[f"{task}_valid"].reshape(-1, width)
        equivalent = baseline.arrays[f"{task}_equivalent"].reshape(-1, width)
        row = report["tasks"][task]
        assert row["baseline_outputs_sha256"] == usefulness_audit._array_sha256(
            output.astype("<f8"), domain=f"feature_usefulness_baseline_outputs:{task}\0".encode(),
        )
        target_domain = b"feature_usefulness_entry_fitted_q_target_v1\0" if task == "entry" else b"feature_usefulness_exit_fitted_q_bellman_target_v1\0"
        assert row["supervision"]["q_targets_bps_sha256"] == usefulness_audit._array_sha256(target.astype("<f8"), domain=target_domain)
        assert row["supervision"]["action_valid_cell_count"] == int(valid.sum())
        assert row["supervision"]["target_tied_row_count"] == int((equivalent.sum(axis=1) > 1).sum())
        assert row["coverage"]["complete"] is True and row["coverage"]["omitted_tokens"] == []
        assert row["forward_variant_count"] == 1 + len(fixture.specs[task])
    exit_times = np.repeat(baseline.arrays["exit_decision_time_ns"][:, None], UNIFIED_EXIT_EPISODE_SIDE_COUNT, axis=1).reshape(-1)
    assert report["tasks"]["exit"]["row_times_sha256"] == usefulness_audit._array_sha256(exit_times, domain=b"feature_usefulness_times:exit\0")
    assert all(not value.flags.writeable for value in baseline.arrays.values())


@pytest.mark.parametrize("budget", [0, True, 1])
def test_native_intervention_budget_fails_before_any_forward(native_intervention_io, budget):
    fixture = native_intervention_io
    fixture.audit_arguments["max_forward_calls"] = budget
    with pytest.raises(RuntimeError, match="FEATURE_USEFULNESS_NATIVE_.*BUDGET"):
        usefulness_audit.audit_native_feature_usefulness(**fixture.audit_arguments)
    assert fixture.calls == [] and fixture.baselines == [] and fixture.interventions == []


@pytest.mark.parametrize("fault", ["entry_input", "episode_omitted", "episode_resealed", "clock", "population", "pair"])
def test_native_interventions_reject_mutation_after_baseline(native_intervention_io, monkeypatch, fault):
    fixture = native_intervention_io
    collect = usefulness_audit.collect_native_usefulness_baseline

    def changed_collect(**arguments):
        result = collect(**arguments)
        if fault == "entry_input":
            original = type(fixture.inputs.dataset).__getitem__
            def changed_sample(self, index):
                sample = original(self, index)
                sample["seq_m15"][0, 0] += 0.125
                return sample
            monkeypatch.setattr(type(fixture.inputs.dataset), "__getitem__", changed_sample)
        elif fault == "episode_omitted":
            fixture.episodes.pop(0)
        elif fault == "episode_resealed":
            fixture.episodes[0]["exit_now_reward_bps"] += 0.125
            fixture.episodes[0] = fixture.support._seal_fixture(fixture.episodes[0])
        elif fault == "clock":
            fixture.inputs.entry_decision_time_ns[0] += 1
        elif fault == "population":
            fixture.lifecycle._episode_pointers.pop((0, 1))
        elif fault == "pair":
            def changed_pair(**_arguments):
                raise RuntimeError("NATIVE_EXIT_SELECTED_TEST_MUTATION")
            monkeypatch.setattr(fixture.native, "require_selected_native_pair_unchanged", changed_pair)
        return result

    def forbidden_report(**_arguments):
        pytest.fail("Changed inputs/model must reject before report assembly")

    monkeypatch.setattr(usefulness_audit, "collect_native_usefulness_baseline", changed_collect)
    monkeypatch.setattr(usefulness_audit, "build_feature_usefulness_report", forbidden_report)
    with pytest.raises(RuntimeError, match="FEATURE_USEFULNESS_NATIVE_|NATIVE_EXIT_SELECTED_TEST_MUTATION"):
        usefulness_audit.audit_native_feature_usefulness(**fixture.audit_arguments)


def test_native_baseline_keeps_full_entry_population_and_fixed_teacher_bridge(native_baseline_io):
    fixture = native_baseline_io
    result = usefulness_audit.collect_native_usefulness_baseline(**fixture.arguments)
    np.testing.assert_array_equal(result.arrays["entry_targets_bps"], [[31, 41, 0], [0, 0, 0], [33, 43, 0], [0, 0, 0]])
    np.testing.assert_array_equal(result.arrays["entry_valid"], [[True, True, True], [False, False, True], [True, True, True], [False, False, True]])
    assert result.episode_pack_sha256 == (fixture.episodes[0]["episode_pack_sha256"], None, fixture.episodes[2]["episode_pack_sha256"], None)
    assert all(value is not None for value in result.fill_binding_sha256[::2])
    assert result.fill_binding_sha256[1::2] == (None, None)
    assert len(result.entry_input_sha256) == 4
    assert result.donor_indices.tolist() == [1, 0]
    assert result.identity["native_mtf_geometry_sha256"] == result.donor_plan["native_mtf_geometry_sha256"]
    assert result.retained_array_bytes == fixture.arguments["max_baseline_bytes"]
    assert result.retained_array_bytes == sum(value.nbytes for value in result.arrays.values()) + result.donor_indices.nbytes
    assert all(not value.flags.writeable for value in (*result.arrays.values(), result.donor_indices))
    assert fixture.calls.count(("pair_check",)) == 2
    assert fixture.calls.count(("input_check",)) == 2
    assert [call[1] for call in fixture.calls if call[0] == "entry"] == list(range(4))
    assert [call[1] for call in fixture.calls if call[0] == "episode"] == list(range(4))
    assert [call[1] for call in fixture.calls if call[0] == "exit_online"] == [0, 2]
    assert [call[1] for call in fixture.calls if call[0] == "exit_teacher"] == [0, 2]
    assert (result.arrays["online_tokens"] == 11).all()
    assert (result.arrays["target_tokens"] == 22).all()
    assert (result.arrays["entry_q_bps"].argmax(axis=1) == 2).all()


@pytest.mark.parametrize("fault", ["budget", "index", "omitted", "spurious", "clock", "pack_hash", "missing_side", "population", "bridge", "target_mask", "output_dtype"])
def test_native_baseline_rejects_partial_population_or_changed_inputs(native_baseline_io, monkeypatch, fault):
    fixture = native_baseline_io
    if fault == "budget":
        fixture.arguments["max_baseline_bytes"] -= 1
        def forbidden_allocation(*arguments, **keywords):
            pytest.fail("Baseline budget must reject before payload allocation")
        monkeypatch.setattr(usefulness_audit.np, "empty", forbidden_allocation)
    elif fault == "index":
        fixture.inputs.dataset.indices = np.array([0, 2, 1, 3])
    elif fault == "omitted":
        fixture.episodes.pop(0)
    elif fault == "spurious":
        fixture.episodes[1] = fixture.episodes[0]
    elif fault == "clock":
        fixture.inputs.entry_decision_time_ns[0] += 1
    elif fault == "pack_hash":
        fixture.episodes[0]["exit_local_history_x"][0, 0] += 1
    elif fault == "missing_side":
        fixture.lifecycle._episode_pointers.pop((0, 1))
    elif fault == "population":
        fixture.lifecycle.state_population_rows -= 1
    elif fault == "target_mask":
        original = fixture.adapter.baseline_supervision
        def wrong_mask(instance, **arguments):
            supervision = original(instance, **arguments)
            supervision.action_valid_mask[..., -1, 0] = True
            return supervision
        monkeypatch.setattr(fixture.adapter, "baseline_supervision", wrong_mask)
    elif fault == "bridge":
        original = fixture.adapter.baseline_supervision
        def wrong_bridge(instance, **arguments):
            supervision = original(instance, **arguments)
            supervision.entry_side_valid_mask[0] = False
            return supervision
        monkeypatch.setattr(fixture.adapter, "baseline_supervision", wrong_bridge)
    else:
        original = fixture.adapter.predict_spec
        def wrong_dtype(instance, **arguments):
            return original(instance, **arguments).astype(np.float64)
        monkeypatch.setattr(fixture.adapter, "predict_spec", wrong_dtype)
    expected_error = (
        "FEATURE_USEFULNESS_NATIVE_VAL_POPULATION_CHANGED"
        if fault in {"index", "clock"} else "FEATURE_USEFULNESS_NATIVE_BASELINE"
    )
    with pytest.raises(RuntimeError, match=expected_error):
        usefulness_audit.collect_native_usefulness_baseline(**fixture.arguments)


def test_native_baseline_real_native_model_and_teacher_bridge_on_synthetic_population(native_baseline_io, monkeypatch):
    """Real native arithmetic/assembly, not trained-state or dataset admission."""

    import torch
    from dataclasses import replace
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256

    fixture = native_baseline_io
    model = fixture.support.native_model.__wrapped__(fixture.normalization).eval().requires_grad_(False)
    teacher = copy.deepcopy(model)
    with torch.no_grad():
        teacher.entry_decision_token[1].bias[0].add_(0.01)
    pair = replace(fixture.pair, model=model, target_model=teacher)
    pair.bindings["online_model_state_sha256"] = canonical_model_state_sha256(model.state_dict())
    pair.bindings["target_model_state_sha256"] = canonical_model_state_sha256(teacher.state_dict())
    pair.bindings["input_normalization_sha256"] = fixture.normalization["contract_sha256"]
    iteration = pair.metadata["unified_exit_training_evidence"]["selected_fitted_q_iteration_state"]
    iteration["target_model_state_sha256"] = pair.bindings["target_model_state_sha256"]
    iteration["normalization_sha256"] = pair.bindings["input_normalization_sha256"]
    def mechanical_admission(**arguments):
        assert arguments["selected_pair"] is pair
    monkeypatch.setattr(fixture.native, "require_selected_native_pair_unchanged", mechanical_admission)
    monkeypatch.setattr(usefulness_audit, "_native_entry_forward", fixture.original_forward)
    monkeypatch.setattr(fixture.native, "CompactNativeExitUsefulnessAdapter", fixture.original_adapter)
    result = usefulness_audit.collect_native_usefulness_baseline(**{**fixture.arguments, "selected_pair": pair})
    adapter = fixture.original_adapter(
        model=model, ordered_signal_names=fixture.inputs.dataset.signal_names,
        device="cpu", max_episode_bytes=fixture.arguments["max_episode_bytes"],
    )
    for pair_index, entry in enumerate(result.arrays["eligible_entry_indices"]):
        episode = fixture.episodes[int(entry)]
        online = torch.from_numpy(result.arrays["online_tokens"][entry:entry + 1].copy())
        target = torch.from_numpy(result.arrays["target_tokens"][entry:entry + 1].copy())
        expected_q = adapter.predict_spec(episode=episode, online_entry_token=online)
        expected = adapter.baseline_supervision(episode=episode, target_model=teacher, target_entry_token=target)
        np.testing.assert_array_equal(result.arrays["exit_q_bps"][pair_index], expected_q)
        np.testing.assert_array_equal(result.arrays["exit_targets_bps"][pair_index], expected.q_targets_bps)
        assert episode["exit_action_valid_mask"].all()
        assert not expected.action_valid_mask[..., -1, 0].any()
        assert expected.action_valid_mask[..., :-1, :].all()
        assert not expected.terminal_mask.any()
        np.testing.assert_array_equal(result.arrays["entry_targets_bps"][entry, :2], expected.entry_first_side_values_bps)
    assert not np.array_equal(result.arrays["online_tokens"], result.arrays["target_tokens"])
    assert canonical_model_state_sha256(model.state_dict()) == result.identity["model_state_sha256"]
    assert canonical_model_state_sha256(teacher.state_dict()) == result.identity["target_model_state_sha256"]


@pytest.mark.parametrize("fault", ["row_index", "dtype", "sequence_length", "mtf_length", "alias"])
def test_native_entry_batch_rejects_mapping_or_tensor_mismatch(native_baseline_io, monkeypatch, fault):
    import torch

    fixture = native_baseline_io
    dataset = fixture.inputs.dataset
    original = type(dataset).__getitem__
    def invalid_sample(instance, index):
        result = original(instance, index)
        if fault == "row_index":
            result["entry_row_index"] = torch.tensor(index + 1)
        elif fault == "dtype":
            result["seq_x"] = result["seq_x"].to(torch.float64)
        elif fault == "sequence_length":
            result["seq_x"] = result["seq_x"][:-1]
        elif fault == "mtf_length":
            result["seq_m15"] = result["seq_m15"][:-1]
        else:
            signal_index = _signal_names().index(f"ctx_cont.{MODEL_NATIVE_CTX_CONT_FIELDS[0]}")
            result["snap_x"][signal_index] += 1
        return result
    monkeypatch.setattr(type(dataset), "__getitem__", invalid_sample)
    with pytest.raises(RuntimeError, match="NATIVE_ENTRY_BATCH|ALIAS_SOURCE_OFF_MANIFOLD"):
        usefulness_audit._native_entry_batch(
            dataset, np.array([0, 1]), task_layout=feature_usefulness_layout(_signal_names())["tasks"]["entry"],
        )


def _identity(tmp_path: Path) -> dict[str, object]:
    from gx1.contracts.entry_model_native_train_launch_v1 import (
        TRAINING_RECIPE_SOURCE_PROVENANCE_SCHEMA,
    )

    source = {"owner": {
        "path": str((tmp_path / "source.py").resolve()), "sha256": "9" * 64,
        "size_bytes": 1, "mtime_ns": 1, "device": 1, "inode": 1,
    }}
    selection = {
        role: {"path": str((tmp_path / role).resolve()), "sha256": "1" * 64}
        for role in (
            "session_contract", "active_pointer", "active_state",
            "selected_checkpoint", "bundle_metadata", "recipe_audit",
        )
    }
    selection["bundle_metadata"]["path"] = str((tmp_path / "bundle/bundle_metadata.json").resolve())
    return {
        "bundle_dir": str((tmp_path / "bundle").resolve()),
        "bundle_metadata_sha256": "1" * 64,
        "model_state_sha256": "2" * 64,
        "dataset_dir": str((tmp_path / "dataset").resolve()),
        "dataset_run_id": "FEATURE_USEFULNESS_SYNTHETIC_V1",
        "val_manifest_path": str((tmp_path / "dataset/val.manifest.json").resolve()),
        "val_manifest_sha256": "3" * 64,
        "val_data_path": str((tmp_path / "dataset/val.parquet").resolve()),
        "val_data_sha256": "4" * 64,
        "val_start_utc": "2026-01-01T00:00:00+00:00",
        "val_end_utc": "2026-01-02T00:00:00+00:00",
        "entry_val_population_row_count": 8,
        "exit_val_population_row_count": 8,
        "normalization_path": selection["bundle_metadata"]["path"],
        "normalization_file_sha256": "1" * 64,
        "normalization_contract_sha256": _exit_fitted_q_iteration()["normalization_sha256"],
        "online_entry_token_stream_sha256": "7" * 64,
        "target_entry_token_stream_sha256": "8" * 64,
        "entry_row_indices_sha256": "9" * 64,
        "native_episode_pack_set_sha256": "9" * 64,
        "native_episode_entry_indices_sha256": "9" * 64,
        "native_episode_pair_count": 2,
        "native_mtf_geometry_sha256": canonical_json_sha256({"0": "a" * 64, "1": "a" * 64}),
        "native_episode_pack_contract": unified_exit_episode_pack_contract(),
        "target_model_state_sha256": "a" * 64,
        "selected_epoch": 5,
        "last_epoch": 7,
        "train_split_sha256": "b" * 64,
        "lifecycle_manifest_path": str((tmp_path / "lifecycle.json").resolve()),
        "lifecycle_manifest_sha256": "d" * 64,
        "selection_artifacts": selection,
        "recipe_source_provenance": {
            "schema_version": TRAINING_RECIPE_SOURCE_PROVENANCE_SCHEMA,
            "recipe_audit_path": selection["recipe_audit"]["path"],
            "recipe_audit_sha256": selection["recipe_audit"]["sha256"],
            "source_commit": "0" * 40, "source_bindings": source,
            "source_bindings_sha256": canonical_json_sha256(source),
        },
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "signal_schema_version": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
        "signal_static_contract_sha256": MODEL_NATIVE_STATIC_CONTRACT_SHA256,
        "entry_decision_token_projection": entry_decision_token_projection_metadata(),
    }


def _states(
    timeframes: tuple[str, ...], *, full_exit_population: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    names = _signal_names()
    state_count = UNIFIED_EXIT_EPISODE_STATE_COUNT if full_exit_population else 2
    rows = 2 * UNIFIED_EXIT_EPISODE_SIDE_COUNT * state_count
    block_sign = np.repeat(
        np.array([1.0, 1.0, -1.0, -1.0] if full_exit_population else [1.0, -1.0, 1.0, -1.0]),
        state_count,
    )
    row = np.arange(rows, dtype=np.float32)
    seq = np.empty((rows, 3, len(names)), dtype=np.float32)
    for index in range(len(names)):
        seq[:, :, index] = (
            row[:, None] * np.float32(0.01 + index / 10000.0)
            + np.arange(3, dtype=np.float32)[None, :] * 0.001
            + np.float32(index / 100.0)
        )
    useful_index = names.index("_v1_atr14")
    noise_index = names.index("atr_z")
    seq[:, :, useful_index] = block_sign[:, None] * np.array(
        [1.6, 1.8, 2.0], dtype=np.float32
    )[None, :]
    snap = seq[:, -1, :].copy()

    ctx = np.empty((rows, len(MODEL_NATIVE_CTX_CONT_FIELDS)), dtype=np.float32)
    for index, field in enumerate(MODEL_NATIVE_CTX_CONT_FIELDS):
        if field in CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS:
            domain = CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS[field]
            ctx[:, index] = np.asarray(
                [domain[position % len(domain)] for position in range(rows)],
                dtype=np.float32,
            )
        else:
            ctx[:, index] = row * np.float32(0.03 + index / 1000.0)
        signal_index = names.index(f"ctx_cont.{field}")
        seq[:, -1, signal_index] = ctx[:, index]
        snap[:, signal_index] = ctx[:, index]
    ctx_cat = np.column_stack(
        [
            np.asarray(
                [domain[position % len(domain)] for position in range(rows)],
                dtype=np.int64,
            )
            for domain in MODEL_NATIVE_CTX_CAT_DOMAINS.values()
        ]
    )

    routing = require_multi_tf_specialist_routing_v4(MULTI_TF_PER_BAR_FEATURES_V4)
    trend_index = routing["trend_ema_encoder"][0]
    momentum_index = routing["momentum_flow_encoder"][0]
    mtf_index = {name: index for index, name in enumerate(MULTI_TF_PER_BAR_FEATURES_V4)}
    states: dict[str, np.ndarray] = {
        "seq_signal": seq,
        "snap_signal": snap,
        "ctx_cont": ctx,
        "ctx_cat": ctx_cat,
    }
    for timeframe in timeframes:
        values = np.empty(
            (rows, 2, len(MULTI_TF_PER_BAR_FEATURES_V4)), dtype=np.float32
        )
        for index in range(len(MULTI_TF_PER_BAR_FEATURES_V4)):
            values[:, :, index] = (
                row[:, None] * np.float32(0.02 + index / 10000.0)
                + np.arange(2, dtype=np.float32)[None, :] * 0.002
            )
        for field, domain in MTF_SEMANTIC_CATEGORICAL_DOMAINS.items():
            index = mtf_index[field]
            values[:, :, index] = np.asarray(
                [domain[position % len(domain)] for position in range(rows)],
                dtype=np.float32,
            )[:, None]
        if timeframe == "H1":
            values[:, :, trend_index] = block_sign[:, None]
            values[:, :, momentum_index] = block_sign[:, None]
        elif timeframe == "M15":
            values[:, :, momentum_index] = block_sign[:, None]
        states[f"seq_{timeframe.lower()}"] = values
    if "M5" in timeframes:
        episode = np.repeat(np.array([0, 1], dtype=np.int64), 2 * state_count)
        state_index = np.tile(np.arange(state_count, dtype=np.int64), 4)
        side_index = np.tile(np.repeat(np.array([0, 1], dtype=np.int64), state_count), 2)
        token = np.zeros(
            (rows, UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM), dtype=np.float32
        )
        token[episode == 0, :] = 0.4
        token[episode == 1, :] = -0.3
        path = np.zeros(
            (rows, 4, len(UNIFIED_EXIT_PATH_FEATURE_ORDER)), dtype=np.float32
        )
        path[:, :, :] = row[:, None, None] * 0.01
        path[:, :, 0] += np.arange(4, dtype=np.float32)[None, :] * 0.1
        states.update(
            {
                "entry_decision_representation": token,
                "exit_path": path,
                "exit_path_lengths": np.resize(np.array([2, 3], dtype=np.int64), rows),
                "exit_side_index": side_index,
                "exit_episode_index": episode,
                "exit_state_index": state_index,
            }
        )
    return states, {
        "useful_index": useful_index,
        "noise_index": noise_index,
        "trend_index": trend_index,
        "momentum_index": momentum_index,
    }


class _SyntheticPredictor:
    def __init__(self, *, task: str, indices: dict[str, int], names: tuple[str, ...]):
        self.task = task
        self.indices = indices
        self.names = names
        self.calls = 0
        self.alias_signal_indices = np.asarray(
            [names.index(f"ctx_cont.{field}") for field in MODEL_NATIVE_CTX_CONT_FIELDS],
            dtype=np.int64,
        )
        self.alias_ctx_indices = np.arange(
            len(MODEL_NATIVE_CTX_CONT_FIELDS), dtype=np.int64
        )

    def __call__(self, states: dict[str, np.ndarray]) -> np.ndarray:
        self.calls += 1
        np.testing.assert_array_equal(
            states["seq_signal"][:, -1, self.alias_signal_indices],
            states["snap_signal"][:, self.alias_signal_indices],
        )
        np.testing.assert_array_equal(
            states["snap_signal"][:, self.alias_signal_indices],
            states["ctx_cont"][:, self.alias_ctx_indices],
        )
        for index, field in enumerate(MODEL_NATIVE_CTX_CAT_FIELDS):
            assert np.isin(states["ctx_cat"][:, index], MODEL_NATIVE_CTX_CAT_DOMAINS[field]).all()
        local = states["snap_signal"][:, self.indices["useful_index"]]
        interaction = (
            states["seq_h1"][:, -1, self.indices["trend_index"]]
            * states["seq_h1"][:, -1, self.indices["momentum_index"]]
        )
        score = 2.5 * local + 0.75 * interaction
        if self.task == "entry":
            return np.column_stack([score, -score, np.zeros_like(score)])
        score = (
            score
            + 0.3 * states["entry_decision_representation"][:, 0]
            + 0.2 * states["exit_path"][:, 0, 0]
            + 0.4 * (0.5 - states["exit_side_index"])
        )
        return np.column_stack([score, -score])


def _audit_task(
    tmp_path: Path, task: str, *, row_times: pd.DatetimeIndex | None = None,
    batch_rows: int | None = None,
    full_exit_population: bool = False,
) -> dict[str, object]:
    names = _signal_names()
    layout = feature_usefulness_layout(names)
    timeframes = tuple(layout["tasks"][task]["timeframes"])
    states, indices = _states(timeframes, full_exit_population=full_exit_population)
    predictor = _SyntheticPredictor(task=task, indices=indices, names=names)
    useful = states["snap_signal"][:, indices["useful_index"]]
    if row_times is None:
        row_times = pd.date_range(
            "2026-01-01", periods=len(useful), freq="min" if full_exit_population else "h", tz="UTC"
        )
        if task == "exit":
            row_times = row_times.take(
                states["exit_episode_index"] * (UNIFIED_EXIT_EPISODE_STATE_COUNT if full_exit_population else 2)
                + states["exit_state_index"]
            )
    common = dict(
        task=task,
        ordered_signal_names=names,
        identity=_identity(tmp_path),
        states=states,
        row_times=row_times,
        row_splits=["val"] * len(useful),
        block_ids=np.repeat(["day0", "day1", "day2", "day3"], 2),
        within_block_positions=np.tile([0, 1], 4),
        predictor=predictor,
        batch_rows=len(useful) if batch_rows is None else batch_rows,
    )
    baseline_q_bps = predictor(states)
    if full_exit_population:
        common["identity"]["exit_val_population_row_count"] = len(useful)
        common["block_ids"] = states["exit_episode_index"].astype(str)
        common["within_block_positions"] = (
            states["exit_side_index"] * UNIFIED_EXIT_EPISODE_STATE_COUNT + states["exit_state_index"]
        )
        common["native_mtf_geometry_by_block"] = {"0": "a" * 64, "1": "a" * 64}
    if task == "entry":
        q_targets = np.asarray(baseline_q_bps, dtype=np.float64).copy()
        action_valid = np.ones((len(useful), 3), dtype=np.bool_)
        q_targets[0] = [0.0, 0.0, 0.0]
        equivalence = action_valid & np.equal(
            q_targets,
            np.max(
                np.where(action_valid, q_targets, -np.inf),
                axis=1,
                keepdims=True,
            ),
        )
        return audit_task_feature_usefulness(
            **common,
            entry_action_q_target_bps=q_targets,
            entry_action_valid_mask=action_valid,
            entry_action_equivalence_mask=equivalence,
            entry_fitted_q_iteration_state=_entry_fitted_q_iteration(),
            exit_fitted_q_iteration_state=_exit_fitted_q_iteration(),
        )
    q_targets = np.asarray(baseline_q_bps, dtype=np.float64).copy()
    action_valid = np.ones((len(useful), 2), dtype=np.bool_)
    terminal = states["exit_state_index"] == (UNIFIED_EXIT_EPISODE_STATE_COUNT - 1 if full_exit_population else 1)
    action_valid[terminal, 0] = False
    q_targets[0] = [0.0, 0.0]
    equivalence = action_valid & np.equal(
        q_targets,
        np.max(np.where(action_valid, q_targets, -np.inf), axis=1, keepdims=True),
    )
    return audit_task_feature_usefulness(
        **common,
        exit_action_q_target_bps=q_targets,
        exit_action_valid_mask=action_valid,
        exit_action_equivalence_mask=equivalence,
        exit_terminal_mask=terminal,
        exit_fitted_q_iteration_state=_exit_fitted_q_iteration(),
    )


@pytest.mark.parametrize("task", ["entry", "exit"])
@pytest.mark.parametrize("batch_rows", [1, 3])
def test_task_metric_stream_preserves_vectors_across_batch_boundaries(
    tmp_path: Path, task: str, batch_rows: int,
) -> None:
    reference = _audit_task(tmp_path, task)
    streamed = _audit_task(tmp_path, task, batch_rows=batch_rows)
    reference_calls = reference.pop("_forward_batch_calls")
    streamed_calls = streamed.pop("_forward_batch_calls")
    assert reference_calls == reference["forward_variant_count"]
    assert streamed_calls == reference_calls * ((8 + batch_rows - 1) // batch_rows)

    def compare(actual: object, expected: object) -> None:
        if isinstance(expected, dict):
            assert isinstance(actual, dict)
            assert actual.keys() == expected.keys()
            for key in expected:
                compare(actual[key], expected[key])
        elif isinstance(expected, float):
            assert actual == pytest.approx(expected, rel=32 * np.finfo(np.float64).eps, abs=0)
        else:
            assert actual == expected

    compare(streamed, reference)


@pytest.fixture(scope="module")
def usefulness_report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Synthetic report arithmetic at full native state count, not native inference.

    Short mock histories and file-binding placeholders do not establish a
    trained-pair or dataset result. Actual native forwards have separate tests.
    """

    tmp_path = tmp_path_factory.mktemp("feature_usefulness")
    entry = _audit_task(tmp_path, "entry")
    exit_task = _audit_task(tmp_path, "exit", full_exit_population=True, batch_rows=256)
    identity = _identity(tmp_path)
    identity["exit_val_population_row_count"] = exit_task["row_count"]
    return build_feature_usefulness_report(
        identity=identity,
        ordered_signal_names=_signal_names(),
        entry_task=entry,
        exit_task=exit_task,
    )


@pytest.mark.parametrize("path_kind", ["absolute", "relative", "home"])
def test_immutable_report_publishes_complete_payload_after_file_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    usefulness_report: dict[str, object],
    path_kind: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    out = (tmp_path / "reports" / "usefulness.json").resolve()
    path = {
        "absolute": out,
        "relative": Path("reports/usefulness.json"),
        "home": Path("~/reports/usefulness.json"),
    }[path_kind]
    expected = (
        json.dumps(usefulness_report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    original_fsync = prediction_evidence.os.fsync
    original_link = prediction_evidence.os.link
    synced_kinds: list[str] = []

    def record_fsync(descriptor: int) -> None:
        original_fsync(descriptor)
        mode = prediction_evidence.os.fstat(descriptor).st_mode
        synced_kinds.append("file" if stat.S_ISREG(mode) else "directory")

    def publish_complete_report(source: Path, destination: Path) -> None:
        assert destination == out
        assert not out.exists()
        assert source.parent == out.parent
        assert source.read_bytes() == expected
        assert synced_kinds == ["file"]
        original_link(source, destination)

    monkeypatch.setattr(prediction_evidence.os, "fsync", record_fsync)
    monkeypatch.setattr(prediction_evidence.os, "link", publish_complete_report)
    assert write_immutable_feature_usefulness_report(path, usefulness_report) == out
    assert out.read_bytes() == expected
    assert json.loads(out.read_text(encoding="utf-8")) == usefulness_report
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert synced_kinds == ["file", "directory"]
    assert list(out.parent.iterdir()) == [out]


def test_immutable_report_duplicate_preserves_existing_output(
    tmp_path: Path,
    usefulness_report: dict[str, object],
) -> None:
    out = write_immutable_feature_usefulness_report(
        tmp_path / "usefulness.json", usefulness_report
    )
    retained = out.read_bytes()
    retained_inode = out.stat().st_ino
    with pytest.raises(RuntimeError, match="FEATURE_USEFULNESS_OUTPUT_EXISTS"):
        write_immutable_feature_usefulness_report(out, usefulness_report)
    assert out.read_bytes() == retained
    assert out.stat().st_ino == retained_inode
    assert list(tmp_path.iterdir()) == [out]


@pytest.mark.parametrize("failure_point", ["write", "file_fsync", "link"])
def test_immutable_report_prepublication_failure_never_exposes_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    usefulness_report: dict[str, object],
    failure_point: str,
) -> None:
    out = (tmp_path / "reports" / "usefulness.json").resolve()

    def fail_before_publication(*args: object) -> None:
        assert not out.exists()
        assert not out.is_symlink()
        raise OSError("injected prepublication failure")

    if failure_point == "write":
        original_open = Path.open

        def fail_during_write(path: Path, mode: str = "r", **kwargs: object):
            if mode == "w" and path.parent == out.parent:
                with original_open(path, mode, **kwargs) as handle:
                    handle.write("{")
                    handle.flush()
                    fail_before_publication()
            return original_open(path, mode, **kwargs)

        monkeypatch.setattr(Path, "open", fail_during_write)
    else:
        monkeypatch.setattr(
            prediction_evidence.os,
            "fsync" if failure_point == "file_fsync" else "link",
            fail_before_publication,
        )
    with pytest.raises(OSError, match="injected prepublication failure"):
        write_immutable_feature_usefulness_report(out, usefulness_report)
    assert not out.exists()
    assert list(out.parent.iterdir()) == []


def test_immutable_report_directory_fsync_failure_retains_complete_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    usefulness_report: dict[str, object],
) -> None:
    out = (tmp_path / "usefulness.json").resolve()
    original_fsync = prediction_evidence.os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(prediction_evidence.os.fstat(descriptor).st_mode):
            assert json.loads(out.read_text(encoding="utf-8")) == usefulness_report
            raise OSError("injected directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(prediction_evidence.os, "fsync", fail_directory_fsync)
    with pytest.raises(OSError, match="injected directory fsync failure"):
        write_immutable_feature_usefulness_report(out, usefulness_report)
    assert json.loads(out.read_text(encoding="utf-8")) == usefulness_report
    assert list(tmp_path.iterdir()) == [out]


def test_immutable_report_publication_race_preserves_competing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    usefulness_report: dict[str, object],
) -> None:
    out = (tmp_path / "usefulness.json").resolve()
    retained = b"already-published output\n"
    original_link = prediction_evidence.os.link

    def publish_competing_output(source: Path, destination: Path) -> None:
        destination.write_bytes(retained)
        original_link(source, destination)

    monkeypatch.setattr(prediction_evidence.os, "link", publish_competing_output)
    with pytest.raises(FileExistsError):
        write_immutable_feature_usefulness_report(out, usefulness_report)
    assert out.read_bytes() == retained
    assert list(tmp_path.iterdir()) == [out]


@pytest.mark.parametrize("target_exists", [False, True])
def test_immutable_report_rejects_leaf_symlink_before_resolution(
    tmp_path: Path,
    usefulness_report: dict[str, object],
    target_exists: bool,
) -> None:
    target = tmp_path / "target" / "usefulness.json"
    retained = b"retained symlink target\n"
    if target_exists:
        target.parent.mkdir()
        target.write_bytes(retained)
    out = tmp_path / "usefulness.json"
    out.symlink_to(target)
    with pytest.raises(RuntimeError, match="FEATURE_USEFULNESS_OUTPUT_EXISTS"):
        write_immutable_feature_usefulness_report(out, usefulness_report)
    assert out.is_symlink()
    assert out.readlink() == target
    if target_exists:
        assert target.read_bytes() == retained
        assert list(target.parent.iterdir()) == [target]
    else:
        assert not target.parent.exists()


@pytest.mark.parametrize("target_exists", [False, True])
def test_immutable_report_preserves_parent_symlink_resolution(
    tmp_path: Path,
    usefulness_report: dict[str, object],
    target_exists: bool,
) -> None:
    target = tmp_path / "target"
    if target_exists:
        target.mkdir()
    parent = tmp_path / "reports"
    parent.symlink_to(target, target_is_directory=True)
    out = write_immutable_feature_usefulness_report(
        parent / "usefulness.json", usefulness_report
    )
    assert out == (target / "usefulness.json").resolve()
    assert parent.is_symlink()
    assert parent.readlink() == target
    assert json.loads(out.read_text(encoding="utf-8")) == usefulness_report
    assert list(target.iterdir()) == [out]


def test_layout_covers_every_logical_field_and_route_without_threshold() -> None:
    layout = feature_usefulness_layout(_signal_names())
    local_count = len(_signal_names())
    ctx_cont_count = len(MODEL_NATIVE_CTX_CONT_FIELDS)
    ctx_cat_count = len(MODEL_NATIVE_CTX_CAT_FIELDS)
    mtf_width = len(MULTI_TF_PER_BAR_FEATURES_V4)
    family_count = 8
    family_pairs = family_count * (family_count - 1) // 2
    for task, timeframe_count, exit_effect_count in (
        ("entry", 4, 0),
        ("exit", 5, 3),
    ):
        mtf_count = mtf_width * timeframe_count
        interaction_count = (
            family_pairs
            + timeframe_count * family_pairs
            + family_count * timeframe_count * (timeframe_count - 1) // 2
        )
        assert layout["tasks"][task]["coverage_counts"] == {
            "local_signal": local_count,
            "ctx_cont": ctx_cont_count,
            "ctx_cat": ctx_cat_count,
            "mtf_fields": mtf_count,
            "physical_field_perturbations": local_count + ctx_cat_count + mtf_count,
            "family_tf_routes": family_count * timeframe_count,
            "local_family_effects": family_count,
            "joint_interaction_effects": interaction_count,
            "interaction_synergy": interaction_count,
            "exit_episode_effects": exit_effect_count,
        }
    assert POLICY["automatic_importance_threshold"] is None
    assert POLICY["automatic_top_k"] is None
    assert POLICY["retirement_authority"] is False


def test_interactions_are_exhaustive_disjoint_owner_pairs() -> None:
    layout = feature_usefulness_layout(_signal_names())
    expected_kinds = {
        "entry": {
            "local_cross_family": 28,
            "per_tf_cross_family": 112,
            "cross_tf_same_family": 48,
        },
        "exit": {
            "local_cross_family": 28,
            "per_tf_cross_family": 140,
            "cross_tf_same_family": 80,
        },
    }
    for task in ("entry", "exit"):
        task_layout = layout["tasks"][task]
        effects = {
            row["physical_id"]: row
            for section in ("local_family_effects", "family_tf_routes")
            for row in task_layout[section]
        }
        observed = {kind: 0 for kind in expected_kinds[task]}
        for row in task_layout["interaction_synergy"]:
            assert row["formula"] == "joint_delta-left_delta-right_delta"
            observed[row["kind"]] += 1
            left = effects[row["left_effect_id"]]
            right = effects[row["right_effect_id"]]
            left_tokens = {
                (target["surface"], index)
                for target in left["targets"]
                for index in target["source_indices"]
            }
            right_tokens = {
                (target["surface"], index)
                for target in right["targets"]
                for index in target["source_indices"]
            }
            assert left_tokens.isdisjoint(right_tokens)
        assert observed == expected_kinds[task]


def test_synthetic_useful_noise_and_interaction_are_measured(
    usefulness_report: dict[str, object],
) -> None:
    checked = require_feature_usefulness_report(usefulness_report)
    assert checked["decision"] == DECISION
    for task in ("entry", "exit"):
        metrics = checked["tasks"][task]["logical_field_metrics"]["local_signal"]
        useful = metrics["local_signal._v1_atr14"]
        noise = metrics["local_signal.atr_z"]
        assert useful["paired_loss_delta"]["mean"] > 0.0
        assert useful["paired_margin_delta"]["mean"] > 0.0
        assert noise["paired_loss_delta"]["mean"] == 0.0
        assert noise["paired_margin_delta"]["mean"] == 0.0
        synergy = checked["tasks"][task]["interaction_synergy_metrics"][
            "interaction.h1.trend_ema_encoder__x__momentum_flow_encoder"
        ]
        assert synergy["formula"] == "joint_delta-left_delta-right_delta"
        assert synergy["paired_loss_delta"]["sample_variance"] > 0.0
        assert synergy["paired_margin_delta"]["sample_variance"] > 0.0
        assert (
            synergy["paired_loss_delta"]["positive_count"]
            + synergy["paired_loss_delta"]["negative_count"]
        ) > 0
    exit_episode = checked["tasks"]["exit"]["exit_episode_effect_metrics"]
    assert set(exit_episode) == {
        "exit_episode.frozen_entry_decision_token",
        "exit_episode.path_and_observed_length",
        "exit_episode.side_entry_binding",
    }
    assert checked["tasks"]["exit"]["side_pair_plan"]["opposite_side"] is True
    for metric in exit_episode.values():
        assert (
            metric["paired_loss_delta"]["positive_count"]
            + metric["paired_loss_delta"]["negative_count"]
        ) > 0


def test_supervision_binds_entry_and_exit_frozen_fitted_q_iterations(
    usefulness_report: dict[str, object],
) -> None:
    entry = usefulness_report["tasks"]["entry"]
    exit_task = usefulness_report["tasks"]["exit"]
    assert entry["comparison_surface"] == (
        "raw_entry_action_q_bps_valid_action_masked_mse_and_unique_target_q_margin"
    )
    assert exit_task["comparison_surface"] == (
        "raw_exit_action_q_bps_frozen_fitted_q_bellman_target_masked_mse_and_unique_target_q_margin"
    )
    entry_supervision = entry["supervision"]
    exit_supervision = exit_task["supervision"]
    assert entry_supervision["schema_version"] == ENTRY_FITTED_Q_SCHEMA_VERSION
    assert exit_supervision["schema_version"] == UNIFIED_EXIT_FITTED_Q_SCHEMA_VERSION
    assert entry_supervision["target_tied_row_count"] == 1
    assert exit_supervision["target_tied_row_count"] == 1
    assert entry_supervision["margin_valid_row_count"] < entry_supervision[
        "loss_valid_row_count"
    ]
    assert exit_supervision["margin_valid_row_count"] < exit_supervision[
        "loss_valid_row_count"
    ]
    assert entry_supervision["exit_fitted_q_iteration_state_sha256"] == (
        exit_supervision["fitted_q_iteration_state_sha256"]
    )


def test_masked_q_mse_and_unique_target_margin_exclude_ties() -> None:
    predicted_q = np.array([[1.0, 1.0], [0.4, -0.1], [7.0, 8.0]])
    q_targets = np.array([[1.0, 1.0], [2.0, 1.0], [7.0, 8.0]])
    valid = np.array([[True, True], [True, True], [False, True]])
    equivalent = np.array([[True, True], [True, False], [False, True]])
    loss, margin, margin_mask = _fitted_q_loss_and_unique_target_margin(
        predicted_q,
        q_targets_bps=q_targets,
        action_valid_mask=valid,
        action_equivalence_mask=equivalent,
    )
    assert loss[0] == 0.0
    assert loss[1] == pytest.approx(((0.4 - 2.0) ** 2 + (-0.1 - 1.0) ** 2) / 2.0)
    assert loss[2] == 0.0
    assert margin_mask.tolist() == [False, True, False]
    assert margin.tolist() == [pytest.approx(0.5)]
    invalid = equivalent.copy()
    invalid[1] = [False, True]
    with pytest.raises(RuntimeError, match="FITTED_Q_EQUIVALENCE_INVALID"):
        _fitted_q_loss_and_unique_target_margin(
            predicted_q,
            q_targets_bps=q_targets,
            action_valid_mask=valid,
            action_equivalence_mask=invalid,
        )

    entry_predicted = np.array([[1.5, 0.25, -0.5], [0.0, 0.0, 0.0]])
    entry_targets = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    entry_valid = np.ones((2, 3), dtype=np.bool_)
    entry_equivalent = np.array([[True, False, False], [True, True, True]])
    entry_loss, entry_margin, entry_margin_mask = (
        _fitted_q_loss_and_unique_target_margin(
            entry_predicted,
            q_targets_bps=entry_targets,
            action_valid_mask=entry_valid,
            action_equivalence_mask=entry_equivalent,
        )
    )
    assert entry_loss.tolist() == pytest.approx(
        [((1.5 - 2.0) ** 2 + (0.25 - 1.0) ** 2 + (-0.5) ** 2) / 3.0, 0.0]
    )
    assert entry_margin_mask.tolist() == [True, False]
    assert entry_margin.tolist() == pytest.approx([1.25])


def test_iteration_state_or_val_test_target_update_mutation_fails_closed(
    usefulness_report: dict[str, object],
) -> None:
    entry_mutation = copy.deepcopy(usefulness_report)
    entry_mutation["tasks"]["entry"]["supervision"][
        "fitted_q_iteration_state"
    ]["target_updated_from_val_or_test"] = True
    with pytest.raises(RuntimeError, match="ENTRY_FITTED_Q"):
        require_feature_usefulness_report(entry_mutation)

    exit_mutation = copy.deepcopy(usefulness_report)
    exit_mutation["tasks"]["exit"]["supervision"][
        "fitted_q_iteration_state"
    ]["target_updated_from_val_or_test"] = True
    with pytest.raises(RuntimeError, match="UNIFIED_EXIT_FITTED_Q"):
        require_feature_usefulness_report(exit_mutation)

    split_brain = copy.deepcopy(usefulness_report)
    entry_supervision = split_brain["tasks"]["entry"]["supervision"]
    alternate_exit = entry_supervision["exit_fitted_q_iteration_state"]
    alternate_exit["iteration_index"] = 5
    alternate_exit["target_model_state_sha256"] = "f" * 64
    alternate_exit_sha = canonical_json_sha256(alternate_exit)
    alternate_entry = entry_supervision["fitted_q_iteration_state"]
    alternate_entry["iteration_index"] = 5
    alternate_entry["entry_target_model_state_sha256"] = "f" * 64
    alternate_entry["exit_target_model_state_sha256"] = "f" * 64
    alternate_entry["exit_fitted_q_iteration_state_sha256"] = alternate_exit_sha
    entry_supervision["exit_fitted_q_iteration_state_sha256"] = alternate_exit_sha
    entry_supervision["fitted_q_iteration_state_sha256"] = canonical_json_sha256(
        alternate_entry
    )
    with pytest.raises(RuntimeError, match="ITERATION_SPLIT_BRAIN"):
        require_feature_usefulness_report(split_brain)


def test_exit_frozen_token_and_same_state_side_pair_fail_closed() -> None:
    timeframes = tuple(
        feature_usefulness_layout(_signal_names())["tasks"]["exit"]["timeframes"]
    )
    states, _indices = _states(timeframes)
    pair, plan = _build_exit_side_pair_plan(states)
    assert np.array_equal(pair[pair], np.arange(len(pair)))
    assert plan["same_episode_state"] is True

    changed_token = {name: value.copy() for name, value in states.items()}
    changed_token["entry_decision_representation"][1, 0] += 1.0
    with pytest.raises(RuntimeError, match="FROZEN_TOKEN_NOT_EPISODE_IMMUTABLE"):
        _build_exit_side_pair_plan(changed_token)

    missing_side = {name: value.copy() for name, value in states.items()}
    missing_side["exit_side_index"][2] = 0
    with pytest.raises(RuntimeError, match="SIDE_PAIR_DUPLICATE"):
        _build_exit_side_pair_plan(missing_side)


def test_alias_and_categorical_perturbations_use_genuine_rows(
    usefulness_report: dict[str, object],
) -> None:
    layout = feature_usefulness_layout(_signal_names())
    alias = next(
        row
        for row in layout["tasks"]["entry"]["physical_field_perturbations"]
        if row["physical_id"].startswith("temporal_alias.")
    )
    assert alias["manifold"] == (
        "genuine_val_joint_seq_snap_ctx_temporal_alias_block_swap"
    )
    assert {target["surface"] for target in alias["targets"]} == {
        "seq_signal",
        "snap_signal",
        "ctx_cont",
    }
    assert usefulness_report["perturbation_policy"]["categorical"] == (
        "swap_observed_valid_category_never_synthesize"
    )
    episode = {
        row["token"]: row
        for row in layout["tasks"]["exit"]["exit_episode_effects"]
    }
    assert episode["exit_episode.frozen_entry_decision_token"]["targets"] == [
        {
            "surface": "entry_decision_representation",
            "source_indices": list(range(UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM)),
        }
    ]
    assert episode["exit_episode.path_and_observed_length"]["targets"][-1] == {
        "surface": "exit_path_lengths",
        "whole_surface": True,
    }
    side_binding = episode["exit_episode.side_entry_binding"]
    assert side_binding["donor_kind"] == "same_state_opposite_side"
    assert {target["surface"] for target in side_binding["targets"]} == {
        "entry_decision_representation",
        "exit_path",
        "exit_path_lengths",
        "exit_side_index",
    }


def test_native_donors_preserve_combined_geometry_without_omitting_rows() -> None:
    blocks = np.repeat(["first", "second", "third", "fourth"], 2)
    positions = np.tile([0, 1], 4)
    geometry = {"first": "a" * 64, "second": "b" * 64, "third": "a" * 64, "fourth": "b" * 64}
    donor, plan = build_structure_preserving_donor_plan(
        block_ids=blocks, within_block_positions=positions,
        native_mtf_geometry_by_block=geometry,
    )
    assert donor.tolist() == [4, 5, 6, 7, 0, 1, 2, 3]
    assert np.array_equal(positions, positions[donor])
    assert plan["row_count"] == len(blocks)
    assert plan["signature_group_count"] == 2
    assert plan["native_mtf_geometry_sha256"] == canonical_json_sha256(geometry)
    assert plan["source_fields"] == ["structure_block_id", "within_block_position", "native_mtf_geometry"]


def test_compact_native_plans_match_full_state_row_hashes() -> None:
    from gx1.contracts.unified_exit_episode_pack_v1 import UNIFIED_EXIT_EPISODE_STATE_COUNT
    from gx1.models.entry_v10.direction_decision_contract import UNIFIED_EXIT_SIDE_ORDER

    entries = np.array([2, 4, 7, 11], dtype=np.int64)
    geometry = {"2": "a" * 64, "4": "b" * 64, "7": "a" * 64, "11": "b" * 64}
    donor, compact_plan, compact_side = build_native_exit_structure_plans(
        entry_row_indices=entries, native_mtf_geometry_by_block=geometry,
    )
    state_count = UNIFIED_EXIT_EPISODE_STATE_COUNT
    side_count = len(UNIFIED_EXIT_SIDE_ORDER)
    block_rows = state_count * side_count
    blocks = np.repeat(entries.astype(str), block_rows)
    full_donor, full_plan = build_structure_preserving_donor_plan(
        block_ids=blocks,
        within_block_positions=np.tile(np.arange(block_rows), len(entries)),
        native_mtf_geometry_by_block=geometry,
    )
    np.testing.assert_array_equal(
        full_donor.reshape(len(entries), block_rows),
        donor[:, None] * block_rows + np.arange(block_rows)[None, :],
    )
    assert compact_plan == full_plan
    states = {
        "exit_episode_index": np.repeat(entries, block_rows),
        "exit_state_index": np.tile(np.arange(state_count), len(entries) * side_count),
        "exit_side_index": np.tile(np.repeat(np.arange(side_count), state_count), len(entries)),
        "entry_decision_representation": np.zeros((len(blocks), UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM)),
    }
    _pair, full_side = _build_exit_side_pair_plan(states)
    assert compact_side == full_side
    assert donor.shape == entries.shape


@pytest.mark.parametrize("entries", [[], [0], [0, 0], [2, 1], [-1, 0], [0.0, 1.0], [False, True], np.array([0, 2**63], dtype=np.uint64)])
def test_compact_native_plans_reject_invalid_entry_identity(entries) -> None:
    with pytest.raises(RuntimeError, match="NATIVE_ENTRY_ROWS_INVALID"):
        build_native_exit_structure_plans(entry_row_indices=entries, native_mtf_geometry_by_block={})


@pytest.mark.parametrize("mutation", ["missing", "extra", "invalid", "singleton"])
def test_native_donor_geometry_rejects_missing_or_unpairable_blocks(mutation: str) -> None:
    geometry = {"first": "a" * 64, "second": "a" * 64}
    if mutation == "missing":
        geometry.pop("first")
    elif mutation == "extra":
        geometry["third"] = "a" * 64
    elif mutation == "invalid":
        geometry["first"] = "not-a-geometry-hash"
    else:
        geometry["second"] = "b" * 64
    with pytest.raises(RuntimeError, match="NATIVE_MTF_GEOMETRY_INVALID|STRUCTURE_HAS_NO_PEER"):
        build_structure_preserving_donor_plan(
            block_ids=["first", "first", "second", "second"],
            within_block_positions=[0, 1, 0, 1],
            native_mtf_geometry_by_block=geometry,
        )


def test_donor_plan_preserves_whole_equal_geometry_blocks() -> None:
    donor, plan = build_structure_preserving_donor_plan(
        block_ids=["a", "a", "b", "b", "c", "c", "d", "d"],
        within_block_positions=[0, 1, 0, 1, 0, 1, 0, 1],
    )
    assert donor.tolist() == [2, 3, 4, 5, 6, 7, 0, 1]
    assert plan["all_rows_deranged"] is True
    assert plan["whole_equal_geometry_blocks_preserved"] is True
    with pytest.raises(RuntimeError, match="STRUCTURE_HAS_NO_PEER"):
        build_structure_preserving_donor_plan(
            block_ids=["a", "a", "b"],
            within_block_positions=[0, 1, 0],
        )


def test_native_geometry_plan_is_bound_by_current_report_contract(usefulness_report) -> None:
    report = copy.deepcopy(usefulness_report)
    plan = report["tasks"]["exit"]["donor_plan"]
    plan["native_mtf_geometry_sha256"] = "a" * 64
    report["identity"]["native_mtf_geometry_sha256"] = "a" * 64
    report["identity_sha256"] = canonical_json_sha256(report["identity"])
    plan["plan_sha256"] = canonical_json_sha256({key: value for key, value in plan.items() if key != "plan_sha256"})
    report["report_sha256"] = canonical_json_sha256({key: value for key, value in report.items() if key != "report_sha256"})
    require_feature_usefulness_report(report)
    plan["native_mtf_geometry_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="DONOR_PLAN_BINDING_INVALID"):
        require_feature_usefulness_report(report)


@pytest.mark.parametrize("mutation", ["legacy_schema", "missing_geometry", "source_fields", "invalid_geometry", "entry_geometry"])
def test_native_geometry_report_rejects_unbound_or_inapplicable_claims(usefulness_report, mutation) -> None:
    report = copy.deepcopy(usefulness_report)
    plan = report["tasks"]["exit"]["donor_plan"]
    if mutation == "legacy_schema":
        report["schema_version"] = "gx1_entry_exit_feature_usefulness_v7"
    elif mutation == "missing_geometry":
        plan.pop("native_mtf_geometry_sha256")
    elif mutation == "source_fields":
        plan["source_fields"].append("native_mtf_geometry")
    elif mutation == "invalid_geometry":
        plan["native_mtf_geometry_sha256"] = "not-a-hash"
    else:
        report["tasks"]["entry"]["donor_plan"]["native_mtf_geometry_sha256"] = "a" * 64
    with pytest.raises(RuntimeError, match="REPORT_POLICY_INVALID|DONOR_PLAN_INVALID|NATIVE_MTF_GEOMETRY_INVALID"):
        require_feature_usefulness_report(report)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("test_split", "NON_VAL_ROW_FORBIDDEN"),
        ("future_time", "FUTURE_OR_OUTSIDE_VAL_ROW_FORBIDDEN"),
        ("missing_time", "ROW_CLOCK_INVALID"),
        ("duplicate_time", "ENTRY_ROW_CLOCK_ORDER_INVALID"),
        ("reversed_time", "ENTRY_ROW_CLOCK_ORDER_INVALID"),
        ("alias", "ALIAS_SOURCE_OFF_MANIFOLD"),
    ],
)
def test_test_future_and_off_manifold_alias_rows_fail_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    names = _signal_names()
    layout = feature_usefulness_layout(names)
    timeframes = tuple(layout["tasks"]["entry"]["timeframes"])
    states, indices = _states(timeframes)
    predictor = _SyntheticPredictor(task="entry", indices=indices, names=names)
    q_targets = predictor(states)
    action_valid = np.ones_like(q_targets, dtype=np.bool_)
    action_equivalent = action_valid & np.equal(
        q_targets,
        np.max(np.where(action_valid, q_targets, -np.inf), axis=1, keepdims=True),
    )
    times = pd.date_range("2026-01-01", periods=len(q_targets), freq="h", tz="UTC")
    splits = ["val"] * len(q_targets)
    if mutation == "test_split":
        splits[-1] = "test"
    elif mutation == "future_time":
        times = times[:-1].append(pd.DatetimeIndex([pd.Timestamp("2026-01-03", tz="UTC")]))
    elif mutation == "missing_time":
        times = times[:-1].append(pd.DatetimeIndex([pd.NaT]))
    elif mutation == "duplicate_time":
        times = times[:-1].append(times[-2:-1])
    elif mutation == "reversed_time":
        times = times[::-1]
    else:
        alias = layout["tasks"]["entry"]["physical_field_perturbations"][0]
        while alias["alias_ctx_cont_index"] is None:
            alias = layout["tasks"]["entry"]["physical_field_perturbations"][
                layout["tasks"]["entry"]["physical_field_perturbations"].index(alias) + 1
            ]
        states["ctx_cont"][0, alias["alias_ctx_cont_index"]] += 1.0
    with pytest.raises(RuntimeError, match=message):
        audit_task_feature_usefulness(
            task="entry",
            ordered_signal_names=names,
            identity=_identity(tmp_path),
            states=states,
            entry_action_q_target_bps=q_targets,
            entry_action_valid_mask=action_valid,
            entry_action_equivalence_mask=action_equivalent,
            entry_fitted_q_iteration_state=_entry_fitted_q_iteration(),
            exit_fitted_q_iteration_state=_exit_fitted_q_iteration(),
            row_times=times,
            row_splits=splits,
            block_ids=np.repeat(["day0", "day1", "day2", "day3"], 2),
            within_block_positions=np.tile([0, 1], 4),
            predictor=predictor,
            batch_rows=len(q_targets),
        )


@pytest.mark.parametrize("mutation", ["missing", "opposite_side_time"])
def test_exit_row_clock_rejects_missing_or_mismatched_side_time(
    tmp_path: Path, mutation: str
) -> None:
    times = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC").take(
        [0, 1, 0, 1, 2, 3, 2, 3]
    )
    values = list(times)
    values[2] = pd.NaT if mutation == "missing" else times[2] + pd.Timedelta(minutes=1)
    expected = "ROW_CLOCK_INVALID" if mutation == "missing" else "SIDE_PAIR_CLOCK_MISMATCH"
    with pytest.raises(RuntimeError, match=expected):
        _audit_task(tmp_path, "exit", row_times=pd.DatetimeIndex(values))


@pytest.mark.parametrize(
    "mutation", ["origin_missing", "hole", "reversed_states", "duplicate_time", "reversed_time"]
)
def test_exit_complete_origin_and_within_episode_clock_are_required(mutation: str) -> None:
    timeframes = tuple(
        feature_usefulness_layout(_signal_names())["tasks"]["exit"]["timeframes"]
    )
    states, _indices = _states(timeframes)
    terminal = states["exit_state_index"] == 1
    valid = np.ones((len(terminal), 2), dtype=np.bool_)
    valid[terminal, 0] = False
    times = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC").take(
        [0, 1, 0, 1, 2, 3, 2, 3]
    )
    _require_exit_fitted_q_state_binding(
        states=states, action_valid_mask=valid, terminal_mask=terminal, row_times=times
    )
    if mutation == "origin_missing":
        states["exit_state_index"] += 1
    elif mutation == "hole":
        states["exit_state_index"][terminal] = 7
    elif mutation == "reversed_states":
        states["exit_state_index"] = 1 - states["exit_state_index"]
        terminal = states["exit_state_index"] == 1
        valid[:, 0] = ~terminal
    elif mutation == "duplicate_time":
        times = times.take([0, 0, 2, 2, 4, 4, 6, 6])
    else:
        times = times.take([1, 0, 3, 2, 5, 4, 7, 6])
    with pytest.raises(RuntimeError, match="FITTED_Q_STATE_BINDING_INVALID"):
        _require_exit_fitted_q_state_binding(
            states=states, action_valid_mask=valid, terminal_mask=terminal, row_times=times
        )


@pytest.mark.parametrize("schema", [
    "gx1_entry_exit_feature_usefulness_v6", "gx1_entry_exit_feature_usefulness_v8",
])
def test_legacy_usefulness_report_cannot_claim_current_clock_contract(
    usefulness_report: dict[str, object],
    schema: str,
) -> None:
    legacy = copy.deepcopy(usefulness_report)
    legacy["schema_version"] = schema
    unsigned = dict(legacy)
    unsigned.pop("report_sha256")
    legacy["report_sha256"] = canonical_json_sha256(unsigned)
    with pytest.raises(RuntimeError, match="FEATURE_USEFULNESS_REPORT_POLICY_INVALID"):
        require_feature_usefulness_report(legacy)


def test_identity_metric_and_coverage_mutations_fail_closed(
    usefulness_report: dict[str, object],
) -> None:
    identity = copy.deepcopy(usefulness_report)
    identity["identity"]["model_state_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="IDENTITY_HASH_INVALID"):
        require_feature_usefulness_report(identity)

    coverage = copy.deepcopy(usefulness_report)
    coverage["tasks"]["entry"]["logical_field_metrics"]["local_signal"].pop(
        "local_signal.atr_z"
    )
    with pytest.raises(RuntimeError, match="LOCAL_SIGNAL_COVERAGE_INVALID"):
        require_feature_usefulness_report(coverage)

    zero_is_valid = copy.deepcopy(usefulness_report)
    require_feature_usefulness_report(zero_is_valid)
    assert zero_is_valid["tasks"]["entry"]["logical_field_metrics"][
        "local_signal"
    ]["local_signal.atr_z"]["interpretation"] == (
        "non_positive_mean_on_both_raw_paired_metrics"
    )


def test_rehashed_normalization_identity_still_must_match_fitted_q_teacher(usefulness_report) -> None:
    report = copy.deepcopy(usefulness_report)
    report["identity"]["normalization_contract_sha256"] = "6" * 64
    report["identity_sha256"] = canonical_json_sha256(report["identity"])
    report["report_sha256"] = canonical_json_sha256({
        key: value for key, value in report.items() if key != "report_sha256"
    })
    with pytest.raises(RuntimeError, match="FEATURE_USEFULNESS_NORMALIZATION_IDENTITY_MISMATCH"):
        require_feature_usefulness_report(report)


@pytest.mark.parametrize("field", [
    "target_model_state_sha256", "train_split_sha256", "lifecycle_manifest_sha256",
    "selected_epoch", "native_episode_pair_count", "native_mtf_geometry_sha256",
])
def test_rehashed_native_identity_cannot_disagree_with_supervision_or_population(usefulness_report, field) -> None:
    report = copy.deepcopy(usefulness_report)
    report["identity"][field] = 3 if field in ("selected_epoch", "native_episode_pair_count") else "f" * 64
    report["identity_sha256"] = canonical_json_sha256(report["identity"])
    report["report_sha256"] = canonical_json_sha256({key: value for key, value in report.items() if key != "report_sha256"})
    with pytest.raises(RuntimeError, match="SELECTED_TEACHER_IDENTITY_MISMATCH|NATIVE_EPISODE_POPULATION_MISMATCH"):
        require_feature_usefulness_report(report)


@pytest.mark.parametrize("mutation", [
    "runtime_snapshot", "runtime_envelope", "missing_teacher_tokens", "negative_pairs",
    "bool_epoch", "reversed_epochs", "collapsed_roles", "wrong_recipe", "wrong_metadata",
    "separate_normalization", "relative_path", "parent_path", "bad_source_hash",
])
def test_native_report_identity_rejects_false_runtime_or_source_bindings(tmp_path, mutation) -> None:
    from gx1.contracts.entry_exit_feature_usefulness_v1 import require_feature_usefulness_identity

    identity = _identity(tmp_path)
    if mutation == "runtime_snapshot":
        identity["entry_decision_token_snapshot_set_sha256"] = "7" * 64
    elif mutation == "runtime_envelope":
        identity["unified_exit_input_envelope_set_sha256"] = "8" * 64
    elif mutation == "missing_teacher_tokens":
        identity.pop("target_entry_token_stream_sha256")
    elif mutation == "negative_pairs":
        identity["native_episode_pair_count"] = -1
    elif mutation == "bool_epoch":
        identity["selected_epoch"] = True
    elif mutation == "reversed_epochs":
        identity["last_epoch"] = 4
    elif mutation == "collapsed_roles":
        identity["selection_artifacts"]["active_state"] = identity["selection_artifacts"]["selected_checkpoint"]
    elif mutation == "wrong_recipe":
        identity["selection_artifacts"]["recipe_audit"]["sha256"] = "f" * 64
    elif mutation == "wrong_metadata":
        identity["selection_artifacts"]["bundle_metadata"]["sha256"] = "f" * 64
    elif mutation == "separate_normalization":
        identity["normalization_path"] = str(tmp_path / "normalization.json")
    elif mutation == "relative_path":
        identity["selection_artifacts"]["active_state"]["path"] = "relative.pt"
    elif mutation == "parent_path":
        identity["lifecycle_manifest_path"] = "/source/../lifecycle.json"
    else:
        identity["recipe_source_provenance"]["source_bindings_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="FEATURE_USEFULNESS"):
        require_feature_usefulness_identity(identity)


def test_native_identity_keeps_online_and_teacher_and_earlier_best_epoch_distinct(tmp_path) -> None:
    from gx1.contracts.entry_exit_feature_usefulness_v1 import require_feature_usefulness_identity

    identity = require_feature_usefulness_identity(_identity(tmp_path))
    assert identity["selected_epoch"] < identity["last_epoch"]
    assert identity["model_state_sha256"] != identity["target_model_state_sha256"]
    assert identity["online_entry_token_stream_sha256"] != identity["target_entry_token_stream_sha256"]
    assert len({binding["sha256"] for binding in identity["selection_artifacts"].values()}) == 1


@pytest.mark.parametrize("task", ["entry", "exit"])
@pytest.mark.parametrize("field", [
    "target_model_state_sha256", "selected_epoch", "train_split_sha256", "lifecycle_manifest_sha256",
])
def test_selected_teacher_mismatch_fails_before_predictor(tmp_path, monkeypatch, task, field) -> None:
    import sys

    identity = _identity(tmp_path)
    identity[field] = 3 if field == "selected_epoch" else "f" * 64
    def wrong_identity(path):
        return identity
    def forbidden_prediction(**arguments):
        pytest.fail("A selected-teacher identity mismatch must fail before audit prediction")
    monkeypatch.setattr(sys.modules[__name__], "_identity", wrong_identity)
    monkeypatch.setattr(usefulness_audit, "_predict", forbidden_prediction)
    with pytest.raises(RuntimeError, match="SELECTED_TEACHER_IDENTITY_MISMATCH"):
        _audit_task(tmp_path, task)


@pytest.mark.parametrize("task", ("entry", "exit"))
@pytest.mark.parametrize("fault", ("normalization", "missing_teacher", "invalid_teacher"))
def test_teacher_identity_is_rejected_before_any_audit_prediction(tmp_path, monkeypatch, task, fault) -> None:
    import sys

    current_module = sys.modules[__name__]
    if fault == "normalization":
        original_identity = _identity

        def wrong_normalization(path):
            return {**original_identity(path), "normalization_contract_sha256": "6" * 64}

        monkeypatch.setattr(current_module, "_identity", wrong_normalization)
        expected = "FEATURE_USEFULNESS_NORMALIZATION_IDENTITY_MISMATCH"
    else:
        original_identity = _identity(tmp_path)
        original_iteration = _exit_fitted_q_iteration()

        def fixed_identity(path):
            return original_identity

        def wrong_teacher():
            return None if fault == "missing_teacher" else {
                **original_iteration, "target_updated_from_val_or_test": True,
            }

        monkeypatch.setattr(current_module, "_identity", fixed_identity)
        monkeypatch.setattr(current_module, "_exit_fitted_q_iteration", wrong_teacher)
        expected = "FITTED_Q_ITERATION_REQUIRED" if fault == "missing_teacher" else "UNIFIED_EXIT_FITTED_Q"

    def forbidden_prediction(**arguments):
        pytest.fail("Invalid fitted-Q identity reached audit inference")

    monkeypatch.setattr(usefulness_audit, "_predict", forbidden_prediction)
    with pytest.raises(RuntimeError, match=expected):
        _audit_task(tmp_path, task)


def test_source_has_no_selection_cutoff_or_hindsight_or_classification_authority() -> None:
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in (
            "gx1/contracts/entry_exit_feature_usefulness_v1.py",
            "gx1/scripts/audit_entry_exit_feature_usefulness_v1.py",
        )
    )
    for forbidden in (
        "exit_action_target",
        "exit_baseline_action_target",
        "unified_exit_optimal_stopping",
        "UNIFIED_EXIT_OPTIMAL_STOPPING",
        "finite_horizon_optimal",
        "pathwise_hindsight",
        "direction_logits",
        "exact_label_ce",
        "labels_sha256",
        "joint_delta-family_delta-timeframe_delta",
        "automatic_importance_threshold\": 0",
        "automatic_top_k\": 133",
    ):
        assert forbidden not in source
