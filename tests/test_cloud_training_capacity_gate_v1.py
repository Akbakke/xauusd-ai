from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from gx1.contracts.cloud_training_capacity_gate_v1 import (
    BENCHMARK_SCHEMA_VERSION,
    FAIL_DECISION,
    GATE_SCHEMA_VERSION,
    HARD_HOST_DEADLINE_SECONDS,
    PASS_DECISION,
    PRECISION_POLICY,
    QUALIFICATION_TIME_LIMIT_SECONDS,
    CloudTrainingCapacityGateError,
    artifact_binding,
    evaluate_cloud_training_capacity,
    require_cloud_training_capacity_gate,
    validate_capacity_gate_payload,
    validate_hopper_benchmark_payload,
)
from gx1.contracts.cloud_training_smoke_measurement_v1 import (
    build_cloud_training_smoke_measurement,
)
from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    MANIFEST_NAME as BUNDLE_COMMIT_MANIFEST_NAME,
    write_bundle_commit_manifest,
)
from gx1.scripts import evaluate_cloud_training_capacity_v1 as cli


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path.resolve(strict=True)


@pytest.fixture
def capacity_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("capacity").resolve(strict=True)


def _benchmark(
    capacity_root: Path, **overrides: object
) -> tuple[dict[str, object], Path]:
    recipe = _write_json(capacity_root / "smoke-recipe.json", {"profile": "smoke"})
    host = _write_json(capacity_root / "cloud-host-profile.json", {"gpu": "H200"})
    batch_size = int(overrides.get("batch_size", 32))
    train_rows = int(overrides.get("train_rows", 313_399))
    val_rows = int(overrides.get("val_rows", 70_880))
    # Synthetic fast-host case, not a hardware measurement or forecast.
    measured_train_seconds = float(overrides.get("measured_train_seconds", 60.0))
    measured_val_seconds = float(overrides.get("measured_val_seconds", 120.0))
    preflight_seconds = float(overrides.get("preflight_seconds", 30.0))
    checkpoint_write_seconds = float(overrides.get("checkpoint_write_seconds", 1.0))
    bundle = (capacity_root / "smoke-bundle").resolve()
    bundle.mkdir()
    checkpoint = bundle / "model_state_dict.pt"
    checkpoint.write_bytes(b"checkpoint")
    measurement = build_cloud_training_smoke_measurement(
        source_commit="a" * 40,
        run_id="ENTRY_CLOUD_SMOKE_20260908T120000Z",
        smoke_recipe_path=recipe,
        cloud_host_profile_path=host,
        batch_size=batch_size,
        physical_train_rows=train_rows,
        physical_val_rows=val_rows,
        sampled_train_rows=(32 + 256) * batch_size,
        sampled_val_rows=(32 + 256) * batch_size,
        measured_train_rows=256 * batch_size,
        measured_train_seconds=measured_train_seconds,
        measured_val_seconds=measured_val_seconds,
        preflight_seconds=preflight_seconds,
        checkpoint_write_seconds=checkpoint_write_seconds,
        checkpoint_path=checkpoint,
    )
    lineage = {
        "training_profile": "smoke",
        "training_run_id": measurement["run_id"],
        "physical_train_rows": train_rows,
        "physical_val_rows": val_rows,
        "effective_train_rows": measurement["sampled_train_rows"],
        "effective_val_rows": measurement["sampled_val_rows"],
    }
    provenance = {
        "recipe_audit_path": str(recipe),
        "recipe_audit_sha256": measurement["smoke_recipe"]["sha256"],
        "source_commit": "a" * 40,
    }
    _write_json(
        bundle / "bundle_metadata.json",
        {
            "git_commit": "a" * 40,
            "execution_tier": "canonical",
            "batch_size": batch_size,
            "run_lineage": lineage,
            "recipe_source_provenance": provenance,
            "cloud_hopper_smoke_measurement": measurement,
        },
    )
    _write_json(
        bundle / "MASTER_TRANSFORMER_LOCK.json",
        {"cloud_hopper_smoke_measurement": measurement},
    )
    write_bundle_commit_manifest(
        bundle_dir=bundle,
        artifact_names=(
            "MASTER_TRANSFORMER_LOCK.json",
            "bundle_metadata.json",
            "model_state_dict.pt",
        ),
        bundle_kind="trained",
        created_at_utc="2026-09-08T12:00:00+00:00",
    )
    payload: dict[str, object] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "source_commit": "a" * 40,
        "smoke_recipe": artifact_binding(recipe, label="smoke recipe"),
        "smoke_bundle_commit": artifact_binding(
            bundle / BUNDLE_COMMIT_MANIFEST_NAME,
            label="smoke bundle commit",
        ),
        "cloud_host_profile": artifact_binding(host, label="cloud host profile"),
        "precision_policy": PRECISION_POLICY,
        "batch_size": batch_size,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "max_epochs": 30,
        "warmup_optimizer_steps": 32,
        "measured_optimizer_steps": 256,
        "measured_train_rows": 256 * batch_size,
        "measured_train_seconds": measured_train_seconds,
        "measured_val_rows": (32 + 256) * batch_size,
        "measured_val_seconds": measured_val_seconds,
        "preflight_seconds": preflight_seconds,
        "checkpoint_write_seconds": checkpoint_write_seconds,
        "provider_price_usd_per_hour": 3.5,
        "nok_per_usd": 10.0,
        "fx_observed_utc": "2026-09-08T12:00:00Z",
        "hard_host_deadline_seconds": HARD_HOST_DEADLINE_SECONDS,
    }
    payload.update(overrides)
    benchmark_path = _write_json(capacity_root / "benchmark.json", payload)
    return payload, benchmark_path


def _evaluate(capacity_root: Path, **overrides: object) -> dict[str, object]:
    payload, path = _benchmark(capacity_root, **overrides)
    return evaluate_cloud_training_capacity(
        payload,
        benchmark_binding=artifact_binding(path, label="benchmark"),
    )


def test_exact_benchmark_and_gate_schemas(capacity_root: Path) -> None:
    payload, path = _benchmark(capacity_root)
    validated = validate_hopper_benchmark_payload(payload)
    gate = evaluate_cloud_training_capacity(
        validated,
        benchmark_binding=artifact_binding(path, label="benchmark"),
    )

    assert gate["schema_version"] == GATE_SCHEMA_VERSION
    assert gate["decision"] == PASS_DECISION
    assert gate["capacity_qualified"] is True
    assert gate["report_only"] is True
    assert gate["activation_authority"] is False
    assert set(gate["authority"].values()) == {False}
    assert gate["side_effects"] == []

    malformed_gate = dict(gate, unexpected=True)
    with pytest.raises(CloudTrainingCapacityGateError, match="keys are not exact"):
        validate_capacity_gate_payload(
            malformed_gate,
            expected_benchmark_path=path,
            expected_benchmark_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    malformed_benchmark = dict(payload, unexpected=True)
    with pytest.raises(CloudTrainingCapacityGateError, match="keys are not exact"):
        validate_hopper_benchmark_payload(malformed_benchmark)

    malformed_binding = dict(payload)
    malformed_binding["smoke_recipe"] = {
        **payload["smoke_recipe"],
        "unexpected": True,
    }
    with pytest.raises(CloudTrainingCapacityGateError, match="binding keys"):
        validate_hopper_benchmark_payload(malformed_binding)


def test_projection_includes_full_val_checkpoints_and_restarts(
    capacity_root: Path,
) -> None:
    payload, path = _benchmark(capacity_root)
    gate = evaluate_cloud_training_capacity(
        payload,
        benchmark_binding=artifact_binding(path, label="benchmark"),
    )

    projection = gate["projection"]
    assert projection["projected_val_seconds"] > payload["measured_val_seconds"]
    assert projection["projected_checkpoint_seconds"] > 0.0
    assert projection["projected_restart_reserve_seconds"] > (
        2 * payload["preflight_seconds"]
    )
    assert gate["decision"] == PASS_DECISION
    assert gate["capacity_qualified"] is True


def test_time_limit_fails_closed(capacity_root: Path) -> None:
    gate = _evaluate(capacity_root, measured_val_seconds=6_000.0)

    assert gate["decision"] == FAIL_DECISION
    assert gate["capacity_qualified"] is False
    assert "PROJECTED_WORST_CASE_EXCEEDS_43_2_HOURS" in gate["failures"]
    assert gate["projection"]["projected_worst_case_seconds"] > QUALIFICATION_TIME_LIMIT_SECONDS
    assert gate["activation_authority"] is False


def test_cost_limit_uses_buffered_hard_deadline(capacity_root: Path) -> None:
    gate = _evaluate(
        capacity_root,
        provider_price_usd_per_hour=5.0,
        nok_per_usd=10.0,
    )

    assert gate["projection"]["projected_buffered_cost_nok"] < 2_500.0
    assert gate["projection"]["hard_deadline_buffered_cost_nok"] > 2_500.0
    assert gate["decision"] == FAIL_DECISION
    assert gate["failures"] == ["BUFFERED_COST_EXCEEDS_2500_NOK"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("warmup_optimizer_steps", 31, "warmup_optimizer_steps"),
        ("measured_optimizer_steps", 255, "measured_optimizer_steps"),
        ("measured_train_seconds", 0.0, "measured_train_seconds"),
        ("measured_train_seconds", -1.0, "measured_train_seconds"),
        ("measured_train_seconds", float("nan"), "measured_train_seconds"),
        ("checkpoint_write_seconds", 0, "checkpoint_write_seconds"),
        ("provider_price_usd_per_hour", float("inf"), "positive and finite"),
        ("nok_per_usd", float("nan"), "positive and finite"),
    ),
)
def test_measurement_minimums_fail_closed(
    capacity_root: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload, _ = _benchmark(capacity_root)
    payload[field] = value
    if field == "measured_optimizer_steps":
        payload["measured_train_rows"] = 255 * 32
    with pytest.raises(CloudTrainingCapacityGateError, match=message):
        validate_hopper_benchmark_payload(payload)


def test_binding_hash_and_path_tamper_fail_closed(capacity_root: Path) -> None:
    payload, benchmark_path = _benchmark(capacity_root)
    gate = evaluate_cloud_training_capacity(
        payload,
        benchmark_binding=artifact_binding(benchmark_path, label="benchmark"),
    )
    output = _write_json(capacity_root / "gate.json", gate)
    output_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    benchmark_sha = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()

    Path(payload["smoke_recipe"]["path"]).write_text("tampered", encoding="utf-8")
    with pytest.raises(CloudTrainingCapacityGateError, match="hash/path mismatch"):
        require_cloud_training_capacity_gate(
            output,
            output_sha,
            expected_benchmark_path=benchmark_path,
            expected_benchmark_sha256=benchmark_sha,
        )

    with pytest.raises(CloudTrainingCapacityGateError, match="SHA-256 mismatch"):
        cli.run(
            argparse.Namespace(
                benchmark=str(benchmark_path),
                benchmark_sha256="0" * 64,
                output=str(capacity_root / "unused.json"),
            )
        )


def test_binding_path_replacement_fails_closed(capacity_root: Path) -> None:
    payload, _ = _benchmark(capacity_root)
    replacement = _write_json(capacity_root / "replacement-recipe.json", {})
    tampered = dict(payload)
    tampered["smoke_recipe"] = {
        "path": str(replacement),
        "sha256": payload["smoke_recipe"]["sha256"],
    }

    with pytest.raises(CloudTrainingCapacityGateError, match="hash/path mismatch"):
        validate_hopper_benchmark_payload(tampered)


def test_test_like_input_path_is_rejected(capacity_root: Path) -> None:
    payload, _ = _benchmark(capacity_root)
    forbidden = _write_json(capacity_root / "TEST_recipe.json", {"split": "forbidden"})
    payload["smoke_recipe"] = {
        "path": str(forbidden),
        "sha256": hashlib.sha256(forbidden.read_bytes()).hexdigest(),
    }

    with pytest.raises(CloudTrainingCapacityGateError, match="TEST-like"):
        validate_hopper_benchmark_payload(payload)


def test_cli_atomic_output_hash_and_no_overwrite(
    capacity_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, benchmark_path = _benchmark(capacity_root)
    benchmark_sha = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    output = capacity_root / "capacity-gate.json"
    argv = [
        "--benchmark",
        str(benchmark_path),
        "--benchmark-sha256",
        benchmark_sha,
        "--output",
        str(output),
    ]

    assert cli.main(argv) == 0
    output_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    assert capsys.readouterr().out.strip() == f"{output_sha}  {output}"
    assert output.stat().st_mode & 0o222 == 0
    require_cloud_training_capacity_gate(
        output,
        output_sha,
        expected_benchmark_path=benchmark_path,
        expected_benchmark_sha256=benchmark_sha,
    )

    with pytest.raises(CloudTrainingCapacityGateError, match="already exists"):
        cli.main(argv)

    symlink_output = capacity_root / "symlink-gate.json"
    symlink_target = _write_json(capacity_root / "symlink-target.json", {})
    symlink_output.symlink_to(symlink_target)
    symlink_args = [*argv[:-1], str(symlink_output)]
    with pytest.raises(CloudTrainingCapacityGateError, match="already exists"):
        cli.main(symlink_args)
    assert json.loads(symlink_target.read_text(encoding="utf-8")) == {}


@pytest.mark.parametrize("batch_size", [32, 64])
def test_slow_fixed_step_measurement_remains_capacity_fail(
    capacity_root: Path, batch_size: int,
) -> None:
    # Preserve the old 600-second case as negative evidence; changing the
    # measurement policy must never turn this slow host into a PASS.
    gate = _evaluate(capacity_root, batch_size=batch_size, measured_train_seconds=600.0)
    assert gate["decision"] == FAIL_DECISION
    assert "PROJECTED_WORST_CASE_EXCEEDS_43_2_HOURS" in gate["failures"]
    assert gate["projection"]["projected_worst_case_seconds"] > QUALIFICATION_TIME_LIMIT_SECONDS


def test_projection_must_fit_remaining_host_life() -> None:
    from datetime import datetime, timezone

    import pytest

    from gx1.contracts.cloud_training_capacity_gate_v1 import (
        CloudTrainingCapacityGateError,
        HARD_HOST_DEADLINE_SECONDS,
        QUALIFICATION_TIME_LIMIT_SECONDS,
        require_projection_fits_remaining_host_life,
    )

    now = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
    fresh_deadline = "2026-09-22T12:00:00Z"  # full 48h remaining
    # A projection at exactly the admission ceiling fits a fresh host:
    # ceiling == reserve_fraction * HARD_DEADLINE by construction.
    require_projection_fits_remaining_host_life(
        projected_worst_case_seconds=float(QUALIFICATION_TIME_LIMIT_SECONDS),
        provider_deadline_utc=fresh_deadline,
        now=now,
    )
    # Four hours of remaining life cannot admit a 43.2h projection.
    with pytest.raises(CloudTrainingCapacityGateError, match="remaining life"):
        require_projection_fits_remaining_host_life(
            projected_worst_case_seconds=float(
                QUALIFICATION_TIME_LIMIT_SECONDS
            ),
            provider_deadline_utc="2026-09-20T16:00:00Z",
            now=now,
        )
    # An expired deadline rejects regardless of projection size.
    with pytest.raises(CloudTrainingCapacityGateError, match="remaining life"):
        require_projection_fits_remaining_host_life(
            projected_worst_case_seconds=1.0,
            provider_deadline_utc="2026-09-20T11:59:59Z",
            now=now,
        )
    # A naive clock is rejected rather than silently compared.
    with pytest.raises(CloudTrainingCapacityGateError, match="timezone-aware"):
        require_projection_fits_remaining_host_life(
            projected_worst_case_seconds=1.0,
            provider_deadline_utc=fresh_deadline,
            now=datetime(2026, 9, 20, 12, 0, 0),
        )
    # The reserve fraction is derived, not restated: consistency check.
    assert QUALIFICATION_TIME_LIMIT_SECONDS * 10 == HARD_HOST_DEADLINE_SECONDS * 9


def test_benchmark_rows_must_match_candidate_dataset() -> None:
    import pytest

    from gx1.contracts.cloud_training_capacity_gate_v1 import (
        CloudTrainingCapacityGateError,
        require_benchmark_rows_match_candidate,
    )

    require_benchmark_rows_match_candidate(
        benchmark_train_rows=313399,
        benchmark_val_rows=5509,
        expected_train_rows=313399,
        expected_val_rows=5509,
    )
    with pytest.raises(CloudTrainingCapacityGateError, match="row populations"):
        require_benchmark_rows_match_candidate(
            benchmark_train_rows=313399,
            benchmark_val_rows=5509,
            expected_train_rows=248028,
            expected_val_rows=5509,
        )
    with pytest.raises(CloudTrainingCapacityGateError, match="row populations"):
        require_benchmark_rows_match_candidate(
            benchmark_train_rows=313399,
            benchmark_val_rows=5509,
            expected_train_rows=313399,
            expected_val_rows=70880,
        )
