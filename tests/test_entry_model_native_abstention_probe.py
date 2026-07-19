from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from gx1.contracts.entry_model_native_abstention_probe_v1 import (
    HISTORICAL_ROLE,
    MODEL_NATIVE_ROLE,
    ROW_SCHEMA_VERSION,
    SCHEMA_VERSION as EVIDENCE_SCHEMA_VERSION,
    UTILITY_DEFINITION,
)
from gx1.scripts import verify_entry_model_native_abstention_probe_v1 as probe


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(path: Path, split: str, counts: tuple[int, int, int]) -> str:
    return _write_json(
        path,
        {
            "schema_version": probe.HISTORICAL_MANIFEST_SCHEMA,
            "manifest_variant": probe.HISTORICAL_MANIFEST_VARIANT,
            "expected_seq_snap_width": probe.HISTORICAL_SIGNAL_DIM,
            "foundation_smoke_dataset_v1": {
                "split": split,
                "sample_rows": sum(counts),
                "label_counts": {str(index): value for index, value in enumerate(counts)},
            },
        },
    )


def _registry(path: Path, *, available: bool, artifact_dir: Path) -> str:
    artifact_dir.mkdir(exist_ok=True)
    return _write_json(
        path,
        {
            "schema_version": "gx1_artifact_selection_v2",
            "active": {
                "entry_iql": {
                    "path": str(artifact_dir) if available else None,
                    "status": "HISTORICAL_COMPARISON_ONLY" if available else "RETIRED_ARTIFACT_ABSENT",
                    "artifact_present": available,
                }
            },
        },
    )


def _base_args(tmp_path: Path, *, registry_available: bool = False) -> argparse.Namespace:
    values: dict[str, object] = {
        "benchmark_evidence_json": None,
        "benchmark_evidence_sha256": None,
        "learned_probe_evidence_json": None,
        "learned_probe_evidence_sha256": None,
        "out_dir": str(tmp_path / "reports"),
        "quiet": True,
    }
    for split, counts in (
        ("train", (1357, 1338, 1400)),
        ("val", (496, 510, 530)),
        ("test", (507, 513, 516)),
    ):
        path = tmp_path / f"{split}.manifest.json"
        values[f"{split}_smoke_manifest"] = str(path)
        values[f"{split}_smoke_manifest_sha256"] = _manifest(path, split, counts)
    registry_path = tmp_path / "PROJECT_STATE_artifacts.json"
    values["artifact_registry_json"] = str(registry_path)
    values["artifact_registry_sha256"] = _registry(
        registry_path,
        available=registry_available,
        artifact_dir=tmp_path / "historical_entry_iql",
    )
    return argparse.Namespace(**values)


def _selection_evidence(
    tmp_path: Path,
    *,
    name: str,
    role: str,
    takes: tuple[bool, ...],
    utilities: tuple[float, ...],
) -> tuple[Path, str]:
    rows_path = tmp_path / f"{name}.jsonl"
    universe_rows: list[tuple[str, str, float]] = []
    rows: list[dict[str, object]] = []
    for index, (take, utility) in enumerate(zip(takes, utilities, strict=True)):
        sample_id = f"sample-{index}"
        time_utc = f"2026-06-0{index + 1}T00:00:00Z"
        row: dict[str, object] = {
            "sample_id": sample_id,
            "time_utc": time_utc,
            "take": take,
            "realized_net_utility_bps": utility,
            "costs_included": True,
        }
        if role == MODEL_NATIVE_ROLE:
            row.update(
                {
                    "model_direction_index": 0 if take else 2,
                    "calibrated_argmax": True,
                }
            )
        rows.append(row)
        universe_rows.append((sample_id, time_utc, utility))
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    take_values = [utility for take, utility in zip(takes, utilities, strict=True) if take]
    skip_values = [utility for take, utility in zip(takes, utilities, strict=True) if not take]
    take_mean = sum(take_values) / len(take_values)
    skip_mean = sum(skip_values) / len(skip_values)
    universe_sha = hashlib.sha256(
        json.dumps(
            sorted(universe_rows), separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    evidence_path = tmp_path / f"{name}.evidence.json"
    evidence_sha = _write_json(
        evidence_path,
        {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "role": role,
            "evaluation_scope": "strict_oot",
            "split": "test",
            "utility_definition": UTILITY_DEFINITION,
            "evaluation_universe_sha256": universe_sha,
            "authority": {
                "direction": False,
                "fallback": False,
                "launch": False,
                "live": False,
            },
            "row_evidence": {
                "schema_version": ROW_SCHEMA_VERSION,
                "format": "jsonl",
                "path": str(rows_path),
                "sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
            },
            "metrics": {
                "rows": len(rows),
                "take_rows": len(take_values),
                "skip_rows": len(skip_values),
                "coverage": len(take_values) / len(rows),
                "take_ev_net_bps": take_mean,
                "skip_ev_net_bps": skip_mean,
                "take_skip_separation_net_bps": take_mean - skip_mean,
            },
        },
    )
    return evidence_path, evidence_sha


def test_missing_benchmark_emits_report_only_block_with_label_probe(tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    report = probe.run(args)

    assert report["decision"] == probe.BLOCK_DECISION
    assert report["empirical_comparison_performed"] is False
    assert report["empirical_gate_passed"] is False
    assert "HISTORICAL_ENTRY_IQL_BENCHMARK_BYTES_NOT_REGISTERED" in report["blockers"]
    assert "EXACT_HISTORICAL_SELECTION_BENCHMARK_EVIDENCE_MISSING" in report["blockers"]
    assert report["inputs"]["historical_smoke_manifests"]["val"]["label_counts"] == {
        "LONG": 496,
        "SHORT": 510,
        "FLAT": 530,
    }
    assert report["probe_diagnostics"]["active_recipe"]["all_required_weights_positive"] is True
    assert report["probe_diagnostics"]["active_recipe"]["empirical_abstention_edge_proven"] is False
    assert report["direction_authority"] is False
    assert report["rebuild_authorized"] is False
    assert report["training_authorized"] is False
    assert report["side_effects"]["parquet_read"] is False


def test_bound_manifest_hash_mismatch_fails_before_report(tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    args.test_smoke_manifest_sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        probe.run(args)
    assert not (tmp_path / "reports").exists()


def test_exact_aligned_evidence_can_pass_comparison_but_grants_no_authority(tmp_path: Path) -> None:
    args = _base_args(tmp_path, registry_available=True)
    utilities = (8.0, 4.0, -3.0, -5.0)
    historical_path, historical_sha = _selection_evidence(
        tmp_path,
        name="historical",
        role=HISTORICAL_ROLE,
        takes=(True, True, False, False),
        utilities=utilities,
    )
    learned_path, learned_sha = _selection_evidence(
        tmp_path,
        name="learned",
        role=MODEL_NATIVE_ROLE,
        takes=(True, True, False, False),
        utilities=utilities,
    )
    args.benchmark_evidence_json = str(historical_path)
    args.benchmark_evidence_sha256 = historical_sha
    args.learned_probe_evidence_json = str(learned_path)
    args.learned_probe_evidence_sha256 = learned_sha

    report = probe.run(args)

    assert report["decision"] == probe.PASS_DECISION
    assert report["empirical_comparison"]["passed"] is True
    assert report["rebuild_authorized"] is False
    assert report["training_authorized"] is False
    assert report["launch_authorized"] is False
    assert report["live_authorized"] is False


def test_model_native_rows_cannot_add_a_threshold_selector(tmp_path: Path) -> None:
    args = _base_args(tmp_path, registry_available=True)
    path, sha = _selection_evidence(
        tmp_path,
        name="learned",
        role=MODEL_NATIVE_ROLE,
        takes=(True, True, False, False),
        utilities=(8.0, 4.0, -3.0, -5.0),
    )
    rows_path = tmp_path / "learned.jsonl"
    rows = rows_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["take"] = False
    rows[0] = json.dumps(first, sort_keys=True)
    rows_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["row_evidence"]["sha256"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    sha = _write_json(path, payload)

    args.learned_probe_evidence_json = str(path)
    args.learned_probe_evidence_sha256 = sha
    report = probe.run(args)
    assert any("take must equal calibrated argmax" in blocker for blocker in report["blockers"])


def test_control_surface_exposes_only_report_route() -> None:
    source = Path("scripts/entry_next_edge_control.sh").read_text(encoding="utf-8")
    assert "model-native-abstention-probe)" in source
    assert "verify_entry_model_native_abstention_probe_v1" in source
    assert "--benchmark-evidence-json" not in source
    assert "--learned-probe-evidence-json" not in source
