import argparse
import hashlib
import json
from pathlib import Path

import pytest

from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    READY_DECISION,
    REQUIRED_PROOF_CHECKS,
    SCHEMA_VERSION,
)
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_CONTRACT_MODE
from gx1.scripts import (
    materialize_entry_model_native_seq513_post_rebuild_readiness_v1 as gate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[argparse.Namespace, dict]:
    run_id = "XAU_SEQ513_POST_REBUILD_PYTEST_V1"
    event_root = (tmp_path / "event").resolve()
    dataset_dir = event_root / "dataset"
    dataset_dir.mkdir(parents=True)
    xau = {
        "schema_version": "m5_tape_current_snapshot_v2",
        "instrument": "XAU_USD",
        "entry_run_id": run_id,
        "tape_root": str(event_root / "tape"),
    }

    split_values: dict[str, str] = {}
    for index, split in enumerate(gate.SPLITS, start=1):
        parquet = dataset_dir / f"v10_seq513_dataset__HOLD_03B_{split}.parquet"
        parquet.write_bytes(f"parquet-{split}".encode())
        manifest = dataset_dir / f"v10_seq513_dataset__HOLD_03B_{split}.manifest.json"
        _write_json(
            manifest,
            {
                "schema_version": gate.SPLIT_MANIFEST_SCHEMA,
                "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
                "output_data_path": str(parquet),
                "expected_seq_snap_width": 513,
                "feature_contract": {
                    "signal_dim": 513,
                    "ctx_cont_dim": 142,
                    "ctx_cat_dim": 5,
                },
                "extra": {
                    "rows": index,
                    "entry_run_id": run_id,
                    "xau_tape_provenance": xau,
                },
            },
        )
        split_values[f"{split}_manifest_json"] = str(manifest)
        split_values[f"{split}_manifest_sha256"] = _sha256(manifest)
        split_values[f"{split}_parquet"] = str(parquet)
        split_values[f"{split}_parquet_sha256"] = _sha256(parquet)

    preflight = _write_json(
        event_root / "preflight" / "ENTRY_MODEL_NATIVE_SEQ513_REBUILD_PREFLIGHT_20260722T000000000001Z.json",
        {
            "schema_version": gate.PREFLIGHT_SCHEMA,
            "decision": gate.PREFLIGHT_DECISION,
            "entry_run_id": run_id,
            "training_allowed": False,
        },
    )
    liveness = _write_json(
        event_root / "audit" / "ENTRY_FULL_INPUT_LIVENESS_CONTRACT_20260722T000000000002Z.json",
        {
            "schema_version": gate.LIVENESS_SCHEMA_VERSION,
            "decision": "PASS",
            "failures": [],
            "field_order_sha256": {"signal": "a", "ctx_cont": "b", "ctx_cat": "c"},
            "expected_field_counts": {"signal": 513, "ctx_cont": 142, "ctx_cat": 5},
            "atr_ood_drift": {"status": "SHIFT_OBSERVED"},
        },
    )
    pretrain = _write_json(
        event_root / "audit" / "XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_20260722T000000000003Z.json",
        {
            "schema_version": gate.PRETRAIN_SCHEMA,
            "decision": "PASS",
            "failures": [],
            "require_xau_provenance": True,
            "data_splits": list(gate.SPLITS),
            "tape_provenance": {split: xau for split in gate.SPLITS},
        },
    )
    terminal = event_root / "CHAIN_TERMINAL_20260722T000000000004Z_GREEN.json"
    _write_json(
        terminal,
        {
            "schema_version": gate.CHAIN_SCHEMA,
            "state": "GREEN",
            "step": "chain-complete",
            "exit_code": 0,
            "entry_run_id": run_id,
            "event_root": str(event_root),
            "terminal_event_path": str(terminal),
            "preflight": {"json_path": str(preflight), "sha256": _sha256(preflight)},
        },
    )
    return (
        argparse.Namespace(
            run_id=run_id,
            event_root=str(event_root),
            repo_dir=str(tmp_path.resolve()),
            chain_terminal_json=str(terminal),
            rebuild_preflight_json=str(preflight),
            full_input_liveness_json=str(liveness),
            pretrain_audit_json=str(pretrain),
            dataset_dir=str(dataset_dir),
            smoke_dataset_dir=str(dataset_dir),
            out_dir=str(event_root / "post_rebuild"),
            quiet=True,
            **split_values,
        ),
        xau,
    )


def test_post_rebuild_readiness_binds_green_chain_and_exact_splits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, xau = _fixture(tmp_path)
    monkeypatch.setattr(gate, "validate_xau_tape_provenance_v1", lambda *a, **k: xau)
    monkeypatch.setattr(
        gate,
        "_git_identity",
        lambda repo: {"repo_dir": str(repo), "head": "a" * 40, "status_short": []},
    )
    monkeypatch.setattr(
        gate,
        "validate_full_input_liveness_artifact",
        lambda *a, **k: {
            "ok": True,
            "schema_version": gate.LIVENESS_SCHEMA_VERSION,
            "decision": "PASS",
            "sha256": _sha256(Path(args.full_input_liveness_json)),
            "field_order_sha256": {"signal": "a", "ctx_cont": "b", "ctx_cat": "c"},
            "atr_ood_status": "SHIFT_OBSERVED",
            "failures": [],
        },
    )

    report = gate.run(args)

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["decision"] == READY_DECISION
    assert report["dataset_dir"] == report["smoke_dataset_dir"]
    assert [row["name"] for row in report["checks"]] == list(REQUIRED_PROOF_CHECKS)
    assert all(row["ok"] for row in report["checks"])
    assert report["failures"] == []
    assert report["xau_tape_provenance"] == xau
    assert report["side_effects_started"] == {
        "dataset_rebuild": False,
        "training": False,
        "replay": False,
        "iql_distillation": False,
        "shadow": False,
        "live": False,
    }


def test_post_rebuild_readiness_rejects_separate_smoke_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        gate,
        "_git_identity",
        lambda repo: {"repo_dir": str(repo), "head": "a" * 40, "status_short": []},
    )
    separate = Path(args.event_root) / "smoke_copy"
    separate.mkdir()
    args.smoke_dataset_dir = str(separate)

    with pytest.raises(RuntimeError, match="separate smoke dataset is forbidden"):
        gate.run(args)
