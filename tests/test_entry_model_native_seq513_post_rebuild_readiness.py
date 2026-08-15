import argparse
import hashlib
import json
from pathlib import Path

import pytest

from gx1.contracts import entry_model_native_train_launch_v1 as launch
from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    READY_DECISION,
    REQUIRED_PROOF_CHECKS,
    SCHEMA_VERSION,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_CONTRACT_MODE,
    model_native_signal_contract_metadata,
)
from gx1.scripts import (
    materialize_entry_model_native_seq513_post_rebuild_readiness_v1 as gate,
)
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    _publish_json_atomic_noreplace,
    publish_prefreeze_test_authority,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_SIGNAL_DIM,
)


def test_post_rebuild_readiness_tracks_current_chain_schema() -> None:
    assert gate.CHAIN_SCHEMA == "seq513_rebuild_chain_status_v7"
    assert SCHEMA_VERSION == "entry_model_native_seq513_post_rebuild_readiness_v2"
    assert gate.TEST_SEAL_SCHEMA_VERSION == (
        "entry_model_native_seq513_untouched_test_seal_v2"
    )


def test_test_authority_publisher_is_atomic_noreplace(tmp_path: Path) -> None:
    target = (tmp_path / "authority" / "event.json").resolve()
    _publish_json_atomic_noreplace(target, {"version": 1})
    original = target.read_bytes()

    with pytest.raises(RuntimeError, match="already exists"):
        _publish_json_atomic_noreplace(target, {"version": 2})

    assert target.read_bytes() == original


def test_launch_consumer_rejects_stale_post_rebuild_v1(tmp_path: Path) -> None:
    with pytest.raises(launch.LaunchContractError, match="schema"):
        launch._validate_post_rebuild_readiness(
            {
                "schema_version": "entry_model_native_seq513_post_rebuild_readiness_v1",
                "decision": READY_DECISION,
                "failures": [],
            },
            artifacts={},
            dataset_dir=tmp_path.resolve(),
            test_seal_lineage={},
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture(
    tmp_path: Path,
    *,
    xau_schema: str = "xau_canonical_native_source_v3",
) -> tuple[argparse.Namespace, dict]:
    run_id = "XAU_SEQ513_POST_REBUILD_PYTEST_V1"
    event_root = (tmp_path / "event").resolve()
    dataset_dir = event_root / "dataset"
    dataset_dir.mkdir(parents=True)
    xau = {
        "schema_version": xau_schema,
        "instrument": "XAU_USD",
        "entry_run_id": run_id,
        "tape_root": str(event_root / "tape"),
    }
    signal_contract = model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.post_rebuild_fixture"
        )
    )

    split_values: dict[str, str] = {}
    for index, split in enumerate(("train", "val", "test"), start=1):
        parquet = dataset_dir / f"v10_seq513_dataset__DIR_TRAIN_FIT_{split}.parquet"
        parquet.write_bytes(f"parquet-{split}".encode())
        manifest = dataset_dir / f"v10_seq513_dataset__DIR_TRAIN_FIT_{split}.manifest.json"
        _write_json(
            manifest,
            {
                "schema_version": gate.SPLIT_MANIFEST_SCHEMA,
                "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
                "output_data_path": str(parquet),
                "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
                "feature_contract": {
                    "signal_bridge_fields": signal_contract["fields"],
                    "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
                    "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
                },
                "extra": {
                    "rows": index,
                    "entry_run_id": run_id,
                    "xau_tape_provenance": xau,
                    "model_native_signal_contract": signal_contract,
                },
            },
        )
        if split in gate.PREFREEZE_SPLITS:
            split_values[f"{split}_manifest_json"] = str(manifest)
            split_values[f"{split}_manifest_sha256"] = _sha256(manifest)
            split_values[f"{split}_parquet"] = str(parquet)
            split_values[f"{split}_parquet_sha256"] = _sha256(parquet)

    rebuild_terminal_path = (
        event_root
        / "rebuild_authority"
        / (
            f"{gate.DATASET_REBUILD_TERMINAL_EVENT_PREFIX}_"
            "20260722T000000000000Z.json"
        )
    )
    seal_path = (
        event_root
        / "rebuild_authority"
        / f"{gate.TEST_SEAL_EVENT_PREFIX}_20260722T000000000001Z.json"
    )
    authority = publish_prefreeze_test_authority(
        entry_run_id=run_id,
        dataset_dir=dataset_dir,
        dataset_stem="v10_seq513_dataset__DIR_TRAIN_FIT",
        pair_lineage={
            "pair_generation_id": "1" * 64,
            "pair_manifest": {"path": "/immutable/pair.json", "sha256": "2" * 64},
            "pair_generation_root": "/immutable/pairs",
            "m1_lifecycle_source": {
                "path": "/immutable/m1.parquet",
                "sha256": "3" * 64,
            },
            "m1_feature_base": {"path": "/immutable/m1-features", "sha256": "4" * 64},
            "m5_feature_base": {"path": "/immutable/m5-features", "sha256": "5" * 64},
            "unified_exit_lifecycle_manifest": {
                "path": "/immutable/exit.json",
                "sha256": "6" * 64,
            },
        },
        source_lineage={
            "dataset_build_proof": {"path": "/immutable/build.json", "sha256": "7" * 64},
            "source_parquet": {"path": "/immutable/source.parquet", "sha256": "8" * 64},
            "canonical_v2_parquet": {"path": "/immutable/v2.parquet", "sha256": "9" * 64},
            "signal_manifest": {"path": "/immutable/signal.json", "sha256": "a" * 64},
            "feature_ranking": {"path": "/immutable/rank.json", "sha256": "b" * 64},
            # The TRAIN rank-reference artifact is retired; the producer's
            # exact source-lineage key set no longer accepts it.
            "position_size_train_ecdf": {
                "path": "/immutable/position-size-ecdf.npy",
                "sha256": "e" * 64,
            },
            "multi_tf_cache": {"path": "/immutable/mtf", "sha256": "d" * 64},
            "xau_tape_provenance": xau,
        },
        rebuild_terminal_json=rebuild_terminal_path,
        test_seal_json=seal_path,
    )
    test_manifest = dataset_dir / "v10_seq513_dataset__DIR_TRAIN_FIT_test.manifest.json"
    test_parquet = dataset_dir / "v10_seq513_dataset__DIR_TRAIN_FIT_test.parquet"
    test_manifest.unlink()
    test_parquet.unlink()

    preflight = _write_json(
        event_root
        / "preflight"
        / "ENTRY_MODEL_NATIVE_SEQ513_REBUILD_PREFLIGHT_20260722T000000000001Z.json",
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
            "data_splits": list(gate.PREFREEZE_SPLITS),
            "tape_provenance": {
                split: xau for split in gate.PREFREEZE_SPLITS
            },
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
            "dataset_rebuild_terminal": authority["rebuild_terminal"],
            "prefreeze_test_seal": authority["prefreeze_test_seal"],
        },
    )
    return (
        argparse.Namespace(
            run_id=run_id,
            event_root=str(event_root),
            repo_dir=str(tmp_path.resolve()),
            chain_terminal_json=str(terminal),
            test_seal_json=str(seal_path),
            test_seal_sha256=_sha256(seal_path),
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
    def _liveness(*_args, **kwargs):
        assert kwargs["expected_manifest_bindings"] == {
            split: {
                "path": split_values[f"{split}_manifest_json"],
                "sha256": split_values[f"{split}_manifest_sha256"],
            }
            for split in gate.SPLITS
        }
        return {
            "ok": True,
            "schema_version": gate.LIVENESS_SCHEMA_VERSION,
            "decision": "PASS",
            "sha256": _sha256(Path(args.full_input_liveness_json)),
            "field_order_sha256": {"signal": "a", "ctx_cont": "b", "ctx_cat": "c"},
            "atr_ood_status": "SHIFT_OBSERVED",
            "failures": [],
        }

    split_values = {
        key: getattr(args, key)
        for split in gate.SPLITS
        for key in (f"{split}_manifest_json", f"{split}_manifest_sha256")
    }
    monkeypatch.setattr(gate, "validate_full_input_liveness_artifact", _liveness)

    forbidden_test_paths = {
        Path(args.dataset_dir)
        / "v10_seq513_dataset__DIR_TRAIN_FIT_test.manifest.json",
        Path(args.dataset_dir) / "v10_seq513_dataset__DIR_TRAIN_FIT_test.parquet",
    }
    original_open = Path.open
    original_stat = Path.stat

    def _guarded_open(self, *open_args, **open_kwargs):
        assert self not in forbidden_test_paths, f"TEST bytes opened: {self}"
        return original_open(self, *open_args, **open_kwargs)

    def _guarded_stat(self, *stat_args, **stat_kwargs):
        assert self not in forbidden_test_paths, f"TEST path statted: {self}"
        return original_stat(self, *stat_args, **stat_kwargs)

    monkeypatch.setattr(Path, "open", _guarded_open)
    monkeypatch.setattr(Path, "stat", _guarded_stat)

    report = gate.run(args)

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["decision"] == READY_DECISION
    assert report["rebuild_completion_mode"] == "seq513_rebuild_chain_v7"
    assert report["dataset_dir"] == report["smoke_dataset_dir"]
    assert [row["name"] for row in report["checks"]] == list(REQUIRED_PROOF_CHECKS)
    assert all(row["ok"] for row in report["checks"])
    assert report["failures"] == []
    assert report["xau_tape_provenance"] == xau
    assert set(report["split_artifacts"]) == {"train", "val", "test"}
    assert report["split_artifacts"]["test"]["access_mode"] == (
        gate.TEST_SEAL_ACCESS_POLICY
    )
    assert report["test_isolation"]["test_dataset_bytes_read"] is False
    assert report["test_isolation"]["test_manifest_bytes_read"] is False
    assert report["test_isolation"]["test_paths_resolved_or_statted"] is False
    for path in forbidden_test_paths:
        with pytest.raises(FileNotFoundError):
            original_stat(path)
    assert report["side_effects_started"] == {
        "dataset_rebuild": False,
        "training": False,
        "replay": False,
        "iql_distillation": False,
        "shadow": False,
        "live": False,
    }


def test_post_rebuild_readiness_accepts_strict_native_v4_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args, xau = _fixture(
        tmp_path,
        xau_schema="xau_canonical_native_source_v4",
    )
    monkeypatch.setattr(
        gate,
        "validate_xau_tape_provenance_v1",
        lambda *a, **k: xau,
    )
    monkeypatch.setattr(
        gate,
        "_git_identity",
        lambda repo: {
            "repo_dir": str(repo),
            "head": "a" * 40,
            "status_short": [],
        },
    )
    monkeypatch.setattr(
        gate,
        "validate_full_input_liveness_artifact",
        lambda *a, **k: {"ok": True},
    )

    report = gate.run(args)

    assert report["decision"] == READY_DECISION
    assert report["xau_tape_provenance"]["schema_version"] == (
        "xau_canonical_native_source_v4"
    )


def test_post_rebuild_readiness_rejects_standalone_build_proof_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, xau = _fixture(tmp_path)
    dataset_dir = Path(args.dataset_dir)
    chain_terminal = json.loads(
        Path(args.chain_terminal_json).read_text(encoding="utf-8")
    )
    proof = _write_json(
        dataset_dir / gate.DIRECT_BUILD_PROOF_FILENAME,
        {
            "entry_run_id": args.run_id,
            "prefreeze_test_seal": chain_terminal["prefreeze_test_seal"],
        },
    )
    args.chain_terminal_json = str(proof)
    monkeypatch.setattr(gate, "validate_xau_tape_provenance_v1", lambda *a, **k: xau)
    monkeypatch.setattr(
        gate,
        "_git_identity",
        lambda repo: {"repo_dir": str(repo), "head": "a" * 40, "status_short": []},
    )
    monkeypatch.setattr(
        gate,
        "validate_full_input_liveness_artifact",
        lambda *a, **k: {"ok": True},
    )

    with pytest.raises(SystemExit) as exc_info:
        gate.run(args)

    assert exc_info.value.code == 2
    reports = list(Path(args.out_dir).glob(f"{gate.EVENT_PREFIX}_*.json"))
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["rebuild_completion_mode"] == "invalid"
    assert report["checks"][0]["ok"] is False
    assert report["checks"][0]["details"]["standalone_build_proof_allowed"] is False
    assert report["checks"][0]["details"]["standalone_build_proof_supplied"] is True


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


def test_post_rebuild_readiness_rejects_stale_v5_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, xau = _fixture(tmp_path)
    preflight_path = Path(args.rebuild_preflight_json)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["schema_version"] = "entry_model_native_seq513_rebuild_preflight_v5"
    _write_json(preflight_path, preflight)
    terminal_path = Path(args.chain_terminal_json)
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["preflight"]["sha256"] = _sha256(preflight_path)
    _write_json(terminal_path, terminal)
    monkeypatch.setattr(
        gate, "validate_xau_tape_provenance_v1", lambda *a, **k: xau
    )
    monkeypatch.setattr(
        gate,
        "_git_identity",
        lambda repo: {"repo_dir": str(repo), "head": "a" * 40, "status_short": []},
    )
    monkeypatch.setattr(
        gate,
        "validate_full_input_liveness_artifact",
        lambda *a, **k: {"ok": True},
    )

    with pytest.raises(SystemExit) as exc_info:
        gate.run(args)

    assert exc_info.value.code == 2
    reports = list(
        Path(args.out_dir).glob("ENTRY_MODEL_NATIVE_SEQ513_POST_REBUILD_*.json")
    )
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["decision"] == "BLOCKED_MODEL_NATIVE_SEQ513_POST_REBUILD"
    assert report["checks"][1]["ok"] is False


def test_post_rebuild_readiness_fails_closed_without_completion_bound_test_seal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args, xau = _fixture(tmp_path)
    terminal_path = Path(args.chain_terminal_json)
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal.pop("prefreeze_test_seal")
    _write_json(terminal_path, terminal)
    monkeypatch.setattr(
        gate, "validate_xau_tape_provenance_v1", lambda *a, **k: xau
    )
    monkeypatch.setattr(
        gate,
        "_git_identity",
        lambda repo: {"repo_dir": str(repo), "head": "a" * 40, "status_short": []},
    )
    monkeypatch.setattr(
        gate,
        "validate_full_input_liveness_artifact",
        lambda *a, **k: {"ok": True},
    )

    with pytest.raises(SystemExit) as exc_info:
        gate.run(args)

    assert exc_info.value.code == 2
    reports = list(Path(args.out_dir).glob(f"{gate.EVENT_PREFIX}_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["checks"][-1]["ok"] is False
    assert "lacks one exact content-bound" in report["checks"][-1]["details"][
        "error"
    ]
    assert set(report["split_artifacts"]) == {"train", "val"}


def test_post_rebuild_readiness_rejects_stale_test_seal_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args, xau = _fixture(tmp_path)
    terminal_path = Path(args.chain_terminal_json)
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    seal_path = Path(terminal["prefreeze_test_seal"]["path"])
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["schema_version"] = "entry_model_native_seq513_untouched_test_seal_v0"
    seal_without_binding = dict(seal)
    seal_without_binding.pop("content_binding_sha256")
    seal["content_binding_sha256"] = _canonical_sha256(seal_without_binding)
    _write_json(seal_path, seal)
    terminal["prefreeze_test_seal"]["sha256"] = _sha256(seal_path)
    _write_json(terminal_path, terminal)
    args.test_seal_sha256 = _sha256(seal_path)
    monkeypatch.setattr(
        gate, "validate_xau_tape_provenance_v1", lambda *a, **k: xau
    )
    monkeypatch.setattr(
        gate,
        "_git_identity",
        lambda repo: {"repo_dir": str(repo), "head": "a" * 40, "status_short": []},
    )
    monkeypatch.setattr(
        gate,
        "validate_full_input_liveness_artifact",
        lambda *a, **k: {"ok": True},
    )

    with pytest.raises(SystemExit) as exc_info:
        gate.run(args)

    assert exc_info.value.code == 2
    reports = list(Path(args.out_dir).glob(f"{gate.EVENT_PREFIX}_*.json"))
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["checks"][-1]["ok"] is False
    assert report["test_isolation"]["decision"] == "REJECTED"


def test_post_rebuild_readiness_rejects_legacy_caller_test_artifacts(
    tmp_path: Path,
) -> None:
    args, _ = _fixture(tmp_path)
    args.test_parquet = str(
        Path(args.dataset_dir) / "v10_seq513_dataset__DIR_TRAIN_FIT_test.parquet"
    )

    with pytest.raises(RuntimeError, match="caller-supplied TEST artifacts"):
        gate.run(args)


def test_post_rebuild_parser_accepts_only_seal_not_direct_test_artifacts() -> None:
    option_strings = {
        option
        for action in gate.build_parser()._actions
        for option in action.option_strings
    }
    assert {"--test-seal-json", "--test-seal-sha256"}.issubset(option_strings)
    assert {
        "--test-manifest-json",
        "--test-manifest-sha256",
        "--test-parquet",
        "--test-parquet-sha256",
    }.isdisjoint(option_strings)
