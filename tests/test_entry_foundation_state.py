import argparse
import hashlib
import json
from pathlib import Path

import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    model_native_signal_contract_metadata,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.scripts.verify_entry_foundation_state_v1 import (
    EVIDENCE_SPECS,
    STATE_BLOCKED_DECISION,
    STATE_EVENT_PREFIX,
    STATE_PROVEN_DECISION,
    build_parser,
    run,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields


REPO = Path(__file__).resolve().parents[1]


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contract() -> dict:
    return model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.foundation_state_fixture"
        )
    )


def _publish(root: Path, prefix: str, payload: dict) -> Path:
    payload = {
        "created_utc": "2026-07-16T12:00:00.123456+00:00",
        **payload,
    }
    path, _ = write_immutable_json_event(root, prefix, payload)
    return path


def _evidence_events(tmp_path: Path, *, broken_preflight: bool = False) -> dict[str, Path]:
    contract = _contract()
    preflight_contract = {
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": "model_native",
        "seq_input_dim": 513,
        "snap_input_dim": 513,
        "seq_len": 96,
        "ctx_cont_dim": 142,
        "ctx_cat_dim": 5,
        "base_signal_dim": 34,
        "selected_feature_count": 479,
        "bridge_dim": 0,
        "bridge_source": None,
        "anchor_source": None,
    }
    if broken_preflight:
        preflight_contract["contract_mode"] = "foundation_seq146"
    preflight = _publish(
        tmp_path / "preflight",
        "ENTRY_MODEL_NATIVE_SEQ513_REBUILD_PREFLIGHT",
        {
            "schema_version": "entry_model_native_seq513_rebuild_preflight_v4",
            "decision": "READY_FOR_MODEL_NATIVE_SEQ513_REBUILD",
            "report_only": True,
            "side_effects_started": {"dataset_rebuild": False, "training": False},
            "failures": [],
            "required_model_native_contract": preflight_contract,
            "rebuild_command_contract": {
                "model_native_signal_contract": contract,
            },
        },
    )

    smoke_manifest_payload = {
        "model_native_signal_contract": contract,
    }
    post_rebuild = {"path": "/immutable/post-rebuild.json", "sha256": "a" * 64}
    specialist = {"path": "/immutable/specialist.json", "sha256": "b" * 64}
    split_artifacts = {
        "train": {"model_native_signal_contract": contract},
        "val": {"model_native_signal_contract": contract},
        "test": {"model_native_signal_contract": contract},
    }
    smoke_manifest = _publish(
        tmp_path / "smoke_manifest",
        "ENTRY_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST",
        {
            "schema_version": "entry_model_native_seq513_smoke_manifest_v2",
            "decision": "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST_REVIEW",
            "report_only": True,
            "side_effects_started": {"training": False, "replay": False},
            "failures": [],
            "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
            "expected_seq_snap_width": 513,
            "smoke_manifest": smoke_manifest_payload,
            "manifest_sha256": _sha256_json(smoke_manifest_payload),
            "post_rebuild_readiness": post_rebuild,
            "specialist_audit": specialist,
            "split_artifacts": split_artifacts,
            "evidence_binding_sha256": _sha256_json(
                {
                    "post_rebuild_readiness": post_rebuild,
                    "specialist_audit": specialist,
                    "split_artifacts": split_artifacts,
                }
            ),
        },
    )

    smoke_inputs = {"manifest": {"path": str(smoke_manifest), "sha256": "c" * 64}}
    smoke_readiness = _publish(
        tmp_path / "smoke_readiness",
        "ENTRY_MODEL_NATIVE_SEQ513_SMOKE_READINESS",
        {
            "schema_version": "entry_model_native_seq513_smoke_readiness_v2",
            "decision": "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_READINESS_REVIEW",
            "report_only": True,
            "side_effects_started": {"training": False, "live": False},
            "failures": [],
            "smart_candidate": {
                "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
                "specialist_contract_mode": MODEL_NATIVE_CONTRACT_MODE,
                "expected_signal_dim": 513,
                "expected_selected_feature_count": 479,
            },
            "inputs": smoke_inputs,
            "evidence_binding_sha256": _sha256_json(smoke_inputs),
        },
    )

    trainability_inputs = {
        "smoke_readiness": {"path": str(smoke_readiness), "sha256": "d" * 64}
    }
    trainability = _publish(
        tmp_path / "trainability",
        "ENTRY_MODEL_NATIVE_SEQ513_TRAINABILITY_READINESS",
        {
            "schema_version": "entry_model_native_seq513_trainability_readiness_v1",
            "decision": "READY_FOR_MODEL_NATIVE_SEQ513_TRAINABILITY_REVIEW",
            "report_only": True,
            "side_effects_started": {"training": False, "live": False},
            "failures": [],
            "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
            "expected_signal_dim": 513,
            "inputs": trainability_inputs,
            "evidence_binding_sha256": _sha256_json(trainability_inputs),
        },
    )

    source = tmp_path / "candidate-source.json"
    source.write_text('{"proof":true}\n', encoding="utf-8")
    candidate = _publish(
        tmp_path / "candidate",
        "ENTRY_CANDIDATE_READINESS",
        {
            "schema_version": "entry_candidate_readiness_model_native_v1",
            "decision": "READY_FOR_CANDIDATE_TRAINING",
            "failures": [],
            "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "expected_signal_dim": 513,
            "edge_test_scope": "strict",
            "promotion_shadow_live_allowed": False,
            "artifact_fingerprints": {
                "proof": {
                    "path": str(source),
                    "exists": True,
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            },
        },
    )
    return {
        "rebuild_preflight": preflight,
        "smoke_manifest": smoke_manifest,
        "smoke_readiness": smoke_readiness,
        "trainability_readiness": trainability,
        "candidate_readiness": candidate,
    }


def _args(events: dict[str, Path], tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        rebuild_preflight_json=str(events["rebuild_preflight"]),
        smoke_manifest_json=str(events["smoke_manifest"]),
        smoke_readiness_json=str(events["smoke_readiness"]),
        trainability_readiness_json=str(events["trainability_readiness"]),
        candidate_readiness_json=str(events["candidate_readiness"]),
        out_dir=str(tmp_path / "state"),
        selftest=False,
        quiet=True,
    )


def _run_blocked(args: argparse.Namespace) -> dict:
    with pytest.raises(SystemExit) as exc_info:
        run(args)
    assert exc_info.value.code == 2
    paths = list(Path(args.out_dir).glob(f"{STATE_EVENT_PREFIX}_*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def test_state_proves_exact_seq513_but_never_launches(tmp_path: Path) -> None:
    report = run(_args(_evidence_events(tmp_path), tmp_path))
    assert report["decision"] == STATE_PROVEN_DECISION
    assert report["model_native_evidence_ready"] is True
    assert report["expected_signal_dim"] == 513
    assert report["launch_allowed"] is False
    assert report["promotion_shadow_live_allowed"] is False
    assert Path(report["json_path"]).name.startswith("ENTRY_MODEL_NATIVE_SEQ513_STATE_")
    assert all(row["ready"] for row in report["evidence"])


def test_state_fails_closed_on_retired_contract_or_missing_events(tmp_path: Path) -> None:
    report = _run_blocked(
        _args(_evidence_events(tmp_path, broken_preflight=True), tmp_path)
    )
    assert report["decision"] == STATE_BLOCKED_DECISION
    assert report["launch_allowed"] is False
    assert any("exact seq513 contract mismatch" in row["failure"] for row in report["failures"])

    parser_args = build_parser().parse_args(
        ["--out-dir", str(tmp_path / "missing_state"), "--quiet"]
    )
    missing = _run_blocked(parser_args)
    assert missing["decision"] == STATE_BLOCKED_DECISION
    assert len(missing["failures"]) == len(EVIDENCE_SPECS)


def test_state_rejects_mutable_latest_and_retired_cli_aliases(tmp_path: Path) -> None:
    events = _evidence_events(tmp_path)
    latest = events["candidate_readiness"].with_name(
        "ENTRY_CANDIDATE_READINESS_latest.json"
    )
    latest.write_bytes(events["candidate_readiness"].read_bytes())
    events["candidate_readiness"] = latest
    report = _run_blocked(_args(events, tmp_path))
    assert report["decision"] == STATE_BLOCKED_DECISION
    assert any("newest immutable" in row["failure"] for row in report["failures"])

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--foundation-seq146"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--challenger-seq215"])
    assert "fail-on-not-ready" not in parser.format_help()


def test_state_selftest_and_control_routes_are_model_native() -> None:
    report = run(argparse.Namespace(selftest=True, quiet=True))
    assert report["decision"] == "MODEL_NATIVE_SEQ513_STATE_SELFTEST_PASS"
    assert report["launch_allowed"] is False

    control = (REPO / "scripts/entry_next_edge_control.sh").read_text(
        encoding="utf-8"
    )
    assert "  model-native-state)" in control
    assert "  model-native-state-selftest)" in control
    assert "verify_entry_foundation_state_v1 --selftest" in control
    assert "verify|state)" not in control
    assert "readiness-report" not in control
