from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from gx1.contracts.entry_foundation_audit_policy_v1 import (
    FOUNDATION_AUDIT_DATA_SPLITS,
    FOUNDATION_AUDIT_POLICY_SHA256,
    foundation_audit_policy_binding,
    foundation_audit_policy_enforcement,
)
from gx1.contracts.entry_dataset_split_artifacts_v1 import (
    ENTRY_DATASET_SPLIT_ARTIFACTS_SCHEMA_VERSION,
)
from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_BASE_ACTIVE_HEADS,
    MODEL_NATIVE_BLOCKED_HEADS,
    MODEL_NATIVE_REQUIRED_SPECIALISTS,
    model_native_readiness_contract_metadata,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_BASE_SIGNAL_DIM,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
    model_native_signal_contract_metadata,
)
from gx1.contracts.immutable_event_authority_v1 import (
    ImmutableEventAuthorityError,
    write_immutable_json_event,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT,
)
from gx1.scripts.verify_entry_foundation_adoption_candidate_v1 import (
    AUDIT_EVENT_PREFIXES,
    EVENT_PREFIX,
    SMOKE_DATASET_SCHEMA,
    SMOKE_EVENT_PREFIX,
    SMOKE_REPORT_DECISION,
    SMOKE_REPORT_SCHEMA,
    SMOKE_SPLIT_SCHEMA,
    build_parser,
    run,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields
from tests.model_native_offline_rl_support import (
    model_native_target_audit_evidence,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_json(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataset(root: Path) -> tuple[Path, dict[str, dict]]:
    dataset = root / "model_native_seq513_dataset"
    dataset.mkdir()
    contract = model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.adoption_fixture"
        )
    )
    rows: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        parquet = dataset / f"candidate_{split}.parquet"
        manifest = dataset / f"candidate_{split}.manifest.json"
        parquet.write_bytes(f"{split}-model-native-seq513".encode("utf-8"))
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": SMOKE_SPLIT_SCHEMA,
                    "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
                    "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
                    "output_data_path": str(parquet.resolve()),
                    "extra": {
                        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
                        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
                        "neutral_xgb_bridge": False,
                        "model_native_signal_contract": contract,
                        "signal_bridge": {
                            "fields": contract["fields"],
                            "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                            "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                        },
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        rows[split] = {
            "parquet": parquet.resolve(),
            "manifest": manifest.resolve(),
            "parquet_sha256": _sha(parquet),
            "manifest_sha256": _sha(manifest),
        }
    return dataset.resolve(), rows


def _event(
    root: Path,
    prefix: str,
    stamp: str,
    payload: dict,
) -> Path:
    created = f"2026-07-16T12:00:{stamp}+00:00"
    path, _ = write_immutable_json_event(
        root,
        prefix,
        {"created_utc": created, **payload},
    )
    return path


def _audits(root: Path, dataset: Path) -> dict[str, Path]:
    audit_dir = root / "audits"
    split_artifacts = {
        split: {
            "manifest_path": str(
                (dataset / f"candidate_{split}.manifest.json").resolve()
            ),
            "manifest_sha256": _sha(
                dataset / f"candidate_{split}.manifest.json"
            ),
            "parquet_path": str(
                (dataset / f"candidate_{split}.parquet").resolve()
            ),
            "parquet_sha256": _sha(dataset / f"candidate_{split}.parquet"),
        }
        for split in FOUNDATION_AUDIT_DATA_SPLITS
    }
    split_identity = {
        "split_artifacts_schema_version": (
            ENTRY_DATASET_SPLIT_ARTIFACTS_SCHEMA_VERSION
        ),
        "split_artifacts": split_artifacts,
    }
    selected = canonical_model_native_selected_fields(
        remainder_prefix="session_regime.adoption_fixture"
    )
    signal_contract = model_native_signal_contract_metadata(selected)
    ranked_remainder = selected[MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT:]
    feature = _event(
        audit_dir,
        AUDIT_EVENT_PREFIXES["feature_audit"],
        "01.000001",
        {
            "schema_version": "entry_feature_foundation_audit_v1",
            **foundation_audit_policy_binding(),
            "foundation_audit_policy_enforcement": (
                foundation_audit_policy_enforcement("feature")
            ),
            "decision": "PASS",
            "failures": [],
            "dataset_dir": str(dataset),
            "data_splits": list(FOUNDATION_AUDIT_DATA_SPLITS),
            **split_identity,
            "model_native_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
            "base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
            "base_signal_fields": list(MODEL_NATIVE_BASE_FIELDS),
            "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
            "manifest_selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
            "mandatory_selected_feature_count": (
                MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
            ),
            "manifest_mandatory_selected_feature_count": (
                MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
            ),
            "ranked_remainder_feature_count": (
                MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
            ),
            "manifest_ranked_remainder_feature_count": (
                MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
            ),
            "ranked_remainder_fields_sha256": _sha_json(ranked_remainder),
            "feature_ranking_fit_scope": "train_only",
            "feature_ranking_sha256": "a" * 64,
            "model_native_signal_contract": signal_contract,
            "foundation_objective_coverage_all_present": True,
            "foundation_objective_liveness_all_live": True,
            "foundation_source_field_liveness_all_live": True,
        },
    )
    target = _event(
        audit_dir,
        AUDIT_EVENT_PREFIXES["target_audit"],
        "02.000002",
        {
            "schema_version": "entry_target_foundation_audit_v2",
            **foundation_audit_policy_binding(),
            "foundation_audit_policy_enforcement": (
                foundation_audit_policy_enforcement("target")
            ),
            "decision": "PASS",
            "failures": [],
            "dataset_dir": str(dataset),
            "data_splits": list(FOUNDATION_AUDIT_DATA_SPLITS),
            **split_identity,
            "target_head_contract": {
                **model_native_target_audit_evidence()["target_head_contract"],
            },
            "model_native_aux_target_contract": (
                model_native_target_audit_evidence()[
                    "model_native_aux_target_contract"
                ]
            ),
            "offline_rl_target_contract": model_native_target_audit_evidence()[
                "offline_rl_target_contract"
            ],
        },
    )
    specialist = _event(
        audit_dir,
        AUDIT_EVENT_PREFIXES["specialist_audit"],
        "03.000003",
        {
            "schema_version": "entry_specialist_feature_group_audit_v1",
            **foundation_audit_policy_binding(),
            "foundation_audit_policy_enforcement": (
                foundation_audit_policy_enforcement("specialist")
            ),
            "decision": "PASS",
            "failures": [],
            "dataset_dir": str(dataset),
            "data_splits": list(FOUNDATION_AUDIT_DATA_SPLITS),
            **split_identity,
            "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "signal_field_count": MODEL_NATIVE_SIGNAL_DIM,
            "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
            "required_training_specialists": list(
                MODEL_NATIVE_REQUIRED_SPECIALISTS
            ),
            "specialist_model_contract": MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT,
            "specialist_model_contract_valid": True,
            "specialist_input_liveness_all_live": True,
            "foundation_objective_routing_all_present_and_expected": True,
            "architecture_contract": {
                "input_dim": MODEL_NATIVE_SIGNAL_DIM,
                "recommended_fusion": {
                    "active_heads": list(MODEL_NATIVE_BASE_ACTIVE_HEADS),
                    "blocked_heads": list(MODEL_NATIVE_BLOCKED_HEADS),
                },
            },
        },
    )
    return {
        "feature_audit": feature,
        "target_audit": target,
        "specialist_audit": specialist,
    }


def _smoke_event(
    root: Path,
    dataset: Path,
    dataset_rows: dict[str, dict],
) -> Path:
    split_artifacts: dict[str, dict] = {}
    embedded_splits: dict[str, dict] = {}
    for split, row in dataset_rows.items():
        split_artifacts[split] = {
            "rows": 3,
            "output_data_path": str(row["parquet"]),
            "manifest_path": str(row["manifest"]),
            "parquet_sha256": row["parquet_sha256"],
            "manifest_sha256": row["manifest_sha256"],
            "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
            "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
            "field_count": MODEL_NATIVE_SIGNAL_DIM,
        }
        embedded_splits[split] = {
            "rows": 3,
            "out_parquet": str(row["parquet"]),
            "out_manifest": str(row["manifest"]),
            "out_parquet_sha256": row["parquet_sha256"],
            "out_manifest_sha256": row["manifest_sha256"],
            "split_manifest_schema_version": SMOKE_SPLIT_SCHEMA,
            "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
            "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
            "field_count": MODEL_NATIVE_SIGNAL_DIM,
        }
    embedded = {
        "schema_version": SMOKE_DATASET_SCHEMA,
        "report_only": True,
        "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
        "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
        "dataset_dir": str(dataset),
        "splits": embedded_splits,
    }
    return _event(
        root / "smoke",
        SMOKE_EVENT_PREFIX,
        "04.000004",
        {
            "schema_version": SMOKE_REPORT_SCHEMA,
            "decision": SMOKE_REPORT_DECISION,
            "report_only": True,
            "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
            "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
            "smart_smoke_dataset_dir": str(dataset),
            "manifest_embedded": True,
            "manifest_sha256": _sha_json(embedded),
            "smoke_manifest": embedded,
            "split_artifacts": split_artifacts,
            "model_native_readiness_contract": (
                model_native_readiness_contract_metadata()
            ),
            "failures": [],
            "training_allowed": False,
            "replay_allowed": False,
            "shadow_live_allowed": False,
            "side_effects_started": {
                "dataset_rebuild": False,
                "training": False,
                "replay": False,
                "shadow": False,
                "live": False,
            },
        },
    )


def _args(
    root: Path,
    dataset: Path,
    audits: dict[str, Path],
    smoke: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_dir=str(dataset),
        feature_audit_json=str(audits["feature_audit"]),
        target_audit_json=str(audits["target_audit"]),
        specialist_audit_json=str(audits["specialist_audit"]),
        smoke_manifest_json=str(smoke),
        out_dir=str(root / "reports"),
        quiet=True,
    )


def test_model_native_adoption_produces_one_report_only_immutable_event(
    tmp_path: Path,
) -> None:
    dataset, rows = _dataset(tmp_path)
    audits = _audits(tmp_path, dataset)
    smoke = _smoke_event(tmp_path, dataset, rows)

    report = run(_args(tmp_path, dataset, audits, smoke))

    assert report["decision"] == "READY_FOR_MODEL_NATIVE_ADOPTION_REVIEW"
    assert report["adoption_evidence_ready"] is True
    assert report["candidate_ready_for_activation"] is False
    assert report["training_allowed"] is False
    assert report["direction_selection_authority"] is False
    assert report["failures"] == []
    assert report["model_native_readiness_contract"] == (
        model_native_readiness_contract_metadata()
    )
    assert report["foundation_audit_policy_sha256"] == (
        FOUNDATION_AUDIT_POLICY_SHA256
    )
    event = Path(report["json_path"])
    assert event.name.startswith(f"{EVENT_PREFIX}_")
    assert list((tmp_path / "reports").iterdir()) == [event]


def test_adoption_uses_smoke_bound_split_identity_not_directory_inventory(
    tmp_path: Path,
) -> None:
    dataset, rows = _dataset(tmp_path)
    audits = _audits(tmp_path, dataset)
    smoke = _smoke_event(tmp_path, dataset, rows)
    (dataset / "unbound_decoy_train.parquet").write_bytes(b"decoy")
    (dataset / "unbound_decoy_train.manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )

    report = run(_args(tmp_path, dataset, audits, smoke))

    assert report["adoption_evidence_ready"] is True
    assert all(
        "unbound_decoy" not in value for value in report["artifacts"].values()
    )


def test_adoption_rejects_smoke_bound_split_hash_mismatch(tmp_path: Path) -> None:
    dataset, rows = _dataset(tmp_path)
    audits = _audits(tmp_path, dataset)
    smoke = _smoke_event(tmp_path, dataset, rows)
    payload = json.loads(smoke.read_text(encoding="utf-8"))
    payload["smoke_manifest"]["splits"]["val"]["out_parquet_sha256"] = "0" * 64
    smoke.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="ARTIFACT_HASH_MISMATCH"):
        run(_args(tmp_path, dataset, audits, smoke))

    assert not (tmp_path / "reports").exists()


def test_adoption_fails_closed_on_audit_dataset_mismatch(tmp_path: Path) -> None:
    dataset, rows = _dataset(tmp_path)
    audits = _audits(tmp_path, dataset)
    feature = json.loads(audits["feature_audit"].read_text(encoding="utf-8"))
    feature["dataset_dir"] = str(tmp_path / "other_dataset")
    audits["feature_audit"].write_text(
        json.dumps(feature, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    smoke = _smoke_event(tmp_path, dataset, rows)

    report = run(_args(tmp_path, dataset, audits, smoke))

    assert report["decision"] == "BLOCKED_MODEL_NATIVE_ADOPTION_REVIEW"
    assert report["adoption_evidence_ready"] is False
    assert any(
        row["gate"] == "feature_audit"
        and row["check"]
        == "evidence dataset matches the explicit seq513 candidate"
        for row in report["failures"]
    )


def test_adoption_rejects_stale_audit_split_artifact_binding(
    tmp_path: Path,
) -> None:
    dataset, rows = _dataset(tmp_path)
    audits = _audits(tmp_path, dataset)
    target = json.loads(audits["target_audit"].read_text(encoding="utf-8"))
    target["split_artifacts"]["val"]["parquet_sha256"] = "0" * 64
    audits["target_audit"].write_text(
        json.dumps(target, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    smoke = _smoke_event(tmp_path, dataset, rows)

    report = run(_args(tmp_path, dataset, audits, smoke))

    assert report["adoption_evidence_ready"] is False
    assert any(
        row["gate"] == "target_audit"
        and row["check"]
        == "foundation audit is content-bound to exact candidate split artifacts"
        for row in report["failures"]
    )


def test_adoption_rejects_swapped_mandatory_signal_prefix(tmp_path: Path) -> None:
    dataset, rows = _dataset(tmp_path)
    audits = _audits(tmp_path, dataset)
    feature = json.loads(audits["feature_audit"].read_text(encoding="utf-8"))
    contract = feature["model_native_signal_contract"]
    contract["selected_fields"][0], contract["selected_fields"][1] = (
        contract["selected_fields"][1],
        contract["selected_fields"][0],
    )
    contract["fields"][MODEL_NATIVE_BASE_SIGNAL_DIM], contract["fields"][
        MODEL_NATIVE_BASE_SIGNAL_DIM + 1
    ] = (
        contract["fields"][MODEL_NATIVE_BASE_SIGNAL_DIM + 1],
        contract["fields"][MODEL_NATIVE_BASE_SIGNAL_DIM],
    )
    audits["feature_audit"].write_text(
        json.dumps(feature, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    smoke = _smoke_event(tmp_path, dataset, rows)

    report = run(_args(tmp_path, dataset, audits, smoke))

    assert report["adoption_evidence_ready"] is False
    partition_failure = next(
        row
        for row in report["failures"]
        if row["gate"] == "feature_audit"
        and row["check"]
            == "feature audit proves exact model-native 34 plus 377 plus 102 partition"
    )
    assert "mandatory_registry_prefix_order_violation" in partition_failure[
        "details"
    ]["contract_error"]


def test_adoption_rejects_stale_partition_count(tmp_path: Path) -> None:
    dataset, rows = _dataset(tmp_path)
    audits = _audits(tmp_path, dataset)
    feature = json.loads(audits["feature_audit"].read_text(encoding="utf-8"))
    feature["mandatory_selected_feature_count"] = (
        MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT - 1
    )
    audits["feature_audit"].write_text(
        json.dumps(feature, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    smoke = _smoke_event(tmp_path, dataset, rows)

    report = run(_args(tmp_path, dataset, audits, smoke))

    assert report["adoption_evidence_ready"] is False
    assert any(
        row["gate"] == "feature_audit"
        and row["check"]
            == "feature audit proves exact model-native 34 plus 377 plus 102 partition"
        for row in report["failures"]
    )


def test_adoption_rejects_mutable_latest_evidence(tmp_path: Path) -> None:
    dataset, rows = _dataset(tmp_path)
    audits = _audits(tmp_path, dataset)
    smoke = _smoke_event(tmp_path, dataset, rows)
    latest = audits["feature_audit"].with_name(
        f'{AUDIT_EVENT_PREFIXES["feature_audit"]}_latest.json'
    )
    latest.write_bytes(audits["feature_audit"].read_bytes())
    args = _args(tmp_path, dataset, audits, smoke)
    args.feature_audit_json = str(latest)

    with pytest.raises(ImmutableEventAuthorityError):
        run(args)

    assert not (tmp_path / "reports").exists()


def test_adoption_fails_closed_on_forged_foundation_audit_policy(
    tmp_path: Path,
) -> None:
    dataset, rows = _dataset(tmp_path)
    audits = _audits(tmp_path, dataset)
    target = json.loads(audits["target_audit"].read_text(encoding="utf-8"))
    target["foundation_audit_policy"]["target_quality"][
        "max_majority_rate"
    ] = 1.0
    audits["target_audit"].write_text(
        json.dumps(target, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    smoke = _smoke_event(tmp_path, dataset, rows)

    report = run(_args(tmp_path, dataset, audits, smoke))

    assert report["adoption_evidence_ready"] is False
    assert any(
        row["gate"] == "target_audit"
        and row["check"]
        == "foundation audit policy identity and full payload are exact"
        for row in report["failures"]
    )


def test_parser_requires_explicit_smoke_event_and_output() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    help_text = parser.format_help()
    assert "--smoke-manifest-json" in help_text
    assert "--smoke-dataset-dir" not in help_text
    assert "_latest.json" not in help_text
    assert "fail-on-not-ready" not in help_text


def test_adoption_source_has_no_split_glob_or_stem_inference() -> None:
    source = Path(
        "gx1/scripts/verify_entry_foundation_adoption_candidate_v1.py"
    ).read_text(encoding="utf-8")
    assert "_split_candidates" not in source
    assert 'glob(f"*_{split}' not in source
