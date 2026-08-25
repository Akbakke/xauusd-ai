import argparse
import json
from pathlib import Path

import pytest

from gx1.contracts.entry_full_input_liveness_v1 import (
    SCHEMA_VERSION as LIVENESS_SCHEMA,
    validate_full_input_liveness_artifact,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTEXT_TAG,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.scripts import verify_entry_model_native_seq513_trainability_readiness_v1 as gate
from gx1.contracts.entry_full_input_liveness_v1 import EXPECTED_FIELD_COUNTS
from tests.entry_full_input_liveness_support import (
    full_input_field_order,
    write_full_input_liveness_fixture,
)


def _liveness_field_order() -> dict[str, list[str]]:
    """Trim the shared liveness fixture to the owner's exact ctx_cat width.

    ``tests/entry_full_input_liveness_support.full_input_field_order`` still
    emits five synthetic ``ctx_cat`` names from the retired five-categorical
    surface; the contract owner declares exactly
    ``EXPECTED_FIELD_COUNTS['ctx_cat']``. Trimming here keeps the fixture at
    the owner's width without restating it.
    """

    order = full_input_field_order()
    order["ctx_cat"] = order["ctx_cat"][: EXPECTED_FIELD_COUNTS["ctx_cat"]]
    return order


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _path_calibration_future_contract(
    wired: bool,
    source_dataset: str,
    smoke_run_id: str,
    dataset_run_id: str,
) -> dict:
    if not wired:
        return {}
    smoke_dataset = source_dataset
    out_bundle = str(
        Path(source_dataset).parent / "v10_entry_model_native_seq513_smoke_<STAMP>"
    )
    wrapper_argv = [
        "scripts/entry_next_edge_control.sh",
        "model-native-smoke-train",
        "--run-id",
        smoke_run_id,
        "--dataset-dir",
        smoke_dataset,
        "--out-bundle-dir",
        out_bundle,
        "--recipe-audit-json",
        "<IMMUTABLE_RECIPE_AUDIT_JSON>",
        "--post-rebuild-readiness-json",
        "<IMMUTABLE_POST_REBUILD_READINESS_JSON>",
    ]
    return {
        "profile": "smoke",
        "control_route": "model-native-smoke-train",
        "wrapper_path": "scripts/run_entry_model_native_seq513_train.sh",
        "entry_run_id": smoke_run_id,
        "dataset_run_id": dataset_run_id,
        "argv_template": wrapper_argv,
        "wrapper_argv_template": wrapper_argv,
        "requires_edge_audit": True,
        "recipe_audit_control_route_exposed": True,
        "recipe_audit_control_route": "model-native-train-recipe-audit",
        "recipe_audit_argv_template": [
            "scripts/entry_next_edge_control.sh",
            "model-native-train-recipe-audit",
        ],
        "post_smoke_prediction_control_route_exposed": True,
        "post_smoke_prediction_control_route": "model-native-selective-edge",
        "post_smoke_prediction_argv_template": [
            "scripts/entry_next_edge_control.sh",
            "model-native-selective-edge",
        ],
        "post_smoke_audit_control_route_exposed": True,
        "post_smoke_audit_control_route": "model-native-smoke-bundle-audit",
        "post_smoke_audit_argv_template": [
            "scripts/entry_next_edge_control.sh",
            "model-native-smoke-bundle-audit",
        ],
        "recipe_audit_schema": gate.RECIPE_AUDIT_SCHEMA,
        "recipe_env_keys": list(gate.MODEL_NATIVE_RECIPE_ENV_KEYS),
        "joint_task_names": list(gate.JOINT_TASK_NAMES),
        "training_objective_schema": gate.TRAINING_OBJECTIVE_SCHEMA,
        "requires_exact_model_native_training_objective": True,
        "requires_direction_diagnostic_recipe_contract": True,
        "direction_diagnostic_recipe_contract": dict(
            gate.DIRECTION_DIAGNOSTIC_RECIPE_CONTRACT
        ),
        "direction_diagnostic_env_template": dict(
            gate.DIRECTION_DIAGNOSTIC_ENV_TEMPLATE
        ),
        "requires_direction_context_slice_contract": True,
        "direction_context_slice_contract": dict(gate.DIRECTION_CONTEXT_SLICE_CONTRACT),
        "requires_canonical_direction_decision_contract": True,
        "canonical_direction_decision_contract": dict(
            gate.CANONICAL_DIRECTION_DECISION_CONTRACT
        ),
    }


def _audited_wrapper_text() -> str:
    return "\n".join(
        (
            "import gx1.contracts.entry_model_native_signal_v1 as s",
            "MODEL_NATIVE_CONTRACT_MODE=s.MODEL_NATIVE_CONTRACT_MODE",
            '--specialist-contract-mode "$MODEL_NATIVE_CONTRACT_MODE"',
            "gx1.contracts.entry_model_native_train_launch_v1",
            "--profile",
            "--recipe-audit-json",
            "--pretrain-audit-json",
            "--train-sequence-integrity-audit-json",
            "--val-sequence-integrity-audit-json",
            "--full-input-liveness-audit-json",
            "--post-rebuild-readiness-json",
            "--prefreeze-test-seal-json",
            "--prefreeze-test-seal-sha256",
            "--trainability-readiness-json",
            "--smoke-manifest-json",
            "--candidate-readiness-json",
            "smoke|candidate",
            "--run-id",
            "--execute",
        )
    )


def _args(tmp_path: Path, *, wired: bool, ctx_tag: str = MODEL_NATIVE_CONTEXT_TAG) -> argparse.Namespace:
    dataset_run_id = "MODEL_NATIVE_SEQ513_DATASET_TRAINABILITY_PYTEST"
    smoke_run_id = "MODEL_NATIVE_SEQ513_SMOKE_TRAINABILITY_PYTEST"
    post_rebuild = (
        tmp_path
        / "ENTRY_SMART_DATASET_POST_REBUILD_READINESS_20260716T120000123456Z.json"
    )
    smoke_readiness = (
        tmp_path / "ENTRY_MODEL_NATIVE_SEQ513_SMOKE_READINESS_20260716T120001123456Z.json"
    )
    source_dataset = str(
        (tmp_path / "fresh_rebuild" / "v10_dataset_6yr_smartctx_xau_direction_repair").resolve()
    )
    smoke_dataset = source_dataset
    full_input_liveness_path, full_input_liveness, _ = write_full_input_liveness_fixture(
        tmp_path / "full_input_liveness",
        dataset_dir=Path(source_dataset),
        field_order=_liveness_field_order(),
    )
    stamped_liveness_path = (
        tmp_path / "ENTRY_FULL_INPUT_LIVENESS_CONTRACT_20260716T120002123456Z.json"
    )
    stamped_liveness_path.write_bytes(full_input_liveness_path.read_bytes())
    full_input_liveness_path = stamped_liveness_path
    full_input_liveness_sha = gate._sha256_file(full_input_liveness_path)
    full_input_validation = validate_full_input_liveness_artifact(
        full_input_liveness_path,
        expected_sha256=full_input_liveness_sha,
        expected_dataset_dir=source_dataset,
        expected_field_order_sha256=full_input_liveness["field_order_sha256"],
    )
    _write_json(
        post_rebuild,
        {
            "schema_version": gate.POST_REBUILD_SCHEMA_VERSION,
            "decision": (
                gate.POST_REBUILD_READY_DECISION
                if wired
                else "BLOCKED_BY_ENTRY_SMART_DATASET_POST_REBUILD_AUDIT"
            ),
            "entry_run_id": dataset_run_id,
            "dataset_dir": source_dataset,
            "post_rebuild_refresh_command_contract": {
                "smoke_dataset_dir": smoke_dataset,
            },
            "full_input_liveness_contract": {
                "path": str(full_input_liveness_path),
                "sha256": full_input_liveness_sha,
                "schema_version": LIVENESS_SCHEMA,
                "decision": full_input_liveness["decision"],
                "field_order_sha256": full_input_liveness["field_order_sha256"],
                "field_counts": full_input_liveness["expected_field_counts"],
                "atr_ood_status": full_input_liveness["atr_ood_drift"]["status"],
            },
            "split_manifests": {
                split: {
                    "ctx_contract": {
                        "tag": ctx_tag,
                        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
                        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
                    }
                }
                for split in ("train", "val", "test")
            },
        },
    )
    _write_json(
        smoke_readiness,
        {
            "decision": (
                "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_READINESS_REVIEW"
                if wired
                else "BLOCKED_MODEL_NATIVE_SEQ513_SMOKE_READINESS"
            ),
            "entry_run_id": smoke_run_id,
            "dataset_run_id": dataset_run_id,
            "future_command_contracts": {
                "smart_smoke_train": {
                    "implemented_in_control_surface": wired,
                    "specialist_contract_mode": MODEL_NATIVE_CONTRACT_MODE,
                    **_path_calibration_future_contract(
                        wired,
                        source_dataset,
                        smoke_run_id,
                        dataset_run_id,
                    ),
                }
            },
            "inputs": {
                "smart_dataset_dir": source_dataset,
                "smart_smoke_dataset_dir": smoke_dataset,
                "full_input_liveness_contract": {
                    "path": str(full_input_liveness_path),
                    "exists": True,
                    "size_bytes": full_input_liveness_path.stat().st_size,
                    "sha256": full_input_liveness_sha,
                },
            },
            "full_input_liveness_validation": full_input_validation,
        },
    )
    if wired:
        control_text = (
            "Usage: model-native-smoke-train --run-id <id>\n"
            "case\n"
            "model-native-smoke-train) exec \"$REPO/scripts/run_entry_model_native_seq513_train.sh\" --profile smoke \"$@\" ;;\n"
            "model-native-candidate-train) exec \"$REPO/scripts/run_entry_model_native_seq513_train.sh\" --profile candidate \"$@\" ;;\n"
            "model-native-train-recipe-audit) exec recipe ;;\n"
            "model-native-selective-edge) exec prediction ;;\n"
            "model-native-smoke-bundle-audit) exec audit ;;\n"
        )
        train_wrapper_text = _audited_wrapper_text()
        smart_script_text = (
            "from gx1.contracts.entry_model_native_signal_v1 import (\n"
            "    MODEL_NATIVE_CONTRACT_MODE,\n"
            "    MODEL_NATIVE_SIGNAL_DIM,\n"
            ")\n"
            "CONTRACT_MODE = MODEL_NATIVE_CONTRACT_MODE\n"
            "EXPECTED_SIGNAL_DIM = MODEL_NATIVE_SIGNAL_DIM\n"
        )
    else:
        control_text = "smart-smoke-readiness)\n"
        train_wrapper_text = "--challenger-seq215 SPECIALIST_CONTRACT_MODE=challenger_seq215\n"
        smart_script_text = "challenger_seq215 215\n"
    trainer_text = (
        "--specialist-contract-mode\n"
        + "\n".join(gate.DIRECTION_DIAGNOSTIC_ENV_KEYS)
        + "\n"
    )
    return argparse.Namespace(
        smart_post_rebuild_readiness_json=str(post_rebuild),
        smart_smoke_readiness_json=str(smoke_readiness),
        full_input_liveness_json=str(full_input_liveness_path),
        control_script=str(_write(tmp_path / "entry_next_edge_control.sh", control_text)),
        trainer_source=str(_write(tmp_path / "entry_v10_ctx_train_v3.py", trainer_text)),
        train_wrapper=str(_write(tmp_path / "run_train.sh", train_wrapper_text)),
        candidate_readiness_script=str(_write(tmp_path / "candidate_readiness.py", smart_script_text)),
        selective_edge_script=str(_write(tmp_path / "selective_edge.py", smart_script_text)),
        out_dir=str(tmp_path / "reports"),
        quiet=True,
    )


def _run_blocked(args: argparse.Namespace) -> dict:
    with pytest.raises(SystemExit) as exc_info:
        gate.run(args)
    assert exc_info.value.code == 2
    paths = list(Path(args.out_dir).glob(f"{gate.EVENT_PREFIX}_*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def test_smart_trainability_blocks_until_train_surface_exists(tmp_path: Path) -> None:
    report = _run_blocked(_args(tmp_path, wired=False))

    assert report["decision"] == gate.BLOCKED_DECISION
    assert report["training_allowed"] is False
    assert report["candidate_training_allowed"] is False
    assert report["replay_allowed"] is False
    assert report["iql_allowed"] is False
    assert not any(report["side_effects_started"].values())
    assert (
        "canonical train wrapper exposes both explicit model-native profiles"
        in report["blockers"]
    )
    assert (
        "both model-native profiles use the canonical wrapper in control surface"
        in report["blockers"]
    )
    assert "smart smoke future contract declares direction context slice audit" in report["blockers"]


def test_smart_trainability_can_pass_when_all_surfaces_are_wired(monkeypatch, tmp_path: Path) -> None:
    report = gate.run(_args(tmp_path, wired=True))

    assert report["decision"] == gate.READY_DECISION
    assert report["expected_signal_dim"] == MODEL_NATIVE_SIGNAL_DIM
    assert report["source_metadata_contract"]["declared_ctx_contracts_match_expected"] is True
    assert report["source_metadata_contract"]["no_stale_ctx6cat6"] is True
    assert report["fresh_source_identity_contract"]["future_train_out_under_source_root"] is True
    assert report["run_lineage_contract"]["ok"] is True
    assert report["entry_run_id"] == "MODEL_NATIVE_SEQ513_SMOKE_TRAINABILITY_PYTEST"
    assert report["dataset_run_id"] == "MODEL_NATIVE_SEQ513_DATASET_TRAINABILITY_PYTEST"
    assert report["full_input_liveness_validation"]["ok"] is True
    assert report["training_allowed"] is False
    assert report["execution_allowed_now"] is False
    assert report["failures"] == []
    assert len(report["evidence_binding_sha256"]) == 64
    event_path = Path(report["json_path"])
    assert event_path.is_file()
    assert list((tmp_path / "reports").iterdir()) == [event_path]


def test_trainability_rejects_wrapper_that_restates_contract_mode(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, wired=True)
    wrapper = Path(args.train_wrapper)
    wrapper.write_text(
        _audited_wrapper_text().replace(
            "import gx1.contracts.entry_model_native_signal_v1 as s\n"
            "MODEL_NATIVE_CONTRACT_MODE=s.MODEL_NATIVE_CONTRACT_MODE\n",
            f"MODEL_NATIVE_CONTRACT_MODE={MODEL_NATIVE_CONTRACT_MODE}\n",
        ),
        encoding="utf-8",
    )

    report = _run_blocked(args)

    assert (
        "canonical train wrapper exposes both explicit model-native profiles"
        in report["blockers"]
    )


def test_smart_trainability_rejects_duplicated_contract_literals_without_ssot_import(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, wired=True)
    Path(args.selective_edge_script).write_text(
        f"CONTRACT_MODE = {MODEL_NATIVE_CONTRACT_MODE!r}\n"
        f"EXPECTED_SIGNAL_DIM = {MODEL_NATIVE_SIGNAL_DIM}\n",
        encoding="utf-8",
    )

    report = _run_blocked(args)

    assert "selective-edge supports model-native seq513" in report["blockers"]
    failed = next(
        row
        for row in report["failures"]
        if row["name"] == "selective-edge supports model-native seq513"
    )
    assert failed["details"]["contract_binding"]["imports_exact_contract_owner"] is False


def test_smart_trainability_blocks_mixed_fresh_and_stale_smoke_reports(tmp_path: Path) -> None:
    args = _args(tmp_path, wired=True)
    smoke_path = Path(args.smart_smoke_readiness_json)
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["inputs"]["smart_smoke_dataset_dir"] = "/tmp/stale_xau_rebuild/v10_dataset_6yr_smartctx_xau_direction_repair_smoke"
    _write_json(smoke_path, smoke)

    report = _run_blocked(args)

    assert report["decision"] == gate.BLOCKED_DECISION
    assert "smart smoke readiness uses same smoke dataset as post-rebuild contract" in report["blockers"]
    assert report["training_allowed"] is False


def test_smart_trainability_blocks_mixed_training_run_lineage(tmp_path: Path) -> None:
    args = _args(tmp_path, wired=True)
    smoke_path = Path(args.smart_smoke_readiness_json)
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["future_command_contracts"]["smart_smoke_train"][
        "entry_run_id"
    ] = "MODEL_NATIVE_SEQ513_OTHER_TRAINABILITY_PYTEST"
    _write_json(smoke_path, smoke)

    report = _run_blocked(args)

    assert (
        "model-native smoke trainability preserves one distinct exact run lineage"
        in report["blockers"]
    )
    assert report["run_lineage_contract"]["ok"] is False


def test_smart_trainability_blocks_liveness_bytes_changed_after_smoke(tmp_path: Path) -> None:
    args = _args(tmp_path, wired=True)
    liveness_path = Path(args.full_input_liveness_json)
    liveness_path.write_text(liveness_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    report = _run_blocked(args)

    assert report["decision"] == gate.BLOCKED_DECISION
    assert (
        "full-input liveness artifact hash schema fields and ATR shift observation validate for trainability"
        in report["blockers"]
    )
    assert any(
        row["code"] == "artifact_sha256_mismatch"
        for row in report["full_input_liveness_validation"]["failures"]
    )


def test_smart_trainability_blocks_stale_ctx6cat6_source_metadata(tmp_path: Path) -> None:
    report = _run_blocked(_args(tmp_path, wired=True, ctx_tag="CTX6CAT6"))

    assert report["decision"] == gate.BLOCKED_DECISION
    assert "smart source metadata has no stale CTX6CAT6 ctx contract" in report["blockers"]
    assert (
        "declared smart source ctx metadata matches "
        f"{MODEL_NATIVE_CONTEXT_TAG}"
    ) in report["blockers"]
    assert report["source_metadata_contract"]["stale_ctx6cat6_paths"]
    assert report["training_allowed"] is False


def test_parser_and_source_require_explicit_evidence_and_publish_one_event() -> None:
    parser = gate.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    assert "_latest.json" not in parser.format_help()

    source = Path(gate.__file__).read_text(encoding="utf-8")
    assert source.count("write_immutable_json_event(") == 1
    assert "replace_latest_json_mirror" not in source
    assert ".md\"" not in source
    assert "smart_seq520" not in source.lower()
    assert "ENTRY_HIER_POCKET" not in source
    assert "ENTRY_TRENDLINE_RAIL_WRONG" not in source
    assert "fail-on-not-ready" not in source
