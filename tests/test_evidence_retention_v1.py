from __future__ import annotations

import json
from pathlib import Path

import pytest

import gx1.contracts.evidence_retention_v1 as retention_contract
import gx1.scripts.cleanup_gx1_evidence_v1 as cleanup_script
from gx1.contracts.evidence_retention_v1 import (
    PLAN_EVENT_PREFIX,
    EvidenceRetentionError,
    build_cleanup_plan_payload,
    sha256_file,
    validate_cleanup_plan,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.scripts.cleanup_gx1_evidence_v1 import (
    APPROVAL_PREFIX,
    CLEARANCE_PREFIX,
    EXECUTION_PREFIX,
    RECOVERY_PREFIX,
    STAGED_PREFIX,
    execute_cleanup,
    publish_cleanup_approval,
    recover_interrupted_cleanup,
    resume_interrupted_cleanup,
)


VEDTAK = "GX1-CLEANUP-TEST"
CREATED_UTC = "2026-07-20T10:00:00+00:00"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _isolated_delete_incident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    incident = tmp_path / "incident.json"
    _write_json(incident, {
        "schema_version": "gx1_entry_iql_delete_incident_v1",
        "project": "XAUUSD",
    })
    monkeypatch.setattr(retention_contract, "CANONICAL_DELETE_INCIDENT", incident)


def _authority_files(
    tmp_path: Path,
    *,
    active: object | None = None,
    retired: object | None = None,
    history: object | None = None,
    launch_extra: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    registry = tmp_path / "registry.json"
    launch = tmp_path / "launch.json"
    _write_json(
        registry,
        {
            "schema_version": "gx1_artifact_selection_v2",
            "project": "XAUUSD",
            "active": {} if active is None else active,
            "retired": {} if retired is None else retired,
            "history": [] if history is None else history,
        },
    )
    _write_json(
        launch,
        {
            "schema_version": "gx1_xau_direction_launch_state_v1",
            "project": "XAUUSD",
            **({} if launch_extra is None else launch_extra),
        },
    )
    return registry, launch


def _published_plan(
    tmp_path: Path,
    *,
    target: Path,
    registry: Path,
    launch: Path,
) -> tuple[Path, str]:
    payload = build_cleanup_plan_payload(
        targets=[target],
        reason="Exact obsolete test evidence",
        vedtak=VEDTAK,
        artifact_registry_json=registry,
        launch_contract_json=launch,
        inventory_dir=tmp_path / "plans",
        created_utc=CREATED_UTC,
        allowed_roots=(tmp_path,),
    )
    plan_path, _ = write_immutable_json_event(
        tmp_path / "plans",
        PLAN_EVENT_PREFIX,
        payload,
    )
    return plan_path, sha256_file(plan_path)


def _published_plan_many(
    tmp_path: Path,
    *,
    targets: list[Path],
    registry: Path,
    launch: Path,
) -> tuple[Path, str]:
    payload = build_cleanup_plan_payload(
        targets=targets,
        reason="Exact obsolete test evidence batch",
        vedtak=VEDTAK,
        artifact_registry_json=registry,
        launch_contract_json=launch,
        inventory_dir=tmp_path / "plans",
        created_utc=CREATED_UTC,
        allowed_roots=(tmp_path,),
    )
    plan_path, _ = write_immutable_json_event(
        tmp_path / "plans",
        PLAN_EVENT_PREFIX,
        payload,
    )
    return plan_path, sha256_file(plan_path)


def test_valid_plan_proves_byte_identity_and_dry_run_deletes_nothing(
    tmp_path: Path,
) -> None:
    target = tmp_path / "evidence" / "obsolete"
    target.mkdir(parents=True)
    (target / "a.bin").write_bytes(b"abc")
    (target / "b.bin").write_bytes(b"defgh")
    registry, launch = _authority_files(tmp_path)
    plan_path, plan_sha = _published_plan(
        tmp_path,
        target=target,
        registry=registry,
        launch=launch,
    )

    validated = validate_cleanup_plan(
        plan_path,
        plan_sha,
        vedtak=VEDTAK,
        allowed_roots=(tmp_path,),
        required_artifact_registry_json=registry,
        required_launch_contract_json=launch,
    )
    assert validated["validated"] is True
    assert validated["targets"][0]["file_count"] == 2
    assert validated["targets"][0]["total_bytes"] == 8
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest_path = Path(plan["targets"][0]["inventory_jsonl"])
    assert sha256_file(manifest_path) == plan["targets"][0][
        "inventory_jsonl_sha256"
    ]
    manifest_rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["relative_path"] for row in manifest_rows] == [".", "a.bin", "b.bin"]

    report_dir = tmp_path / "reports"
    assert (
        execute_cleanup(
            plan_json=plan_path,
            plan_sha256=plan_sha,
            vedtak=VEDTAK,
            out_dir=report_dir,
            execute=False,
            quiet=True,
            allowed_roots=(tmp_path,),
            required_artifact_registry_json=registry,
            required_launch_contract_json=launch,
        )
        == 0
    )
    assert target.exists()
    assert not report_dir.exists()


def test_stage_rejects_open_writer_before_moving_the_canonical_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "evidence" / "active.bin"
    target.parent.mkdir()
    target.write_bytes(b"still-writing")
    registry, launch = _authority_files(tmp_path)
    plan_path, plan_sha = _published_plan(
        tmp_path,
        target=target,
        registry=registry,
        launch=launch,
    )
    validated = validate_cleanup_plan(
        plan_path,
        plan_sha,
        vedtak=VEDTAK,
        allowed_roots=(tmp_path,),
        required_artifact_registry_json=registry,
        required_launch_contract_json=launch,
    )
    mapping = cleanup_script._stage_plan(
        validated["targets"], plan_sha256=plan_sha
    )[0]
    monkeypatch.setattr(
        cleanup_script,
        "_reject_open_writer_fds",
        lambda _path: (_ for _ in ()).throw(RuntimeError("open writer")),
    )

    with pytest.raises(RuntimeError, match="open writer"):
        cleanup_script._stage_exact_target(validated["targets"][0], mapping)

    assert target.read_bytes() == b"still-writing"
    assert not Path(mapping["quarantine_wrapper"]).exists()


def test_explicit_execution_writes_clearance_and_deletes_exact_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "evidence" / "obsolete.bin"
    target.parent.mkdir()
    target.write_bytes(b"obsolete")
    registry, launch = _authority_files(tmp_path)
    plan_path, plan_sha = _published_plan(
        tmp_path,
        target=target,
        registry=registry,
        launch=launch,
    )
    report_dir = tmp_path / "reports"
    approval_path, approval_sha = publish_cleanup_approval(
        plan_json=plan_path,
        plan_sha256=plan_sha,
        vedtak=VEDTAK,
        approved_by="test-operator",
        out_dir=tmp_path / "approvals",
        allowed_roots=(tmp_path,),
        required_artifact_registry_json=registry,
        required_launch_contract_json=launch,
    )

    assert (
        execute_cleanup(
            plan_json=plan_path,
            plan_sha256=plan_sha,
            vedtak=VEDTAK,
            out_dir=report_dir,
            execute=True,
            quiet=True,
            approval_json=approval_path,
            approval_sha256=approval_sha,
            allowed_roots=(tmp_path,),
            required_artifact_registry_json=registry,
            required_launch_contract_json=launch,
        )
        == 0
    )
    assert not target.exists()
    assert len(list(report_dir.glob(f"{CLEARANCE_PREFIX}_*.json"))) == 1
    assert len(list(report_dir.glob(f"{STAGED_PREFIX}_*.json"))) == 1
    assert len(list(report_dir.glob(f"{EXECUTION_PREFIX}_*.json"))) == 1
    assert len(list((tmp_path / "approvals").glob(f"{APPROVAL_PREFIX}_*.json"))) == 1


def test_execution_without_separate_approval_fails_before_side_effects(
    tmp_path: Path,
) -> None:
    target = tmp_path / "evidence" / "obsolete.bin"
    target.parent.mkdir()
    target.write_bytes(b"obsolete")
    registry, launch = _authority_files(tmp_path)
    plan_path, plan_sha = _published_plan(
        tmp_path,
        target=target,
        registry=registry,
        launch=launch,
    )
    report_dir = tmp_path / "reports"

    with pytest.raises(RuntimeError, match="requires exact --approval"):
        execute_cleanup(
            plan_json=plan_path,
            plan_sha256=plan_sha,
            vedtak=VEDTAK,
            out_dir=report_dir,
            execute=True,
            quiet=True,
            allowed_roots=(tmp_path,),
            required_artifact_registry_json=registry,
            required_launch_contract_json=launch,
        )
    assert target.exists()
    assert not report_dir.exists()


def test_execute_rejects_plan_bound_to_noncanonical_authority(tmp_path: Path) -> None:
    target = tmp_path / "evidence" / "obsolete.bin"
    target.parent.mkdir()
    target.write_bytes(b"obsolete")
    fake_registry, fake_launch = _authority_files(tmp_path)
    plan_path, plan_sha = _published_plan(
        tmp_path,
        target=target,
        registry=fake_registry,
        launch=fake_launch,
    )
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    canonical_registry, canonical_launch = _authority_files(canonical)

    with pytest.raises(EvidenceRetentionError, match="pinned canonical"):
        execute_cleanup(
            plan_json=plan_path,
            plan_sha256=plan_sha,
            vedtak=VEDTAK,
            out_dir=tmp_path / "reports",
            execute=False,
            quiet=True,
            allowed_roots=(tmp_path,),
            required_artifact_registry_json=canonical_registry,
            required_launch_contract_json=canonical_launch,
        )
    assert target.exists()


def test_atomic_quarantine_wrapper_is_never_replaced(tmp_path: Path) -> None:
    target = tmp_path / "evidence" / "obsolete.bin"
    target.parent.mkdir()
    target.write_bytes(b"obsolete")
    registry, launch = _authority_files(tmp_path)
    plan_path, plan_sha = _published_plan(
        tmp_path,
        target=target,
        registry=registry,
        launch=launch,
    )
    approval_path, approval_sha = publish_cleanup_approval(
        plan_json=plan_path,
        plan_sha256=plan_sha,
        vedtak=VEDTAK,
        approved_by="test-operator",
        out_dir=tmp_path / "approvals",
        allowed_roots=(tmp_path,),
        required_artifact_registry_json=registry,
        required_launch_contract_json=launch,
    )
    wrapper = target.parent / f".gx1_delete_{plan_sha[:16]}_0000"
    wrapper.mkdir()
    sentinel = wrapper / "sentinel"
    sentinel.write_bytes(b"keep")

    with pytest.raises(RuntimeError, match="partial failure"):
        execute_cleanup(
            plan_json=plan_path,
            plan_sha256=plan_sha,
            vedtak=VEDTAK,
            out_dir=tmp_path / "reports",
            execute=True,
            quiet=True,
            approval_json=approval_path,
            approval_sha256=approval_sha,
            allowed_roots=(tmp_path,),
            required_artifact_registry_json=registry,
            required_launch_contract_json=launch,
        )
    assert target.read_bytes() == b"obsolete"
    assert sentinel.read_bytes() == b"keep"


def test_post_rename_mismatch_keeps_payload_and_durable_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "evidence" / "obsolete.bin"
    target.parent.mkdir()
    target.write_bytes(b"obsolete")
    registry, launch = _authority_files(tmp_path)
    plan_path, plan_sha = _published_plan(
        tmp_path,
        target=target,
        registry=registry,
        launch=launch,
    )
    approval_path, approval_sha = publish_cleanup_approval(
        plan_json=plan_path,
        plan_sha256=plan_sha,
        vedtak=VEDTAK,
        approved_by="test-operator",
        out_dir=tmp_path / "approvals",
        allowed_roots=(tmp_path,),
        required_artifact_registry_json=registry,
        required_launch_contract_json=launch,
    )
    real_inventory = cleanup_script.inventory_path

    def mismatched_inventory(path: Path) -> dict[str, object]:
        observed = real_inventory(path)
        if path.name == "payload":
            observed["inventory_sha256"] = "0" * 64
        return observed

    monkeypatch.setattr(cleanup_script, "inventory_path", mismatched_inventory)
    report_dir = tmp_path / "reports"
    with pytest.raises(RuntimeError, match="partial failure"):
        execute_cleanup(
            plan_json=plan_path,
            plan_sha256=plan_sha,
            vedtak=VEDTAK,
            out_dir=report_dir,
            execute=True,
            quiet=True,
            approval_json=approval_path,
            approval_sha256=approval_sha,
            allowed_roots=(tmp_path,),
            required_artifact_registry_json=registry,
            required_launch_contract_json=launch,
        )
    wrapper = target.parent / f".gx1_delete_{plan_sha[:16]}_0000"
    assert not target.exists()
    assert (wrapper / "payload").read_bytes() == b"obsolete"
    started_path = next(report_dir.glob(f"{cleanup_script.STARTED_PREFIX}_*.json"))
    started = json.loads(started_path.read_text(encoding="utf-8"))
    assert started["stage_plan"][0]["source_path"] == str(target)
    assert started["stage_plan"][0]["quarantine_path"] == str(wrapper / "payload")


def test_manifest_delete_never_removes_unapproved_racing_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "evidence" / "obsolete"
    target.mkdir(parents=True)
    (target / "approved.bin").write_bytes(b"approved")
    registry, launch = _authority_files(tmp_path)
    plan_path, plan_sha = _published_plan(
        tmp_path,
        target=target,
        registry=registry,
        launch=launch,
    )
    approval_path, approval_sha = publish_cleanup_approval(
        plan_json=plan_path,
        plan_sha256=plan_sha,
        vedtak=VEDTAK,
        approved_by="test-operator",
        out_dir=tmp_path / "approvals",
        allowed_roots=(tmp_path,),
        required_artifact_registry_json=registry,
        required_launch_contract_json=launch,
    )
    real_delete = cleanup_script._delete_staged_manifest_exact

    def inject_unapproved_entry(
        staged_target: dict[str, object],
        plan_target: dict[str, object],
    ) -> None:
        payload = Path(str(staged_target["quarantine_path"]))
        (payload / "unapproved.bin").write_bytes(b"must-not-delete")
        real_delete(staged_target, plan_target)

    monkeypatch.setattr(
        cleanup_script,
        "_delete_staged_manifest_exact",
        inject_unapproved_entry,
    )
    with pytest.raises(RuntimeError, match="partial failure"):
        execute_cleanup(
            plan_json=plan_path,
            plan_sha256=plan_sha,
            vedtak=VEDTAK,
            out_dir=tmp_path / "reports",
            execute=True,
            quiet=True,
            approval_json=approval_path,
            approval_sha256=approval_sha,
            allowed_roots=(tmp_path,),
            required_artifact_registry_json=registry,
            required_launch_contract_json=launch,
        )
    wrapper = target.parent / f".gx1_delete_{plan_sha[:16]}_0000"
    assert (wrapper / "payload" / "unapproved.bin").read_bytes() == b"must-not-delete"


def test_batch_delete_validates_full_target_plan_only_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = [
        tmp_path / "evidence" / "obsolete-a.bin",
        tmp_path / "evidence" / "obsolete-b.bin",
    ]
    targets[0].parent.mkdir()
    for index, target in enumerate(targets):
        target.write_bytes(f"obsolete-{index}".encode())
    registry, launch = _authority_files(tmp_path)
    plan_path, plan_sha = _published_plan_many(
        tmp_path,
        targets=targets,
        registry=registry,
        launch=launch,
    )
    approval_path, approval_sha = publish_cleanup_approval(
        plan_json=plan_path,
        plan_sha256=plan_sha,
        vedtak=VEDTAK,
        approved_by="test-operator",
        out_dir=tmp_path / "approvals",
        allowed_roots=(tmp_path,),
        required_artifact_registry_json=registry,
        required_launch_contract_json=launch,
    )
    real_validate = cleanup_script.validate_cleanup_plan
    validation_calls = 0

    def counted_validate(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal validation_calls
        validation_calls += 1
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(cleanup_script, "validate_cleanup_plan", counted_validate)
    assert (
        execute_cleanup(
            plan_json=plan_path,
            plan_sha256=plan_sha,
            vedtak=VEDTAK,
            out_dir=tmp_path / "reports",
            execute=True,
            quiet=True,
            approval_json=approval_path,
            approval_sha256=approval_sha,
            allowed_roots=(tmp_path,),
            required_artifact_registry_json=registry,
            required_launch_contract_json=launch,
        )
        == 0
    )
    assert validation_calls == 2
    assert all(not target.exists() for target in targets)


def test_authority_change_between_staged_deletes_stops_remaining_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = [
        tmp_path / "evidence" / "obsolete-a.bin",
        tmp_path / "evidence" / "obsolete-b.bin",
    ]
    targets[0].parent.mkdir()
    for index, target in enumerate(targets):
        target.write_bytes(f"obsolete-{index}".encode())
    registry, launch = _authority_files(tmp_path)
    plan_path, plan_sha = _published_plan_many(
        tmp_path,
        targets=targets,
        registry=registry,
        launch=launch,
    )
    approval_path, approval_sha = publish_cleanup_approval(
        plan_json=plan_path,
        plan_sha256=plan_sha,
        vedtak=VEDTAK,
        approved_by="test-operator",
        out_dir=tmp_path / "approvals",
        allowed_roots=(tmp_path,),
        required_artifact_registry_json=registry,
        required_launch_contract_json=launch,
    )
    real_delete = cleanup_script._delete_staged_manifest_exact
    delete_calls = 0

    def mutate_authority_after_first_delete(
        staged_target: dict[str, object],
        plan_target: dict[str, object],
    ) -> None:
        nonlocal delete_calls
        real_delete(staged_target, plan_target)
        delete_calls += 1
        if delete_calls == 1:
            registry.write_text(
                registry.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        cleanup_script,
        "_delete_staged_manifest_exact",
        mutate_authority_after_first_delete,
    )
    with pytest.raises(RuntimeError, match="partial failure"):
        execute_cleanup(
            plan_json=plan_path,
            plan_sha256=plan_sha,
            vedtak=VEDTAK,
            out_dir=tmp_path / "reports",
            execute=True,
            quiet=True,
            approval_json=approval_path,
            approval_sha256=approval_sha,
            allowed_roots=(tmp_path,),
            required_artifact_registry_json=registry,
            required_launch_contract_json=launch,
        )
    assert delete_calls == 1
    assert not targets[0].exists()
    second_wrapper = targets[1].parent / f".gx1_delete_{plan_sha[:16]}_0001"
    assert (second_wrapper / "payload").read_bytes() == b"obsolete-1"


def test_interrupted_pre_staged_cleanup_restores_exact_source_paths(
    tmp_path: Path,
) -> None:
    targets = [
        tmp_path / "evidence" / "obsolete-a.bin",
        tmp_path / "evidence" / "obsolete-b.bin",
    ]
    targets[0].parent.mkdir()
    for index, target in enumerate(targets):
        target.write_bytes(f"obsolete-{index}".encode())
    registry, launch = _authority_files(tmp_path)
    plan_path, plan_sha = _published_plan_many(
        tmp_path,
        targets=targets,
        registry=registry,
        launch=launch,
    )
    approval_path, approval_sha = publish_cleanup_approval(
        plan_json=plan_path,
        plan_sha256=plan_sha,
        vedtak=VEDTAK,
        approved_by="test-operator",
        out_dir=tmp_path / "approvals",
        allowed_roots=(tmp_path,),
        required_artifact_registry_json=registry,
        required_launch_contract_json=launch,
    )
    validated = validate_cleanup_plan(
        plan_path,
        plan_sha,
        vedtak=VEDTAK,
        allowed_roots=(tmp_path,),
        required_artifact_registry_json=registry,
        required_launch_contract_json=launch,
    )
    stage_plan = cleanup_script._stage_plan(
        validated["targets"],
        plan_sha256=plan_sha,
    )
    cleanup_script._stage_exact_target(validated["targets"][0], stage_plan[0])
    started_path, _ = write_immutable_json_event(
        tmp_path / "reports",
        cleanup_script.STARTED_PREFIX,
        {
            "schema_version": "gx1_evidence_cleanup_started_v1",
            "created_utc": CREATED_UTC,
            "decision": "ATOMIC_STAGING_STARTED",
            "plan_json": str(plan_path),
            "plan_sha256": plan_sha,
            "approval_json": str(approval_path),
            "approval_sha256": approval_sha,
            "vedtak": VEDTAK,
            "stage_plan": stage_plan,
            "direction_authority": False,
            "launch_authority": False,
        },
    )

    assert (
        recover_interrupted_cleanup(
            plan_json=plan_path,
            plan_sha256=plan_sha,
            vedtak=VEDTAK,
            approval_json=approval_path,
            approval_sha256=approval_sha,
            started_json=started_path,
            started_sha256=sha256_file(started_path),
            out_dir=tmp_path / "reports",
            recover=True,
            quiet=True,
            allowed_roots=(tmp_path,),
            required_artifact_registry_json=registry,
            required_launch_contract_json=launch,
        )
        == 0
    )
    assert [target.read_bytes() for target in targets] == [
        b"obsolete-0",
        b"obsolete-1",
    ]
    assert not Path(stage_plan[0]["quarantine_wrapper"]).exists()
    recovery_path = next((tmp_path / "reports").glob(f"{RECOVERY_PREFIX}_*.json"))
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert recovery["decision"] == "RESTORE_COMPLETE"
    assert len(recovery["restored"]) == 1
    assert recovery["failure"] is None


def _staged_transaction(tmp_path: Path) -> dict[str, object]:
    """Build the exact post-STAGED state an interrupted execution leaves behind."""

    targets = [
        tmp_path / "evidence" / "obsolete-a.bin",
        tmp_path / "evidence" / "obsolete-b.bin",
    ]
    targets[0].parent.mkdir()
    for index, target in enumerate(targets):
        target.write_bytes(f"obsolete-{index}".encode())
    registry, launch = _authority_files(tmp_path)
    plan_path, plan_sha = _published_plan_many(
        tmp_path,
        targets=targets,
        registry=registry,
        launch=launch,
    )
    approval_path, approval_sha = publish_cleanup_approval(
        plan_json=plan_path,
        plan_sha256=plan_sha,
        vedtak=VEDTAK,
        approved_by="test-operator",
        out_dir=tmp_path / "approvals",
        allowed_roots=(tmp_path,),
        required_artifact_registry_json=registry,
        required_launch_contract_json=launch,
    )
    validated = validate_cleanup_plan(
        plan_path,
        plan_sha,
        vedtak=VEDTAK,
        allowed_roots=(tmp_path,),
        required_artifact_registry_json=registry,
        required_launch_contract_json=launch,
    )
    stage_plan = cleanup_script._stage_plan(validated["targets"], plan_sha256=plan_sha)
    staged = [
        cleanup_script._stage_exact_target(target, mapping)
        for target, mapping in zip(validated["targets"], stage_plan, strict=True)
    ]
    staged_path, _ = write_immutable_json_event(
        tmp_path / "reports",
        STAGED_PREFIX,
        {
            "schema_version": "gx1_evidence_cleanup_staged_v1",
            "created_utc": CREATED_UTC,
            "decision": "EXACT_TARGETS_STAGED_AND_REVALIDATED",
            "plan_json": str(plan_path),
            "plan_sha256": plan_sha,
            "approval_json": str(approval_path),
            "approval_sha256": approval_sha,
            "vedtak": VEDTAK,
            "stage_plan": stage_plan,
            "staged": staged,
            "direction_authority": False,
            "launch_authority": False,
        },
    )
    return {
        "targets": targets,
        "registry": registry,
        "launch": launch,
        "plan_path": plan_path,
        "plan_sha": plan_sha,
        "approval_path": approval_path,
        "approval_sha": approval_sha,
        "stage_plan": stage_plan,
        "staged_path": staged_path,
        "validated_targets": validated["targets"],
    }


def _resume(
    state: dict[str, object],
    tmp_path: Path,
    *,
    allow_interrupted_payload: bool = False,
) -> int:
    return resume_interrupted_cleanup(
        plan_json=state["plan_path"],
        plan_sha256=state["plan_sha"],
        vedtak=VEDTAK,
        approval_json=state["approval_path"],
        approval_sha256=state["approval_sha"],
        staged_json=state["staged_path"],
        staged_sha256=sha256_file(state["staged_path"]),
        out_dir=tmp_path / "reports",
        resume=True,
        quiet=True,
        allow_interrupted_payload=allow_interrupted_payload,
        allowed_roots=(tmp_path,),
        required_artifact_registry_json=state["registry"],
        required_launch_contract_json=state["launch"],
    )


def _staged_tree_transaction(tmp_path: Path) -> dict[str, object]:
    """A single directory target, staged, so a partial delete can be simulated."""

    target = tmp_path / "evidence" / "obsolete-tree"
    (target / "nested").mkdir(parents=True)
    (target / "keep.bin").write_bytes(b"keep-payload")
    (target / "nested" / "gone.bin").write_bytes(b"gone-payload")
    registry, launch = _authority_files(tmp_path)
    plan_path, plan_sha = _published_plan_many(
        tmp_path,
        targets=[target],
        registry=registry,
        launch=launch,
    )
    approval_path, approval_sha = publish_cleanup_approval(
        plan_json=plan_path,
        plan_sha256=plan_sha,
        vedtak=VEDTAK,
        approved_by="test-operator",
        out_dir=tmp_path / "approvals",
        allowed_roots=(tmp_path,),
        required_artifact_registry_json=registry,
        required_launch_contract_json=launch,
    )
    validated = validate_cleanup_plan(
        plan_path,
        plan_sha,
        vedtak=VEDTAK,
        allowed_roots=(tmp_path,),
        required_artifact_registry_json=registry,
        required_launch_contract_json=launch,
    )
    stage_plan = cleanup_script._stage_plan(validated["targets"], plan_sha256=plan_sha)
    staged = [cleanup_script._stage_exact_target(validated["targets"][0], stage_plan[0])]
    staged_path, _ = write_immutable_json_event(
        tmp_path / "reports",
        STAGED_PREFIX,
        {
            "schema_version": "gx1_evidence_cleanup_staged_v1",
            "created_utc": CREATED_UTC,
            "decision": "EXACT_TARGETS_STAGED_AND_REVALIDATED",
            "plan_json": str(plan_path),
            "plan_sha256": plan_sha,
            "approval_json": str(approval_path),
            "approval_sha256": approval_sha,
            "vedtak": VEDTAK,
            "stage_plan": stage_plan,
            "staged": staged,
            "direction_authority": False,
            "launch_authority": False,
        },
    )
    return {
        "targets": [target],
        "registry": registry,
        "launch": launch,
        "plan_path": plan_path,
        "plan_sha": plan_sha,
        "approval_path": approval_path,
        "approval_sha": approval_sha,
        "stage_plan": stage_plan,
        "staged_path": staged_path,
    }


def test_resume_finishes_a_payload_the_interrupted_run_died_inside(
    tmp_path: Path,
) -> None:
    state = _staged_tree_transaction(tmp_path)
    quarantine = Path(state["stage_plan"][0]["quarantine_path"])
    # The killed run had already unlinked one file and removed its directory.
    (quarantine / "nested" / "gone.bin").unlink()
    (quarantine / "nested").rmdir()

    # Without the explicit flag this state must fail closed and delete nothing.
    with pytest.raises(RuntimeError, match="quarantine inventory differs"):
        _resume(state, tmp_path)
    assert (quarantine / "keep.bin").read_bytes() == b"keep-payload"

    assert _resume(state, tmp_path, allow_interrupted_payload=True) == 0
    assert not quarantine.exists()
    assert not Path(state["stage_plan"][0]["quarantine_wrapper"]).exists()
    assert not state["targets"][0].exists()
    execution_path = retention_contract.immutable_events.select_latest_immutable_event(
        tmp_path / "reports", EXECUTION_PREFIX,
    )
    assert execution_path is not None
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    assert execution["decision"] == "DELETE_COMPLETE"
    assert execution["failure"] is None
    assert len(execution["interrupted_payloads"]) == 1
    assert execution["interrupted_payloads"][0]["absent_relative_paths"] == [
        "nested",
        "nested/gone.bin",
    ]


def test_interrupted_payload_mode_still_rejects_a_foreign_path(
    tmp_path: Path,
) -> None:
    state = _staged_tree_transaction(tmp_path)
    quarantine = Path(state["stage_plan"][0]["quarantine_path"])
    (quarantine / "nested" / "gone.bin").unlink()
    (quarantine / "nested" / "planted.bin").write_bytes(b"planted")
    with pytest.raises(RuntimeError, match="foreign file"):
        _resume(state, tmp_path, allow_interrupted_payload=True)
    assert (quarantine / "keep.bin").read_bytes() == b"keep-payload"
    assert (quarantine / "nested" / "planted.bin").exists()


def test_interrupted_payload_mode_still_rejects_changed_surviving_bytes(
    tmp_path: Path,
) -> None:
    state = _staged_tree_transaction(tmp_path)
    quarantine = Path(state["stage_plan"][0]["quarantine_path"])
    (quarantine / "nested" / "gone.bin").unlink()
    (quarantine / "nested").rmdir()
    (quarantine / "keep.bin").write_bytes(b"keep-payloaD")
    with pytest.raises(RuntimeError, match="bytes changed"):
        _resume(state, tmp_path, allow_interrupted_payload=True)
    assert (quarantine / "keep.bin").exists()


def test_interrupted_payload_mode_rejects_a_complete_payload(
    tmp_path: Path,
) -> None:
    # An intact payload is not an interrupted delete; the normal exact-inventory
    # path already handles it, so the relaxed proof must refuse to be invoked.
    state = _staged_tree_transaction(tmp_path)
    quarantine = Path(state["stage_plan"][0]["quarantine_path"])
    assert (
        cleanup_script._prove_interrupted_payload_subset.__name__
        == "_prove_interrupted_payload_subset"
    )
    validated = validate_cleanup_plan(
        state["plan_path"],
        state["plan_sha"],
        vedtak=VEDTAK,
        allowed_roots=(tmp_path,),
        required_artifact_registry_json=state["registry"],
        required_launch_contract_json=state["launch"],
        verify_target_bytes=False,
        require_targets_exist=False,
    )
    with pytest.raises(RuntimeError, match="not an interrupted delete"):
        cleanup_script._prove_interrupted_payload_subset(
            quarantine,
            validated["targets"][0],
        )


def test_resume_finishes_a_delete_loop_interrupted_after_staging(
    tmp_path: Path,
) -> None:
    state = _staged_transaction(tmp_path)
    stage_plan = state["stage_plan"]
    # The interrupted execution deleted the first staged target, then died
    # before the second one and before writing any execution event.
    cleanup_script._delete_staged_manifest_exact(
        {
            "quarantine_path": stage_plan[0]["quarantine_path"],
            "kind": "file",
        },
        validate_cleanup_plan(
            state["plan_path"],
            state["plan_sha"],
            vedtak=VEDTAK,
            allowed_roots=(tmp_path,),
            required_artifact_registry_json=state["registry"],
            required_launch_contract_json=state["launch"],
            verify_target_bytes=False,
            require_targets_exist=False,
        )["targets"][0],
    )
    Path(stage_plan[0]["quarantine_wrapper"]).rmdir()

    assert _resume(state, tmp_path) == 0

    assert all(not target.exists() for target in state["targets"])
    assert not Path(stage_plan[1]["quarantine_wrapper"]).exists()
    execution_path = retention_contract.immutable_events.select_latest_immutable_event(
        tmp_path / "reports", EXECUTION_PREFIX,
    )
    assert execution_path is not None
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    assert execution["decision"] == "DELETE_COMPLETE"
    assert execution["failure"] is None
    assert execution["deleted"] == [str(state["targets"][1])]
    assert execution["already_deleted_before_resume"] == [str(state["targets"][0])]
    assert execution["resumed_from_staged_json"] == str(state["staged_path"])


def test_resume_refuses_a_target_whose_source_is_still_present(
    tmp_path: Path,
) -> None:
    # A transaction someone partially restored: the second target is back at its
    # source path, so this is no longer a delete to finish. Resume must refuse
    # it whole rather than delete the half that is still staged.
    state = _staged_transaction(tmp_path)
    cleanup_script._restore_staged_target(
        state["validated_targets"][1],
        state["stage_plan"][1],
    )
    with pytest.raises(RuntimeError, match="source is present"):
        _resume(state, tmp_path)
    assert state["targets"][1].read_bytes() == b"obsolete-1"
    assert (
        Path(state["stage_plan"][0]["quarantine_path"]).read_bytes() == b"obsolete-0"
    )


def test_resume_rejects_quarantined_bytes_that_changed_after_staging(
    tmp_path: Path,
) -> None:
    state = _staged_transaction(tmp_path)
    Path(state["stage_plan"][1]["quarantine_path"]).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="quarantine inventory differs"):
        _resume(state, tmp_path)
    # Fail closed before the first delete: the untampered payload survives too.
    assert (
        Path(state["stage_plan"][0]["quarantine_path"]).read_bytes() == b"obsolete-0"
    )


def test_resume_requires_the_explicit_flag(tmp_path: Path) -> None:
    state = _staged_transaction(tmp_path)
    with pytest.raises(RuntimeError, match="requires explicit --resume"):
        resume_interrupted_cleanup(
            plan_json=state["plan_path"],
            plan_sha256=state["plan_sha"],
            vedtak=VEDTAK,
            approval_json=state["approval_path"],
            approval_sha256=state["approval_sha"],
            staged_json=state["staged_path"],
            staged_sha256=sha256_file(state["staged_path"]),
            out_dir=tmp_path / "reports",
            resume=False,
            quiet=True,
            allowed_roots=(tmp_path,),
            required_artifact_registry_json=state["registry"],
            required_launch_contract_json=state["launch"],
        )
    assert (
        Path(state["stage_plan"][0]["quarantine_path"]).read_bytes() == b"obsolete-0"
    )


def test_changed_target_bytes_invalidate_published_plan(tmp_path: Path) -> None:
    target = tmp_path / "evidence" / "obsolete.bin"
    target.parent.mkdir()
    target.write_bytes(b"before")
    registry, launch = _authority_files(tmp_path)
    plan_path, plan_sha = _published_plan(
        tmp_path,
        target=target,
        registry=registry,
        launch=launch,
    )
    target.write_bytes(b"after")

    with pytest.raises(EvidenceRetentionError, match="changed"):
        validate_cleanup_plan(
            plan_path,
            plan_sha,
            vedtak=VEDTAK,
            allowed_roots=(tmp_path,),
            required_artifact_registry_json=registry,
            required_launch_contract_json=launch,
        )


def test_changed_empty_directory_topology_invalidates_plan(tmp_path: Path) -> None:
    target = tmp_path / "evidence" / "obsolete"
    target.mkdir(parents=True)
    registry, launch = _authority_files(tmp_path)
    plan_path, plan_sha = _published_plan(
        tmp_path,
        target=target,
        registry=registry,
        launch=launch,
    )
    (target / "new-empty-directory").mkdir()

    with pytest.raises(EvidenceRetentionError, match="changed"):
        validate_cleanup_plan(
            plan_path,
            plan_sha,
            vedtak=VEDTAK,
            allowed_roots=(tmp_path,),
            required_artifact_registry_json=registry,
            required_launch_contract_json=launch,
        )


@pytest.mark.parametrize("authority_owner", ["active", "retired", "history", "launch"])
def test_every_authority_inventory_blocks_overlapping_target(
    tmp_path: Path,
    authority_owner: str,
) -> None:
    target = tmp_path / "evidence" / "protected"
    target.mkdir(parents=True)
    protected = target / "model.bin"
    protected.write_bytes(b"model")
    kwargs: dict[str, object] = {}
    if authority_owner == "launch":
        kwargs["launch_extra"] = {"model_path": str(protected)}
    elif authority_owner == "history":
        kwargs["history"] = [{"model_path": str(protected)}]
    else:
        kwargs[authority_owner] = {"model_path": str(protected)}
    registry, launch = _authority_files(tmp_path, **kwargs)

    with pytest.raises(EvidenceRetentionError, match="authority-protected"):
        build_cleanup_plan_payload(
            targets=[target],
            reason="Attempted protected evidence cleanup",
            vedtak=VEDTAK,
            artifact_registry_json=registry,
            launch_contract_json=launch,
            inventory_dir=tmp_path / "plans",
            created_utc=CREATED_UTC,
            allowed_roots=(tmp_path,),
        )


def test_nonempty_exclusions_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "evidence" / "obsolete.bin"
    target.parent.mkdir()
    target.write_bytes(b"obsolete")
    registry, launch = _authority_files(tmp_path)
    payload = build_cleanup_plan_payload(
        targets=[target],
        reason="Exact obsolete test evidence",
        vedtak=VEDTAK,
        artifact_registry_json=registry,
        launch_contract_json=launch,
        inventory_dir=tmp_path / "plans",
        created_utc=CREATED_UTC,
        allowed_roots=(tmp_path,),
    )
    payload["exclusions"] = [str(target / "...")]
    plan_path, _ = write_immutable_json_event(
        tmp_path / "plans",
        PLAN_EVENT_PREFIX,
        payload,
    )

    with pytest.raises(EvidenceRetentionError, match="exclusions are forbidden"):
        validate_cleanup_plan(
            plan_path,
            sha256_file(plan_path),
            vedtak=VEDTAK,
            allowed_roots=(tmp_path,),
            required_artifact_registry_json=registry,
            required_launch_contract_json=launch,
        )


def test_relative_nonexistent_ellipsis_and_symlink_targets_fail_closed(
    tmp_path: Path,
) -> None:
    registry, launch = _authority_files(tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    symlink = tmp_path / "linked"
    symlink.symlink_to(real, target_is_directory=True)
    ellipsis = tmp_path / "..."
    ellipsis.mkdir()
    targets = [Path("relative"), tmp_path / "missing", ellipsis, symlink]

    for target in targets:
        with pytest.raises(EvidenceRetentionError):
            build_cleanup_plan_payload(
                targets=[target],
                reason="Invalid cleanup target is rejected",
                vedtak=VEDTAK,
                artifact_registry_json=registry,
                launch_contract_json=launch,
                inventory_dir=tmp_path / "plans",
                created_utc=CREATED_UTC,
                allowed_roots=(tmp_path,),
            )


def test_mount_boundary_inventory_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "evidence"
    target.mkdir()
    registry, launch = _authority_files(tmp_path)
    monkeypatch.setattr(retention_contract, "_mount_points", lambda: frozenset({target}))

    with pytest.raises(EvidenceRetentionError, match="mount point"):
        build_cleanup_plan_payload(
            targets=[target],
            reason="Mounted target must remain protected",
            vedtak=VEDTAK,
            artifact_registry_json=registry,
            launch_contract_json=launch,
            inventory_dir=tmp_path / "plans",
            created_utc=CREATED_UTC,
            allowed_roots=(tmp_path,),
        )


def _graph_paths(tmp_path: Path, **authority: object) -> tuple[Path, ...]:
    registry, launch = _authority_files(tmp_path, **authority)
    incident = retention_contract.CANONICAL_DELETE_INCIDENT
    return retention_contract.authority_protected_paths(
        json.loads(registry.read_text(encoding="utf-8")),
        json.loads(launch.read_text(encoding="utf-8")),
        json.loads(incident.read_text(encoding="utf-8")),
        artifact_registry_json=registry,
        launch_contract_json=launch,
        delete_incident_json=incident,
    )


@pytest.mark.parametrize("owner", ["active", "retired", "history", "launch", "incident"])
def test_authority_graph_follows_parent_lineage_and_relative_cache_files(
    tmp_path: Path, owner: str,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    array = cache / "m1_feats.npy"
    array.write_bytes(b"mechanical fixture, not financial data")
    ancestor = tmp_path / "ancestor.json"
    _write_json(ancestor, {"tfs": {"m1": {"feats_npy": "cache/m1_feats.npy"}}})
    manifest = cache / "manifest.json"
    _write_json(manifest, {"parent_source": {
        "manifest_path": "../ancestor.json",
        "manifest_sha256": sha256_file(ancestor),
    }})
    binding = {"manifest_path": "cache/manifest.json", "manifest_sha256": sha256_file(manifest)}
    authority: dict[str, object] = {}
    if owner == "incident":
        incident = retention_contract.CANONICAL_DELETE_INCIDENT
        payload = json.loads(incident.read_text(encoding="utf-8"))
        payload["lineage"] = binding
        _write_json(incident, payload)
    elif owner == "launch":
        authority["launch_extra"] = {"lineage": binding}
    elif owner == "history":
        authority[owner] = [binding]
    else:
        authority[owner] = {"lineage": binding}
    assert set(_graph_paths(tmp_path, **authority)) == {manifest, ancestor, array}


def _semantic_metadata_fixture(kind: str) -> tuple[str, dict[str, object], str]:
    if kind == "causal_target":
        from gx1.contracts.entry_causal_m1_outcomes_v1 import causal_m1_target_contract

        return "target_contract", causal_m1_target_contract(), "missing_or_gapped_m1_path"
    if kind == "ranking_target":
        from gx1.contracts.entry_causal_m1_target_policy_v1 import (
            train_feature_ranking_target_contract,
        )

        return (
            "target_contract",
            train_feature_ranking_target_contract(),
            "missing_or_gapped_m1_path",
        )
    if kind == "blocked_heads":
        from gx1.contracts.entry_model_native_readiness_v1 import model_native_blocked_head_reasons

        return "blocked_head_reasons", model_native_blocked_head_reasons(), "bad_path"
    from gx1.features.entry_specialist_feature_groups_v1 import model_native_recommended_fusion_metadata

    assert kind == "fusion"
    return "recommended_fusion", model_native_recommended_fusion_metadata(), "direction_path"


@pytest.mark.parametrize("kind", ["causal_target", "ranking_target", "blocked_heads", "fusion"])
def test_authority_graph_exact_semantic_objects_do_not_invent_file_dependencies(
    tmp_path: Path, kind: str,
) -> None:
    key, semantic, _field = _semantic_metadata_fixture(kind)
    upstream = tmp_path / "upstream.bin"
    upstream.write_bytes(b"mechanical fixture")
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {key: semantic, "upstream_path": str(upstream)})
    assert set(_graph_paths(tmp_path, active={"manifest_path": str(manifest)})) == {
        manifest, upstream,
    }


@pytest.mark.parametrize("kind", ["causal_target", "ranking_target", "blocked_heads", "fusion"])
def test_authority_graph_semantic_looking_field_cannot_hide_a_real_path(
    tmp_path: Path, kind: str,
) -> None:
    key, semantic, field = _semantic_metadata_fixture(kind)
    upstream = tmp_path / "upstream.bin"
    upstream.write_bytes(b"mechanical fixture")
    semantic[field] = str(upstream)
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {key: semantic})
    assert set(_graph_paths(tmp_path, active={"manifest_path": str(manifest)})) == {
        manifest, upstream,
    }


@pytest.mark.parametrize("kind", ["causal_target", "ranking_target", "blocked_heads", "fusion"])
@pytest.mark.parametrize("mutation", ["extra_reference", "incomplete", "wrong_container", "list", "nested_list"])
def test_authority_graph_semantic_exemption_requires_the_complete_owned_object(
    tmp_path: Path, kind: str, mutation: str,
) -> None:
    key, semantic, field = _semantic_metadata_fixture(kind)
    upstream = tmp_path / "upstream.bin"
    upstream.write_bytes(b"mechanical fixture")
    if mutation == "extra_reference":
        semantic["upstream_path"] = str(upstream)
    elif mutation == "incomplete":
        del semantic[next(name for name in semantic if name != field)]
    elif mutation == "list":
        semantic = [semantic]
    elif mutation == "nested_list":
        semantic = [[semantic]]
    else:
        key = "unrecognized_" + key
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {key: semantic})
    with pytest.raises(EvidenceRetentionError, match="unresolved authority reference"):
        _graph_paths(tmp_path, active={"manifest_path": str(manifest)})
    assert upstream.read_bytes() == b"mechanical fixture"


@pytest.mark.parametrize("kind", ["causal_target", "ranking_target", "fusion"])
def test_authority_graph_semantic_exemption_preserves_json_boolean_types(
    tmp_path: Path, kind: str,
) -> None:
    key, semantic, _field = _semantic_metadata_fixture(kind)
    boolean_field = next(name for name, value in semantic.items() if value is False)
    semantic[boolean_field] = 0
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {key: semantic})
    with pytest.raises(EvidenceRetentionError, match="unresolved authority reference"):
        _graph_paths(tmp_path, active={"manifest_path": str(manifest)})


@pytest.mark.parametrize("kind", ["causal_target", "ranking_target", "blocked_heads", "fusion"])
def test_authority_graph_semantic_exemption_never_bypasses_inherited_test_seal(
    tmp_path: Path, kind: str,
) -> None:
    key, semantic, _field = _semantic_metadata_fixture(kind)
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"test": {key: semantic}})
    with pytest.raises(EvidenceRetentionError, match="sealed TEST reachability"):
        _graph_paths(tmp_path, active={"manifest_path": str(manifest)})


def test_audit_semantic_metadata_uses_the_shared_owners(monkeypatch: pytest.MonkeyPatch) -> None:
    from gx1.contracts.entry_model_native_signal_v1 import model_native_signal_contract_metadata
    from gx1.scripts import audit_entry_foundation_targets_v1 as target_audit
    from gx1.scripts import audit_entry_specialist_feature_groups_v1 as specialist_audit
    from tests.model_native_signal_support import canonical_model_native_selected_fields

    target_owner = target_audit.model_native_blocked_head_reasons
    fusion_owner = specialist_audit.model_native_recommended_fusion_metadata
    calls: list[str] = []

    def target_metadata():
        calls.append("target")
        return target_owner()

    def fusion_metadata():
        calls.append("fusion")
        return fusion_owner()

    monkeypatch.setattr(target_audit, "model_native_blocked_head_reasons", target_metadata)
    monkeypatch.setattr(specialist_audit, "model_native_recommended_fusion_metadata", fusion_metadata)
    contract = model_native_signal_contract_metadata(canonical_model_native_selected_fields())
    assert target_audit._head_contract([])["blocked_head_reasons"] == target_owner()
    assert specialist_audit._architecture(contract["fields"])["recommended_fusion"] == fusion_owner()
    assert calls == ["target", "fusion"]


@pytest.mark.parametrize("timeframe", ["M1", "M5"])
def test_authority_graph_native_source_labels_keep_exact_snapshot_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeframe: str,
) -> None:
    from tests.test_oanda_backfill_vedtak_gate import materialize_native_xau_test_bundle

    native_root = tmp_path / "native"
    manifest = materialize_native_xau_test_bundle(native_root, timeframe=timeframe)
    manifest_path = native_root / "MANIFEST.json"
    recorded_bindings: dict[str, str | None] = {}
    hash_owner = retention_contract._reference_hash

    def record_hash(payload, key):
        digest = hash_owner(payload, key)
        if key == "path" and isinstance(payload.get(key), str) and payload[key].endswith(".py"):
            recorded_bindings[payload[key]] = digest
        return digest

    monkeypatch.setattr(retention_contract, "_reference_hash", record_hash)
    protected = set(_graph_paths(tmp_path, active={"manifest_path": str(manifest_path)}))
    assert native_root in protected and manifest_path in protected
    for entry in manifest["producer_source_files"]:
        assert native_root / entry["snapshot_relative_path"] in protected
        assert native_root / entry["repo_relative_path"] not in protected
        assert recorded_bindings[entry["snapshot_relative_path"]] == entry["sha256"]
    assert Path(manifest["source_endpoint"]) not in protected


@pytest.mark.parametrize("mutation", ["owner", "endpoint", "inventory_hash", "snapshot_relationship"])
def test_authority_graph_native_source_label_adapter_rejects_unowned_metadata(
    tmp_path: Path, mutation: str,
) -> None:
    from gx1.contracts.xau_tape_provenance_v1 import canonical_json_sha256
    from tests.test_oanda_backfill_vedtak_gate import materialize_native_xau_test_bundle

    native_root = tmp_path / "native"
    manifest = materialize_native_xau_test_bundle(native_root)
    if mutation == "owner":
        manifest["producer_owner"] = "unrecognized.owner"
    elif mutation == "endpoint":
        manifest["source_endpoint"] = str(tmp_path / "not-an-api-label.bin")
    elif mutation == "inventory_hash":
        manifest["producer_source_inventory_sha256"] = "0" * 64
    else:
        manifest["producer_source_files"][0]["snapshot_relative_path"] = "another.bin"
        manifest["producer_source_inventory_sha256"] = canonical_json_sha256(manifest["producer_source_files"])
    manifest_path = native_root / "MANIFEST.json"
    _write_json(manifest_path, manifest)
    with pytest.raises(EvidenceRetentionError, match="native producer source metadata invalid"):
        _graph_paths(tmp_path, active={"manifest_path": str(manifest_path)})


@pytest.mark.parametrize("mutation", ["unknown_schema", "nested_lookalike"])
def test_authority_graph_native_source_label_adapter_is_top_level_and_schema_scoped(
    tmp_path: Path, mutation: str,
) -> None:
    from tests.test_oanda_backfill_vedtak_gate import materialize_native_xau_test_bundle

    native_root = tmp_path / "native"
    manifest = materialize_native_xau_test_bundle(native_root)
    manifest_path = native_root / "MANIFEST.json"
    if mutation == "unknown_schema":
        manifest["schema_version"] = "unknown_native_source"
    else:
        manifest = {"nested_metadata": manifest}
    _write_json(manifest_path, manifest)
    with pytest.raises(EvidenceRetentionError, match="unresolved authority reference"):
        _graph_paths(tmp_path, active={"manifest_path": str(manifest_path)})


@pytest.mark.parametrize("mutation", ["missing", "symlink"])
def test_authority_graph_native_source_snapshots_remain_required(
    tmp_path: Path, mutation: str,
) -> None:
    from tests.test_oanda_backfill_vedtak_gate import materialize_native_xau_test_bundle

    native_root = tmp_path / "native"
    manifest = materialize_native_xau_test_bundle(native_root)
    source = native_root / manifest["producer_source_files"][0]["snapshot_relative_path"]
    source.unlink()
    if mutation == "symlink":
        alternate = tmp_path / "alternate.bin"
        alternate.write_bytes(b"mechanical fixture")
        source.symlink_to(alternate)
    with pytest.raises(EvidenceRetentionError, match="unresolved|symlink"):
        _graph_paths(tmp_path, active={"manifest_path": str(native_root / "MANIFEST.json")})


def test_authority_graph_native_source_adapter_does_not_open_sealed_test_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_oanda_backfill_vedtak_gate import materialize_native_xau_test_bundle

    native_root = tmp_path / "native"
    materialize_native_xau_test_bundle(native_root)
    manifest_path = native_root / "MANIFEST.json"

    def reject_read(*args, **kwargs):
        raise AssertionError("sealed native metadata must not be opened")

    monkeypatch.setattr(retention_contract, "_authority_json", reject_read)
    with pytest.raises(EvidenceRetentionError, match="sealed TEST reachability"):
        _graph_paths(tmp_path, active={"test": {"manifest_path": str(manifest_path)}})


def test_authority_graph_cycles_and_diamonds_read_each_json_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    shared = tmp_path / "shared.json"
    _write_json(shared, {"parent_manifest": "first.json"})
    _write_json(second, {"manifest_path": "shared.json"})
    _write_json(first, {"paths": ["second.json", "shared.json"]})
    reads: list[Path] = []
    reader = retention_contract._authority_json

    def counted_reader(path: Path, **kwargs: int):
        reads.append(path)
        return reader(path, **kwargs)

    monkeypatch.setattr(retention_contract, "_authority_json", counted_reader)
    assert set(_graph_paths(tmp_path, active={"manifest_path": str(first)})) == {
        first, second, shared,
    }
    assert sorted(reads) == sorted([first, second, shared])


def test_authority_graph_directory_manifest_protects_descendants_not_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "cache"
    directory.mkdir()
    manifest = directory / "manifest.json"
    upstream = tmp_path / "upstream.bin"
    upstream.write_bytes(b"mechanical dependency")
    _write_json(manifest, {"source_path": str(upstream)})
    (tmp_path / "unreachable.json").write_text("malformed and not reachable", encoding="utf-8")
    monkeypatch.setattr(
        retention_contract.os, "scandir",
        lambda *_args, **_kwargs: pytest.fail("data-tree enumeration"),
    )
    protected = _graph_paths(tmp_path, active={"cache_dir": str(directory)})
    assert set(protected) == {directory, manifest, upstream}
    overlap = retention_contract._paths_overlap
    assert any(overlap(directory / "nested" / "payload.bin", path) for path in protected)
    assert any(overlap(tmp_path, path) for path in protected)
    assert not any(overlap(tmp_path / "cache-sibling", path) for path in protected)


@pytest.mark.parametrize("reference", ["missing.json", "missing.npy", "opaque-directory"])
def test_authority_graph_unresolved_references_fail_closed(tmp_path: Path, reference: str) -> None:
    if reference == "opaque-directory":
        (tmp_path / reference).mkdir()
    with pytest.raises(EvidenceRetentionError, match="authority"):
        _graph_paths(tmp_path, active={"path": reference})


@pytest.mark.parametrize("encoded", [
    "[]", "{", '{"path":"first.npy","path":"second.npy"}',
    '{"value":NaN}', '{"value":Infinity}', '{"value":1e999}',
    '{"path":17}', '{"path":{}}',
    '{"path":[]}', '{"paths":17}', '{"manifest_path":""}',
    '{"manifest_path":null,"manifest_sha256":"' + "a" * 64 + '"}',
])
def test_authority_graph_malformed_metadata_fails_closed(tmp_path: Path, encoded: str) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(encoded, encoding="utf-8")
    with pytest.raises(EvidenceRetentionError, match="authority"):
        _graph_paths(tmp_path, active={"manifest_path": str(manifest)})


@pytest.mark.parametrize("digest", ["0" * 64, "not-an-exact-hash"])
def test_authority_graph_checks_declared_manifest_hashes(tmp_path: Path, digest: str) -> None:
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {})
    with pytest.raises(EvidenceRetentionError, match="SHA-256"):
        _graph_paths(tmp_path, active={"manifest_path": str(manifest), "manifest_sha256": digest})


def test_authority_graph_checks_hash_on_already_visited_json(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {})
    with pytest.raises(EvidenceRetentionError, match="SHA-256"):
        _graph_paths(tmp_path, history=[
            {"manifest_path": str(manifest), "manifest_sha256": "0" * 64},
            {"manifest_path": str(manifest), "manifest_sha256": sha256_file(manifest)},
        ])


@pytest.mark.parametrize("directory_link", [False, True])
def test_authority_graph_rejects_symlinks_before_opening_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory_link: bool,
) -> None:
    directory = tmp_path / "real"
    directory.mkdir()
    manifest = directory / "manifest.json"
    _write_json(manifest, {})
    link = tmp_path / "link"
    link.symlink_to(directory if directory_link else manifest)
    reference = link / "manifest.json" if directory_link else link
    monkeypatch.setattr(
        retention_contract, "_authority_json",
        lambda *_args, **_kwargs: pytest.fail("opened symlink"),
    )
    with pytest.raises(EvidenceRetentionError, match="symlink"):
        _graph_paths(tmp_path, active={"manifest_path": str(reference)})


def test_authority_graph_refuses_sealed_test_without_path_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "sealed" / "test_unified_exit_lifecycle.manifest.json"
    monkeypatch.setattr(
        retention_contract, "_canonical_path",
        lambda *_args, **_kwargs: pytest.fail("resolved a TEST path"),
    )
    monkeypatch.setattr(
        retention_contract, "_authority_json",
        lambda *_args, **_kwargs: pytest.fail("opened TEST metadata"),
    )
    with pytest.raises(EvidenceRetentionError, match="sealed TEST"):
        _graph_paths(tmp_path, active={"splits": {"test": {"lifecycle_manifest": str(reference)}}})


@pytest.mark.parametrize("limit", [
    "MAX_AUTHORITY_JSON_BYTES", "MAX_AUTHORITY_TOTAL_JSON_BYTES",
    "MAX_AUTHORITY_JSON_FILES", "MAX_AUTHORITY_VALUES", "MAX_AUTHORITY_DEPTH",
])
def test_authority_graph_limits_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str,
) -> None:
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"nested": {"value": "metadata"}})
    monkeypatch.setattr(retention_contract, limit, 1)
    if limit == "MAX_AUTHORITY_JSON_FILES":
        second = tmp_path / "second.json"
        _write_json(second, {})
        _write_json(manifest, {"manifest_path": str(second)})
    with pytest.raises(EvidenceRetentionError, match="limit exceeded"):
        _graph_paths(tmp_path, active={"manifest_path": str(manifest)})


def test_authority_graph_binary_dependencies_are_not_read_or_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = tmp_path / "physical.npy"
    payload.write_bytes(b"mechanical fixture")
    monkeypatch.setattr(
        retention_contract, "sha256_file", lambda *_args: pytest.fail("hashed data"),
    )
    monkeypatch.setattr(
        retention_contract, "_authority_json",
        lambda *_args, **_kwargs: pytest.fail("opened data"),
    )
    assert _graph_paths(tmp_path, active={
        "feats_npy": str(payload), "feats_npy_sha256": "a" * 64,
    }) == (payload,)


def test_authority_graph_blocks_transitive_plan_before_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "keep.bin"
    target.write_bytes(b"mechanical dependency")
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"source_path": str(target)})
    registry, launch = _authority_files(tmp_path, active={"manifest_path": "manifest.json"})
    monkeypatch.setattr(
        retention_contract, "write_inventory_manifest",
        lambda *_args, **_kwargs: pytest.fail("inventoried protected target"),
    )
    with pytest.raises(EvidenceRetentionError, match="authority-protected"):
        _published_plan(tmp_path, target=target, registry=registry, launch=launch)
    assert target.read_bytes() == b"mechanical dependency"
    assert not (tmp_path / "plans").exists()


def test_authority_graph_is_recomputed_when_validating_a_plan(tmp_path: Path) -> None:
    target = tmp_path / "newly-protected.bin"
    target.write_bytes(b"mechanical dependency")
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {})
    registry, launch = _authority_files(tmp_path, active={"manifest_path": str(manifest)})
    plan_path, plan_sha = _published_plan(
        tmp_path, target=target, registry=registry, launch=launch,
    )
    _write_json(manifest, {"source_path": str(target)})
    with pytest.raises(EvidenceRetentionError, match="authority-protected"):
        validate_cleanup_plan(
            plan_path, plan_sha, vedtak=VEDTAK, allowed_roots=(tmp_path,),
            required_artifact_registry_json=registry,
            required_launch_contract_json=launch,
        )
    assert target.exists()


@pytest.mark.parametrize("size_bytes", [34_085_492, 91_059_197])
def test_authority_graph_accepts_reported_metadata_sizes(
    tmp_path: Path, size_bytes: int,
) -> None:
    manifest = tmp_path / "manifest.json"
    encoded = b'{"schema_version":"mechanical-size-check"}'
    manifest.write_bytes(encoded + b" " * (size_bytes - len(encoded)))
    assert _graph_paths(tmp_path, active={"manifest_path": str(manifest)}) == (manifest,)


def test_authority_graph_lifecycle_relative_split_manifest(tmp_path: Path) -> None:
    root = tmp_path / "lifecycle"
    root.mkdir()
    binary = root / "train_unified_exit_lifecycle.parquet"
    binary.write_bytes(b"mechanical fixture")
    split_manifest = root / "train_unified_exit_lifecycle.manifest.json"
    _write_json(split_manifest, {"lifecycle_parquet": binary.name})
    manifest = root / "UNIFIED_EXIT_LIFECYCLE_MANIFEST.json"
    _write_json(manifest, {"splits": {"train": {
        "lifecycle_manifest": split_manifest.name,
        "lifecycle_manifest_sha256": sha256_file(split_manifest),
    }}})
    assert set(_graph_paths(tmp_path, active={"manifest_path": str(manifest)})) == {
        manifest, split_manifest, binary,
    }


@pytest.mark.parametrize("reference", [
    ".../manifest.json", "nested/../manifest.json", "~/manifest.json",
    "https://example.invalid/manifest.json", " manifest.json", "manifest.json\x00",
])
def test_authority_graph_ambiguous_references_fail_closed(
    tmp_path: Path, reference: str,
) -> None:
    with pytest.raises(EvidenceRetentionError, match="authority"):
        _graph_paths(tmp_path, active={"manifest_path": reference})


def test_authority_graph_reader_itself_rejects_a_symlink(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {})
    link = tmp_path / "link.json"
    link.symlink_to(manifest)
    with pytest.raises(EvidenceRetentionError, match="authority JSON"):
        retention_contract._authority_json(link, byte_limit=1024)


def test_authority_graph_bounds_transitive_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retention_contract, "MAX_AUTHORITY_DEPTH", 5)
    for index in range(8):
        _write_json(tmp_path / f"manifest-{index}.json", {
            "manifest_path": f"manifest-{index + 1}.json",
        })
    with pytest.raises(EvidenceRetentionError, match="graph depth limit exceeded"):
        _graph_paths(tmp_path, active={"manifest_path": "manifest-0.json"})


def test_authority_graph_existing_output_directory_is_not_an_opaque_leaf(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    manifest = root / "manifest.json"
    dependency = tmp_path / "dependency.npy"
    dependency.write_bytes(b"mechanical fixture")
    _write_json(manifest, {"feats_npy": str(dependency)})
    assert set(_graph_paths(tmp_path, active={"out_dir": str(root)})) == {
        root, manifest, dependency,
    }


def test_authority_graph_event_history_fails_closed_on_missing_outbound_reference(
    tmp_path: Path,
) -> None:
    history = tmp_path / "events"
    earlier, _ = write_immutable_json_event(history, "RETAIN", {
        "created_utc": "2026-09-06T10:00:00+00:00", "decision": "PASS",
        "source_path": str(tmp_path / "unresolved-upstream.bin"),
    })
    current, _ = write_immutable_json_event(history, "RETAIN", {
        "created_utc": "2026-09-06T09:59:00+00:00", "decision": "FAIL",
    })
    witness = retention_contract.immutable_events._order_path(earlier)
    assert retention_contract._authority_event_prefix(current) == "RETAIN"
    with pytest.raises(EvidenceRetentionError, match="unresolved authority reference"):
        _graph_paths(tmp_path, active={"event_path": str(current)})
    assert earlier.exists() and current.exists() and witness.exists()


def test_authority_graph_does_not_parse_order_witness_as_generic_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    event, _ = write_immutable_json_event(tmp_path / "events", "RETAIN", {
        "created_utc": CREATED_UTC, "decision": "PASS",
    })
    witness = retention_contract.immutable_events._order_path(event)
    monkeypatch.setattr(
        retention_contract, "_authority_json",
        lambda *_args, **_kwargs: pytest.fail("generic witness JSON parse"),
    )
    with pytest.raises(EvidenceRetentionError, match="publication-order witnesses"):
        _graph_paths(tmp_path, active={"path": str(witness)})


def test_authority_graph_directory_manifest_cannot_hide_event_history(tmp_path: Path) -> None:
    history = tmp_path / "events"
    event, _ = write_immutable_json_event(history, "RETAIN", {
        "created_utc": CREATED_UTC, "decision": "PASS",
    })
    _write_json(history / "manifest.json", {"event_path": str(event)})
    protected = _graph_paths(tmp_path, active={"root": str(history)})
    assert set(protected) == {
        history, event, history / "manifest.json",
        retention_contract.immutable_events._order_path(event),
    }


def test_plan_directory_protects_its_publication_witness(tmp_path: Path) -> None:
    target = tmp_path / "ordinary.bin"
    target.write_bytes(b"mechanical fixture")
    registry, launch = _authority_files(tmp_path)
    plan_path, _ = _published_plan(tmp_path, target=target, registry=registry, launch=launch)
    witness = retention_contract.immutable_events._order_path(plan_path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["targets"][0]["path"] = str(witness)
    payload["created_utc"] = "2026-07-20T10:00:01+00:00"
    successor, _ = write_immutable_json_event(plan_path.parent, PLAN_EVENT_PREFIX, payload)
    with pytest.raises(EvidenceRetentionError, match="authority-protected"):
        validate_cleanup_plan(
            successor, sha256_file(successor), vedtak=VEDTAK,
            allowed_roots=(tmp_path,), required_artifact_registry_json=registry,
            required_launch_contract_json=launch,
        )
    assert witness.exists() and plan_path.exists()


def test_authority_graph_retains_scoped_cross_run_history_and_outbound_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = tmp_path / "history"
    dependency = tmp_path / "outside-scope.npy"
    dependency.write_bytes(b"mechanical fixture")
    manifest = tmp_path / "upstream.json"
    _write_json(manifest, {"feats_npy": dependency.name})
    earlier, _ = write_immutable_json_event(scope / "run_one", "RETAIN", {
        "created_utc": "2026-09-06T10:00:00+00:00", "decision": "PASS",
        "manifest_path": str(manifest), "manifest_sha256": sha256_file(manifest),
    }, authority_root=scope, scope_dir_glob="run_*")
    current, _ = write_immutable_json_event(scope / "run_two", "RETAIN", {
        "created_utc": "2026-09-06T09:59:00+00:00", "decision": "FAIL",
    }, authority_root=scope, scope_dir_glob="run_*")
    reader = retention_contract._authority_json
    reads: list[Path] = []

    def metadata_reader(path: Path, **kwargs: int):
        assert path.suffix == ".json"
        reads.append(path)
        return reader(path, **kwargs)

    monkeypatch.setattr(retention_contract, "_authority_json", metadata_reader)
    protected = _graph_paths(tmp_path, active={"event_path": str(current)})
    assert set(protected) == {
        scope, earlier, current, manifest, dependency,
        retention_contract.immutable_events._order_path(earlier),
        retention_contract.immutable_events._order_path(current),
    }
    assert set(reads) == {earlier, current, manifest}
    assert len(reads) == 3
    assert retention_contract._paths_overlap(scope / "run_other" / "future.order", scope)


def test_authority_graph_rejects_orphan_publication_witness(tmp_path: Path) -> None:
    root = tmp_path / "history"
    event, _ = write_immutable_json_event(root, "RETAIN", {
        "created_utc": CREATED_UTC, "decision": "PASS",
    })
    witness = retention_contract.immutable_events._order_path(event)
    order = json.loads(witness.read_text(encoding="utf-8"))
    missing = root / "RETAIN_20260720T100001000000Z.json"
    order["json_path"] = str(missing)
    orphan = retention_contract.immutable_events._order_path(missing)
    _write_json(orphan, order)
    with pytest.raises(EvidenceRetentionError, match="witness has no event"):
        _graph_paths(tmp_path, active={"event_path": str(event)})
    assert witness.exists() and orphan.exists() and event.exists()


def test_authority_graph_rejects_legacy_event_without_declared_scope(tmp_path: Path) -> None:
    event = tmp_path / "RETAIN_20260720T100000000000Z.json"
    _write_json(event, {"created_utc": CREATED_UTC, "json_path": str(event)})
    with pytest.raises(
        EvidenceRetentionError, match="legacy event has no declared authority scope",
    ):
        _graph_paths(tmp_path, active={"event_path": str(event)})


def test_plan_validation_rechecks_the_complete_publication_history(tmp_path: Path) -> None:
    target = tmp_path / "ordinary.bin"
    target.write_bytes(b"mechanical fixture")
    registry, launch = _authority_files(tmp_path)
    earlier, _ = _published_plan(tmp_path, target=target, registry=registry, launch=launch)
    payload = json.loads(earlier.read_text(encoding="utf-8"))
    payload["created_utc"] = "2026-07-20T10:00:01+00:00"
    current, _ = write_immutable_json_event(earlier.parent, PLAN_EVENT_PREFIX, payload)
    historical = json.loads(earlier.read_text(encoding="utf-8"))
    historical["vedtak"] = "GX1-CLEANUP-TAMPERED"
    _write_json(earlier, historical)
    with pytest.raises(EvidenceRetentionError, match="immutable authority"):
        validate_cleanup_plan(
            current, sha256_file(current), vedtak=VEDTAK,
            allowed_roots=(tmp_path,), required_artifact_registry_json=registry,
            required_launch_contract_json=launch,
        )
    assert target.exists()


def test_authority_graph_event_self_reference_is_not_a_payload_sha_binding(tmp_path: Path) -> None:
    root = tmp_path / "events"
    event, _ = write_immutable_json_event(root, "RETAIN", {
        "created_utc": CREATED_UTC, "sha256": "a" * 64,
    })
    assert set(_graph_paths(tmp_path, active={"event_path": str(event)})) == {
        root, event, retention_contract.immutable_events._order_path(event),
    }


def test_authority_graph_rejects_event_hash_before_discovering_its_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    event, _ = write_immutable_json_event(tmp_path / "events", "RETAIN", {
        "created_utc": CREATED_UTC,
    })
    monkeypatch.setattr(
        retention_contract.immutable_events, "validated_immutable_event_authority_inventory",
        lambda *_args, **_kwargs: pytest.fail("followed an unverified event scope"),
    )
    with pytest.raises(EvidenceRetentionError, match="SHA-256 mismatch"):
        _graph_paths(tmp_path, active={"path": str(event), "sha256": "0" * 64})


@pytest.mark.parametrize("limit", ["event_bytes", "witness_bytes", "total_bytes", "events"])
def test_authority_graph_inventory_limits_precede_historical_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str,
) -> None:
    owner = retention_contract.immutable_events
    earlier, _ = write_immutable_json_event(tmp_path / "events", "RETAIN", {
        "created_utc": CREATED_UTC, "padding": "mechanical" * 1024,
    })
    current, _ = write_immutable_json_event(tmp_path / "events", "RETAIN", {
        "created_utc": "2026-07-20T10:00:01+00:00",
    })
    earlier_witness = owner._order_path(earlier)
    current_witness = owner._order_path(current)
    documents = (earlier, current, earlier_witness, current_witness)
    if limit == "event_bytes":
        ceiling = max(path.stat().st_size for path in documents if path != earlier)
        monkeypatch.setattr(retention_contract, "MAX_AUTHORITY_JSON_BYTES", ceiling)
        expected = "max_document_bytes"
    elif limit == "witness_bytes":
        ceiling = max(path.stat().st_size for path in documents)
        earlier_witness.write_bytes(earlier_witness.read_bytes() + b" " * ceiling)
        monkeypatch.setattr(retention_contract, "MAX_AUTHORITY_JSON_BYTES", ceiling)
        expected = "max_document_bytes"
    elif limit == "total_bytes":
        ceiling = sum(path.stat().st_size for path in documents) - 1
        monkeypatch.setattr(retention_contract, "MAX_AUTHORITY_TOTAL_JSON_BYTES", ceiling)
        expected = "max_total_bytes"
    else:
        monkeypatch.setattr(retention_contract, "MAX_AUTHORITY_JSON_FILES", 1)
        expected = "max_events"
    reader = owner._read_json_object
    reads: list[Path] = []

    def bounded_bootstrap(path: Path, *, max_bytes: int | None = None):
        assert path == current_witness
        assert max_bytes == current_witness.stat().st_size
        reads.append(path)
        return reader(path, max_bytes=max_bytes)

    monkeypatch.setattr(owner, "_read_json_object", bounded_bootstrap)
    with pytest.raises(EvidenceRetentionError, match=expected):
        _graph_paths(tmp_path, active={"event_path": str(current)})
    assert reads == [current_witness]
    assert all(path.exists() for path in documents)


def test_authority_graph_passes_remaining_inventory_budgets_and_rechecks_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = retention_contract.immutable_events
    upstream = tmp_path / "upstream.json"
    consumed = tmp_path / "consumed.json"
    leading = tmp_path / "leading.json"
    _write_json(upstream, {})
    _write_json(consumed, {"value": "mechanical metadata"})
    earlier, _ = write_immutable_json_event(tmp_path / "events", "RETAIN", {
        "created_utc": CREATED_UTC, "manifest_path": str(upstream),
    })
    current, _ = write_immutable_json_event(tmp_path / "events", "RETAIN", {
        "created_utc": "2026-07-20T10:00:01+00:00",
    })
    _write_json(leading, {"paths": [str(consumed), str(current)]})
    witnesses = (owner._order_path(earlier), owner._order_path(current))
    scope_bytes = sum(path.stat().st_size for path in (earlier, current, *witnesses))
    budget = scope_bytes + sum(path.stat().st_size for path in (leading, consumed, upstream))
    monkeypatch.setattr(retention_contract, "MAX_AUTHORITY_TOTAL_JSON_BYTES", budget)
    monkeypatch.setattr(retention_contract, "MAX_AUTHORITY_JSON_FILES", 5)
    inventory_reader = owner.validated_immutable_event_authority_inventory
    calls: list[dict[str, object]] = []

    def bounded_inventory(path: Path, prefix: str, **kwargs: object):
        calls.append(kwargs)
        return inventory_reader(path, prefix, **kwargs)

    monkeypatch.setattr(owner, "validated_immutable_event_authority_inventory", bounded_inventory)
    protected = _graph_paths(tmp_path, active={"manifest_path": str(leading)})
    assert set(protected) == {
        leading, consumed, upstream, earlier, current, *witnesses, current.parent,
    }
    assert len(calls) == 2
    assert calls[0]["max_total_bytes"] == scope_bytes + upstream.stat().st_size
    assert calls[0]["max_events"] == 3
    assert calls[1]["max_total_bytes"] == scope_bytes
    assert calls[1]["max_events"] == 2
    assert calls[1]["authority_root"] == current.parent
    assert all(call["max_document_bytes"] == 128 * 1024 * 1024 for call in calls)


def test_authority_graph_owner_rejects_history_growth_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = retention_contract.immutable_events
    earlier, _ = write_immutable_json_event(tmp_path / "events", "RETAIN", {
        "created_utc": CREATED_UTC,
    })
    current, _ = write_immutable_json_event(tmp_path / "events", "RETAIN", {
        "created_utc": "2026-07-20T10:00:01+00:00",
    })
    original_size = earlier.stat().st_size
    reader = owner._read_json_object

    def grow_before_read(path: Path, *, max_bytes: int | None = None):
        assert max_bytes is not None
        if path == earlier:
            assert max_bytes == original_size
            path.write_bytes(path.read_bytes() + b" ")
        return reader(path, max_bytes=max_bytes)

    monkeypatch.setattr(owner, "_read_json_object", grow_before_read)
    with pytest.raises(EvidenceRetentionError, match="bounded read"):
        _graph_paths(tmp_path, active={"event_path": str(current)})
    assert earlier.exists() and current.exists()


def test_plan_history_is_bounded_and_reserves_graph_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "ordinary.bin"
    target.write_bytes(b"mechanical fixture")
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"value": "mechanical metadata"})
    registry, launch = _authority_files(tmp_path, active={"manifest_path": str(manifest)})
    plan, plan_sha = _published_plan(tmp_path, target=target, registry=registry, launch=launch)
    owner = retention_contract.immutable_events
    scope_bytes = plan.stat().st_size + owner._order_path(plan).stat().st_size
    monkeypatch.setattr(
        retention_contract, "MAX_AUTHORITY_TOTAL_JSON_BYTES",
        scope_bytes + manifest.stat().st_size - 1,
    )
    reader = owner._read_json_object
    reads: list[Path] = []

    def bounded_plan_reader(path: Path, *, max_bytes: int | None = None):
        assert max_bytes is not None
        reads.append(path)
        return reader(path, max_bytes=max_bytes)

    monkeypatch.setattr(owner, "_read_json_object", bounded_plan_reader)
    with pytest.raises(EvidenceRetentionError, match="authority JSON byte limit exceeded"):
        validate_cleanup_plan(
            plan, plan_sha, vedtak=VEDTAK, allowed_roots=(tmp_path,),
            required_artifact_registry_json=registry,
            required_launch_contract_json=launch,
        )
    assert set(reads) == {plan, owner._order_path(plan)}
    assert target.exists()


def test_plan_requires_bounded_owner_selected_publication(tmp_path: Path) -> None:
    target = tmp_path / "ordinary.bin"
    target.write_bytes(b"mechanical fixture")
    registry, launch = _authority_files(tmp_path)
    earlier, earlier_sha = _published_plan(
        tmp_path, target=target, registry=registry, launch=launch,
    )
    payload = json.loads(earlier.read_text(encoding="utf-8"))
    payload["created_utc"] = "2026-07-20T09:59:00+00:00"
    current, _ = write_immutable_json_event(earlier.parent, PLAN_EVENT_PREFIX, payload)
    with pytest.raises(EvidenceRetentionError, match="not newest immutable authority"):
        validate_cleanup_plan(
            earlier, earlier_sha, vedtak=VEDTAK, allowed_roots=(tmp_path,),
            required_artifact_registry_json=registry,
            required_launch_contract_json=launch,
        )
    assert current.exists() and target.exists()


@pytest.mark.parametrize("limit", ["max_total_bytes", "max_events"])
def test_publication_inventory_zero_remaining_budget_never_decodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str,
) -> None:
    owner = retention_contract.immutable_events
    event, _ = write_immutable_json_event(tmp_path / "events", "RETAIN", {
        "created_utc": CREATED_UTC,
    })
    monkeypatch.setattr(
        owner, "_read_json_object",
        lambda *_args, **_kwargs: pytest.fail("decoded with exhausted budget"),
    )
    with pytest.raises(EvidenceRetentionError, match=limit):
        retention_contract._authority_event_inventory(event, "RETAIN", **{limit: 0})
