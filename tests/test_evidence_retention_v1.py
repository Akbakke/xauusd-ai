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
)


VEDTAK = "GX1-CLEANUP-TEST"
CREATED_UTC = "2026-07-20T10:00:00+00:00"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
