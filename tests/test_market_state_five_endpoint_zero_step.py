"""Small preclaim checks for the one-row source audit; no model execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from gx1.audit import verify_market_state_five_endpoint_zero_step_v1 as audit
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as target


def _owners():
    return {path: audit._sha(audit.REPO / path) for path in audit.REQUIRED_OWNERS}


def _plan_and_policy():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    output = audit.RUNS / audit.STAGE.upper()
    plan_ref = {"path": str(audit.REPO / "configs/research" / (audit.STAGE.upper() + ".json")),
                "sha256": "a" * 64}
    plan = {"schema_version": audit.SCHEMA, "stage": audit.STAGE,
            "status": "PREPARED_NOT_ADMITTED", "source_base_head": head,
            "output_root": str(output), "max_invocations": 1,
            "fold_id": 1, "partition": "fit", "sample_parent_rows": [0],
            "seed": 20261107, "model_forwards_authorized": 1,
            "optimizer_steps_authorized": 0, "training_authorized": False,
            "test_data_used": False,
            "corrected_mask": {"path": target.REFS["mask"][0],
                               "sha256": target.REFS["mask"][1]},
            "source_owners": _owners(),
            "expected_view_sha256": "b" * 64,
            "expected_objective_sha256": "c" * 64,
            "max_json_bytes": 1024 * 1024,
            "limits": {"memory_max": 8 * 1024**3,
                       "memory_high": 8 * 1024**3,
                       "swap": 512 * 1024**2,
                       "max_wall_seconds": 3600}}
    plan.update({name: audit._fixed_ref(name) for name in audit.FIXED})
    gate = {"decision": "ADMITTED", "plan": plan_ref,
            "execution_authorized": True, "data_reads_authorized": True,
            "model_forwards_authorized": 1, "optimizer_steps_authorized": 0,
            "training_authorized": False, "test_authorized": False,
            "claimed_invocations": 0, "completed_invocations": 0,
            "max_invocations": 1, "output_root": str(output)}
    policy = {"training_enabled": False, "full_val_allowed": False,
              audit.STAGE: gate}
    cap = {"memory_max": 8 * 1024**3, "memory_high": 8 * 1024**3,
           "swap": 512 * 1024**2, "pids": 128}
    return plan, plan_ref, policy, cap, head, output


def _qualify(plan, ref, policy, cap, head, output):
    audit._qualify_plan(plan, ref, policy, head=head, out_dir=output,
                        device="cuda", cap=cap)


def test_plan_accepts_bound_fit_only_scope_without_claim():
    plan, ref, policy, cap, head, output = _plan_and_policy()
    previous = subprocess.check_output(["git", "rev-parse", "HEAD^"], text=True).strip()
    plan["source_base_head"] = previous
    _qualify(plan, ref, policy, cap, head, output)
    audit._qualify_plan(plan, ref, None, head=head, out_dir=output,
                        device="cuda", cap=None)
    assert not output.exists()


@pytest.mark.parametrize("change", [
    lambda p, r, g, c: p.update(source_base_head="0" * 40),
    lambda p, r, g, c: p.update(sample_parent_rows=[29849]),
    lambda p, r, g, c: p.update(partition="score"),
    lambda p, r, g, c: p.update(training_authorized=True),
    lambda p, r, g, c: p["corrected_mask"].update(sha256="0" * 64),
    lambda p, r, g, c: p["source_owners"].pop(
        "gx1/contracts/market_state_five_endpoint_model_v1.py"),
    lambda p, r, g, c: p["source_owners"].update(
        {"gx1/contracts/market_state_five_endpoint_model_v1.py": "0" * 64}),
    lambda p, r, g, c: p["limits"].update(memory_max=4 * 1024**3),
    lambda p, r, g, c: c.update(memory_max=4 * 1024**3),
    lambda p, r, g, c: g[audit.STAGE].update(model_forwards_authorized=2),
    lambda p, r, g, c: g.update(training_enabled=True),
])
def test_plan_rejects_scope_hash_cap_and_policy_changes(change):
    plan, ref, policy, cap, head, output = _plan_and_policy()
    change(plan, ref, policy, cap)
    with pytest.raises(RuntimeError, match="FIVE_ENDPOINT_ZERO_STEP_"):
        _qualify(plan, ref, policy, cap, head, output)
    assert not output.exists()


def test_source_refs_drop_prior_target_and_bind_corrected_mask(tmp_path):
    legacy = tmp_path / "legacy.txt"
    legacy.write_text("small source")
    digest = hashlib.sha256(legacy.read_bytes()).hexdigest()
    old_view = {"source_bindings": {
        "dynamic_state_target_source": {"path": str(legacy), "sha256": digest},
        "legacy_feature_source": {"path": str(legacy), "sha256": digest}}}
    mask_ref = {"path": target.REFS["mask"][0], "sha256": target.REFS["mask"][1]}
    sources = audit._source_refs(old_view, _owners(), mask_ref)
    assert "dynamic_state_target_source" not in sources
    assert sources["five_endpoint_corrected_mask"] == mask_ref
    assert set(audit.NEW_SOURCE_KEYS) <= set(sources)
    old_view["source_bindings"].pop("dynamic_state_target_source")
    with pytest.raises(RuntimeError, match="OLD_VIEW_TARGET_SOURCE"):
        audit._source_refs(old_view, _owners(), mask_ref)



def test_writable_october_clock_headers_are_sealed_without_value_changes():
    import numpy as np
    signal = np.zeros((3, 2), dtype=np.float32)
    signal.setflags(write=False)
    times = np.array([10, 20, 30], dtype=np.int64)
    positions = np.array([0, 2], dtype=np.int64)
    source = {"signal": signal, "times_ns": times,
              "sample_positions": positions, "audit": {"source": "synthetic"}}
    sealed = audit._readonly_sequence_source_headers(source)
    assert sealed["signal"] is signal and sealed["audit"] is source["audit"]
    assert np.array_equal(sealed["times_ns"], times)
    assert np.array_equal(sealed["sample_positions"], positions)
    assert not sealed["times_ns"].flags.writeable
    assert not sealed["sample_positions"].flags.writeable
    assert times.flags.writeable and positions.flags.writeable
    with pytest.raises(ValueError):
        sealed["sample_positions"][0] = 1
    source["times_ns"] = times.astype(np.float32)
    with pytest.raises(RuntimeError, match="SEQUENCE_HEADER_INPUT:times_ns"):
        audit._readonly_sequence_source_headers(source)


def test_real_october_preclaim_receipts_without_payload():
    audit._inspect_reusable_receipts()


def test_metadata_only_view_identity_checked_before_claim():
    plan, _, _, _, _, output = _plan_and_policy()
    plan["expected_view_sha256"] = "094e6e64f7b5092c91a9e2c8f3259f176a90c753767d585d4169247ba2ae92ae"
    plan["expected_objective_sha256"] = "bc4f261a5ee2380879d4eb53a118e5aac5479e37dcb073126dab91c05f6420ed"
    view, _, _, _, _ = audit._bind_view_metadata(plan)
    assert view["contract_sha256"] == plan["expected_view_sha256"]
    assert len(view["source_bindings"]) == 15
    assert "dynamic_state_target_source" not in view["source_bindings"]
    plan["expected_view_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="NEW_VIEW_IDENTITY"):
        audit._bind_view_metadata(plan)
    plan["expected_view_sha256"] = view["contract_sha256"]
    plan["expected_objective_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="NEW_VIEW_IDENTITY"):
        audit._bind_view_metadata(plan)
    assert not output.exists()


def test_exclusive_small_publication_without_active_run(tmp_path):
    ref = audit._write(tmp_path, "RECEIPT.json", {"status": "synthetic"}, limit=1024)
    path = tmp_path / "RECEIPT.json"
    first = path.read_bytes()
    assert ref == {"path": str(path), "sha256": hashlib.sha256(first).hexdigest()}
    with pytest.raises(RuntimeError, match="already exists"):
        audit._write(tmp_path, "RECEIPT.json", {"status": "second"}, limit=1024)
    assert path.read_bytes() == first
    with pytest.raises(RuntimeError, match="PUBLICATION_SIZE"):
        audit._write(tmp_path, "TOO_LARGE.json", {"text": "x" * 1024}, limit=128)
    assert not (tmp_path / "TOO_LARGE.json").exists()


def test_read_only_preflight_checks_view_before_claim(monkeypatch):
    plan, ref, _, _, head, output = _plan_and_policy()
    plan["expected_view_sha256"] = "094e6e64f7b5092c91a9e2c8f3259f176a90c753767d585d4169247ba2ae92ae"
    plan["expected_objective_sha256"] = "bc4f261a5ee2380879d4eb53a118e5aac5479e37dcb073126dab91c05f6420ed"
    monkeypatch.setattr(audit, "_clean_head", lambda: head)
    monkeypatch.setattr(audit, "_checked_plan_ref", lambda *_: (plan, ref))
    ready = audit.preflight_read_only(Path(ref["path"]), ref["sha256"], output)
    assert ready["view_sha256"] == plan["expected_view_sha256"]
    assert ready["claim_created"] is False and ready["model_forwards"] == 0
    plan["expected_objective_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="NEW_VIEW_IDENTITY"):
        audit.preflight_read_only(Path(ref["path"]), ref["sha256"], output)
    assert not output.exists()


def test_preclaim_rejects_changed_ancestral_worker(monkeypatch):
    input_plan = audit._read_ref(audit._fixed_ref("input_plan"))
    worker_path = Path(input_plan["ancestral_264_worker"]["path"])
    original = audit._sha
    monkeypatch.setattr(audit, "_sha", lambda path: "0" * 64 if path == worker_path else original(path))
    with pytest.raises(RuntimeError, match="ANCESTRAL_WORKER_HASH"):
        audit._inspect_reusable_receipts()


def _synthetic_preflight(monkeypatch):
    plan = {"max_json_bytes": 4096}
    plan_ref = {"path": "/synthetic/PLAN.json", "sha256": "a" * 64}
    policy_ref = {"path": "/synthetic/POLICY.json", "sha256": "b" * 64}
    monkeypatch.setattr(audit, "_preflight", lambda *args: (plan, plan_ref, policy_ref, "c" * 40))


def test_caught_forward_failure_publishes_terminal_without_relaunch(tmp_path, monkeypatch):
    _synthetic_preflight(monkeypatch)
    monkeypatch.setattr(audit, "_one_forward", lambda _: (_ for _ in ()).throw(RuntimeError("synthetic forward")))
    output = tmp_path / "synthetic-run"
    with pytest.raises(RuntimeError, match="synthetic forward"):
        audit.run(Path("/synthetic/PLAN.json"), "a" * 64, "cuda", output)
    assert {p.name for p in output.iterdir()} == {"CLAIM.json", "FAILED.json", "TERMINAL.json"}
    failure = json.loads((output / "FAILED.json").read_text())
    terminal = json.loads((output / "TERMINAL.json").read_text())
    assert terminal["status"] == "FAILED" and terminal["exit_code"] == 1
    assert terminal["failure"] == audit._ref(output / "FAILED.json", audit._sha(output / "FAILED.json"))
    assert terminal["claim"] == failure["claim"]
    assert terminal["no_relaunch_under_this_plan"] is True
    with pytest.raises(FileExistsError):
        audit.run(Path("/synthetic/PLAN.json"), "a" * 64, "cuda", output)


def test_caught_claim_failure_still_publishes_failure_terminal(tmp_path, monkeypatch):
    _synthetic_preflight(monkeypatch)
    original = audit._write
    def fail_claim(root, name, value, *, limit):
        if name == "CLAIM.json":
            raise RuntimeError("synthetic claim publication")
        return original(root, name, value, limit=limit)
    monkeypatch.setattr(audit, "_write", fail_claim)
    output = tmp_path / "synthetic-claim-failure"
    with pytest.raises(RuntimeError, match="synthetic claim publication"):
        audit.run(Path("/synthetic/PLAN.json"), "a" * 64, "cuda", output)
    assert {p.name for p in output.iterdir()} == {"FAILED.json", "TERMINAL.json"}
    terminal = json.loads((output / "TERMINAL.json").read_text())
    assert terminal["claim"] is None and terminal["status"] == "FAILED"
    assert terminal["no_relaunch_under_this_plan"] is True


def test_synthetic_success_publishes_exact_claim_result_terminal_chain(tmp_path, monkeypatch):
    _synthetic_preflight(monkeypatch)
    calls = []
    def fake_forward(plan):
        calls.append(plan)
        return {"model_forwards": 1, "optimizer_steps": 0,
                "test_data_used": False, "synthetic_only": True}
    monkeypatch.setattr(audit, "_one_forward", fake_forward)
    monkeypatch.setattr(audit, "_read_ref", lambda *_args, **_kwargs: pytest.fail("data read"))
    original_sha = audit._sha
    def fake_sha(path):
        if str(path) == "/synthetic/PLAN.json":
            return "a" * 64
        if str(path) == "/synthetic/POLICY.json":
            return "b" * 64
        return original_sha(path)
    monkeypatch.setattr(audit, "_sha", fake_sha)
    def fake_git(args, *, text):
        assert text is True
        if args == ["git", "rev-parse", "HEAD"]:
            return "c" * 40 + "\n"
        if args == ["git", "status", "--porcelain"]:
            return ""
        pytest.fail("unexpected source check")
    monkeypatch.setattr(audit.subprocess, "check_output", fake_git)

    output = tmp_path / "synthetic-success"
    published = audit.run(Path("/synthetic/PLAN.json"), "a" * 64, "cuda", output)
    assert calls == [{"max_json_bytes": 4096}]
    assert {path.name for path in output.iterdir()} == {"CLAIM.json", "RESULT.json", "TERMINAL.json"}
    claim_path, result_path, terminal_path = (output / name for name in
        ("CLAIM.json", "RESULT.json", "TERMINAL.json"))
    claim = json.loads(claim_path.read_text())
    result = json.loads(result_path.read_text())
    terminal = json.loads(terminal_path.read_text())
    claim_ref = {"path": str(claim_path), "sha256": hashlib.sha256(claim_path.read_bytes()).hexdigest()}
    result_ref = {"path": str(result_path), "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest()}
    terminal_ref = {"path": str(terminal_path), "sha256": hashlib.sha256(terminal_path.read_bytes()).hexdigest()}
    assert claim["plan"] == result["plan"] == terminal["plan"]
    assert claim["policy"] == {"path": "/synthetic/POLICY.json", "sha256": "b" * 64}
    assert claim["source_head"] == result["source_head"] == "c" * 40
    assert result["claim"] == terminal["claim"] == claim_ref
    assert terminal["result"] == result_ref
    assert published == {"result": result_ref, "terminal": terminal_ref}
    assert result["synthetic_only"] is True and result["model_forwards"] == 1
    assert terminal["output_inventory"] == ["CLAIM.json", "RESULT.json", "TERMINAL.json"]
    assert terminal["status"] == "COMPLETE" and terminal["no_relaunch_under_this_plan"] is True
