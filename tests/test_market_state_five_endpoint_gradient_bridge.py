"""Synthetic, CPU-only checks for the one-use TRAIN gradient witness."""
import copy
import hashlib
import json
import signal
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from gx1.audit import verify_market_state_five_endpoint_gradient_bridge_v1 as audit
from gx1.audit import verify_market_state_five_endpoint_zero_step_v1 as zero
from gx1.contracts.market_state_five_endpoint_target_binding_v1 import (
    BoundFiveEndpointTargets,
)


def _plan(tmp_path, monkeypatch):
    plan = json.loads((
        audit.REPO / "configs/research/MARKET_STATE_FIVE_ENDPOINT_ZERO_STEP_20261107_003.json"
    ).read_text())
    output = tmp_path / "gradient_bridge"
    monkeypatch.setattr(audit, "OUT_ROOT", output)
    plan.update({
        "schema_version": audit.SCHEMA, "stage": audit.STAGE,
        "source_base_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=audit.REPO, text=True).strip(),
        "output_root": str(output),
        "sample_parent_rows": audit.SAMPLE_ROWS,
        "sample_issue_ns": audit.ISSUE_NS,
        "backward_calls_authorized": 1,
        "limits": {"memory_max": audit.MEMORY_BYTES,
                   "memory_high": audit.MEMORY_BYTES,
                   "swap": audit.SWAP_BYTES,
                   "max_wall_seconds": audit.WALL_SECONDS},
        "execution": audit.EXECUTION,
        "prior_zero_result": audit.PRIOR_ZERO["result"],
        "prior_zero_terminal": audit.PRIOR_ZERO["terminal"],
    })
    for name in audit.REQUIRED_OWNERS:
        plan["source_owners"][name] = zero._sha(audit.REPO / name)
    return plan, output


def test_plan_scope_and_guard_fail_closed(tmp_path, monkeypatch):
    plan, output = _plan(tmp_path, monkeypatch)
    ref = {"path": "/synthetic/plan.json", "sha256": "a" * 64}
    head = plan["source_base_head"]
    audit._qualify_plan(plan, ref, None, head=head, out_dir=output,
                        device="cuda", cap=None)
    changed = copy.deepcopy(plan)
    changed["sample_parent_rows"] = [0]
    with pytest.raises(RuntimeError, match="PLAN_SCOPE"):
        audit._qualify_plan(changed, ref, None, head=head, out_dir=output,
                            device="cuda", cap=None)
    changed = copy.deepcopy(plan)
    changed["limits"]["memory_max"] = 8 * 1024**3
    with pytest.raises(RuntimeError, match="CAPS"):
        audit._qualify_plan(changed, ref, None, head=head, out_dir=output,
                            device="cuda", cap=None)
    cap = {"class": "producer", "execution_mode": "cuda_producer",
           "memory_max": audit.MEMORY_BYTES, "memory_high": audit.MEMORY_BYTES,
           "swap": audit.SWAP_BYTES, "pids": 64, "cgroup_path": "/sys/fs/cgroup/synthetic.scope",
           "lock": {"scope": "/synthetic.scope"}}
    policy = {"training_enabled": False, "native_training_enabled": False,
              "native_launch_authorized": False, "full_val_allowed": False,
              audit.STAGE: {
                  "decision": "ADMITTED", "plan": ref,
                  "execution_authorized": True, "data_reads_authorized": True,
                  "model_forwards_authorized": 1,
                  "backward_calls_authorized": 1,
                  "optimizer_steps_authorized": 0,
                  "training_authorized": False, "test_authorized": False,
                  "claimed_invocations": 0, "completed_invocations": 0,
                  "max_invocations": 1, "output_root": str(output),
                  "relaunch_allowed": False,
                  "checkpoint_reads_authorized": False,
                  "broker_access_allowed": False,
                  "test_data_used": False,
                  "memory_max": "12G", "swap_max": "512M",
                  "wall_seconds": audit.WALL_SECONDS,
                  "test_data_used": False,
              }}
    audit._qualify_plan(plan, ref, policy, head=head, out_dir=output,
                        device="cuda", cap=cap)
    broken = copy.deepcopy(policy)
    broken[audit.STAGE]["claimed_invocations"] = 1
    with pytest.raises(RuntimeError, match="POLICY"):
        audit._qualify_plan(plan, ref, broken, head=head, out_dir=output,
                            device="cuda", cap=cap)
    broken = copy.deepcopy(policy)
    broken[audit.STAGE]["test_data_used"] = True
    with pytest.raises(RuntimeError, match="POLICY"):
        audit._qualify_plan(plan, ref, broken, head=head, out_dir=output,
                            device="cuda", cap=cap)
    broken_cap = dict(cap, memory_max=8 * 1024**3)
    with pytest.raises(RuntimeError, match="ACTUAL_CAPS"):
        audit._qualify_plan(plan, ref, policy, head=head, out_dir=output,
                            device="cuda", cap=broken_cap)
    broken_cap = dict(cap, lock={"scope": "/other.scope"})
    with pytest.raises(RuntimeError, match="LOCK_CGROUP_SCOPE"):
        audit._qualify_plan(plan, ref, policy, head=head, out_dir=output,
                            device="cuda", cap=broken_cap)
    broken = dict(policy, native_training_enabled=True)
    with pytest.raises(RuntimeError, match="POLICY"):
        audit._qualify_plan(plan, ref, broken, head=head, out_dir=output,
                            device="cuda", cap=cap)
    assert not output.exists()


def test_read_only_preflight_does_not_claim_or_open_payload(tmp_path, monkeypatch):
    plan, output = _plan(tmp_path, monkeypatch)
    ref = {"path": "/synthetic/plan.json", "sha256": "a" * 64}
    monkeypatch.setattr(zero, "_clean_head", lambda: plan["source_base_head"])
    monkeypatch.setattr(audit, "_checked_plan_ref", lambda *_: (plan, ref))
    monkeypatch.setattr(audit, "_inspect_priors", lambda: None)
    monkeypatch.setattr(zero, "_bind_view_metadata", lambda _: (
        {"contract_sha256": "v" * 64,
         "research_objective": {"contract_sha256": "o" * 64}},
        None, None, None, None))
    result = audit.preflight_read_only(Path("/synthetic/plan.json"), "a" * 64, output)
    assert result["status"] == "READY_NO_CLAIM_NO_PAYLOAD"
    assert result["claim_created"] is False
    assert result["model_forwards"] == result["backward_calls"] == 0
    assert not output.exists()


class _TinyFiveEndpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx_cat_embeddings = nn.Embedding(3, 3)
        self.backbone = nn.Module()
        self.backbone.entry_q_joint_in = nn.Linear(3, 3)
        self.endpoint_head = nn.Linear(3, 3)

    def forward(self, **features):
        assert set(features) == {"seq_x", "seq_m5", "ctx_cat"}
        hidden = (features["seq_x"][:, 0, :] + features["seq_m5"][:, 0, :]
                  + self.ctx_cat_embeddings(features["ctx_cat"][:, 0]))
        return self.endpoint_head(
            self.backbone.entry_q_joint_in(hidden))[:, None, :].expand(-1, 5, -1)


def test_real_loss_owner_backward_skips_immature_row_and_preserves_state():
    torch.manual_seed(3)
    selected = np.asarray(audit.SAMPLE_ROWS, dtype=np.int64)
    issues = np.asarray(audit.ISSUE_NS, dtype=np.int64)
    allowed = np.full((2, 5), 2, dtype=np.uint8)
    bound = BoundFiveEndpointTargets(
        parent_rows=selected, decision_ns=issues, allowed_set=allowed,
        m1_source_row=np.zeros((2, 5), dtype=np.int64),
        m5_source_row=np.asarray([0, 1], dtype=np.int64),
        unjoined_target_rows=0)
    model = _TinyFiveEndpoint()
    features = {
        "seq_x": torch.randn(2, 1, 3, requires_grad=True),
        "seq_m5": torch.randn(2, 1, 3, requires_grad=True),
        "ctx_cat": torch.tensor([[0], [1]], dtype=torch.int64),
    }
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    initial = canonical_model_state_sha256(model.state_dict())
    logits = model(**features)
    result = audit._evaluate_and_backward(model, logits, features, selected,
                                           issues, bound, initial)
    assert result["active_count"] == 5
    assert result["skipped_count"] == 0 and result["skipped_update"] is True
    assert result["head_weight_gradient"]["nonzero"] is True
    assert result["backbone_entry_q_joint_in_weight_gradient"]["nonzero"] is True
    assert result["input_gradients"]["seq_x"]["informative_row"]["nonzero"] is True
    assert result["input_gradients"]["seq_m5"]["informative_row"]["nonzero"] is True
    assert result["input_gradients"]["seq_x"]["skipped_row"]["nonzero"] is False
    assert result["categorical_embedding_gradients"][
        "ctx_cat_embeddings.weight"]["nonzero"] is True
    assert result["initial_model_state_sha256"] == result["after_model_state_sha256"]
    assert result["optimizer_steps"] == result["ema_updates"] == 0
    model.zero_grad(set_to_none=True)
    model.backbone.entry_q_joint_in.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="BACKBONE_BACKWARD"):
        audit._evaluate_and_backward(
            model, model(**features), features, selected, issues, bound,
            canonical_model_state_sha256(model.state_dict()))


def _publication_setup(tmp_path, monkeypatch):
    output = tmp_path / "one_use"
    plan_file = tmp_path / "plan.json"
    plan_file.write_text('{"synthetic":true}\n')
    ref = zero._ref(plan_file, hashlib.sha256(plan_file.read_bytes()).hexdigest())
    policy_ref = zero._ref(audit.POLICY, zero._sha(audit.POLICY))
    head = "c" * 40
    plan = {"max_json_bytes": 1024 * 1024}
    cap = {"cgroup_path": "/synthetic", "memory_max": audit.MEMORY_BYTES,
           "memory_high": audit.MEMORY_BYTES, "swap": audit.SWAP_BYTES,
           "pids": 64}
    monkeypatch.setattr(audit, "_preflight",
                        lambda *_: (plan, ref, policy_ref, head, cap))
    original_check_output = audit.subprocess.check_output

    def fake_git(args, **kwargs):
        if args == ["git", "rev-parse", "HEAD"]:
            return head
        if args == ["git", "status", "--porcelain"]:
            return ""
        return original_check_output(args, **kwargs)

    monkeypatch.setattr(audit.subprocess, "check_output", fake_git)
    return output, ref


def test_synthetic_success_has_claim_result_terminal_chain(tmp_path, monkeypatch):
    output, plan_ref = _publication_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "_one_gradient_witness",
                        lambda *_: {"model_forwards": 1, "backward_calls": 1,
                                    "optimizer_steps": 0, "test_data_used": False})
    report = audit.run(Path(plan_ref["path"]), plan_ref["sha256"], "cuda", output)
    assert {path.name for path in output.iterdir()} == {
        "CLAIM.json", "RESULT.json", "TERMINAL.json"}
    claim = json.loads((output / "CLAIM.json").read_text())
    result = json.loads((output / "RESULT.json").read_text())
    terminal = json.loads((output / "TERMINAL.json").read_text())
    assert result["claim"] == zero._ref(output / "CLAIM.json",
                                        hashlib.sha256((output / "CLAIM.json").read_bytes()).hexdigest())
    assert terminal["result"] == report["result"]
    assert report["terminal"] == zero._ref(
        output / "TERMINAL.json",
        hashlib.sha256((output / "TERMINAL.json").read_bytes()).hexdigest())
    assert terminal["no_relaunch_under_this_plan"] is True
    assert terminal["optimizer_steps"] == 0
    assert claim["selected_parent_rows"] == audit.SAMPLE_ROWS


def test_synthetic_failure_is_terminal_and_cannot_relaunch(tmp_path, monkeypatch):
    output, plan_ref = _publication_setup(tmp_path, monkeypatch)

    def fail(*_):
        raise RuntimeError("synthetic backward failure")

    monkeypatch.setattr(audit, "_one_gradient_witness", fail)
    with pytest.raises(RuntimeError, match="synthetic backward failure"):
        audit.run(Path(plan_ref["path"]), plan_ref["sha256"], "cuda", output)
    assert {path.name for path in output.iterdir()} == {
        "CLAIM.json", "FAILED.json", "TERMINAL.json"}
    terminal = json.loads((output / "TERMINAL.json").read_text())
    assert terminal["status"] == "FAILED"
    assert terminal["no_relaunch_under_this_plan"] is True
    assert terminal["optimizer_steps"] == 0
    with pytest.raises(FileExistsError):
        audit.run(Path(plan_ref["path"]), plan_ref["sha256"], "cuda", output)


def test_synthetic_timeout_publishes_failed_terminal(tmp_path, monkeypatch):
    output, plan_ref = _publication_setup(tmp_path, monkeypatch)
    def expire_on_arm(which, seconds):
        assert which == signal.ITIMER_REAL
        if seconds > 0:
            signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
    monkeypatch.setattr(audit.signal, "setitimer", expire_on_arm)
    monkeypatch.setattr(audit, "_one_gradient_witness",
                        lambda *_: pytest.fail("work ran after timeout"))
    with pytest.raises(TimeoutError, match="WORKER_WALL_LIMIT"):
        audit.run(Path(plan_ref["path"]), plan_ref["sha256"], "cuda", output)
    terminal = json.loads((output / "TERMINAL.json").read_text())
    failure = json.loads((output / "FAILED.json").read_text())
    assert terminal["status"] == "FAILED"
    assert terminal["no_relaunch_under_this_plan"] is True
    assert failure["error_type"] == "TimeoutError"


def test_dispatcher_routes_exact_run_and_read_only_preflight(monkeypatch, capsys):
    from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as cli
    calls = []

    def fake_run(**kwargs):
        calls.append(("run", kwargs))
        return {"route": "run"}

    def fake_preflight(**kwargs):
        calls.append(("preflight", kwargs))
        return {"route": "preflight"}

    monkeypatch.setattr(audit, "run", fake_run)
    monkeypatch.setattr(audit, "preflight_read_only", fake_preflight)
    args = [audit.EXECUTION["flag"], "/synthetic/PLAN.json",
            audit.EXECUTION["flag"] + "-sha256", "a" * 64,
            "--device", "cuda", "--out-dir", "/synthetic/OUT"]
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out) == {"route": "run"}
    assert calls[-1] == ("run", {
        "plan_path": Path("/synthetic/PLAN.json"),
        "plan_sha256": "a" * 64, "device": "cuda",
        "out_dir": Path("/synthetic/OUT")})
    assert cli.main(args + ["--preflight"]) == 0
    assert json.loads(capsys.readouterr().out) == {"route": "preflight"}
    assert calls[-1] == ("preflight", {
        "plan_path": Path("/synthetic/PLAN.json"),
        "plan_sha256": "a" * 64, "out_dir": Path("/synthetic/OUT")})


def test_dispatcher_rejects_duplicate_and_extra_flags_before_run(monkeypatch):
    from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as cli
    monkeypatch.setattr(audit, "run", lambda **_: pytest.fail("unexpected run"))
    monkeypatch.setattr(audit, "preflight_read_only",
                        lambda **_: pytest.fail("unexpected preflight"))
    args = [audit.EXECUTION["flag"], "/synthetic/PLAN.json",
            audit.EXECUTION["flag"] + "-sha256", "a" * 64,
            "--device", "cuda", "--out-dir", "/synthetic/OUT"]
    for suffix in (["--device", "cuda"], ["--unknown"],
                   ["--preflight", "--preflight"]):
        with pytest.raises(SystemExit) as error:
            cli.main(args + suffix)
        assert error.value.code == 2
