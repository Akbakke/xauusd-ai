"""Synthetic fold clocks and one-use publication for TRAIN prior audit."""
import hashlib
import json

import numpy as np
import pytest

from gx1.audit import measure_market_state_five_endpoint_prior_v1 as audit
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as target


def _bound(*, score_bits=None):
    lower, upper = target.FOLD_BOUNDARIES_NS[:2]
    parent = np.arange(4, dtype=np.int64)
    issues = np.array([target.TRAIN_START_NS + target.M5_NS,
                       lower - target.M5_NS,
                       lower + target.M5_NS,
                       upper - target.M5_NS], dtype=np.int64)
    allowed = np.array([[5, 1, 2, 4, 7], [4, 4, 4, 4, 4],
                        [1, 2, 4, 5, 7], [2, 1, 5, 4, 5]], dtype=np.uint8)
    if score_bits is not None:
        allowed[2:] = np.asarray(score_bits, dtype=np.uint8)
    return target.BoundFiveEndpointTargets(
        parent_rows=parent, decision_ns=issues, allowed_set=allowed,
        m1_source_row=np.zeros((4, 5), dtype=np.int64),
        m5_source_row=np.arange(4, dtype=np.int64), unjoined_target_rows=0)


def test_score_labels_cannot_change_earlier_fit_prior():
    before = audit._fold_report(_bound(), 1)
    changed = audit._fold_report(_bound(score_bits=[[2]*5, [2]*5]), 1)
    assert before["fit_rows"] == changed["fit_rows"] == 2
    assert before["score_rows"] == changed["score_rows"] == 2
    assert before["prior_probabilities"] == changed["prior_probabilities"]
    assert before["fit_counts"] == changed["fit_counts"]
    assert before["score_sums"] != changed["score_sums"]


def test_strict_maturity_and_separate_score_denominators():
    report = audit._fold_report(_bound(), 1)
    lower, upper = target.FOLD_BOUNDARIES_NS[:2]
    assert report["fit_last_issue_ns"] + target.M5_NS == lower
    assert report["score_last_issue_ns"] + target.M5_NS == upper
    # The second fit and score rows are too close to their right fold boundary.
    assert report["fit_counts"]["M5"] == {
        "activity": 1, "active": 1, "certified_side": 0, "up": 0}
    expected = {"M5": (1, 1, 1), "M15": (1, 0, 1),
                "H1": (1, 1, 1), "H4": (1, 0, 0), "D1": (0, 0, 0)}
    for tf, counts in expected.items():
        actual = report["score_sums"][tf]
        assert tuple(actual[name]["count"] for name in
                     ("activity", "conditional_side", "three_class")) == counts
        assert report["analytic_activity"][tf]["count"] == counts[0]
        assert report["score_coverage"][tf]["eligible_mature"] == (0 if tf == "D1" else 1)
    assert report["analytic_activity"]["M15"]["always_active_brier_sum"] == 1
    assert report["analytic_activity"]["H4"]["always_flat_brier_sum"] == 1


def _synthetic_run_setup(tmp_path, monkeypatch):
    plan_path = tmp_path / "PLAN.json"
    plan_path.write_text('{"synthetic":true}\n')
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    plan_ref = {"path": str(plan_path), "sha256": plan_sha}
    head = "c" * 40
    policy_ref = {"path": str(audit.POLICY), "sha256": "p" * 64}
    monkeypatch.setattr(audit, "preflight_read_only",
                        lambda *_args, **_kwargs: ({"fold_ids": [1, 2, 3, 4, 5]},
                                                  plan_ref, head, policy_ref))
    monkeypatch.setattr(audit, "_clean_head", lambda: head)
    original_sha = audit._sha
    monkeypatch.setattr(audit, "_sha",
                        lambda path: "p" * 64 if path == audit.POLICY
                        else original_sha(path))
    return plan_path, plan_sha, tmp_path / "RUN"


def test_synthetic_success_publishes_exact_chain_once(tmp_path, monkeypatch):
    plan_path, plan_sha, output = _synthetic_run_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "_measure",
                        lambda: {"synthetic_only": True, "model_forwards": 0,
                                 "optimizer_steps": 0, "test_data_used": False})
    published = audit.run(plan_path, plan_sha, output)
    assert {p.name for p in output.iterdir()} == {"CLAIM.json", "RESULT.json", "TERMINAL.json"}
    claim, result, terminal = (json.loads((output / name).read_text())
                               for name in ("CLAIM.json", "RESULT.json", "TERMINAL.json"))
    claim_ref = audit._ref(output / "CLAIM.json")
    result_ref = audit._ref(output / "RESULT.json")
    assert claim["plan"] == result["plan"] == terminal["plan"]
    assert result["claim"] == terminal["claim"] == claim_ref
    assert terminal["result"] == result_ref
    assert terminal["exit_code"] == 0 and terminal["test_data_used"] is False
    assert published["result"] == result_ref
    with pytest.raises(FileExistsError):
        audit.run(plan_path, plan_sha, output)


def test_synthetic_failure_publishes_terminal_without_relaunch(tmp_path, monkeypatch):
    plan_path, plan_sha, output = _synthetic_run_setup(tmp_path, monkeypatch)
    def fail():
        raise RuntimeError("synthetic measurement failure")
    monkeypatch.setattr(audit, "_measure", fail)
    with pytest.raises(RuntimeError, match="synthetic measurement failure"):
        audit.run(plan_path, plan_sha, output)
    assert {p.name for p in output.iterdir()} == {"CLAIM.json", "FAILED.json", "TERMINAL.json"}
    failure = json.loads((output / "FAILED.json").read_text())
    terminal = json.loads((output / "TERMINAL.json").read_text())
    assert terminal["status"] == "FAILED" and terminal["exit_code"] == 1
    assert failure["claim"] == terminal["claim"] == audit._ref(output / "CLAIM.json")
    assert terminal["output_inventory"] == ["CLAIM.json", "FAILED.json", "TERMINAL.json"]
    assert terminal["failure"] == audit._ref(output / "FAILED.json")
    assert terminal["no_relaunch_under_this_plan"] is True
    assert terminal["model_forwards"] == terminal["optimizer_steps"] == 0
    assert terminal["test_data_used"] is False


def test_directory_fsync_failure_still_closes_one_use_root(tmp_path, monkeypatch):
    plan_path, plan_sha, output = _synthetic_run_setup(tmp_path, monkeypatch)
    def fail_fsync(_path):
        raise RuntimeError("synthetic directory fsync failure")
    monkeypatch.setattr(audit, "_fsync_directory", fail_fsync)
    with pytest.raises(RuntimeError, match="synthetic directory fsync failure"):
        audit.run(plan_path, plan_sha, output)
    assert {p.name for p in output.iterdir()} == {"FAILED.json", "TERMINAL.json"}
    failure = json.loads((output / "FAILED.json").read_text())
    terminal = json.loads((output / "TERMINAL.json").read_text())
    assert failure["claim"] is None and terminal["claim"] is None
    assert terminal["output_inventory"] == ["FAILED.json", "TERMINAL.json"]
    assert terminal["failure"] == audit._ref(output / "FAILED.json")
    assert terminal["status"] == "FAILED" and terminal["exit_code"] == 1
    assert terminal["no_relaunch_under_this_plan"] is True
    assert terminal["model_forwards"] == terminal["optimizer_steps"] == 0
    assert terminal["test_data_used"] is False
    with pytest.raises(FileExistsError):
        audit.run(plan_path, plan_sha, output)
