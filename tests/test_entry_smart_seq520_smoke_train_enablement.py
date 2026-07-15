import argparse
import json
from pathlib import Path

from gx1.scripts import materialize_entry_smart_seq520_smoke_train_enablement_package_v1 as gate


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _future_train_contract() -> dict:
    return {
        "starts_trainer": True,
        "starts_replay": False,
        "starts_iql_distillation": False,
        "touches_shadow_or_live": False,
        "requires_ram_cap": True,
        "requires_edge_audit": True,
        "direction_balance_env_template": dict(gate.REQUIRED_DIRECTION_ENV),
    }


def _write_inputs(tmp_path: Path, *, closed: bool = True) -> dict[str, Path]:
    root = tmp_path / "reports"
    return {
        "smoke": _write_json(
            root / "smoke_readiness.json",
            {
                "decision": gate.READY_SMOKE_DECISION,
                "training_allowed": False,
                "execution_allowed_now": False,
                "future_command_contracts": {"smart_smoke_train": _future_train_contract()},
            },
        ),
        "trainability": _write_json(
            root / "trainability_readiness.json",
            {
                "decision": gate.READY_TRAINABILITY_DECISION,
                "training_allowed": False,
                "execution_allowed_now": False,
                "candidate_training_allowed": False,
                "replay_allowed": False,
                "iql_allowed": False if closed else True,
                "shadow_live_promotion_allowed": False,
                "future_train_contract": _future_train_contract(),
            },
        ),
    }


def _args(
    tmp_path: Path,
    paths: dict[str, Path],
    *,
    vedtak: str = "SMART_SEQ520_XAU_SMOKE_TEST_V1",
) -> argparse.Namespace:
    return argparse.Namespace(
        vedtak=vedtak,
        smoke_readiness_json=str(paths["smoke"]),
        trainability_readiness_json=str(paths["trainability"]),
        out_dir=str(tmp_path / "out"),
        device="cpu",
        epochs=1,
        batch_size=8,
        mem_cap="22G",
        swap_cap="2G",
        fail_on_not_ready=False,
        quiet=True,
    )


def _dry_run_stub(**kwargs) -> dict:
    return {
        "argv": ["scripts/entry_next_edge_control.sh", "smart-smoke-train", "--dry-run"],
        "returncode": 0,
        "stdout_tail": "",
        "stderr_tail": "",
        "capped_smoke_train_command": (
            "Capped smoke train command: scripts/gx1_capped_run.sh --mem 22G --swap 2G -- "
            "env ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT=8.00 "
            "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE=0.02 "
            "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE=0.10 "
            "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT=3.00 "
            "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE=0.02 "
            "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE=3 "
            "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS=6 "
            "ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT=4.00 "
            "ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS=15.0 "
            "ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN=0.10 "
            "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT=6.00 "
            "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS=15.0 "
            "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN=0.10 "
            "ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT=8.00 "
            "ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE=0.10 "
            "ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS=8 "
            "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION=0.50 "
            "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR=0.10 "
            "ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN=0.10 "
            "python -m gx1.models.entry_v10.entry_v10_ctx_train_v3 --num-workers 0 "
            "--enable-xau-direction-repair-heads"
        ),
        "smoke_train_command": "",
        "post_smoke_audit_command": "Post-smoke audit command: audit --require-edge --edge-test-scope strict",
        "has_capped_run": True,
        "has_mem_cap": True,
        "has_swap_cap": True,
        "has_num_workers_zero": True,
        "has_global_prior_match": True,
        "has_prior_match": True,
        "has_hard_red_stop": True,
        "has_utility_margin": True,
        "has_side_utility_conviction": True,
        "has_flat_starvation": True,
        "has_xau_repair_heads": True,
        "has_strict_edge_audit": True,
        "trainer_started": False,
    }


def test_smart_smoke_train_enablement_passes_with_clean_package(monkeypatch, tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    monkeypatch.setattr(gate, "_git_status_short", lambda: [])
    monkeypatch.setattr(gate, "_dry_run_wrapper", _dry_run_stub)

    report = gate.run(_args(tmp_path, paths))

    assert report["decision"] == gate.READY_DECISION
    assert report["training_allowed"] is False
    assert report["smart_smoke_training_allowed_with_this_package"] is True
    assert report["candidate_training_allowed"] is False
    assert report["replay_allowed"] is False
    assert report["iql_allowed"] is False
    assert report["trainer_started"] is False
    assert report["promotion_shadow_live_allowed"] is False
    assert report["wrapper_dry_run"]["has_capped_run"] is True


def test_smart_smoke_train_enablement_blocks_missing_vedtak(monkeypatch, tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    monkeypatch.setattr(gate, "_git_status_short", lambda: [])
    monkeypatch.setattr(gate, "_dry_run_wrapper", _dry_run_stub)

    report = gate.run(_args(tmp_path, paths, vedtak=""))

    assert report["decision"] == gate.BLOCKED_DECISION
    failed = {row["check"] for row in report["failures"]}
    assert "explicit smart XAU smoke train vedtak is present" in failed


def test_smart_smoke_train_enablement_blocks_dirty_git(monkeypatch, tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    monkeypatch.setattr(gate, "_git_status_short", lambda: [" M gx1/models/entry_v10/entry_v10_ctx_train_v3.py"])
    monkeypatch.setattr(gate, "_dry_run_wrapper", _dry_run_stub)

    report = gate.run(_args(tmp_path, paths))

    assert report["decision"] == gate.BLOCKED_DECISION
    failed = {row["check"] for row in report["failures"]}
    assert "worktree is clean before smart smoke train enablement package" in failed


def test_smart_smoke_train_enablement_blocks_open_iql(monkeypatch, tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, closed=False)
    monkeypatch.setattr(gate, "_git_status_short", lambda: [])
    monkeypatch.setattr(gate, "_dry_run_wrapper", _dry_run_stub)

    report = gate.run(_args(tmp_path, paths))

    assert report["decision"] == gate.BLOCKED_DECISION
    failed = {row["check"] for row in report["failures"]}
    assert "upstream readiness remains report-only and keeps candidate/replay/IQL/shadow/live closed" in failed
