import argparse
import json
from pathlib import Path

from gx1.scripts import verify_entry_smart_seq520_trainability_readiness_v1 as gate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _args(tmp_path: Path, *, wired: bool) -> argparse.Namespace:
    post_rebuild = tmp_path / "post_rebuild.json"
    smoke_readiness = tmp_path / "smoke_readiness.json"
    _write_json(
        post_rebuild,
        {
            "decision": (
                "ENTRY_SMART_DATASET_READY_FOR_TRAIN_READINESS_REVIEW"
                if wired
                else "BLOCKED_BY_ENTRY_SMART_DATASET_POST_REBUILD_AUDIT"
            )
        },
    )
    _write_json(
        smoke_readiness,
        {
            "decision": (
                "READY_FOR_SMART_SEQ520_SMOKE_MANIFEST_REVIEW"
                if wired
                else "BLOCKED_SMART_SEQ520_SMOKE_READINESS"
            ),
            "future_command_contracts": {
                "smart_smoke_train": {
                    "implemented_in_control_surface": wired,
                    "specialist_contract_mode": "smart_seq520_candidate",
                }
            },
        },
    )
    if wired:
        control_text = "Usage: smart-smoke-train --vedtak <id>\ncase\nsmart-smoke-train) exec wrapper ;;\n"
        smoke_wrapper_text = "--smart-seq520 SPECIALIST_CONTRACT_MODE=smart_seq520_candidate\n"
        candidate_wrapper_text = "--smart-seq520 SPECIALIST_CONTRACT_MODE=smart_seq520_candidate\n"
        smart_script_text = "smart_seq520_candidate 520\n"
    else:
        control_text = "smart-smoke-readiness)\n"
        smoke_wrapper_text = "--challenger-seq215 SPECIALIST_CONTRACT_MODE=challenger_seq215\n"
        candidate_wrapper_text = "--challenger-seq215 SPECIALIST_CONTRACT_MODE=challenger_seq215\n"
        smart_script_text = "challenger_seq215 215\n"
    return argparse.Namespace(
        smart_post_rebuild_readiness_json=str(post_rebuild),
        smart_smoke_readiness_json=str(smoke_readiness),
        control_script=str(_write(tmp_path / "entry_next_edge_control.sh", control_text)),
        trainer_source=str(_write(tmp_path / "entry_v10_ctx_train_v3.py", "--specialist-contract-mode\n")),
        smoke_wrapper=str(_write(tmp_path / "run_smoke.sh", smoke_wrapper_text)),
        candidate_wrapper=str(_write(tmp_path / "run_candidate.sh", candidate_wrapper_text)),
        candidate_readiness_script=str(_write(tmp_path / "candidate_readiness.py", smart_script_text)),
        selective_edge_script=str(_write(tmp_path / "selective_edge.py", smart_script_text)),
        replay_evidence_script=str(_write(tmp_path / "replay_evidence.py", smart_script_text)),
        replay_readiness_script=str(_write(tmp_path / "replay_readiness.py", smart_script_text)),
        out_dir=str(tmp_path / "reports"),
        fail_on_not_ready=False,
        quiet=True,
    )


def test_smart_trainability_blocks_until_train_surface_exists(tmp_path: Path) -> None:
    report = gate.run(_args(tmp_path, wired=False))

    assert report["decision"] == gate.BLOCKED_DECISION
    assert report["training_allowed"] is False
    assert report["candidate_training_allowed"] is False
    assert report["replay_allowed"] is False
    assert report["iql_allowed"] is False
    assert not any(report["side_effects_started"].values())
    assert "smart smoke wrapper exposes --smart-seq520 lane" in report["blockers"]
    assert "smart smoke train is wired in control surface" in report["blockers"]


def test_smart_trainability_can_pass_when_all_surfaces_are_wired(monkeypatch, tmp_path: Path) -> None:
    report = gate.run(_args(tmp_path, wired=True))

    assert report["decision"] == gate.READY_DECISION
    assert report["expected_signal_dim"] == 520
    assert report["training_allowed"] is False
    assert report["execution_allowed_now"] is False
    assert report["failures"] == []
