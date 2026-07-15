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


def _path_calibration_future_contract(wired: bool) -> dict:
    if not wired:
        return {}
    return {
        "requires_path_calibration_recipe_contract": True,
        "path_calibration_recipe_contract": dict(gate.PATH_CALIBRATION_RECIPE_CONTRACT),
        "path_calibration_env_template": dict(gate.PATH_CALIBRATION_ENV_TEMPLATE),
        "requires_direction_balance_recipe_contract": True,
        "direction_balance_recipe_contract": dict(gate.DIRECTION_BALANCE_RECIPE_CONTRACT),
        "direction_balance_env_template": dict(gate.DIRECTION_BALANCE_ENV_TEMPLATE),
        "requires_tail_direction_recipe_contract": True,
        "tail_direction_recipe_contract": dict(gate.TAIL_DIRECTION_RECIPE_CONTRACT),
        "tail_direction_env_template": dict(gate.TAIL_DIRECTION_ENV_TEMPLATE),
        "requires_direction_context_slice_contract": True,
        "direction_context_slice_contract": dict(gate.DIRECTION_CONTEXT_SLICE_CONTRACT),
        "inner_train_argv_template": [
            "env",
            *[f"{key}={value}" for key, value in gate.PATH_CALIBRATION_ENV_TEMPLATE.items()],
            *[f"{key}={value}" for key, value in gate.DIRECTION_BALANCE_ENV_TEMPLATE.items()],
            *[f"{key}={value}" for key, value in gate.TAIL_DIRECTION_ENV_TEMPLATE.items()],
            ".venv/bin/python",
            "--enable-xau-direction-repair-heads",
            "--anchor-gate-init",
            "0.0",
        ],
    }


def _path_calibration_wrapper_text(kind: str) -> str:
    prefix = "ENTRY_FOUNDATION_SMOKE_" if kind == "smoke" else "ENTRY_FOUNDATION_CANDIDATE_"
    upstream = "\n".join(key.replace("ENTRY_", prefix) for key in gate.PATH_CALIBRATION_ENV_KEYS)
    downstream = "\n".join(gate.PATH_CALIBRATION_ENV_KEYS)
    return f"{upstream}\n{downstream}\n"


def _direction_balance_wrapper_text(kind: str) -> str:
    prefix = "ENTRY_FOUNDATION_SMOKE_" if kind == "smoke" else "ENTRY_FOUNDATION_CANDIDATE_"
    upstream = "\n".join(
        key.replace("ENTRY_", prefix).replace("GX1_V10_", prefix)
        for key in gate.DIRECTION_BALANCE_ENV_KEYS
    )
    downstream = "\n".join(gate.DIRECTION_BALANCE_ENV_KEYS)
    smart_recipe = "\n".join(
        [
            "PRED_BALANCE_ALPHA=0.50",
            "PRED_BALANCE_CLASS_WEIGHTS=1.0,1.0,4.0",
            "DIRECTION_CE_SCALE=4.00",
            "CKPT_CLASS_BALANCE_GUARD_WEIGHT=0.50",
            "CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL=0.35",
            "CKPT_CLASS_BALANCE_MIN_PRED_RATE=0.05",
            "CKPT_DIRECTION_SLICE_GUARD=1",
            "DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT=12.00",
            "DIRECTION_MIN_PRED_RATE_FRACTION=0.50",
            "DIRECTION_MIN_PRED_RATE_FLOOR=0.05",
            "DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE=0.15",
            "DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT=8.00",
            "DIRECTION_SLICE_MIN_PRED_RATE_FRACTION=0.50",
            "DIRECTION_SLICE_MIN_PRED_RATE_FLOOR=0.05",
            "DIRECTION_SLICE_RECALL_LOSS_WEIGHT=4.00",
            "DIRECTION_SLICE_BALANCED_CE_WEIGHT=2.00",
            "DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE=0.10",
            "DIRECTION_SLICE_BALANCED_CE_MIN_ROWS=8",
            "DIRECTION_SLICE_TRUE_MARGIN_WEIGHT=2.00",
            "DIRECTION_SLICE_TRUE_MARGIN=0.10",
            "DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE=0.10",
            "DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS=8",
            "DIRECTION_SLICE_LOSS_AGGREGATION=mean_max",
            "DIRECTION_VS_FLAT_MARGIN_WEIGHT=4.00",
            "DIRECTION_VS_FLAT_MARGIN=0.10",
            "HIER_LEGACY_CE_MULT=1.00",
            "HIER_SIDE_VALIDITY_WEIGHT=1.50",
            "HIER_POCKET_ABSTAIN_WEIGHT=5.00",
            "HIER_POCKET_SIDE_MARGIN_WEIGHT=3.00",
            "TRENDLINE_RAIL_AUX_WEIGHT=1.00",
            "TRENDLINE_RAIL_WRONG_SIDE_WEIGHT=1.50",
            "TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT=5.00",
            "ANCHOR_GATE_INIT=0.0",
            "--enable-xau-direction-repair-heads",
            "--anchor-gate-init",
        ]
    )
    return f"{upstream}\n{downstream}\n{smart_recipe}\n"


def _tail_direction_wrapper_text(kind: str) -> str:
    prefix = "ENTRY_FOUNDATION_SMOKE_" if kind == "smoke" else "ENTRY_FOUNDATION_CANDIDATE_"
    upstream = "\n".join(key.replace("ENTRY_", prefix) for key in gate.TAIL_DIRECTION_ENV_KEYS)
    downstream = "\n".join(gate.TAIL_DIRECTION_ENV_KEYS)
    return f"{upstream}\n{downstream}\n"


def _args(tmp_path: Path, *, wired: bool, ctx_tag: str = "CTX6CAT5") -> argparse.Namespace:
    post_rebuild = tmp_path / "post_rebuild.json"
    smoke_readiness = tmp_path / "smoke_readiness.json"
    _write_json(
        post_rebuild,
        {
            "decision": (
                "ENTRY_SMART_DATASET_READY_FOR_TRAIN_READINESS_REVIEW"
                if wired
                else "BLOCKED_BY_ENTRY_SMART_DATASET_POST_REBUILD_AUDIT"
            ),
            "split_manifests": {
                split: {
                    "ctx_contract": {
                        "tag": ctx_tag,
                        "ctx_cont_dim": 142,
                        "ctx_cat_dim": 5,
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
                "READY_FOR_SMART_SEQ520_SMOKE_MANIFEST_REVIEW"
                if wired
                else "BLOCKED_SMART_SEQ520_SMOKE_READINESS"
            ),
            "future_command_contracts": {
                "smart_smoke_train": {
                    "implemented_in_control_surface": wired,
                    "specialist_contract_mode": "smart_seq520_candidate",
                    **_path_calibration_future_contract(wired),
                }
            },
        },
    )
    if wired:
        control_text = "Usage: smart-smoke-train --vedtak <id>\ncase\nsmart-smoke-train) exec wrapper ;;\n"
        smoke_wrapper_text = (
            "--smart-seq520 SPECIALIST_CONTRACT_MODE=smart_seq520_candidate\n"
            + _path_calibration_wrapper_text("smoke")
            + _direction_balance_wrapper_text("smoke")
            + _tail_direction_wrapper_text("smoke")
        )
        candidate_wrapper_text = (
            "--smart-seq520 SPECIALIST_CONTRACT_MODE=smart_seq520_candidate\n"
            + _path_calibration_wrapper_text("candidate")
            + _direction_balance_wrapper_text("candidate")
            + _tail_direction_wrapper_text("candidate")
        )
        smart_script_text = "smart_seq520_candidate 520\n"
    else:
        control_text = "smart-smoke-readiness)\n"
        smoke_wrapper_text = "--challenger-seq215 SPECIALIST_CONTRACT_MODE=challenger_seq215\n"
        candidate_wrapper_text = "--challenger-seq215 SPECIALIST_CONTRACT_MODE=challenger_seq215\n"
        smart_script_text = "challenger_seq215 215\n"
    trainer_text = (
        "--specialist-contract-mode\n"
        + "\n".join(gate.PATH_CALIBRATION_ENV_KEYS)
        + "\n"
        + "\n".join(gate.DIRECTION_BALANCE_ENV_KEYS)
        + "\n"
        + "\n".join(gate.TAIL_DIRECTION_ENV_KEYS)
        + "\n"
    )
    return argparse.Namespace(
        smart_post_rebuild_readiness_json=str(post_rebuild),
        smart_smoke_readiness_json=str(smoke_readiness),
        control_script=str(_write(tmp_path / "entry_next_edge_control.sh", control_text)),
        trainer_source=str(_write(tmp_path / "entry_v10_ctx_train_v3.py", trainer_text)),
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
    assert "smart smoke future contract declares direction context slice audit" in report["blockers"]


def test_smart_trainability_can_pass_when_all_surfaces_are_wired(monkeypatch, tmp_path: Path) -> None:
    report = gate.run(_args(tmp_path, wired=True))

    assert report["decision"] == gate.READY_DECISION
    assert report["expected_signal_dim"] == 520
    assert report["source_metadata_contract"]["declared_ctx_contracts_match_expected"] is True
    assert report["source_metadata_contract"]["no_stale_ctx6cat6"] is True
    assert report["training_allowed"] is False
    assert report["execution_allowed_now"] is False
    assert report["failures"] == []


def test_smart_trainability_blocks_stale_ctx6cat6_source_metadata(tmp_path: Path) -> None:
    report = gate.run(_args(tmp_path, wired=True, ctx_tag="CTX6CAT6"))

    assert report["decision"] == gate.BLOCKED_DECISION
    assert "smart source metadata has no stale CTX6CAT6 ctx contract" in report["blockers"]
    assert "declared smart source ctx metadata matches CTX6CAT5" in report["blockers"]
    assert report["source_metadata_contract"]["stale_ctx6cat6_paths"]
    assert report["training_allowed"] is False
