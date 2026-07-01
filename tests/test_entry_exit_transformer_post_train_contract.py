import argparse
import hashlib
import json
from pathlib import Path

from gx1.scripts.audit_entry_exit_transformer_post_train_contract_v1 import run


EXPECTED_HEADS = [
    "exit_now_logit",
    "hold_value_bps",
    "exit_now_reward_bps",
    "giveback_risk_bps",
    "mfe_capture_ratio",
]


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_inputs(
    tmp_path: Path,
    *,
    train_review_ready: bool = True,
    weak_policy_ready: bool = True,
    head_mismatch: bool = False,
) -> Path:
    root = tmp_path / "reports"
    heads = EXPECTED_HEADS[:-1] if head_mismatch else EXPECTED_HEADS
    feature_schema = _write_json(root / "feature_schema.json", {"state_feature_names": ["running_pnl_bps"]})
    training_plan = _write_json(
        root / "training_plan.json",
        {
            "decision": "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW",
            "training_plan": {
                "architecture": {"output_heads": heads},
                "dataset": {
                    "feature_schema_json": str(feature_schema),
                    "feature_schema_json_sha256": _sha256(feature_schema),
                    "normalization_json": "/tmp/normalization.json",
                },
                "resource_guardrails": {
                    "num_workers": 0,
                    "max_process_rss_gib": 8,
                    "abort_if_mem_available_below_gib": 8,
                },
            },
        },
    )
    pretrain = _write_json(
        root / "pretrain.json",
        {
            "decision": "ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_READY_FOR_TRAIN_EXECUTION_REVIEW",
            "preflight_manifest": {"output_heads": heads},
        },
    )
    slice_report = _write_json(
        root / "slice.json",
        {
            "decision": "ENTRY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS_READY_WITH_WEAK_SLICE_DISCLOSURE",
            "slice_review": {"weak_slice_count": 2, "unsupported_slice_count": 0},
        },
    )
    requirements = {
        "must_report_session_regime_side_metrics": True,
        "must_report_direction_and_tail_metrics": True,
        "must_not_promote_from_broad_average": True,
        "must_compare_weak_slices_separately": True,
        "must_block_shadow_live_on_unsupported_slice": True,
        "must_keep_weak_slice_disclosure_in_post_train_audit": True,
    }
    weak_policy = {
        "weak_slice_count": 2,
        "unsupported_slice_count": 0,
        "train_execution_requirements": requirements if weak_policy_ready else {},
    }
    train_review = _write_json(
        root / "train_review.json",
        {
            "decision": (
                "ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_READY_FOR_EXPLICIT_VEDTAK_PACKAGE"
                if train_review_ready
                else "BLOCKED_BY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW"
            ),
            "training_plan_json": str(training_plan),
            "training_plan_json_sha256": _sha256(training_plan),
            "pretrain_manifest_json": str(pretrain),
            "pretrain_manifest_json_sha256": _sha256(pretrain),
            "slice_robustness_json": str(slice_report),
            "slice_robustness_json_sha256": _sha256(slice_report),
            "review_contract": {"weak_slice_policy": weak_policy},
            "exit_training_allowed": False,
            "exit_training_allowed_with_explicit_vedtak": False,
        },
    )
    return train_review


def _args(tmp_path: Path, train_review: Path) -> argparse.Namespace:
    return argparse.Namespace(
        train_execution_review_json=str(train_review),
        out_dir=str(tmp_path / "out"),
        fail_on_not_ready=False,
        quiet=True,
    )


def test_entry_exit_transformer_post_train_contract_passes_but_keeps_downstream_closed(tmp_path: Path) -> None:
    train_review = _write_inputs(tmp_path)

    report = run(_args(tmp_path, train_review))

    assert report["decision"] == "ENTRY_EXIT_TRANSFORMER_POST_TRAIN_AUDIT_CONTRACT_READY"
    assert report["head_contract"]["training_plan_heads"] == EXPECTED_HEADS
    assert report["post_train_audit_contract"]["required_metric_slices"]["weak_slice_count_to_preserve"] == 2
    identity = report["post_train_audit_contract"]["required_bundle_identity"]
    assert identity["feature_schema_json"]
    assert len(identity["feature_schema_json_sha256"]) == 64
    assert report["exit_training_allowed"] is False
    assert report["exit_iql_allowed"] is False
    assert report["trainer_started"] is False


def test_entry_exit_transformer_post_train_contract_blocks_unready_train_review(tmp_path: Path) -> None:
    train_review = _write_inputs(tmp_path, train_review_ready=False)

    report = run(_args(tmp_path, train_review))

    assert report["decision"] == "BLOCKED_BY_EXIT_TRANSFORMER_POST_TRAIN_AUDIT_CONTRACT"
    failed = {row["check"] for row in report["failures"]}
    assert "active Exit Transformer train-execution review is ready" in failed


def test_entry_exit_transformer_post_train_contract_blocks_missing_weak_slice_policy(tmp_path: Path) -> None:
    train_review = _write_inputs(tmp_path, weak_policy_ready=False)

    report = run(_args(tmp_path, train_review))

    assert report["decision"] == "BLOCKED_BY_EXIT_TRANSFORMER_POST_TRAIN_AUDIT_CONTRACT"
    failed = {row["check"] for row in report["failures"]}
    assert "weak-slice policy is preserved for post-train audit" in failed


def test_entry_exit_transformer_post_train_contract_blocks_head_mismatch(tmp_path: Path) -> None:
    train_review = _write_inputs(tmp_path, head_mismatch=True)

    report = run(_args(tmp_path, train_review))

    assert report["decision"] == "BLOCKED_BY_EXIT_TRANSFORMER_POST_TRAIN_AUDIT_CONTRACT"
    failed = {row["check"] for row in report["failures"]}
    assert "exact Exit output heads are locked across plan and preflight" in failed
