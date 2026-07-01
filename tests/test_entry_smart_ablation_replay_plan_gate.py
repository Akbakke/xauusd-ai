import argparse
import json
from pathlib import Path

import pandas as pd

from gx1.scripts import materialize_entry_smart_ablation_replay_plan_gate_v1 as gate


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_smart_preflight(path: Path) -> None:
    _write_json(
        path,
        {
            "decision": "READY_FOR_SMART_REBUILD_VEDTAK_REVIEW",
            "report_only": True,
            "training_allowed": False,
            "counts": {
                "manifest_variant": "smart_seq520_candidate",
                "expected_seq_snap_width": 520,
                "smart_layer_features": 305,
            },
        },
    )


def _write_candidate_bundle_audit(path: Path, bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        path,
        {
            "decision": "PASS",
            "failures": [],
            "bundle_dir": str(bundle_dir),
            "manifest_variant": "smart_seq520_candidate",
            "bundle_summary": {
                "seq_input_dim": 520,
                "snap_input_dim": 520,
                "manifest_variant": "smart_seq520_candidate",
                "specialist_contract_mode": "challenger_seq215",
            },
        },
    )


def _write_replay_dir(
    path: Path,
    *,
    variant: str,
    bundle_dir: Path | None = None,
    candidate_bundle_audit_sha256: str = "",
) -> None:
    _write_csv(
        path / "replay_policy_metrics.csv",
        [
            {
                "scope": "aggregate",
                "policy_id": f"{variant}_policy",
                "n_trades": 12,
                "net_sum_bps": 180.0,
                "net_mean_bps": 15.0,
                "profit_factor": 1.4,
                "max_drawdown_bps": 35.0,
                "mean_mae_bps": 8.0,
                "mae_p95_bps": 16.0,
                "bad_path_rate": 0.08,
                "mean_bad_path_prob": 0.18,
                "mean_path_quality_pred": 0.72,
                "path_quality_p10": 0.38,
            }
        ],
    )
    _write_csv(
        path / "replay_policy_monthly.csv",
        [{"policy_id": f"{variant}_policy", "month": "2026-01", "net_sum_bps": 180.0}],
    )
    _write_csv(
        path / "replay_policy_trades.csv",
        [
            {
                "entry_time": "2026-01-05T08:00:00Z",
                "policy_id": f"{variant}_policy",
                "session": "EU",
                "trend_regime": "UP",
                "side": "LONG",
                "tail_bucket": "normal",
                "net_pnl_bps": 25.0,
                "mae_bps": 6.0,
                "bad_path_prob": 0.12,
                "path_quality_pred": 0.8,
            },
            {
                "entry_time": "2026-01-06T13:00:00Z",
                "policy_id": f"{variant}_policy",
                "session": "US",
                "trend_regime": "DOWN",
                "side": "SHORT",
                "tail_bucket": "tail_loss_watch",
                "net_pnl_bps": -8.0,
                "mae_bps": 18.0,
                "bad_path_prob": 0.32,
                "path_quality_pred": 0.45,
            },
        ],
    )
    artifact_hashes = {
        "replay_policy_metrics.csv": gate._sha256_file(path / "replay_policy_metrics.csv"),
        "replay_policy_monthly.csv": gate._sha256_file(path / "replay_policy_monthly.csv"),
        "replay_policy_trades.csv": gate._sha256_file(path / "replay_policy_trades.csv"),
    }
    _write_json(
        path / "REPLAY_EVIDENCE_MANIFEST.json",
        {
            "decision": "PASS",
            "failures": [],
            "manifest_variant": variant,
            "candidate_bundle_dir": str(bundle_dir or ""),
            "candidate_bundle_audit_sha256": candidate_bundle_audit_sha256,
            "artifact_hashes": artifact_hashes,
            "replay_identity_contract": {
                "ready": True,
                "manifest_variant": variant,
                "candidate_bundle_dir": str(bundle_dir or ""),
                "candidate_bundle_audit_sha256": candidate_bundle_audit_sha256,
                "artifact_hashes": artifact_hashes,
            },
        },
    )


def _args(tmp_path: Path, *, require_baselines: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        smart_preflight_json=str(tmp_path / "smart_preflight.json"),
        candidate_bundle_audit_json=str(tmp_path / "candidate_audit.json"),
        candidate_bundle_dir="",
        candidate_replay_dir=str(tmp_path / "smart_replay"),
        seq146_replay_dir=str(tmp_path / "seq146_replay"),
        seq215_replay_dir=str(tmp_path / "seq215_replay"),
        out_dir=str(tmp_path / "out"),
        require_baseline_replay_evidence=require_baselines,
        fail_on_not_ready=False,
        quiet=True,
    )


def test_smart_ablation_plan_gate_materializes_exact_report_only_matrix(tmp_path: Path) -> None:
    args = _args(tmp_path)
    bundle_dir = tmp_path / "smart_bundle"
    _write_smart_preflight(Path(args.smart_preflight_json))
    candidate_audit_path = Path(args.candidate_bundle_audit_json)
    _write_candidate_bundle_audit(candidate_audit_path, bundle_dir)
    _write_replay_dir(
        Path(args.candidate_replay_dir),
        variant="smart_seq520_candidate",
        bundle_dir=bundle_dir,
        candidate_bundle_audit_sha256=gate._sha256_file(candidate_audit_path) or "",
    )
    _write_replay_dir(Path(args.seq146_replay_dir), variant="foundation_seq146")
    _write_replay_dir(Path(args.seq215_replay_dir), variant="challenger_seq215")

    report = gate.run(args)

    assert report["decision"] == "READY_FOR_SMART_ABLATION_REPLAY_PLAN_REVIEW"
    assert report["report_only"] is True
    assert report["training_allowed"] is False
    assert report["replay_allowed_by_this_gate"] is False
    assert all(value is False for value in report["side_effects_started"].values())
    plan = report["required_ablation_plan"]
    names = [row["ablation_name"] for row in plan["required_ablations"]]
    assert names[:4] == ["with-old+smart", "smart-only", "old-only", "no-XGB"]
    assert plan["ablation_count"] == 14
    assert plan["drop_family_count"] == 10
    assert {row["dropped_smart_family"]["family_label"] for row in plan["required_ablations"] if row["ablation_type"] == "drop_smart_family"} == set(gate.SMART_LAYER_FEATURES)
    assert set(report["required_replay_evidence_contract"]["required_metric_families"]) == {
        "pnl",
        "drawdown",
        "mae",
        "bad_path",
        "path_quality",
    }
    assert set(report["required_replay_evidence_contract"]["required_slice_dimensions"]) == {
        "session",
        "regime",
        "direction",
        "tail",
    }
    assert Path(report["json_path"]).exists()
    assert Path(report["md_path"]).exists()


def test_smart_ablation_plan_gate_accepts_legacy_audit_variant_from_contract_mode(tmp_path: Path) -> None:
    args = _args(tmp_path, require_baselines=False)
    bundle_dir = tmp_path / "smart_bundle"
    _write_smart_preflight(Path(args.smart_preflight_json))
    candidate_audit_path = Path(args.candidate_bundle_audit_json)
    _write_candidate_bundle_audit(candidate_audit_path, bundle_dir)
    audit = json.loads(candidate_audit_path.read_text(encoding="utf-8"))
    audit.pop("manifest_variant", None)
    audit.pop("candidate_variant", None)
    audit["specialist_contract_mode"] = "smart_seq520_candidate"
    audit["bundle_summary"].pop("manifest_variant", None)
    audit["bundle_summary"].pop("candidate_variant", None)
    audit["bundle_summary"]["specialist_contract_mode"] = "smart_seq520_candidate"
    _write_json(candidate_audit_path, audit)
    _write_replay_dir(
        Path(args.candidate_replay_dir),
        variant="smart_seq520_candidate",
        bundle_dir=bundle_dir,
        candidate_bundle_audit_sha256=gate._sha256_file(candidate_audit_path) or "",
    )

    report = gate.run(args)

    assert report["decision"] == "READY_FOR_SMART_ABLATION_REPLAY_PLAN_REVIEW"


def test_smart_ablation_plan_gate_fails_closed_without_candidate_bundle_or_replay(tmp_path: Path) -> None:
    args = _args(tmp_path, require_baselines=False)
    _write_smart_preflight(Path(args.smart_preflight_json))

    report = gate.run(args)

    assert report["decision"] == "BLOCKED_SMART_ABLATION_REPLAY_PLAN_GATE"
    failed = {row["check"] for row in report["failures"]}
    assert "smart candidate bundle audit exists" in failed
    assert "smart candidate replay manifest exists" in failed
    assert report["training_allowed"] is False
    assert report["replay_allowed_by_this_gate"] is False
    assert all(value is False for value in report["side_effects_started"].values())


def test_smart_ablation_plan_gate_requires_path_quality_bad_path_and_tail_slice(tmp_path: Path) -> None:
    args = _args(tmp_path, require_baselines=False)
    bundle_dir = tmp_path / "smart_bundle"
    _write_smart_preflight(Path(args.smart_preflight_json))
    candidate_audit_path = Path(args.candidate_bundle_audit_json)
    _write_candidate_bundle_audit(candidate_audit_path, bundle_dir)
    _write_replay_dir(
        Path(args.candidate_replay_dir),
        variant="smart_seq520_candidate",
        bundle_dir=bundle_dir,
        candidate_bundle_audit_sha256=gate._sha256_file(candidate_audit_path) or "",
    )
    trades_path = Path(args.candidate_replay_dir) / "replay_policy_trades.csv"
    trades = pd.read_csv(trades_path).drop(columns=["bad_path_prob", "path_quality_pred", "tail_bucket"])
    trades.to_csv(trades_path, index=False)
    metrics_path = Path(args.candidate_replay_dir) / "replay_policy_metrics.csv"
    metrics = pd.read_csv(metrics_path).drop(
        columns=["bad_path_rate", "mean_bad_path_prob", "mean_path_quality_pred", "path_quality_p10"]
    )
    metrics.to_csv(metrics_path, index=False)

    report = gate.run(args)

    assert report["decision"] == "BLOCKED_SMART_ABLATION_REPLAY_PLAN_GATE"
    failed = {row["check"] for row in report["failures"]}
    assert "smart candidate replay supports metric family bad_path" in failed
    assert "smart candidate replay supports metric family path_quality" in failed
    assert "smart candidate replay supports slice tail" in failed
