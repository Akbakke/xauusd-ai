#!/usr/bin/env python3
"""Materialize the report-only smart Entry ablation/replay plan gate.

This gate does not run replay, train, distill IQL, promote, shadow, or touch
live paths. It only binds existing smart candidate bundle/replay evidence to
the exact ablation matrix that must be run before smart_seq520_candidate can be
compared against seq146/seq215.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.scripts.materialize_entry_specialist_challenger_extension_manifest_v1 import SMART_LAYER_FEATURES
from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


SMART_VARIANT = "smart_seq520_candidate"
SMART_SEQ_SNAP_WIDTH = 520
BASE_SIGNAL_FEATURE_COUNT = 41
OLD_EXTENSION_FEATURE_COUNT = 105 + 41 + 28
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_smart_ablation_replay_plan_gate_20260630_v1"
DEFAULT_SMART_PREFLIGHT_JSON = (
    REPORTS_ROOT
    / "entry_smart_seq_rebuild_preflight_20260630_v1/ENTRY_SMART_REBUILD_PREFLIGHT_latest.json"
)
DEFAULT_SMART_CANDIDATE_BUNDLE_AUDIT_JSON = (
    REPORTS_ROOT
    / "entry_candidate_bundle_audit_20260628_v1/smart_seq520_candidate/ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.json"
)
DEFAULT_SMART_CANDIDATE_REPLAY_DIR = (
    REPORTS_ROOT / "entry_candidate_replay_20260628_v1/smart_seq520_candidate"
)
DEFAULT_SEQ146_REPLAY_DIR = REPORTS_ROOT / "entry_candidate_replay_20260628_v1"
DEFAULT_SEQ215_REPLAY_DIR = (
    REPORTS_ROOT / "entry_candidate_replay_20260628_v1/challenger_seq215_20260630"
)

REQUIRED_METRIC_FAMILIES: "OrderedDict[str, dict[str, Any]]" = OrderedDict(
    [
        (
            "pnl",
            {
                "metric_columns_any": ("net_sum_bps", "net_mean_bps", "pnl_sum_bps"),
                "trade_columns_any": ("net_pnl_bps", "realized_pnl_bps", "pnl_bps"),
            },
        ),
        (
            "drawdown",
            {
                "metric_columns_any": ("max_drawdown_bps", "max_drawdown_signed_bps"),
                "trade_columns_any": (),
            },
        ),
        (
            "mae",
            {
                "metric_columns_any": ("mean_mae_bps", "mae_p95_bps", "max_mae_bps"),
                "trade_columns_any": ("mae_bps",),
            },
        ),
        (
            "bad_path",
            {
                "metric_columns_any": ("bad_path_rate", "mean_bad_path_prob", "bad_path_count"),
                "trade_columns_any": ("bad_path_prob", "bad_path", "bad_path_flag"),
            },
        ),
        (
            "path_quality",
            {
                "metric_columns_any": (
                    "mean_path_quality_pred",
                    "path_quality_mean",
                    "path_quality_p10",
                ),
                "trade_columns_any": ("path_quality_pred", "path_quality", "path_quality_score"),
            },
        ),
    ]
)

REQUIRED_SLICE_DIMENSIONS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    [
        ("session", ("session", "session_id", "session_name")),
        ("regime", ("regime", "trend_regime", "vol_regime", "regime_id", "vol_regime_id")),
        ("direction", ("side", "direction", "trade_side", "pred_direction")),
        ("tail", ("tail_bucket", "tail_risk_bucket", "tail_event", "tail_loss_bucket")),
    ]
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": _sha256_file(path) if path.exists() else None,
    }


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _check(name: str, ok: bool, details: Any = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details}


def _first_int(*values: Any) -> int:
    for value in values:
        try:
            if value not in (None, ""):
                return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _smart_layer_specs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, (version, features, builder, source_path) in SMART_LAYER_FEATURES.items():
        rows.append(
            {
                "family_label": label,
                "feature_count": int(len(features)),
                "feature_version": version,
                "builder": builder,
                "source_path": str(source_path),
            }
        )
    return rows


def _required_replay_evidence_contract() -> dict[str, Any]:
    return {
        "required_files": [
            "REPLAY_EVIDENCE_MANIFEST.json",
            "replay_policy_metrics.csv",
            "replay_policy_monthly.csv",
            "replay_policy_trades.csv",
        ],
        "required_metric_families": {
            name: {
                "metric_columns_any": list(spec["metric_columns_any"]),
                "trade_columns_any": list(spec["trade_columns_any"]),
            }
            for name, spec in REQUIRED_METRIC_FAMILIES.items()
        },
        "required_slice_dimensions": {
            name: list(columns) for name, columns in REQUIRED_SLICE_DIMENSIONS.items()
        },
        "minimum_scope": "candidate and every ablation arm must report aggregate plus session/regime/direction/tail slices",
        "identity_required": [
            "candidate_bundle_dir",
            "manifest_variant",
            "bundle/replay artifact hashes",
            "policy_id",
            "ablation_id",
        ],
    }


def build_required_ablation_plan() -> dict[str, Any]:
    smart_layers = _smart_layer_specs()
    smart_feature_count = int(sum(row["feature_count"] for row in smart_layers))
    full_blocks = {
        "base_signal_bridge": BASE_SIGNAL_FEATURE_COUNT,
        "old_seq215_extension": OLD_EXTENSION_FEATURE_COUNT,
        "smart_layers": smart_feature_count,
    }

    def arm(
        *,
        ablation_id: str,
        ablation_name: str,
        ablation_type: str,
        expected_seq_snap_width: int,
        included_blocks: list[str],
        excluded_blocks: list[str],
        xgb_bridge_mode: str = "active",
        dropped_family: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "ablation_id": ablation_id,
            "ablation_name": ablation_name,
            "ablation_type": ablation_type,
            "manifest_variant": SMART_VARIANT,
            "expected_seq_snap_width": int(expected_seq_snap_width),
            "included_feature_blocks": included_blocks,
            "excluded_feature_blocks": excluded_blocks,
            "xgb_bridge_mode": xgb_bridge_mode,
            "dropped_smart_family": dropped_family,
            "must_materialize_replay_evidence": True,
            "must_use_same_splits_costs_overlays_and_threshold_policy": True,
            "compare_against": ["with-old+smart", "baseline_seq146", "baseline_seq215"],
            "replay_evidence_contract": _required_replay_evidence_contract(),
        }

    arms = [
        arm(
            ablation_id="with_old_plus_smart",
            ablation_name="with-old+smart",
            ablation_type="full_smart_control",
            expected_seq_snap_width=BASE_SIGNAL_FEATURE_COUNT + OLD_EXTENSION_FEATURE_COUNT + smart_feature_count,
            included_blocks=["base_signal_bridge", "old_seq215_extension", "smart_layers"],
            excluded_blocks=[],
        ),
        arm(
            ablation_id="smart_only",
            ablation_name="smart-only",
            ablation_type="feature_set_ablation",
            expected_seq_snap_width=BASE_SIGNAL_FEATURE_COUNT + smart_feature_count,
            included_blocks=["base_signal_bridge", "smart_layers"],
            excluded_blocks=["old_seq215_extension"],
        ),
        arm(
            ablation_id="old_only",
            ablation_name="old-only",
            ablation_type="feature_set_ablation",
            expected_seq_snap_width=BASE_SIGNAL_FEATURE_COUNT + OLD_EXTENSION_FEATURE_COUNT,
            included_blocks=["base_signal_bridge", "old_seq215_extension"],
            excluded_blocks=["smart_layers"],
        ),
        arm(
            ablation_id="no_xgb",
            ablation_name="no-XGB",
            ablation_type="bridge_ablation",
            expected_seq_snap_width=BASE_SIGNAL_FEATURE_COUNT + OLD_EXTENSION_FEATURE_COUNT + smart_feature_count,
            included_blocks=["base_signal_bridge_neutralized", "old_seq215_extension", "smart_layers"],
            excluded_blocks=["live_xgb_bridge_signal"],
            xgb_bridge_mode="neutralize_signal_bridge",
        ),
    ]
    for row in smart_layers:
        width = BASE_SIGNAL_FEATURE_COUNT + OLD_EXTENSION_FEATURE_COUNT + smart_feature_count - int(row["feature_count"])
        label = str(row["family_label"])
        arms.append(
            arm(
                ablation_id=f"drop_family_{label}",
                ablation_name=f"drop-family:{label}",
                ablation_type="drop_smart_family",
                expected_seq_snap_width=width,
                included_blocks=["base_signal_bridge", "old_seq215_extension", "smart_layers_minus_one_family"],
                excluded_blocks=[label],
                dropped_family=row,
            )
        )
    return {
        "schema_version": "entry_smart_ablation_matrix_v1",
        "smart_variant": SMART_VARIANT,
        "feature_block_counts": full_blocks,
        "smart_layer_count": int(len(smart_layers)),
        "smart_layer_feature_count": smart_feature_count,
        "reference_baselines": [
            {
                "baseline_id": "baseline_seq146",
                "contract_mode": "foundation_seq146",
                "expected_seq_snap_width": 146,
                "role": "active foundation baseline",
            },
            {
                "baseline_id": "baseline_seq215",
                "contract_mode": "challenger_seq215",
                "expected_seq_snap_width": 215,
                "role": "old seq215 challenger baseline before smart layers",
            },
        ],
        "ablation_count": int(len(arms)),
        "drop_family_count": int(sum(1 for row in arms if row["ablation_type"] == "drop_smart_family")),
        "required_ablations": arms,
    }


def _smart_preflight_checks(path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    return [
        _check("smart rebuild preflight exists", path.exists(), _artifact_meta(path)),
        _check(
            "smart rebuild preflight is ready",
            str(report.get("decision") or "") == "READY_FOR_SMART_REBUILD_VEDTAK_REVIEW",
            {"decision": report.get("decision")},
        ),
        _check(
            "smart preflight variant is smart_seq520_candidate",
            str(counts.get("manifest_variant") or report.get("manifest_variant") or "") == SMART_VARIANT,
            counts,
        ),
        _check(
            "smart preflight expected seq/snap width is 520",
            int(counts.get("expected_seq_snap_width") or 0) == SMART_SEQ_SNAP_WIDTH,
            counts,
        ),
        _check("smart preflight remains report-only", bool(report.get("report_only")) is True, report.get("report_only")),
        _check(
            "smart preflight keeps training closed",
            bool(report.get("training_allowed")) is False,
            report.get("training_allowed"),
        ),
    ]


def _candidate_bundle_checks(path: Path, report: dict[str, Any], candidate_bundle_dir_arg: str = "") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundle = report.get("bundle_summary") if isinstance(report.get("bundle_summary"), dict) else {}
    pretrain = report.get("pretrain_manifest_contract") if isinstance(report.get("pretrain_manifest_contract"), dict) else {}
    bundle_dir = _first_text(
        candidate_bundle_dir_arg,
        report.get("bundle_dir"),
        bundle.get("bundle_dir"),
        pretrain.get("bundle_dir"),
    )
    bundle_dir_path = Path(bundle_dir).expanduser() if bundle_dir else Path("")
    seq_dim = _first_int(
        report.get("seq_input_dim"),
        report.get("bundle_seq_input_dim"),
        bundle.get("seq_input_dim"),
        pretrain.get("seq_input_dim"),
    )
    snap_dim = _first_int(
        report.get("snap_input_dim"),
        report.get("bundle_snap_input_dim"),
        bundle.get("snap_input_dim"),
        pretrain.get("snap_input_dim"),
    )
    variant = _first_text(
        report.get("manifest_variant"),
        report.get("candidate_variant"),
        bundle.get("manifest_variant"),
        bundle.get("candidate_variant"),
        pretrain.get("manifest_variant"),
        pretrain.get("candidate_variant"),
    )
    identity = {
        "candidate_bundle_audit_json": str(path),
        "candidate_bundle_dir": bundle_dir,
        "manifest_variant": variant,
        "seq_input_dim": seq_dim,
        "snap_input_dim": snap_dim,
        "decision": str(report.get("decision") or ""),
    }
    return [
        _check("smart candidate bundle audit exists", path.exists(), _artifact_meta(path)),
        _check(
            "smart candidate bundle audit PASS",
            path.exists() and str(report.get("decision") or "") == "PASS",
            {"decision": report.get("decision"), "failures": report.get("failures")},
        ),
        _check("smart candidate bundle dir is declared", bool(bundle_dir), identity),
        _check(
            "smart candidate bundle dir exists",
            bool(bundle_dir) and bundle_dir_path.exists(),
            {"candidate_bundle_dir": bundle_dir},
        ),
        _check(
            "smart candidate bundle variant is smart_seq520_candidate",
            variant == SMART_VARIANT,
            identity,
        ),
        _check(
            "smart candidate bundle seq/snap dims are 520",
            seq_dim == SMART_SEQ_SNAP_WIDTH and snap_dim == SMART_SEQ_SNAP_WIDTH,
            identity,
        ),
    ], identity


def _slice_available_from_file(slices: pd.DataFrame, dimension: str) -> bool:
    if slices.empty:
        return False
    for col in ("slice_family", "slice_dimension", "dimension", "group", "slice"):
        if col not in slices.columns:
            continue
        values = {str(value).strip().lower() for value in slices[col].dropna().unique()}
        if dimension.lower() in values:
            return True
        if any(str(value).lower().startswith(f"{dimension.lower()}=") for value in values):
            return True
    return False


def _replay_evidence_checks(
    *,
    label: str,
    replay_dir: Path,
    expected_variant: str,
    expected_candidate_bundle_dir: str = "",
    required: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = replay_dir / "REPLAY_EVIDENCE_MANIFEST.json"
    metrics_path = replay_dir / "replay_policy_metrics.csv"
    monthly_path = replay_dir / "replay_policy_monthly.csv"
    trades_path = replay_dir / "replay_policy_trades.csv"
    slices_path = replay_dir / "replay_policy_slices.csv"
    manifest = _read_json_if_exists(manifest_path)
    metrics = _read_csv_if_exists(metrics_path)
    monthly = _read_csv_if_exists(monthly_path)
    trades = _read_csv_if_exists(trades_path)
    slices = _read_csv_if_exists(slices_path)
    metric_cols = set(str(col) for col in metrics.columns)
    trade_cols = set(str(col) for col in trades.columns)
    manifest_identity = (
        manifest.get("replay_identity_contract")
        if isinstance(manifest.get("replay_identity_contract"), dict)
        else {}
    )
    manifest_variant = _first_text(
        manifest.get("manifest_variant"),
        manifest.get("candidate_variant"),
        manifest_identity.get("manifest_variant"),
        manifest_identity.get("candidate_variant"),
        manifest.get("contract_mode"),
    )
    manifest_bundle_dir = _first_text(
        manifest.get("candidate_bundle_dir"),
        manifest_identity.get("candidate_bundle_dir"),
        manifest_identity.get("replay_identity_candidate_bundle_dir"),
    )

    checks = [
        _check(f"{label} replay dir exists", replay_dir.exists(), {"replay_dir": str(replay_dir)}),
        _check(f"{label} replay manifest exists", manifest_path.exists(), _artifact_meta(manifest_path)),
        _check(
            f"{label} replay manifest PASS",
            manifest_path.exists() and str(manifest.get("decision") or "") == "PASS",
            {"decision": manifest.get("decision"), "failures": manifest.get("failures")},
        ),
        _check(f"{label} replay metrics exists", metrics_path.exists() and not metrics.empty, _artifact_meta(metrics_path)),
        _check(f"{label} replay monthly exists", monthly_path.exists() and not monthly.empty, _artifact_meta(monthly_path)),
        _check(f"{label} replay trades exists", trades_path.exists() and not trades.empty, _artifact_meta(trades_path)),
        _check(
            f"{label} replay manifest variant matches",
            manifest_variant == expected_variant,
            {"expected_variant": expected_variant, "manifest_variant": manifest_variant},
        ),
    ]
    if expected_candidate_bundle_dir:
        checks.append(
            _check(
                f"{label} replay identity matches smart candidate bundle",
                manifest_bundle_dir == expected_candidate_bundle_dir,
                {
                    "expected_candidate_bundle_dir": expected_candidate_bundle_dir,
                    "manifest_candidate_bundle_dir": manifest_bundle_dir,
                },
            )
        )

    for family, spec in REQUIRED_METRIC_FAMILIES.items():
        metric_ok = bool(set(spec["metric_columns_any"]) & metric_cols)
        trade_ok = bool(set(spec["trade_columns_any"]) & trade_cols)
        checks.append(
            _check(
                f"{label} replay supports metric family {family}",
                metric_ok or trade_ok,
                {
                    "metric_columns_any": list(spec["metric_columns_any"]),
                    "trade_columns_any": list(spec["trade_columns_any"]),
                    "metric_columns_present": sorted(metric_cols),
                    "trade_columns_present": sorted(trade_cols),
                },
            )
        )

    for dimension, aliases in REQUIRED_SLICE_DIMENSIONS.items():
        checks.append(
            _check(
                f"{label} replay supports slice {dimension}",
                bool(set(aliases) & trade_cols) or _slice_available_from_file(slices, dimension),
                {
                    "accepted_trade_columns": list(aliases),
                    "slice_file": str(slices_path),
                    "slice_file_exists": slices_path.exists(),
                },
            )
        )

    if not required:
        checks = [
            check if check["ok"] else {**check, "ok": True, "details": {"optional_missing": True, "original": check["details"]}}
            for check in checks
        ]

    identity = {
        "label": label,
        "replay_dir": str(replay_dir),
        "manifest_json": str(manifest_path),
        "metrics_csv": str(metrics_path),
        "monthly_csv": str(monthly_path),
        "trades_csv": str(trades_path),
        "slices_csv": str(slices_path),
        "manifest_variant": manifest_variant,
        "candidate_bundle_dir": manifest_bundle_dir,
        "metrics_columns": sorted(metric_cols),
        "trade_columns": sorted(trade_cols),
        "required": bool(required),
    }
    return checks, identity


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Smart Ablation Replay Plan Gate",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Report only: `{str(report['report_only']).lower()}`",
        f"- Replay started: `{str(report['side_effects_started']['replay']).lower()}`",
        f"- Training started: `{str(report['side_effects_started']['training']).lower()}`",
        f"- Ablations required: `{report['required_ablation_plan']['ablation_count']}`",
        f"- Failures: `{len(report['failures'])}`",
        "",
        "## Required Ablations",
        "",
    ]
    for row in report["required_ablation_plan"]["required_ablations"]:
        lines.append(
            f"- `{row['ablation_name']}`: width `{row['expected_seq_snap_width']}`, "
            f"xgb `{row['xgb_bridge_mode']}`"
        )
    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        lines.extend(f"- `{failure['gate']}`: {failure['check']}" for failure in report["failures"])
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    smart_preflight_path = Path(args.smart_preflight_json).expanduser().resolve()
    candidate_bundle_audit_path = Path(args.candidate_bundle_audit_json).expanduser().resolve()
    candidate_replay_dir = Path(args.candidate_replay_dir).expanduser().resolve()
    seq146_replay_dir = Path(args.seq146_replay_dir).expanduser().resolve()
    seq215_replay_dir = Path(args.seq215_replay_dir).expanduser().resolve()

    smart_preflight = _read_json_if_exists(smart_preflight_path)
    candidate_bundle_audit = _read_json_if_exists(candidate_bundle_audit_path)
    ablation_plan = build_required_ablation_plan()
    gate_checks: dict[str, list[dict[str, Any]]] = {
        "smart_preflight": _smart_preflight_checks(smart_preflight_path, smart_preflight),
    }
    bundle_checks, candidate_identity = _candidate_bundle_checks(
        candidate_bundle_audit_path,
        candidate_bundle_audit,
        str(getattr(args, "candidate_bundle_dir", "") or ""),
    )
    gate_checks["smart_candidate_bundle"] = bundle_checks
    candidate_replay_checks, candidate_replay_identity = _replay_evidence_checks(
        label="smart candidate",
        replay_dir=candidate_replay_dir,
        expected_variant=SMART_VARIANT,
        expected_candidate_bundle_dir=str(candidate_identity.get("candidate_bundle_dir") or ""),
        required=True,
    )
    gate_checks["smart_candidate_replay_evidence"] = candidate_replay_checks
    seq146_checks, seq146_identity = _replay_evidence_checks(
        label="seq146 baseline",
        replay_dir=seq146_replay_dir,
        expected_variant="foundation_seq146",
        required=bool(args.require_baseline_replay_evidence),
    )
    seq215_checks, seq215_identity = _replay_evidence_checks(
        label="seq215 baseline",
        replay_dir=seq215_replay_dir,
        expected_variant="challenger_seq215",
        required=bool(args.require_baseline_replay_evidence),
    )
    gate_checks["seq146_baseline_replay_evidence"] = seq146_checks
    gate_checks["seq215_baseline_replay_evidence"] = seq215_checks
    gate_checks["side_effect_guard"] = [
        _check("gate never starts replay", True),
        _check("gate never starts training", True),
        _check("gate never starts IQL", True),
        _check("gate never touches shadow/live", True),
    ]

    gates = []
    for name, checks in gate_checks.items():
        passed = int(sum(1 for check in checks if check["ok"]))
        gates.append(
            {
                "name": name,
                "decision": "PASS" if passed == len(checks) else "FAIL",
                "passed": passed,
                "total": int(len(checks)),
                "checks": checks,
            }
        )
    failures = [
        {"gate": gate["name"], "check": check["name"], "details": check.get("details")}
        for gate in gates
        for check in gate["checks"]
        if not check["ok"]
    ]
    ready = not failures
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"ENTRY_SMART_ABLATION_REPLAY_PLAN_GATE_{timestamp}.json"
    md_path = out_dir / f"ENTRY_SMART_ABLATION_REPLAY_PLAN_GATE_{timestamp}.md"
    report = {
        "schema_version": "entry_smart_ablation_replay_plan_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "READY_FOR_SMART_ABLATION_REPLAY_PLAN_REVIEW" if ready else "BLOCKED_SMART_ABLATION_REPLAY_PLAN_GATE",
        "report_only": True,
        "training_allowed": False,
        "replay_allowed_by_this_gate": False,
        "iql_allowed_by_this_gate": False,
        "shadow_live_promotion_allowed": False,
        "side_effects_started": {
            "training": False,
            "replay": False,
            "iql_distillation": False,
            "shadow": False,
            "live": False,
            "promotion": False,
        },
        "smart_variant": SMART_VARIANT,
        "smart_expected_seq_snap_width": SMART_SEQ_SNAP_WIDTH,
        "candidate_identity": candidate_identity,
        "candidate_replay_identity": candidate_replay_identity,
        "baseline_replay_identities": {
            "seq146": seq146_identity,
            "seq215": seq215_identity,
        },
        "required_replay_evidence_contract": _required_replay_evidence_contract(),
        "required_ablation_plan": ablation_plan,
        "gates": gates,
        "failures": failures,
        "next_required_gate": (
            "manual review, then explicit ablation replay vedtak using the exact ablation matrix; "
            "this gate itself cannot start replay"
            if ready
            else "produce smart candidate bundle audit plus replay evidence, and seq146/seq215 baseline replay evidence if required"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_SMART_ABLATION_REPLAY_PLAN_GATE_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_SMART_ABLATION_REPLAY_PLAN_GATE_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failures": report["failures"],
                    "json_path": report["json_path"],
                    "md_path": report["md_path"],
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
    if args.fail_on_not_ready and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smart-preflight-json", default=str(DEFAULT_SMART_PREFLIGHT_JSON))
    ap.add_argument("--candidate-bundle-audit-json", default=str(DEFAULT_SMART_CANDIDATE_BUNDLE_AUDIT_JSON))
    ap.add_argument("--candidate-bundle-dir", default="")
    ap.add_argument("--candidate-replay-dir", default=str(DEFAULT_SMART_CANDIDATE_REPLAY_DIR))
    ap.add_argument("--seq146-replay-dir", default=str(DEFAULT_SEQ146_REPLAY_DIR))
    ap.add_argument("--seq215-replay-dir", default=str(DEFAULT_SEQ215_REPLAY_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--require-baseline-replay-evidence", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
