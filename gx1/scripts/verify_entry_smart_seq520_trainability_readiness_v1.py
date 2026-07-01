#!/usr/bin/env python3
"""Report-only smart_seq520 trainability readiness gate.

This gate is deliberately stricter than smoke-readiness. It does not train,
replay, distill IQL, rebuild data, or touch shadow/live paths. It only proves
whether the smart_seq520 candidate has a fully wired train/proof lane before a
future trainer can be reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.features.entry_specialist_feature_groups_v1 import (
    SPECIALIST_CONTRACT_MODES,
    required_training_specialists_for_mode,
    specialist_contract_training_allowed_for_mode,
)
from gx1.scripts.verify_entry_foundation_state_v1 import REPO, REPORTS_ROOT


CONTRACT_MODE = "smart_seq520_candidate"
EXPECTED_SIGNAL_DIM = 520
EXPECTED_SPECIALIST_COUNT = 8
EXPECTED_CTX_TAG = "CTX6CAT5"
EXPECTED_CTX_CONT_DIM = 142
EXPECTED_CTX_CAT_DIM = 5
READY_DECISION = "READY_FOR_SMART_SEQ520_TRAINABILITY_REVIEW"
BLOCKED_DECISION = "BLOCKED_SMART_SEQ520_TRAINABILITY_READINESS"

DEFAULT_POST_REBUILD_READINESS_JSON = (
    REPORTS_ROOT
    / "entry_smart_dataset_post_rebuild_readiness_20260630_v1"
    / "ENTRY_SMART_DATASET_POST_REBUILD_READINESS_latest.json"
)
DEFAULT_SMOKE_READINESS_JSON = (
    REPORTS_ROOT
    / "entry_smart_seq520_smoke_readiness_20260630_v1"
    / "ENTRY_SMART_SEQ520_SMOKE_READINESS_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_smart_seq520_trainability_readiness_20260630_v1"
DEFAULT_CONTROL_SCRIPT = REPO / "scripts/entry_next_edge_control.sh"
DEFAULT_TRAINER_SOURCE = REPO / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"
DEFAULT_SMOKE_WRAPPER = REPO / "scripts/run_entry_foundation_seq146_smoke_train.sh"
DEFAULT_CANDIDATE_WRAPPER = REPO / "scripts/run_entry_foundation_seq146_candidate_train.sh"
DEFAULT_CANDIDATE_READINESS = REPO / "gx1/scripts/verify_entry_candidate_readiness_v1.py"
DEFAULT_SELECTIVE_EDGE = REPO / "gx1/scripts/evaluate_entry_candidate_selective_edge_v1.py"
DEFAULT_REPLAY_EVIDENCE = REPO / "gx1/scripts/materialize_entry_candidate_replay_evidence_v1.py"
DEFAULT_REPLAY_READINESS = REPO / "gx1/scripts/verify_entry_replay_readiness_v1.py"

SIDE_EFFECTS_CLOSED = {
    "dataset_rebuild": False,
    "training": False,
    "replay": False,
    "iql_distillation": False,
    "shadow": False,
    "live": False,
    "promotion": False,
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "size_bytes": int(path.stat().st_size) if path.exists() else None,
        "sha256": _sha256_file(path),
    }


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _check(name: str, ok: bool, details: Any = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details if details is not None else {}}


def _future_train_contract(smoke_readiness: dict[str, Any]) -> dict[str, Any]:
    contracts = smoke_readiness.get("future_command_contracts")
    if not isinstance(contracts, dict):
        return {}
    contract = contracts.get("smart_smoke_train")
    return contract if isinstance(contract, dict) else {}


def _walk_json(value: Any, *, path: str = "$"):
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_json(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            yield from _walk_json(item, path=f"{path}[{idx}]")


def _contains_exact_string(payloads: dict[str, Any], needle: str) -> list[str]:
    matches: list[str] = []
    for root, payload in payloads.items():
        for path, value in _walk_json(payload, path=root):
            if value == needle:
                matches.append(path)
    return matches


def _ctx_contract_rows(payloads: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root, payload in payloads.items():
        for path, value in _walk_json(payload, path=root):
            if not isinstance(value, dict):
                continue
            if "ctx_contract" in value and isinstance(value.get("ctx_contract"), dict):
                ctx = value["ctx_contract"]
                rows.append(
                    {
                        "path": f"{path}.ctx_contract",
                        "tag": ctx.get("tag") or ctx.get("ctx_tag"),
                        "ctx_cont_dim": ctx.get("ctx_cont_dim"),
                        "ctx_cat_dim": ctx.get("ctx_cat_dim"),
                    }
                )
            elif ("ctx_tag" in value or "tag" in value) and (
                "ctx_cont_dim" in value or "ctx_cat_dim" in value
            ):
                rows.append(
                    {
                        "path": path,
                        "tag": value.get("ctx_tag") or value.get("tag"),
                        "ctx_cont_dim": value.get("ctx_cont_dim"),
                        "ctx_cat_dim": value.get("ctx_cat_dim"),
                    }
                )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = (row.get("path"), row.get("tag"), row.get("ctx_cont_dim"), row.get("ctx_cat_dim"))
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def _ctx_metadata_contract(payloads: dict[str, Any]) -> dict[str, Any]:
    rows = _ctx_contract_rows(payloads)
    stale_ctx6cat6_paths = _contains_exact_string(payloads, "CTX6CAT6")
    declared_rows = [row for row in rows if row.get("tag") or row.get("ctx_cont_dim") or row.get("ctx_cat_dim")]
    mismatched_rows = [
        row
        for row in declared_rows
        if not (
            row.get("tag") == EXPECTED_CTX_TAG
            and int(row.get("ctx_cont_dim") or 0) == EXPECTED_CTX_CONT_DIM
            and int(row.get("ctx_cat_dim") or 0) == EXPECTED_CTX_CAT_DIM
        )
    ]
    return {
        "expected": {
            "ctx_tag": EXPECTED_CTX_TAG,
            "ctx_cont_dim": EXPECTED_CTX_CONT_DIM,
            "ctx_cat_dim": EXPECTED_CTX_CAT_DIM,
        },
        "declared_ctx_contract_count": int(len(declared_rows)),
        "declared_ctx_contracts": declared_rows,
        "mismatched_ctx_contracts": mismatched_rows,
        "stale_ctx6cat6_paths": stale_ctx6cat6_paths,
        "no_stale_ctx6cat6": not stale_ctx6cat6_paths,
        "declared_ctx_contracts_match_expected": not mismatched_rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    post_rebuild_json = Path(args.smart_post_rebuild_readiness_json).expanduser().resolve()
    smoke_readiness_json = Path(args.smart_smoke_readiness_json).expanduser().resolve()
    control_script = Path(args.control_script).expanduser().resolve()
    trainer_source = Path(args.trainer_source).expanduser().resolve()
    smoke_wrapper = Path(args.smoke_wrapper).expanduser().resolve()
    candidate_wrapper = Path(args.candidate_wrapper).expanduser().resolve()
    candidate_readiness_script = Path(args.candidate_readiness_script).expanduser().resolve()
    selective_edge_script = Path(args.selective_edge_script).expanduser().resolve()
    replay_evidence_script = Path(args.replay_evidence_script).expanduser().resolve()
    replay_readiness_script = Path(args.replay_readiness_script).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    post_rebuild = _read_json_or_empty(post_rebuild_json)
    smoke_readiness = _read_json_or_empty(smoke_readiness_json)
    future_train = _future_train_contract(smoke_readiness)
    source_metadata_contract = _ctx_metadata_contract(
        {
            "smart_post_rebuild_readiness": post_rebuild,
            "smart_smoke_readiness": smoke_readiness,
        }
    )

    control_text = _read_text(control_script)
    trainer_text = _read_text(trainer_source)
    smoke_wrapper_text = _read_text(smoke_wrapper)
    candidate_wrapper_text = _read_text(candidate_wrapper)
    candidate_readiness_text = _read_text(candidate_readiness_script)
    selective_edge_text = _read_text(selective_edge_script)
    replay_evidence_text = _read_text(replay_evidence_script)
    replay_readiness_text = _read_text(replay_readiness_script)

    try:
        required_specialists = tuple(required_training_specialists_for_mode(CONTRACT_MODE))
    except Exception:
        required_specialists = ()
    try:
        registry_training_allowed = specialist_contract_training_allowed_for_mode(CONTRACT_MODE)
    except Exception:
        registry_training_allowed = False

    checks = [
        _check("smart post-rebuild dataset audit is ready", post_rebuild.get("decision") == "ENTRY_SMART_DATASET_READY_FOR_TRAIN_READINESS_REVIEW", post_rebuild.get("decision")),
        _check("smart smoke readiness is ready", smoke_readiness.get("decision") == "READY_FOR_SMART_SEQ520_SMOKE_MANIFEST_REVIEW", smoke_readiness.get("decision")),
        _check("smart specialist mode is accepted by trainer contract modes", CONTRACT_MODE in SPECIALIST_CONTRACT_MODES, list(SPECIALIST_CONTRACT_MODES)),
        _check("smart specialist registry keeps training disabled until explicit train gate", registry_training_allowed is False, registry_training_allowed),
        _check("smart required specialist count is eight", len(required_specialists) == EXPECTED_SPECIALIST_COUNT, list(required_specialists)),
        _check("trainer CLI can accept smart specialist contract mode", CONTRACT_MODE in SPECIALIST_CONTRACT_MODES and "specialist-contract-mode" in trainer_text, _artifact_meta(trainer_source)),
        _check(
            "trainer ctx contract is not hard-coded to stale CTX6CAT6 for smart",
            "CTX6CAT6" not in trainer_text,
            _artifact_meta(trainer_source),
        ),
        _check(
            "smart source metadata has no stale CTX6CAT6 ctx contract",
            source_metadata_contract["no_stale_ctx6cat6"],
            source_metadata_contract,
        ),
        _check(
            "declared smart source ctx metadata matches CTX6CAT5",
            source_metadata_contract["declared_ctx_contracts_match_expected"],
            source_metadata_contract,
        ),
        _check("smart smoke wrapper exposes --smart-seq520 lane", "--smart-seq520" in smoke_wrapper_text and CONTRACT_MODE in smoke_wrapper_text, _artifact_meta(smoke_wrapper)),
        _check("smart smoke train is wired in control surface", "smart-smoke-train)" in control_text and "smart-smoke-train --vedtak <id>" in control_text, _artifact_meta(control_script)),
        _check("smart smoke future contract is implemented in control surface", future_train.get("implemented_in_control_surface") is True, future_train),
        _check("smart candidate-readiness supports smart_seq520", CONTRACT_MODE in candidate_readiness_text and str(EXPECTED_SIGNAL_DIM) in candidate_readiness_text, _artifact_meta(candidate_readiness_script)),
        _check("smart candidate train wrapper exposes --smart-seq520 lane", "--smart-seq520" in candidate_wrapper_text and CONTRACT_MODE in candidate_wrapper_text, _artifact_meta(candidate_wrapper)),
        _check("smart selective-edge supports smart_seq520", CONTRACT_MODE in selective_edge_text and str(EXPECTED_SIGNAL_DIM) in selective_edge_text, _artifact_meta(selective_edge_script)),
        _check("smart replay evidence supports smart_seq520", CONTRACT_MODE in replay_evidence_text and str(EXPECTED_SIGNAL_DIM) in replay_evidence_text, _artifact_meta(replay_evidence_script)),
        _check("smart replay-readiness supports smart_seq520", CONTRACT_MODE in replay_readiness_text and str(EXPECTED_SIGNAL_DIM) in replay_readiness_text, _artifact_meta(replay_readiness_script)),
        _check("side effects remain closed", all(value is False for value in SIDE_EFFECTS_CLOSED.values()), SIDE_EFFECTS_CLOSED),
    ]
    failures = [row for row in checks if not row["ok"]]
    ready = not failures
    decision = READY_DECISION if ready else BLOCKED_DECISION
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"ENTRY_SMART_SEQ520_TRAINABILITY_READINESS_{stamp}.json"
    md_path = out_dir / f"ENTRY_SMART_SEQ520_TRAINABILITY_READINESS_{stamp}.md"
    report = {
        "schema_version": "entry_smart_seq520_trainability_readiness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "report_only": True,
        "manifest_variant": CONTRACT_MODE,
        "expected_signal_dim": EXPECTED_SIGNAL_DIM,
        "required_training_specialists": list(required_specialists),
        "inputs": {
            "smart_post_rebuild_readiness": _artifact_meta(post_rebuild_json),
            "smart_smoke_readiness": _artifact_meta(smoke_readiness_json),
            "control_script": _artifact_meta(control_script),
            "trainer_source": _artifact_meta(trainer_source),
            "smoke_wrapper": _artifact_meta(smoke_wrapper),
            "candidate_wrapper": _artifact_meta(candidate_wrapper),
            "candidate_readiness_script": _artifact_meta(candidate_readiness_script),
            "selective_edge_script": _artifact_meta(selective_edge_script),
            "replay_evidence_script": _artifact_meta(replay_evidence_script),
            "replay_readiness_script": _artifact_meta(replay_readiness_script),
        },
        "future_train_contract": future_train,
        "source_metadata_contract": source_metadata_contract,
        "checks": checks,
        "failures": failures,
        "blockers": [row["name"] for row in failures],
        "training_allowed": False,
        "candidate_training_allowed": False,
        "replay_allowed": False,
        "iql_allowed": False,
        "shadow_live_promotion_allowed": False,
        "execution_allowed_now": False,
        "side_effects_started": dict(SIDE_EFFECTS_CLOSED),
        "next_required_gate": (
            "review explicit smart smoke-train implementation package"
            if ready
            else "wire smart_seq520 trainer/wrapper/control/candidate/replay surfaces before any smart training vedtak"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    latest_json = out_dir / "ENTRY_SMART_SEQ520_TRAINABILITY_READINESS_latest.json"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    md_lines = [
        "# Entry Smart Seq520 Trainability Readiness",
        "",
        f"- decision: `{decision}`",
        "- report_only: `true`",
        "- training_allowed: `false`",
        f"- failures: `{len(failures)}`",
        "",
        "## Failures",
        "",
    ]
    md_lines.extend([f"- `{row['name']}`" for row in failures] or ["- None"])
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (out_dir / "ENTRY_SMART_SEQ520_TRAINABILITY_READINESS_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    if failures and bool(args.fail_on_not_ready):
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smart-post-rebuild-readiness-json", default=str(DEFAULT_POST_REBUILD_READINESS_JSON))
    ap.add_argument("--smart-smoke-readiness-json", default=str(DEFAULT_SMOKE_READINESS_JSON))
    ap.add_argument("--control-script", default=str(DEFAULT_CONTROL_SCRIPT))
    ap.add_argument("--trainer-source", default=str(DEFAULT_TRAINER_SOURCE))
    ap.add_argument("--smoke-wrapper", default=str(DEFAULT_SMOKE_WRAPPER))
    ap.add_argument("--candidate-wrapper", default=str(DEFAULT_CANDIDATE_WRAPPER))
    ap.add_argument("--candidate-readiness-script", default=str(DEFAULT_CANDIDATE_READINESS))
    ap.add_argument("--selective-edge-script", default=str(DEFAULT_SELECTIVE_EDGE))
    ap.add_argument("--replay-evidence-script", default=str(DEFAULT_REPLAY_EVIDENCE))
    ap.add_argument("--replay-readiness-script", default=str(DEFAULT_REPLAY_READINESS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
