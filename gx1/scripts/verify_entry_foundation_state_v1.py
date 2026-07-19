#!/usr/bin/env python3
"""Verify the immutable model-native Entry seq513 evidence state.

The historical foundation/seq146 mega-verifier is retired.  This module keeps
the small set of filesystem constants imported by adjacent report-only tools,
but state authority comes only from explicit timestamped seq513 events.  It
never selects ``*_latest.json``, starts work, or authorizes launch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_SIGNAL_DIM,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
    model_native_signal_contract_metadata,
    require_model_native_signal_contract,
)
from gx1.contracts.immutable_event_authority_v1 import (
    require_newest_immutable_event,
    write_immutable_json_event,
)


STATE_SCHEMA_VERSION = "entry_model_native_seq513_state_v1"
STATE_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SEQ513_STATE"
STATE_PROVEN_DECISION = "MODEL_NATIVE_SEQ513_STATE_PROVEN_LAUNCH_BLOCKED"
STATE_BLOCKED_DECISION = "BLOCKED_MODEL_NATIVE_SEQ513_STATE"


@dataclass(frozen=True)
class EvidenceSpec:
    name: str
    arg_name: str
    event_prefix: str
    schema_version: str
    ready_decision: str


EVIDENCE_SPECS = (
    EvidenceSpec(
        "rebuild_preflight",
        "rebuild_preflight_json",
        "ENTRY_MODEL_NATIVE_SEQ513_REBUILD_PREFLIGHT",
        "entry_model_native_seq513_rebuild_preflight_v1",
        "READY_FOR_MODEL_NATIVE_SEQ513_REBUILD_VEDTAK_REVIEW",
    ),
    EvidenceSpec(
        "smoke_manifest",
        "smoke_manifest_json",
        "ENTRY_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST",
        "entry_model_native_seq513_smoke_manifest_v1",
        "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST_REVIEW",
    ),
    EvidenceSpec(
        "smoke_readiness",
        "smoke_readiness_json",
        "ENTRY_MODEL_NATIVE_SEQ513_SMOKE_READINESS",
        "entry_model_native_seq513_smoke_readiness_v1",
        "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_READINESS_REVIEW",
    ),
    EvidenceSpec(
        "trainability_readiness",
        "trainability_readiness_json",
        "ENTRY_MODEL_NATIVE_SEQ513_TRAINABILITY_READINESS",
        "entry_model_native_seq513_trainability_readiness_v1",
        "READY_FOR_MODEL_NATIVE_SEQ513_TRAINABILITY_REVIEW",
    ),
    EvidenceSpec(
        "candidate_readiness",
        "candidate_readiness_json",
        "ENTRY_CANDIDATE_READINESS",
        "entry_candidate_readiness_model_native_v1",
        "READY_FOR_CANDIDATE_TRAINING_VEDTAK",
    ),
)

_FORBIDDEN_DIRECTION_KEYS = frozenset(
    {
        "anchor_logits",
        "delta_logits",
        "anchor_gate",
        "neutralize_signal_bridge",
        "selection_score_threshold",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"immutable evidence root is not an object: {path}")
    return payload


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _signal_contracts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _walk_dicts(payload):
        candidate = row.get("model_native_signal_contract")
        if not isinstance(candidate, dict):
            continue
        identity = _sha256_json(candidate)
        if identity not in seen:
            seen.add(identity)
            contracts.append(candidate)
    return contracts


def _forbidden_direction_keys(payload: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    for row in _walk_dicts(payload):
        found.update(_FORBIDDEN_DIRECTION_KEYS.intersection(row))
    return sorted(found)


def _contract_failures(spec: EvidenceSpec, payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if payload.get("schema_version") != spec.schema_version:
        failures.append(
            f"schema_version={payload.get('schema_version')!r} "
            f"expected={spec.schema_version!r}"
        )
    if payload.get("decision") != spec.ready_decision:
        failures.append(
            f"decision={payload.get('decision')!r} expected={spec.ready_decision!r}"
        )
    if payload.get("failures"):
        failures.append("event declares failures")
    if spec.name != "candidate_readiness" and payload.get("report_only") is not True:
        failures.append("readiness/preflight event is not explicitly report-only")

    side_effects = payload.get("side_effects_started")
    if spec.name != "candidate_readiness":
        if not isinstance(side_effects, dict) or not side_effects:
            failures.append("readiness/preflight event lacks side-effect closure proof")
        elif any(value is not False for value in side_effects.values()):
            failures.append("readiness/preflight event reports a started side effect")
    for key in (
        "shadow_live_allowed",
        "shadow_live_promotion_allowed",
        "promotion_shadow_live_allowed",
    ):
        if payload.get(key) is True:
            failures.append(f"event improperly authorizes {key}")

    forbidden = _forbidden_direction_keys(payload)
    if forbidden:
        failures.append(f"forbidden direction keys present: {forbidden}")

    contracts = _signal_contracts(payload)
    for index, contract in enumerate(contracts):
        try:
            require_model_native_signal_contract(
                contract,
                context=f"MODEL_NATIVE_STATE_{spec.name}_{index}",
            )
        except RuntimeError as exc:
            failures.append(str(exc))

    if spec.name == "rebuild_preflight":
        required = payload.get("required_model_native_contract")
        required = required if isinstance(required, dict) else {}
        expected = {
            "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
            "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
            "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
            "seq_len": MODEL_NATIVE_SEQ_LEN,
            "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
            "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
            "base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
            "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
            "bridge_dim": 0,
            "bridge_source": None,
            "anchor_source": None,
        }
        if any(required.get(key) != value for key, value in expected.items()):
            failures.append("rebuild preflight exact seq513 contract mismatch")
        if not contracts:
            failures.append("rebuild preflight lacks exact model-native signal contract")

    elif spec.name == "smoke_manifest":
        if payload.get("manifest_variant") != MODEL_NATIVE_CONTRACT_MODE:
            failures.append("smoke manifest contract mode mismatch")
        if payload.get("expected_seq_snap_width") != MODEL_NATIVE_SIGNAL_DIM:
            failures.append("smoke manifest signal width mismatch")
        smoke_manifest = payload.get("smoke_manifest")
        smoke_manifest = smoke_manifest if isinstance(smoke_manifest, dict) else {}
        if not smoke_manifest or payload.get("manifest_sha256") != _sha256_json(
            smoke_manifest
        ):
            failures.append("smoke manifest embedded hash mismatch")
        binding = {
            "post_rebuild_readiness": payload.get("post_rebuild_readiness"),
            "specialist_audit": payload.get("specialist_audit"),
            "split_artifacts": payload.get("split_artifacts"),
        }
        if payload.get("evidence_binding_sha256") != _sha256_json(binding):
            failures.append("smoke manifest evidence binding hash mismatch")
        if not contracts:
            failures.append("smoke manifest lacks exact model-native signal contract")

    elif spec.name == "smoke_readiness":
        candidate = payload.get("smart_candidate")
        candidate = candidate if isinstance(candidate, dict) else {}
        if (
            candidate.get("manifest_variant") != MODEL_NATIVE_CONTRACT_MODE
            or candidate.get("specialist_contract_mode")
            != MODEL_NATIVE_CONTRACT_MODE
            or candidate.get("expected_signal_dim") != MODEL_NATIVE_SIGNAL_DIM
            or candidate.get("expected_selected_feature_count")
            != MODEL_NATIVE_SELECTED_FEATURE_COUNT
        ):
            failures.append("smoke readiness exact seq513 contract mismatch")
        inputs = payload.get("inputs")
        inputs = inputs if isinstance(inputs, dict) else {}
        if payload.get("evidence_binding_sha256") != _sha256_json(inputs):
            failures.append("smoke readiness evidence binding hash mismatch")

    elif spec.name == "trainability_readiness":
        if (
            payload.get("manifest_variant") != MODEL_NATIVE_CONTRACT_MODE
            or payload.get("expected_signal_dim") != MODEL_NATIVE_SIGNAL_DIM
        ):
            failures.append("trainability readiness exact seq513 contract mismatch")
        inputs = payload.get("inputs")
        inputs = inputs if isinstance(inputs, dict) else {}
        if payload.get("evidence_binding_sha256") != _sha256_json(inputs):
            failures.append("trainability readiness evidence binding hash mismatch")

    elif spec.name == "candidate_readiness":
        if (
            payload.get("contract_mode") != MODEL_NATIVE_CONTRACT_MODE
            or payload.get("expected_signal_dim") != MODEL_NATIVE_SIGNAL_DIM
            or payload.get("edge_test_scope") != "strict"
            or payload.get("promotion_shadow_live_allowed") is not False
        ):
            failures.append("candidate readiness exact seq513 contract mismatch")
        fingerprints = payload.get("artifact_fingerprints")
        fingerprints = fingerprints if isinstance(fingerprints, dict) else {}
        if not fingerprints:
            failures.append("candidate readiness lacks artifact fingerprints")
        for name, row in fingerprints.items():
            row = row if isinstance(row, dict) else {}
            raw_path = str(row.get("path") or "").strip()
            if not raw_path or len(str(row.get("sha256") or "")) != 64:
                failures.append(f"candidate fingerprint incomplete: {name}")
                continue
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file() or _sha256_file(path) != row.get("sha256"):
                failures.append(f"candidate fingerprint rehash mismatch: {name}")

    return failures


def _validate_evidence(spec: EvidenceSpec, raw_path: str) -> dict[str, Any]:
    failures: list[str] = []
    if not str(raw_path or "").strip():
        return {
            "name": spec.name,
            "ready": False,
            "path": None,
            "sha256": None,
            "schema_version": None,
            "decision": None,
            "failures": ["explicit immutable event path is required"],
        }

    path = Path(raw_path).expanduser().resolve()
    payload: dict[str, Any] = {}
    sha256: str | None = None
    try:
        require_newest_immutable_event(path, spec.event_prefix)
        payload = _read_json(path)
        sha256 = _sha256_file(path)
        failures.extend(_contract_failures(spec, payload))
    except Exception as exc:
        failures.append(str(exc))
    return {
        "name": spec.name,
        "ready": not failures,
        "path": str(path),
        "sha256": sha256,
        "schema_version": payload.get("schema_version"),
        "decision": payload.get("decision"),
        "failures": failures,
    }


def _selftest() -> dict[str, Any]:
    selected = [
        *MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
        *[
            f"session_regime.state_selftest_{index:03d}"
            for index in range(MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT)
        ],
    ]
    contract = model_native_signal_contract_metadata(selected)
    checks: list[dict[str, Any]] = []
    try:
        require_model_native_signal_contract(contract, context="STATE_SELFTEST")
        checks.append({"name": "exact 34+479=513 contract passes", "ok": True})
    except RuntimeError as exc:
        checks.append(
            {"name": "exact 34+479=513 contract passes", "ok": False, "error": str(exc)}
        )

    broken = json.loads(json.dumps(contract))
    broken["bridge_dim"] = 7
    rejected = False
    try:
        require_model_native_signal_contract(broken, context="STATE_SELFTEST_RETIRED")
    except RuntimeError:
        rejected = True
    checks.append({"name": "retired bridge contract is rejected", "ok": rejected})
    checks.append(
        {
            "name": "state never authorizes launch",
            "ok": True,
            "launch_allowed": False,
        }
    )
    passed = all(bool(row.get("ok")) for row in checks)
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "MODEL_NATIVE_SEQ513_STATE_SELFTEST_PASS" if passed else "MODEL_NATIVE_SEQ513_STATE_SELFTEST_FAIL",
        "selftest": True,
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
        "expected_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "launch_allowed": False,
        "promotion_shadow_live_allowed": False,
        "checks": checks,
        "failures": [row for row in checks if not row.get("ok")],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if bool(getattr(args, "selftest", False)):
        report = _selftest()
        if not bool(getattr(args, "quiet", False)):
            print(json.dumps(report, indent=2, sort_keys=True))
        if report["failures"]:
            raise SystemExit(2)
        return report

    evidence = [
        _validate_evidence(spec, str(getattr(args, spec.arg_name, "") or ""))
        for spec in EVIDENCE_SPECS
    ]
    failures = [
        {"gate": row["name"], "failure": failure}
        for row in evidence
        for failure in row["failures"]
    ]
    state_proven = not failures
    report = {
        "schema_version": STATE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": STATE_PROVEN_DECISION if state_proven else STATE_BLOCKED_DECISION,
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
        "base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "expected_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "model_native_evidence_ready": state_proven,
        "launch_allowed": False,
        "promotion_shadow_live_allowed": False,
        "training_started": False,
        "replay_started": False,
        "live_started": False,
        "evidence": evidence,
        "evidence_sha256": _sha256_json(
            {row["name"]: row.get("sha256") for row in evidence}
        ),
        "failures": failures,
        "next_required_gate": (
            "explicit candidate-train review; replay, serve-parity and launch proof remain required"
            if state_proven
            else "provide and repair every explicit immutable model-native seq513 evidence event"
        ),
    }

    out_dir_raw = str(getattr(args, "out_dir", "") or "").strip()
    if out_dir_raw:
        _, report = write_immutable_json_event(
            Path(out_dir_raw), STATE_EVENT_PREFIX, report
        )
    if not bool(getattr(args, "quiet", False)):
        print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-preflight-json", default="")
    parser.add_argument("--smoke-manifest-json", default="")
    parser.add_argument("--smoke-readiness-json", default="")
    parser.add_argument("--trainability-readiness-json", default="")
    parser.add_argument("--candidate-readiness-json", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
