#!/usr/bin/env python3
"""Repair Entry dataset manifest provenance fields without touching parquet data."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.contracts.signal_bridge_v3 import CONTRACT_SHA256_V3, SIGNAL_BRIDGE_ID_V3


DEFAULT_SEQ215_DATASET_DIR = Path(
    "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/"
    "v10_6yr_rebuild_20260628_foundation_seq146/"
    "v10_dataset_challenger_seq215_neutral_20260630"
)
DEFAULT_SEQ215_OUT_DIR = Path(
    "/home/andre2/GX1_DATA/reports/entry_seq215_manifest_provenance_repair_20260630_v1"
)
DEFAULT_FOUNDATION_DATASET_DIR = Path(
    "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/"
    "v10_6yr_rebuild_20260628_foundation_seq146/"
    "v10_dataset_foundation_seq146_neutral"
)
DEFAULT_FOUNDATION_OUT_DIR = Path(
    "/home/andre2/GX1_DATA/reports/entry_foundation_manifest_provenance_repair_20260701_v1"
)
DEFAULT_DATASET_DIR = DEFAULT_SEQ215_DATASET_DIR
DEFAULT_OUT_DIR = DEFAULT_SEQ215_OUT_DIR
DATASET_KINDS = ("challenger_seq215", "foundation_seq146")
DATASET_DIR_BY_KIND = {
    "challenger_seq215": DEFAULT_SEQ215_DATASET_DIR,
    "foundation_seq146": DEFAULT_FOUNDATION_DATASET_DIR,
}
OUT_DIR_BY_KIND = {
    "challenger_seq215": DEFAULT_SEQ215_OUT_DIR,
    "foundation_seq146": DEFAULT_FOUNDATION_OUT_DIR,
}
SCHEMA_BY_KIND = {
    "challenger_seq215": "entry_seq215_manifest_provenance_repair_v1",
    "foundation_seq146": "entry_foundation_manifest_provenance_repair_v1",
}
REPORT_PREFIX_BY_KIND = {
    "challenger_seq215": "ENTRY_SEQ215_MANIFEST_PROVENANCE_REPAIR",
    "foundation_seq146": "ENTRY_FOUNDATION_MANIFEST_PROVENANCE_REPAIR",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _expected_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    ctx = extra.get("ctx_contract") if isinstance(extra.get("ctx_contract"), dict) else {}
    sig = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
    ctx_cont_dim = int(ctx.get("ctx_cont_dim") or len(ctx.get("ctx_cont_names") or []))
    ctx_cat_dim = int(ctx.get("ctx_cat_dim") or len(ctx.get("ctx_cat_names") or []))
    return {
        "ctx_tag": str(ctx.get("tag") or f"CTX6CAT{ctx_cat_dim}"),
        "ctx_cont_dim": ctx_cont_dim,
        "ctx_cat_dim": ctx_cat_dim,
        "ctx_cont_base_dim": int(ctx.get("ctx_cont_base_dim") or 6),
        "signal_bridge_id": str(sig.get("signal_bridge_id") or sig.get("id") or SIGNAL_BRIDGE_ID_V3),
        "signal_bridge_contract_sha256": str(sig.get("contract_sha256") or CONTRACT_SHA256_V3),
        "signal_bridge_fields": [str(x) for x in (sig.get("fields") or [])],
    }


def _contract_anchor_failures(path: Path, manifest: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    feature_contract = manifest.get("feature_contract")
    if not isinstance(feature_contract, dict):
        return [f"{path}: feature_contract is not an object"]
    extra = manifest.get("extra")
    if not isinstance(extra, dict):
        return [f"{path}: extra is not an object"]
    ctx = extra.get("ctx_contract")
    if not isinstance(ctx, dict):
        failures.append(f"{path}: extra.ctx_contract is not an object")
    sig = extra.get("signal_bridge")
    if not isinstance(sig, dict):
        failures.append(f"{path}: extra.signal_bridge is not an object")
    if failures:
        return failures

    top_sha = str(feature_contract.get("signal_bridge_contract_sha256") or "")
    extra_sha = str(sig.get("contract_sha256") or "")
    expected_sha = str(expected.get("signal_bridge_contract_sha256") or "")
    if not top_sha or not extra_sha:
        failures.append(
            f"{path}: signal bridge contract sha missing; "
            f"feature_contract={top_sha!r} extra.signal_bridge={extra_sha!r}"
        )
    elif top_sha != extra_sha:
        failures.append(
            f"{path}: signal bridge contract sha mismatch; "
            f"feature_contract={top_sha!r} extra.signal_bridge={extra_sha!r}"
        )
    elif expected_sha != extra_sha:
        failures.append(
            f"{path}: expected signal bridge sha does not match extra.signal_bridge; "
            f"expected={expected_sha!r} extra.signal_bridge={extra_sha!r}"
        )

    if not str(ctx.get("tag") or "").strip():
        failures.append(f"{path}: extra.ctx_contract.tag missing")
    if int(expected.get("ctx_cont_dim") or 0) <= 0 or int(expected.get("ctx_cat_dim") or 0) <= 0:
        failures.append(f"{path}: extra.ctx_contract dimensions are not positive")
    if not str(sig.get("signal_bridge_id") or sig.get("id") or "").strip():
        failures.append(f"{path}: extra.signal_bridge id missing")
    return failures


def _plan_manifest_repair(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    before_sha = _sha256(path)
    manifest = _read_json(path)
    expected = _expected_from_manifest(manifest)
    failures = _contract_anchor_failures(path, manifest, expected)
    changed_fields: dict[str, dict[str, Any]] = {}
    if not failures:
        feature_contract = manifest.setdefault("feature_contract", {})
        if not isinstance(feature_contract, dict):
            raise RuntimeError(f"{path}: feature_contract is not an object")
        for key, value in expected.items():
            if key == "signal_bridge_fields" and not value:
                continue
            old = feature_contract.get(key)
            if old != value:
                changed_fields[f"feature_contract.{key}"] = {"old": old, "new": value}
                feature_contract[key] = value

        extra = manifest.setdefault("extra", {})
        if not isinstance(extra, dict):
            raise RuntimeError(f"{path}: extra is not an object")
        sig = extra.setdefault("signal_bridge", {})
        if not isinstance(sig, dict):
            raise RuntimeError(f"{path}: extra.signal_bridge is not an object")
        for key, value in (
            ("signal_bridge_id", expected["signal_bridge_id"]),
            ("ctx_cont_dim", expected["ctx_cont_dim"]),
            ("ctx_cat_dim", expected["ctx_cat_dim"]),
        ):
            old = sig.get(key)
            if old != value:
                changed_fields[f"extra.signal_bridge.{key}"] = {"old": old, "new": value}
                sig[key] = value

    row = {
        "path": str(path),
        "before_sha256": before_sha,
        "after_sha256": before_sha,
        "changed": bool(changed_fields),
        "changed_fields": changed_fields,
        "repair_allowed": not failures,
        "failures": failures,
    }
    return row, manifest if not failures else None


def _plan_build_proof_repair(path: Path, expected: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    before_sha = _sha256(path)
    proof = _read_json(path)
    failures: list[str] = []
    expected_sha = str(expected.get("signal_bridge_contract_sha256") or "")
    proof_sha = str(proof.get("signal_bridge_contract_sha256") or "")
    if not proof_sha:
        failures.append(f"{path}: DATASET_BUILD_PROOF.signal_bridge_contract_sha256 missing")
    elif proof_sha != expected_sha:
        failures.append(
            f"{path}: DATASET_BUILD_PROOF signal bridge sha mismatch; "
            f"proof={proof_sha!r} expected={expected_sha!r}"
        )
    changes: dict[str, dict[str, Any]] = {}
    if not failures:
        for key in ("ctx_tag", "ctx_cont_dim", "ctx_cat_dim", "signal_bridge_id", "signal_bridge_contract_sha256"):
            old = proof.get(key)
            new = expected.get(key)
            if old != new:
                changes[key] = {"old": old, "new": new}
                proof[key] = new
    row = {
        "path": str(path),
        "before_sha256": before_sha,
        "after_sha256": before_sha,
        "changed": bool(changes),
        "changed_fields": changes,
        "repair_allowed": not failures,
        "failures": failures,
    }
    return row, proof if not failures else None


def _apply_json_plan(path: Path, row: dict[str, Any], payload: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(row)
    if not out.get("changed"):
        return out
    if payload is None:
        out["repair_allowed"] = False
        out["failures"] = list(out.get("failures") or []) + [f"{path}: repair payload missing"]
        return out
    current_sha = _sha256(path)
    if current_sha != out["before_sha256"]:
        out["after_sha256"] = current_sha
        out["repair_allowed"] = False
        out["failures"] = list(out.get("failures") or []) + [
            f"{path}: file changed after repair planning; refusing apply"
        ]
        return out
    _write_json(path, payload)
    out["after_sha256"] = _sha256(path)
    return out


def _repair_manifest(path: Path, *, apply: bool) -> dict[str, Any]:
    row, payload = _plan_manifest_repair(path)
    return _apply_json_plan(path, row, payload) if apply and not row["failures"] else row


def _repair_build_proof(path: Path, expected: dict[str, Any], *, apply: bool) -> dict[str, Any]:
    row, payload = _plan_build_proof_repair(path, expected)
    return _apply_json_plan(path, row, payload) if apply and not row["failures"] else row


def _dataset_kind(raw: Any) -> str:
    kind = str(raw or "challenger_seq215")
    if kind == "seq215":
        kind = "challenger_seq215"
    if kind not in DATASET_KINDS:
        raise RuntimeError(f"unsupported dataset kind: {kind}")
    return kind


def run(args: argparse.Namespace) -> dict[str, Any]:
    kind = _dataset_kind(getattr(args, "dataset_kind", "challenger_seq215"))
    dataset_dir_arg = str(getattr(args, "dataset_dir", "") or "")
    out_dir_arg = str(getattr(args, "out_dir", "") or "")
    dataset_dir = Path(dataset_dir_arg or DATASET_DIR_BY_KIND[kind]).expanduser().resolve()
    out_dir = Path(out_dir_arg or OUT_DIR_BY_KIND[kind]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = sorted(dataset_dir.glob("*.manifest.json"))
    if not manifests:
        raise RuntimeError(f"no split manifests found under {dataset_dir}")

    manifest_plans = [_plan_manifest_repair(path) for path in manifests]
    manifest_rows = [row for row, _payload in manifest_plans]
    first_payload = _read_json(manifests[0])
    expected = _expected_from_manifest(first_payload)
    build_proof_path = dataset_dir / "DATASET_BUILD_PROOF.json"
    build_proof_payload: dict[str, Any] | None = None
    if build_proof_path.exists():
        build_proof_row, build_proof_payload = _plan_build_proof_repair(build_proof_path, expected)
    else:
        build_proof_row = {"path": str(build_proof_path), "changed": False, "missing": True, "failures": []}

    failures: list[str] = []
    for row in manifest_rows:
        failures.extend(str(item) for item in row.get("failures") or [])
    failures.extend(str(item) for item in build_proof_row.get("failures") or [])
    if expected["ctx_tag"] != "CTX6CAT5":
        failures.append(f"unexpected ctx_tag after repair planning: {expected['ctx_tag']}")
    if expected["ctx_cont_dim"] != 142 or expected["ctx_cat_dim"] != 5:
        failures.append(f"unexpected ctx dims after repair planning: {expected}")
    if expected["signal_bridge_id"] != SIGNAL_BRIDGE_ID_V3:
        failures.append(f"unexpected signal bridge id after repair planning: {expected['signal_bridge_id']}")

    if bool(args.apply) and not failures:
        for row in manifest_rows:
            path = Path(str(row["path"]))
            current_sha = _sha256(path)
            if current_sha != row["before_sha256"]:
                failures.append(f"{path}: file changed after repair planning; refusing apply")
        if build_proof_path.exists():
            current_sha = _sha256(build_proof_path)
            if current_sha != build_proof_row["before_sha256"]:
                failures.append(f"{build_proof_path}: file changed after repair planning; refusing apply")

    if bool(args.apply) and not failures:
        manifest_rows = [
            _apply_json_plan(Path(str(row["path"])), row, payload)
            for row, payload in manifest_plans
        ]
        if build_proof_path.exists():
            build_proof_row = _apply_json_plan(build_proof_path, build_proof_row, build_proof_payload)
        for row in manifest_rows:
            failures.extend(str(item) for item in row.get("failures") or [])
        failures.extend(str(item) for item in build_proof_row.get("failures") or [])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    any_changes = any(bool(row.get("changed")) for row in manifest_rows) or bool(build_proof_row.get("changed"))
    decision = "FAIL"
    if not failures:
        if bool(args.apply):
            decision = "APPLIED" if any_changes else "NOOP"
        else:
            decision = "DRY_RUN_READY" if any_changes else "PASS"
    report = {
        "schema_version": SCHEMA_BY_KIND[kind],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "dataset_kind": kind,
        "apply": bool(args.apply),
        "dataset_dir": str(dataset_dir),
        "expected_contract": expected,
        "manifest_repairs": manifest_rows,
        "build_proof_repair": build_proof_row,
        "parquet_data_modified": False,
        "failures": failures,
    }
    report_prefix = REPORT_PREFIX_BY_KIND[kind]
    json_path = out_dir / f"{report_prefix}_{stamp}.json"
    latest_path = out_dir / f"{report_prefix}_latest.json"
    _write_json(json_path, report)
    _write_json(latest_path, report)
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-kind", choices=DATASET_KINDS, default="challenger_seq215")
    ap.add_argument("--dataset-dir", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
