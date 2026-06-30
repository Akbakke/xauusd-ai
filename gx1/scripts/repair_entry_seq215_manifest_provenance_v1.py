#!/usr/bin/env python3
"""Repair seq215 dataset manifest provenance fields without touching parquet data."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.contracts.signal_bridge_v3 import CONTRACT_SHA256_V3, SIGNAL_BRIDGE_ID_V3


DEFAULT_DATASET_DIR = Path(
    "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/"
    "v10_6yr_rebuild_20260628_foundation_seq146/"
    "v10_dataset_challenger_seq215_neutral_20260630"
)
DEFAULT_OUT_DIR = Path(
    "/home/andre2/GX1_DATA/reports/entry_seq215_manifest_provenance_repair_20260630_v1"
)


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


def _repair_manifest(path: Path, *, apply: bool) -> dict[str, Any]:
    before_sha = _sha256(path)
    manifest = _read_json(path)
    expected = _expected_from_manifest(manifest)
    changed_fields: dict[str, dict[str, Any]] = {}
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

    after_sha = before_sha
    if apply and changed_fields:
        _write_json(path, manifest)
        after_sha = _sha256(path)
    return {
        "path": str(path),
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "changed": bool(changed_fields),
        "changed_fields": changed_fields,
    }


def _repair_build_proof(path: Path, expected: dict[str, Any], *, apply: bool) -> dict[str, Any]:
    before_sha = _sha256(path)
    proof = _read_json(path)
    changes: dict[str, dict[str, Any]] = {}
    for key in ("ctx_tag", "ctx_cont_dim", "ctx_cat_dim", "signal_bridge_id", "signal_bridge_contract_sha256"):
        old = proof.get(key)
        new = expected.get(key)
        if old != new:
            changes[key] = {"old": old, "new": new}
            proof[key] = new
    after_sha = before_sha
    if apply and changes:
        _write_json(path, proof)
        after_sha = _sha256(path)
    return {
        "path": str(path),
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "changed": bool(changes),
        "changed_fields": changes,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = sorted(dataset_dir.glob("*.manifest.json"))
    if not manifests:
        raise RuntimeError(f"no split manifests found under {dataset_dir}")

    manifest_rows = [_repair_manifest(path, apply=bool(args.apply)) for path in manifests]
    first_payload = _read_json(manifests[0])
    expected = _expected_from_manifest(first_payload)
    build_proof_path = dataset_dir / "DATASET_BUILD_PROOF.json"
    build_proof_row = (
        _repair_build_proof(build_proof_path, expected, apply=bool(args.apply))
        if build_proof_path.exists()
        else {"path": str(build_proof_path), "changed": False, "missing": True}
    )
    failures: list[str] = []
    if expected["ctx_tag"] != "CTX6CAT5":
        failures.append(f"unexpected ctx_tag after repair planning: {expected['ctx_tag']}")
    if expected["ctx_cont_dim"] != 142 or expected["ctx_cat_dim"] != 5:
        failures.append(f"unexpected ctx dims after repair planning: {expected}")
    if expected["signal_bridge_id"] != SIGNAL_BRIDGE_ID_V3:
        failures.append(f"unexpected signal bridge id after repair planning: {expected['signal_bridge_id']}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "entry_seq215_manifest_provenance_repair_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "APPLIED" if args.apply and not failures else ("DRY_RUN_READY" if not failures else "FAIL"),
        "apply": bool(args.apply),
        "dataset_dir": str(dataset_dir),
        "expected_contract": expected,
        "manifest_repairs": manifest_rows,
        "build_proof_repair": build_proof_row,
        "parquet_data_modified": False,
        "failures": failures,
    }
    json_path = out_dir / f"ENTRY_SEQ215_MANIFEST_PROVENANCE_REPAIR_{stamp}.json"
    latest_path = out_dir / "ENTRY_SEQ215_MANIFEST_PROVENANCE_REPAIR_latest.json"
    _write_json(json_path, report)
    _write_json(latest_path, report)
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
