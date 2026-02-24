#!/usr/bin/env python3
raise RuntimeError("LEGACY_DISABLED: Use gx1/scripts/README_TRUTH_XGB.md (TRUTH XGB lane only).")
# -*- coding: utf-8 -*-
"""
TRUTH purge report (dry-run by default).

Purpose:
- List legacy truth artifacts (prebuilts/bundles) on disk.
- Produce a deterministic archive plan.
- Apply only with --apply: move candidates under:
    $GX1_DATA/_ARCHIVE_TRUTH_PURGE/<utc_ts>/...

Hard rules:
- No network calls.
- Default mode is report-only.
- In --apply, we MOVE to archive (no hard delete).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _utc_ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_gx1_data_root() -> Path:
    gx1_data_env = os.environ.get("GX1_DATA_DIR") or os.environ.get("GX1_DATA_ROOT") or os.environ.get("GX1_DATA")
    if not gx1_data_env:
        raise RuntimeError("[TRUTH_PURGE] GX1_DATA root env missing (set GX1_DATA_DIR or GX1_DATA_ROOT or GX1_DATA)")
    p = Path(gx1_data_env).expanduser().resolve()
    if p.name != "GX1_DATA":
        raise RuntimeError(f"[TRUTH_PURGE] GX1_DATA root must end with 'GX1_DATA', got: {p}")
    return p


@dataclass(frozen=True)
class ArtifactHit:
    kind: str  # "bundle" | "prebuilt"
    path: str
    feature_contract_id: Optional[str]
    feature_list_sha256: Optional[str]
    notes: Optional[str] = None


def _scan_bundles(entry_root: Path) -> List[ArtifactHit]:
    hits: List[ArtifactHit] = []
    if not entry_root.exists():
        return hits
    for lock_path in sorted(entry_root.glob("**/MASTER_MODEL_LOCK.json")):
        try:
            lock = _read_json(lock_path)
            hits.append(
                ArtifactHit(
                    kind="bundle",
                    path=str(lock_path.parent.resolve()),
                    feature_contract_id=str(lock.get("feature_contract_id") or "") or None,
                    feature_list_sha256=str(lock.get("feature_list_sha256") or "") or None,
                    notes=str(lock.get("bundle_arm") or "") or None,
                )
            )
        except Exception:
            hits.append(ArtifactHit(kind="bundle", path=str(lock_path.parent.resolve()), feature_contract_id=None, feature_list_sha256=None, notes="unreadable_lock"))
    return hits


def _scan_prebuilts(prebuilt_root: Path) -> List[ArtifactHit]:
    hits: List[ArtifactHit] = []
    if not prebuilt_root.exists():
        return hits
    for schema_path in sorted(prebuilt_root.glob("**/*.schema_manifest.json")):
        try:
            schema = _read_json(schema_path)
            hits.append(
                ArtifactHit(
                    kind="prebuilt",
                    path=str(Path(str(schema.get("output_parquet") or schema_path.with_suffix(".parquet"))).expanduser().resolve()),
                    feature_contract_id=str(schema.get("feature_contract_id") or "") or None,
                    feature_list_sha256=str(schema.get("feature_list_sha256") or "") or None,
                    notes=str(schema_path.resolve()),
                )
            )
        except Exception:
            hits.append(ArtifactHit(kind="prebuilt", path=str(schema_path.resolve()), feature_contract_id=None, feature_list_sha256=None, notes="unreadable_schema_manifest"))
    return hits


def _is_legacy_contract(contract_id: Optional[str]) -> bool:
    if not contract_id:
        return True
    c = contract_id.lower()
    if "v10" in c or "v12" in c or "v13_core" in c or "core" == c:
        return True
    # Treat non-refined3 as legacy for purge reporting purposes.
    return "refined3" not in c


def _build_archive_plan(
    *,
    hits: List[ArtifactHit],
    canonical_bundle_dir: Optional[Path],
    canonical_prebuilt_parquet: Optional[Path],
    archive_root: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    keep: List[Dict[str, Any]] = []
    move: List[Dict[str, Any]] = []
    for h in hits:
        p = Path(h.path).expanduser().resolve()
        is_keep = False
        if canonical_bundle_dir and h.kind == "bundle":
            if p == canonical_bundle_dir:
                is_keep = True
        if canonical_prebuilt_parquet and h.kind == "prebuilt":
            if p == canonical_prebuilt_parquet:
                is_keep = True

        entry = {
            "kind": h.kind,
            "path": str(p),
            "feature_contract_id": h.feature_contract_id,
            "feature_list_sha256": h.feature_list_sha256,
            "notes": h.notes,
            "legacy": _is_legacy_contract(h.feature_contract_id),
        }
        if is_keep:
            entry["action"] = "KEEP_CANONICAL"
            keep.append(entry)
            continue

        if entry["legacy"]:
            rel = str(p).lstrip("/")
            dst = (archive_root / rel).resolve()
            entry["action"] = "ARCHIVE_MOVE"
            entry["archive_dst"] = str(dst)
            move.append(entry)
        else:
            entry["action"] = "KEEP_NON_LEGACY"
            keep.append(entry)
    return keep, move


def _apply_moves(move_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for item in move_plan:
        src = Path(str(item["path"])).resolve()
        dst = Path(str(item["archive_dst"])).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            results.append({"path": str(src), "status": "SKIP_MISSING"})
            continue
        # Move file or directory
        shutil.move(str(src), str(dst))
        results.append({"path": str(src), "status": "MOVED", "dst": str(dst)})
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-bundle-dir", type=str, default="", help="Canonical XGB bundle dir to KEEP (absolute)")
    ap.add_argument("--canonical-prebuilt-parquet", type=str, default="", help="Canonical prebuilt parquet to KEEP (absolute)")
    ap.add_argument("--apply", action="store_true", help="Apply archive moves (default: report only)")
    ap.add_argument("--out", type=str, default="", help="Write report JSON to this path (absolute). If empty, prints JSON only.")
    args = ap.parse_args()

    gx1_data_root = _resolve_gx1_data_root()
    archive_root = (gx1_data_root / "_ARCHIVE_TRUTH_PURGE" / _utc_ts_compact()).resolve()

    canonical_bundle_dir = Path(args.canonical_bundle_dir).expanduser().resolve() if args.canonical_bundle_dir else None
    canonical_prebuilt = Path(args.canonical_prebuilt_parquet).expanduser().resolve() if args.canonical_prebuilt_parquet else None

    # Scan targets (bounded to GX1_DATA)
    bundles_root = gx1_data_root / "models" / "models" / "entry_v10_ctx"
    prebuilts_root = gx1_data_root / "data" / "data" / "prebuilt"
    hits = _scan_bundles(bundles_root) + _scan_prebuilts(prebuilts_root)

    keep, move = _build_archive_plan(
        hits=hits,
        canonical_bundle_dir=canonical_bundle_dir,
        canonical_prebuilt_parquet=canonical_prebuilt,
        archive_root=archive_root,
    )

    report = {
        "utc_ts": datetime.now(timezone.utc).isoformat(),
        "gx1_data_root": str(gx1_data_root),
        "archive_root": str(archive_root),
        "canonical_keep": {
            "bundle_dir": str(canonical_bundle_dir) if canonical_bundle_dir else None,
            "prebuilt_parquet": str(canonical_prebuilt) if canonical_prebuilt else None,
        },
        "counts": {
            "hits_total": len(hits),
            "keep_total": len(keep),
            "move_total": len(move),
        },
        "keep": keep,
        "move_plan": move,
        "apply": bool(args.apply),
        "apply_results": None,
    }

    if args.apply:
        report["apply_results"] = _apply_moves(move)

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        if not out_path.is_absolute():
            raise RuntimeError(f"--out must be absolute (got: {out_path})")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

