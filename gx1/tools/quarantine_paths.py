#!/usr/bin/env python3
"""
Quarantine paths under a root into a timestamped quarantine directory.
Includes tripwires to protect critical runtime and PREBUILT artifacts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
import shutil
import sys

def sha256_file(p: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _normalize_rel(path_str: str) -> str:
    return path_str.replace("\\", "/")

def _match_any(path_str: str, patterns: list[str]) -> bool:
    normalized = _normalize_rel(path_str)
    for pat in patterns:
        if fnmatch(normalized, pat):
            return True
    return False

def _load_paths_from_list(path: Path) -> list[str]:
    paths: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            paths.append(s)
    return paths

def _load_paths_from_inventory(inventory_path: Path, classes: set[str]) -> list[str]:
    data = json.loads(inventory_path.read_text(encoding="utf-8"))
    items = data.get("items", [])
    paths: list[str] = []
    for item in items:
        if item.get("classification") in classes:
            p = item.get("path")
            if p:
                paths.append(str(p))
    return paths

def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total
    return 0

def _load_master_lock_paths(root: Path, master_lock_path: Path | None) -> list[str]:
    if master_lock_path is None:
        return []
    if not master_lock_path.exists():
        return []
    try:
        lock = json.loads(master_lock_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    artifacts = lock.get("artifacts", [])
    protected: list[str] = []
    for item in artifacts:
        rel = item.get("path")
        if rel:
            protected.append(str(rel))
    return protected

def _build_forbidden_patterns(root_kind: str, root: Path, master_lock_path: Path | None) -> list[str]:
    forbidden = []
    if root_kind == "data":
        forbidden.extend([
            "MASTER_MODEL_LOCK.json",
            "data/data/entry_v10/*.parquet",
            "data/data/entry_v10/*manifest*",
            "data/data/entry_v10/*sha*",
            "data/data/entry_v10/*fingerprint*",
            "**/TRUTH_BASELINE_LOCK*",
        ])
        protected_from_lock = _load_master_lock_paths(root, master_lock_path)
        forbidden.extend([_normalize_rel(p) for p in protected_from_lock])
        active_bundle = root / "models" / "models" / "entry_v10_ctx" / "FULLYEAR_2025_GATED_FUSION"
        if active_bundle.exists():
            forbidden.append("models/models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION/**")
    if root_kind == "engine":
        from gx1.utils.truth_banlist import TRUTH_CANONICAL_POLICY_RELATIVE
        policy_glob = (Path(TRUTH_CANONICAL_POLICY_RELATIVE).parent.parent.parent / "**").as_posix()
        forbidden.extend([
            "gx1/execution/oanda_demo_runner.py",
            "gx1/execution/entry_manager.py",
            "gx1/execution/entry_feature_telemetry.py",
            "gx1/runtime/feature_contract_v10_ctx.py",
            "gx1/runtime/feature_fingerprint.py",
            "gx1/scripts/replay_eval_gated_parallel.py",
            "gx1/scripts/replay_worker.py",
            "gx1/scripts/run_phase1c_baseline_eval.py",
            "gx1/scripts/project_truth_scan.py",
            "gx1/scripts/repo_cleanup_inventory.py",
            "gx1/tools/quarantine_paths.py",
            policy_glob,
            "gx1/configs/xgb_session_policy.json",
            "MASTER_MODEL_LOCK.json",
        ])
    return forbidden

def _bucket_key(rel: str) -> str:
    normalized = _normalize_rel(rel)
    return normalized.split("/", 1)[0] if "/" in normalized else normalized

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Repo/data root where paths are relative from")
    ap.add_argument("--list", default=None, help="Text file with relative paths, one per line")
    ap.add_argument("--inventory-json", default=None, help="Inventory JSON with items + classification")
    ap.add_argument("--inventory-classes", default="UNREFERENCED,DANGEROUS", help="Comma list of classes to include")
    ap.add_argument("--quarantine-root", required=True, help="Where to place quarantined files")
    ap.add_argument("--tag", default=None, help="Optional tag (e.g. DEL2_UNREFERENCED)")
    ap.add_argument("--apply", action="store_true", help="Actually move files (default is dry-run)")
    ap.add_argument("--allow-missing", action="store_true", help="Skip missing files instead of failing")
    ap.add_argument("--root-kind", default="auto", choices=["auto", "data", "engine"], help="Tripwire mode")
    ap.add_argument("--forbid-pattern", action="append", default=[], help="Extra forbidden glob patterns")
    ap.add_argument("--allow-pattern", action="append", default=[], help="Override forbidden globs")
    ap.add_argument("--master-lock", default=None, help="Path to MASTER_MODEL_LOCK.json")
    ap.add_argument("--require-master-lock", action="store_true", help="Fail if MASTER_MODEL_LOCK is missing")
    ap.add_argument("--smoke-test", action="store_true", help="Run a quick preflight smoke test")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    qroot = Path(args.quarantine_root).resolve()
    master_lock_path = Path(args.master_lock).resolve() if args.master_lock else None

    if not root.exists():
        raise SystemExit(f"ROOT does not exist: {root}")

    if args.smoke_test:
        print(f"[SMOKE] root={root} quarantine_root={qroot}")
        return 0

    if args.list is None and args.inventory_json is None:
        raise SystemExit("Must provide --list or --inventory-json")

    if args.root_kind == "auto":
        if "GX1_DATA" in root.parts:
            root_kind = "data"
        elif "GX1_ENGINE" in root.parts:
            root_kind = "engine"
        else:
            root_kind = "auto"
    else:
        root_kind = args.root_kind

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or "QUARANTINE"
    out_dir = qroot / f"{tag}_{ts}"
    manifest_path = out_dir / "MANIFEST.json"

    paths: list[str] = []
    if args.list:
        paths.extend(_load_paths_from_list(Path(args.list)))
    if args.inventory_json:
        classes = {c.strip() for c in args.inventory_classes.split(",") if c.strip()}
        paths.extend(_load_paths_from_inventory(Path(args.inventory_json), classes))
    paths = [p for p in paths if p]

    if root_kind == "data" and "GX1_DATA" not in root.parts:
        raise SystemExit(f"Root kind=data but root is not under GX1_DATA: {root}")
    if root_kind == "engine" and "GX1_ENGINE" not in root.parts:
        raise SystemExit(f"Root kind=engine but root is not under GX1_ENGINE: {root}")

    if root_kind == "data" and args.require_master_lock:
        default_lock = root / "MASTER_MODEL_LOCK.json"
        if master_lock_path is None:
            master_lock_path = default_lock
        if not master_lock_path.exists():
            raise SystemExit(f"MASTER_MODEL_LOCK missing: {master_lock_path}")

    forbidden_patterns = _build_forbidden_patterns(root_kind, root, master_lock_path)
    forbidden_patterns.extend(args.forbid_pattern or [])
    allow_patterns = args.allow_pattern or []

    # Preflight: resolve + existence + target collisions + tripwires
    moves = []
    missing = []
    blocked = []
    for rel in paths:
        rel_norm = _normalize_rel(rel)
        if _match_any(rel_norm, forbidden_patterns) and not _match_any(rel_norm, allow_patterns):
            blocked.append({"rel": rel, "reason": "FORBIDDEN_PATTERN"})
            continue
        if rel_norm.endswith(".db") and not _match_any(rel_norm, allow_patterns):
            blocked.append({"rel": rel, "reason": "FORBIDDEN_DB"})
            continue

        src = (root / rel).resolve()
        if not str(src).startswith(str(root) + os.sep):
            raise SystemExit(f"Refuses path outside root: {rel} -> {src}")

        dst = (out_dir / rel).resolve()
        if not str(dst).startswith(str(out_dir) + os.sep):
            raise SystemExit(f"Bad dst resolution: {rel} -> {dst}")

        if not src.exists():
            missing.append({"rel": rel, "src": str(src)})
            continue

        if dst.exists():
            raise SystemExit(f"Target already exists (collision): {dst}")

        moves.append((rel, src, dst))

    if blocked:
        print("FATAL: blocked paths:", file=sys.stderr)
        for b in blocked[:50]:
            print(f"  - {b['rel']}  ({b['reason']})", file=sys.stderr)
        if len(blocked) > 50:
            print(f"  ... +{len(blocked)-50} more", file=sys.stderr)
        return 3

    if missing and not args.allow_missing:
        print("FATAL: missing files:", file=sys.stderr)
        for m in missing[:50]:
            print(f"  - {m['rel']}  ({m['src']})", file=sys.stderr)
        if len(missing) > 50:
            print(f"  ... +{len(missing)-50} more", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_counts: dict[str, int] = defaultdict(int)
    bucket_sizes: dict[str, int] = defaultdict(int)
    total_size = 0
    for rel, src, _ in moves:
        size = _path_size_bytes(src)
        total_size += size
        key = _bucket_key(rel)
        bucket_counts[key] += 1
        bucket_sizes[key] += size

    manifest = {
        "tag": tag,
        "created_utc": ts,
        "root": str(root),
        "quarantine_dir": str(out_dir),
        "apply": bool(args.apply),
        "total_listed": len(paths),
        "total_found": len(moves),
        "total_missing": len(missing),
        "total_size_bytes": total_size,
        "bucket_counts": dict(bucket_counts),
        "bucket_sizes_bytes": dict(bucket_sizes),
        "missing": missing,
        "moved": [],
    }

    # Dry-run prints what would happen
    if not args.apply:
        print(f"[DRY_RUN] root={root}")
        print(f"[DRY_RUN] quarantine_dir={out_dir}")
        print(f"[DRY_RUN] total_found={len(moves)} total_missing={len(missing)} total_size_bytes={total_size}")
        print("[DRY_RUN] buckets:")
        for key in sorted(bucket_counts.keys()):
            print(f"  - {key}: count={bucket_counts[key]} size_bytes={bucket_sizes[key]}")
        for rel, src, dst in moves[:200]:
            print(f"[DRY_RUN] MOVE {rel}")
            print(f"          {src} -> {dst}")
        if len(moves) > 200:
            print(f"[DRY_RUN] ... +{len(moves)-200} more")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[DRY_RUN] wrote manifest stub: {manifest_path}")
        return 0

    # Apply: move + hash
    for rel, src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        size = src.stat().st_size
        sha = sha256_file(src)
        shutil.move(str(src), str(dst))
        manifest["moved"].append({
            "rel": rel,
            "src": str(src),
            "dst": str(dst),
            "size_bytes": size,
            "sha256": sha,
        })

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    actions_path = out_dir / "QUARANTINE_ACTIONS.json"
    actions_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path = out_dir / "QUARANTINE_REPORT.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Quarantine Report\n\n")
        f.write(f"- tag: `{tag}`\n")
        f.write(f"- created_utc: `{ts}`\n")
        f.write(f"- root: `{root}`\n")
        f.write(f"- quarantine_dir: `{out_dir}`\n")
        f.write(f"- total_found: `{len(moves)}`\n")
        f.write(f"- total_missing: `{len(missing)}`\n")
        f.write(f"- total_size_bytes: `{total_size}`\n\n")
        f.write("## Buckets\n\n")
        for key in sorted(bucket_counts.keys()):
            f.write(f"- `{key}`: count={bucket_counts[key]} size_bytes={bucket_sizes[key]}\n")
        f.write("\n## Files (moved)\n\n")
        for item in manifest["moved"]:
            f.write(f"- `{item['rel']}` ({item['size_bytes']} bytes)\n")
    print(f"[OK] moved={len(manifest['moved'])} missing={len(missing)}")
    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] report: {report_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
