#!/usr/bin/env python3
"""Deterministic run-manifest builder + preconditions gate for GX1 experiments.

This is the mechanical half of the `/run-experiment` skill. It exists so that "we
thought we ran config X but actually ran Y" is impossible: every run is pinned to a
commit hash, a config hash, a dataset hash + row count, a checkpoint hash, a seed,
and a feature-set version — all written to disk next to the results.

Three non-negotiable principles (mirror CLAUDE.md):
  1. NO SILENT DEFAULTS  — `--config` is required; a missing/absent config is a hard
     error, never a guessed default. Same for `--seed` and `--feature-set`.
  2. HASH EVERYTHING     — anything that identifies the run is sha256'd (config,
     dataset, checkpoint) and recorded.
  3. GIT CLEAN           — refuses to emit a "real" manifest if the target repo's
     working tree is dirty (a dirty tree means we don't know what code ran).

Usage (preview, never blocks, never writes):
  run_manifest.py --dry-run --config <cfg> --seed 42 --feature-set <ver> [--dataset <p>] ...

Usage (real, writes manifest.json under <runs-root>/<utc>-<commit8>-<cfg8>/):
  run_manifest.py --config <cfg> --seed 42 --feature-set <ver> --dataset <p> \
                  --checkpoint <ckpt> --run-cmd "<the exact command>" --commit

Exit codes: 0 ok | 2 precondition failed (dirty git / missing config / bad input).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

DEFAULT_REPO = Path("/home/andre2/src/GX1_ENGINE")
DEFAULT_RUNS_ROOT = Path("/home/andre2/GX1_DATA/runs")
_CHUNK = 1 << 20  # 1 MiB


def _fail(msg: str) -> NoReturn:
    print(f"[run-manifest] PRECONDITION FAILED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        _fail(f"git {' '.join(args)} -> {out.stderr.strip() or out.stdout.strip()}")
    return out.stdout.rstrip("\n")


def _sha256_file(p: Path, mode: str) -> dict:
    """Content hash of a file. mode='full' streams the bytes; mode='meta' is a fast
    size+mtime fingerprint (use only when a multi-GB content hash is impractical)."""
    st = p.stat()
    if mode == "meta":
        h = hashlib.sha256(f"{p.name}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()
        return {"sha256": h, "hash_mode": "meta(size+mtime)", "size_bytes": st.st_size}
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "hash_mode": "full", "size_bytes": st.st_size}


def _dataset_rows(p: Path) -> int | None:
    suf = p.suffix.lower()
    try:
        if suf == ".parquet":
            import pyarrow.parquet as pq  # local import: only needed for parquet
            return int(pq.read_metadata(str(p)).num_rows)
        if suf in (".csv", ".tsv"):
            with p.open("rb") as fh:
                n = sum(1 for _ in fh)
            return max(0, n - 1)  # minus header
    except Exception as e:  # noqa: BLE001 — row-count is informative, not a gate
        print(f"[run-manifest] WARN: could not count rows of {p.name}: {e}", file=sys.stderr)
    return None


def _hash_artifact(label: str, raw: str | None, hash_mode: str, required: bool) -> dict | None:
    if raw is None:
        if required:
            _fail(f"--{label} is required (no silent default)")
        return None
    if not raw.strip():
        _fail(f"--{label} is empty (no silent default)")
    p = Path(raw)
    if not p.exists():
        _fail(f"--{label} path does not exist: {p}")
    if not p.is_file():
        _fail(f"--{label} must be a file, not a directory: {p}")
    info = {"path": str(p.resolve()), **_sha256_file(p, hash_mode)}
    if label == "dataset":
        info["rows"] = _dataset_rows(p)
    return info


def build_manifest(args: argparse.Namespace) -> dict:
    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists() and not (repo / ".git").is_file():
        _fail(f"--repo is not a git repo: {repo}")

    # ---- GATE: git must be clean (unless explicitly previewing a dirty tree) ----
    status = _git(repo, "status", "--short")
    dirty = bool(status.strip())
    if dirty and not args.dry_run:
        _fail(
            "git working tree is DIRTY — refusing to pin a real run to an unknown state.\n"
            f"  repo: {repo}\n  uncommitted:\n"
            + "\n".join(f"    {ln}" for ln in status.splitlines())
            + "\n  Commit/stash first, or use --dry-run to preview only."
        )

    commit = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")

    # ---- GATE: explicit identifying inputs, no silent defaults ----
    config = _hash_artifact("config", args.config, args.hash_mode, required=True)
    dataset = _hash_artifact("dataset", args.dataset, args.hash_mode, required=False)
    checkpoint = _hash_artifact("checkpoint", args.checkpoint, args.hash_mode, required=False)
    if args.seed is None:
        _fail("--seed is required (no silent default)")
    if not args.feature_set:
        _fail("--feature-set is required (no silent default)")

    manifest = {
        "manifest_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "repo": str(repo),
            "commit": commit,
            "commit_short": commit[:8],
            "branch": branch,
            "clean": not dirty,
            "dirty_preview": status.splitlines() if dirty else [],
        },
        "config": config,
        "dataset": dataset,
        "checkpoint": checkpoint,
        "seed": int(args.seed),
        "feature_set": args.feature_set,
        "run_cmd": args.run_cmd,
        "label": args.label,
    }
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="GX1 deterministic run-manifest builder")
    ap.add_argument("--config", help="REQUIRED: path to the run config (no silent default)")
    ap.add_argument("--dataset", help="dataset path (parquet/csv) — hashed + row-counted")
    ap.add_argument("--checkpoint", help="model checkpoint / bundle path — hashed")
    ap.add_argument("--seed", help="REQUIRED: random seed")
    ap.add_argument("--feature-set", dest="feature_set", help="REQUIRED: feature-set version id")
    ap.add_argument("--run-cmd", dest="run_cmd", help="the exact command this run will execute")
    ap.add_argument("--label", default="", help="freeform run label")
    ap.add_argument("--repo", default=str(DEFAULT_REPO), help="git repo to pin (default: main GX1_ENGINE)")
    ap.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT), help="where the run dir is created")
    ap.add_argument("--hash-mode", choices=["full", "meta"], default="full",
                    help="full=stream sha256 (default); meta=fast size+mtime for multi-GB files")
    ap.add_argument("--dry-run", action="store_true", help="build + print manifest, write nothing, allow dirty git")
    ap.add_argument("--commit", action="store_true", help="actually create the run dir + write manifest.json")
    args = ap.parse_args()

    manifest = build_manifest(args)
    blob = json.dumps(manifest, indent=2)
    print(blob)

    if args.dry_run:
        print("\n[run-manifest] DRY-RUN — nothing written.", file=sys.stderr)
        return 0
    if not args.commit:
        print("\n[run-manifest] built but not persisted (pass --commit to write the run dir).", file=sys.stderr)
        return 0

    run_dir = Path(args.runs_root) / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{manifest['git']['commit_short']}-{manifest['config']['sha256'][:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "manifest.json").write_text(blob + "\n")
    print(f"\n[run-manifest] WROTE {run_dir / 'manifest.json'}", file=sys.stderr)
    print(str(run_dir))  # last stdout line = the run dir, for the caller to chain into
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
