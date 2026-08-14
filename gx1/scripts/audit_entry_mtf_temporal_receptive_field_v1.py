#!/usr/bin/env python3
"""Deterministically inventory receptive-field migration callsites.

This is read-only.  It records the current owner, field plumbing and frozen
legacy literals that an eventual atomic migration must update or rebuild.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from gx1.contracts.entry_mtf_temporal_receptive_field_policy_v1 import (
    canonical_json_sha256,
    temporal_receptive_field_policy,
)


SCHEMA_VERSION = "gx1_entry_mtf_temporal_migration_surface_audit_v1"
DECISION = "MIGRATION_SURFACE_INVENTORIED_NOT_INTEGRATED"
SOURCE_ROOTS = ("gx1", "scripts", "tests")
SOURCE_SUFFIXES = (".py", ".sh")
EXCLUDED_PATHS = (
    "gx1/contracts/entry_mtf_temporal_receptive_field_policy_v1.py",
    "gx1/scripts/audit_entry_mtf_temporal_receptive_field_v1.py",
    "gx1/scripts/benchmark_entry_mtf_temporal_receptive_field_v1.py",
    "tests/test_entry_mtf_temporal_receptive_field_policy_v1.py",
)
FIELD_TOKENS = (
    "PRODUCTION_MTF_PER_TF_WINDOW_BARS",
    "per_tf_seq_lens",
    "per_tf_seq_len_m5",
    "per_tf_seq_len_m15",
    "per_tf_seq_len_h1",
    "per_tf_seq_len_h4",
    "per_tf_seq_len_d1",
    "m5_seq_len",
    "m15_seq_len",
    "h1_seq_len",
    "h4_seq_len",
    "d1_seq_len",
    "--per-tf-seq-len-m5",
    "--per-tf-seq-len-m15",
    "--per-tf-seq-len-h1",
    "--per-tf-seq-len-h4",
    "--per-tf-seq-len-d1",
    "PER_TF_SEQ_LEN_M5",
    "PER_TF_SEQ_LEN_M15",
    "PER_TF_SEQ_LEN_H1",
    "PER_TF_SEQ_LEN_H4",
    "PER_TF_SEQ_LEN_D1",
)
_LEGACY_LITERAL_PATTERNS = {
    "legacy_literal_m5_16": re.compile(
        r"(?:[\"']M5[\"']|m5_seq_len|per_tf_seq_len_m5)[^\n]{0,40}(?:[:=,]|default=)\s*16\b"
    ),
    "legacy_literal_m15_64": re.compile(
        r"(?:[\"']M15[\"']|m15_seq_len|per_tf_seq_len_m15)[^\n]{0,40}(?:[:=,]|default=)\s*64\b"
    ),
    "legacy_literal_h1_96": re.compile(
        r"(?:[\"']H1[\"']|h1_seq_len|per_tf_seq_len_h1)[^\n]{0,40}(?:[:=,]|default=)\s*96\b"
    ),
    "legacy_literal_h4_96": re.compile(
        r"(?:[\"']H4[\"']|h4_seq_len|per_tf_seq_len_h4)[^\n]{0,40}(?:[:=,]|default=)\s*96\b"
    ),
    "legacy_literal_d1_252": re.compile(
        r"(?:[\"']D1[\"']|d1_seq_len|per_tf_seq_len_d1)[^\n]{0,40}(?:[:=,]|default=)\s*252\b"
    ),
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_evidence(line: str) -> tuple[list[str], list[str]]:
    fields = [token for token in FIELD_TOKENS if token in line]
    literals = [
        token for token, pattern in _LEGACY_LITERAL_PATTERNS.items()
        if pattern.search(line)
    ]
    return fields, literals


def audit_temporal_receptive_field_migration_surface(
    repository_root: Path,
) -> dict[str, Any]:
    root = repository_root.expanduser().resolve()
    if not (root / "gx1").is_dir() or not (root / "tests").is_dir():
        raise RuntimeError("TEMPORAL_MIGRATION_AUDIT_REPOSITORY_INVALID")
    source_paths: list[Path] = []
    for source_root in SOURCE_ROOTS:
        base = root / source_root
        if not base.exists():
            continue
        source_paths.extend(
            path for path in base.rglob("*")
            if path.is_file()
            and path.suffix in SOURCE_SUFFIXES
            and path.relative_to(root).as_posix() not in EXCLUDED_PATHS
        )
    source_paths = sorted(set(source_paths))
    hits: list[dict[str, Any]] = []
    matched_files: dict[str, str] = {}
    for path in source_paths:
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        file_sha = _file_sha256(path)
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            fields, literals = _line_evidence(raw_line)
            if not fields and not literals:
                continue
            matched_files[relative] = file_sha
            if literals:
                kind = "legacy_exact_literal"
            elif "PRODUCTION_MTF_PER_TF_WINDOW_BARS" in fields:
                kind = "window_owner_or_derived_owner_reference"
            else:
                kind = "per_timeframe_length_field_plumbing"
            normalized = raw_line.strip()
            hits.append(
                {
                    "path": relative,
                    "line": line_number,
                    "kind": kind,
                    "field_tokens": fields,
                    "legacy_literal_tokens": literals,
                    "line_sha256": hashlib.sha256(
                        normalized.encode("utf-8")
                    ).hexdigest(),
                }
            )
    if not hits:
        raise RuntimeError("TEMPORAL_MIGRATION_AUDIT_EMPTY")
    counts = Counter(row["kind"] for row in hits)
    policy = temporal_receptive_field_policy()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "integrated": False,
        "policy_contract_sha256": policy["contract_sha256"],
        "source_roots": list(SOURCE_ROOTS),
        "source_suffixes": list(SOURCE_SUFFIXES),
        "excluded_paths": list(EXCLUDED_PATHS),
        "source_files_scanned": len(source_paths),
        "matched_file_count": len(matched_files),
        "matched_files": matched_files,
        "matched_files_sha256": canonical_json_sha256(matched_files),
        "hit_counts_by_kind": dict(sorted(counts.items())),
        "migration_hits": hits,
        "migration_hits_sha256": canonical_json_sha256(hits),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def write_immutable_audit(path: Path, report: dict[str, Any]) -> Path:
    output = path.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"TEMPORAL_MIGRATION_AUDIT_OUTPUT_EXISTS: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(output, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_temporal_receptive_field_migration_surface(
        args.repository_root
    )
    if args.output is not None:
        write_immutable_audit(args.output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = (
    "DECISION",
    "EXCLUDED_PATHS",
    "FIELD_TOKENS",
    "SCHEMA_VERSION",
    "audit_temporal_receptive_field_migration_surface",
    "write_immutable_audit",
)
