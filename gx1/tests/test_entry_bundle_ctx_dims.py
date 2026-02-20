#!/usr/bin/env python3
"""
Guardrail: canonical entry bundle must declare ctx dims 6/6 (no silent drift).

Reads GX1_CANONICAL_TRUTH_FILE (or canonical_truth_signal_only.json), resolves
canonical_transformer_bundle_dir, and verifies bundle_metadata.json has
ctx_cont_dim=6 and ctx_cat_dim=6 (or expected_* equivalents).
Does not load torch checkpoint (metadata only).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _truth_path() -> Path:
    path = os.environ.get("GX1_CANONICAL_TRUTH_FILE")
    if path:
        return Path(path)
    root = Path(__file__).resolve().parents[2]
    return root / "gx1" / "configs" / "canonical_truth_signal_only.json"


def _get_canonical_entry_bundle_dir() -> Path:
    truth_path = _truth_path()
    if not truth_path.exists():
        pytest.skip(f"Canonical truth not found: {truth_path}")
    with open(truth_path, "r", encoding="utf-8") as f:
        truth = json.load(f)
    bundle_dir = truth.get("canonical_transformer_bundle_dir") or ""
    if not bundle_dir:
        pytest.skip("canonical_transformer_bundle_dir not set in truth")
    return Path(bundle_dir)


def _get_bundle_metadata(bundle_dir: Path) -> dict:
    meta_path = bundle_dir / "bundle_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Bundle metadata not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_canonical_entry_bundle_ctx_dims_6_6():
    """Canonical entry bundle metadata must declare ctx_cont_dim=6 and ctx_cat_dim=6."""
    bundle_dir = _get_canonical_entry_bundle_dir()
    if not bundle_dir.exists():
        pytest.skip(f"Bundle dir not present (GX1_DATA): {bundle_dir}")
    meta = _get_bundle_metadata(bundle_dir)
    ctx_cont = meta.get("expected_ctx_cont_dim") or meta.get("ctx_cont_dim")
    ctx_cat = meta.get("expected_ctx_cat_dim") or meta.get("ctx_cat_dim")
    assert ctx_cont == 6, (
        f"Entry bundle must have ctx_cont_dim=6 (no drift). "
        f"Got ctx_cont_dim={ctx_cont} in {bundle_dir}"
    )
    assert ctx_cat == 6, (
        f"Entry bundle must have ctx_cat_dim=6 (no drift). "
        f"Got ctx_cat_dim={ctx_cat} in {bundle_dir}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
