#!/usr/bin/env python3
"""GOLDEN characterization test — locks the CEMENT V10 entry model's output behaviour.

Targets the exact failure mode this project keeps hitting: "fungerende kode endres ved et
uhell". If the cement bundle's WEIGHTS or the FORWARD code (architecture / ctx-fusion /
multi-TF assembly / heads / dims) change, the locked output checksum shifts and this test
fails loudly — before anything ships.

Deterministic by construction:
  - CPU forward (no cuDNN nondeterminism), fixed torch seed, eval()/no_grad().
  - Inputs shaped from the bundle's OWN metadata (so it self-adjusts to dims, not hardcoded).
  - Logits rounded before hashing → robust to last-bit float jitter.

The fasit (tests/golden/cement_entry_v10_fasit.json) is updated DELIBERATELY only when a
cement change is intended (via vedtak). A surprise diff here = an accidental regression.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent
PROJECT_STATE = REPO / "PROJECT_STATE_artifacts.json"
FASIT = Path(__file__).parent / "golden" / "cement_entry_v10_fasit.json"
SEED = 1234
BATCH = 64
ROUND = 4  # decimals kept before hashing logits


def _cement_bundle_dir() -> Path:
    """One-truth: resolve the ACTIVE V10 bundle from PROJECT_STATE_artifacts.json."""
    if not PROJECT_STATE.exists():
        pytest.skip("PROJECT_STATE_artifacts.json not found")
    d = json.loads(PROJECT_STATE.read_text())
    try:
        path = Path(d["active"]["v10_entry"]["path"])
    except (KeyError, TypeError):
        pytest.skip("active/v10_entry/path missing from PROJECT_STATE_artifacts.json")
    if not path.exists():
        pytest.skip(f"cement V10 bundle not on disk: {path}")
    return path


def _seeded_inputs(meta: dict) -> dict:
    torch.manual_seed(SEED)
    mtf = meta.get("multi_tf", {}) or {}
    seq_len = int(meta.get("seq_len", 96))
    seq_dim = int(meta.get("seq_input_dim", 41))
    snap_dim = int(meta.get("snap_input_dim", 41))
    cat_dim = int(meta.get("expected_ctx_cat_dim", meta.get("ctx_cat_dim", 6)))
    cont_dim = int(meta.get("expected_ctx_cont_dim", meta.get("ctx_cont_dim", 105)))

    kw = {
        "seq_x": torch.randn(BATCH, seq_len, seq_dim),
        "snap_x": torch.randn(BATCH, snap_dim),
        "ctx_cat": torch.zeros(BATCH, cat_dim, dtype=torch.long),  # category 0 — deterministic
        "ctx_cont": torch.randn(BATCH, cont_dim),
    }
    if mtf.get("enabled"):
        for tf in ("m5", "m15", "h1", "h4", "d1"):
            sl = int(mtf.get(f"{tf}_seq_len", seq_len))
            sd = int(mtf.get(f"{tf}_seq_dim", 25))
            kw[f"seq_{tf}"] = torch.randn(BATCH, sl, sd)
    return kw


def _characterize(bundle_dir: Path) -> dict:
    from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle

    bundle = load_entry_v10_ctx_bundle(bundle_dir=bundle_dir, device="cpu", xgb_models=None)
    model = bundle.transformer_model
    model.eval()
    meta = bundle.metadata or {}
    kw = _seeded_inputs(meta)

    with torch.no_grad():
        out = model(kw.pop("seq_x"), kw.pop("snap_x"),
                    ctx_cat=kw.pop("ctx_cat"), ctx_cont=kw.pop("ctx_cont"), **kw)

    logits = out["direction_logits"].cpu().numpy().astype(np.float64)
    logits_r = np.round(logits, ROUND)
    calls = logits.argmax(axis=1)
    n_classes = logits.shape[1]

    return {
        "bundle_name": bundle_dir.name,
        "state_dict_sha256": meta.get("state_dict_sha256"),
        "n": int(logits.shape[0]),
        "n_classes": int(n_classes),
        "logits_sha256": hashlib.sha256(logits_r.tobytes()).hexdigest(),
        "per_class_logit_mean": [round(float(x), ROUND) for x in logits.mean(axis=0)],
        "argmax_counts": [int((calls == c).sum()) for c in range(n_classes)],
    }


def test_golden_cement_entry_v10():
    bundle_dir = _cement_bundle_dir()
    got = _characterize(bundle_dir)

    if not FASIT.exists():
        FASIT.parent.mkdir(parents=True, exist_ok=True)
        FASIT.write_text(json.dumps(got, indent=2) + "\n")
        pytest.skip(f"fasit CREATED at {FASIT} — review and commit it; re-run to lock.")

    want = json.loads(FASIT.read_text())

    # Same cement bundle?
    assert got["bundle_name"] == want["bundle_name"], (
        f"ACTIVE V10 bundle changed: {want['bundle_name']} -> {got['bundle_name']}. "
        "If intentional (vedtak), regenerate the fasit."
    )
    assert got["state_dict_sha256"] == want["state_dict_sha256"], "bundle weights file changed"

    # Exact output lock — the regression tripwire.
    assert got["logits_sha256"] == want["logits_sha256"], (
        "CEMENT V10 OUTPUT CHANGED — weights or forward code drifted.\n"
        f"  argmax_counts:        {want['argmax_counts']} -> {got['argmax_counts']}\n"
        f"  per_class_logit_mean: {want['per_class_logit_mean']} -> {got['per_class_logit_mean']}\n"
        "If this change is intentional (vedtak), delete the fasit and re-run to relock."
    )
    assert got["argmax_counts"] == want["argmax_counts"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
