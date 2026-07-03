"""Fit post-hoc direction calibration (temperature + per-class bias) for an
Entry V10 candidate bundle and write it into bundle_metadata.json.

One-truth calibration leg for the FLAT-rate non-stationarity finding
(vedtak SMART_SEQ520_candidate_train_20260703): the smart520 candidate
DISCRIMINATES direction well OOT but its FLAT prediction RATE drifts with the
market regime. This fits softmax(log(p)/T + b) by NLL on a recent held-out
split (never train) and stores the result in
bundle_metadata["direction_calibration"]; the bundle loader installs it into
the model's canonical forward so the bundle audit and live serve see identical
calibrated logits by construction.

Fail-closed: refuses to fit on the train split, refuses non-finite params,
backs up bundle_metadata.json before writing, and records full provenance
(fit split, rows, NLL before/after, predictions source, vedtak).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

CLASS_COLUMNS = ("p_long", "p_short", "p_flat")
LABEL_TO_INDEX = {0: 0, 1: 1, 2: 2}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _nll(params: np.ndarray, logp: np.ndarray, y: np.ndarray) -> float:
    temperature = float(np.exp(params[0]))
    bias = params[1:4]
    z = logp / temperature + bias
    z = z - z.max(axis=1, keepdims=True)
    prob = np.exp(z)
    prob /= prob.sum(axis=1, keepdims=True)
    return float(-np.log(prob[np.arange(len(y)), y] + 1e-12).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--predictions-parquet", required=True,
                    help="selective_edge_predictions.parquet produced from the SAME bundle, pre-calibration")
    ap.add_argument("--fit-split", default="val", help="held-out split to fit on (never 'train')")
    ap.add_argument("--model-name", default="candidate")
    ap.add_argument("--vedtak", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.fit_split == "train":
        print("FATAL: refusing to fit calibration on the train split", file=sys.stderr)
        return 2

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    meta_path = bundle_dir / "bundle_metadata.json"
    if not meta_path.exists():
        print(f"FATAL: missing {meta_path}", file=sys.stderr)
        return 2
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if isinstance(meta.get("direction_calibration"), dict):
        print("FATAL: bundle already carries direction_calibration; refusing to re-fit on "
              "calibrated predictions. Remove the key (restore backup) first.", file=sys.stderr)
        return 2

    pred_path = Path(args.predictions_parquet).expanduser().resolve()
    frame = pd.read_parquet(pred_path, columns=["split", "model", "y_direction", *CLASS_COLUMNS])
    frame = frame[(frame["split"] == args.fit_split) & (frame["model"] == args.model_name)]
    if frame.empty:
        print(f"FATAL: no rows for split={args.fit_split} model={args.model_name} in {pred_path}", file=sys.stderr)
        return 2

    probs = frame[list(CLASS_COLUMNS)].to_numpy(dtype=np.float64)
    y = frame["y_direction"].to_numpy(dtype=np.int64)
    if not np.isfinite(probs).all() or probs.min() < 0.0:
        print("FATAL: non-finite/negative probabilities in predictions", file=sys.stderr)
        return 2
    logp = np.log(np.clip(probs, 1e-12, 1.0))

    nll_before = _nll(np.array([0.0, 0.0, 0.0, 0.0]), logp, y)
    res = minimize(_nll, np.zeros(4), args=(logp, y), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-7})
    temperature = float(np.exp(res.x[0]))
    bias = [float(b) for b in res.x[1:4]]
    nll_after = _nll(res.x, logp, y)
    if not np.isfinite([temperature, *bias, nll_after]).all() or temperature <= 0.0:
        print(f"FATAL: fit produced invalid params T={temperature} bias={bias}", file=sys.stderr)
        return 2
    if nll_after > nll_before:
        print(f"FATAL: calibration did not improve NLL ({nll_before:.5f} -> {nll_after:.5f})", file=sys.stderr)
        return 2

    payload = {
        "enabled": True,
        "temperature": temperature,
        "bias": bias,
        "class_order": ["LONG", "SHORT", "FLAT"],
        "fitted_on_split": args.fit_split,
        "fitted_rows": int(len(y)),
        "nll_before": nll_before,
        "nll_after": nll_after,
        "predictions_parquet": str(pred_path),
        "predictions_sha256": _sha256_file(pred_path),
        "fitted_utc": datetime.now(timezone.utc).isoformat(),
        "vedtak": args.vedtak,
        "note": ("Post-hoc direction calibration (FLAT-rate non-stationarity leg). "
                 "Installed by load_entry_v10_ctx_bundle into the canonical forward; "
                 "audit == serve by construction."),
    }
    print(json.dumps({"fit": payload}, indent=1))
    if args.dry_run:
        print("DRY-RUN: not writing bundle metadata")
        return 0

    backup = bundle_dir / "bundle_metadata.pre_direction_cal.json"
    shutil.copy(meta_path, backup)
    meta["direction_calibration"] = payload
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"WROTE direction_calibration -> {meta_path} (backup: {backup})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
