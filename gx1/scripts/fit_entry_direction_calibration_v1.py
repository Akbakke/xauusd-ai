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


def _fit_path_calibration(args, bundle_dir: Path, meta_path: Path, meta: dict) -> int:
    """Affine recal for path_quality_pred (vs realized path_quality_bps) + Platt
    for bad_path_prob (vs y_bad_path), fitted on the held-out split only."""
    if isinstance(meta.get("path_calibration"), dict):
        print("FATAL: bundle already carries path_calibration; refusing to re-fit on "
              "calibrated predictions. Remove the key (restore backup) first.", file=sys.stderr)
        return 2
    pred_path = Path(args.predictions_parquet).expanduser().resolve()
    cols = ["split", "model", "path_quality_pred", "bad_path_prob", "path_quality_bps", "y_bad_path"]
    frame = pd.read_parquet(pred_path, columns=cols)
    frame = frame[(frame["split"] == args.fit_split) & (frame["model"] == args.model_name)].dropna()
    if len(frame) < 200:
        print(f"FATAL: too few rows ({len(frame)}) for split={args.fit_split}", file=sys.stderr)
        return 2
    x_pq = frame["path_quality_pred"].to_numpy(np.float64)
    y_pq = frame["path_quality_bps"].to_numpy(np.float64)
    # affine least-squares
    A = np.vstack([x_pq, np.ones_like(x_pq)]).T
    (a, b), *_ = np.linalg.lstsq(A, y_pq, rcond=None)
    mse_before = float(np.mean((x_pq - y_pq) ** 2))
    mse_after = float(np.mean((a * x_pq + b - y_pq) ** 2))
    # Platt on bad_path logit
    p_bp = np.clip(frame["bad_path_prob"].to_numpy(np.float64), 1e-9, 1 - 1e-9)
    logit = np.log(p_bp / (1 - p_bp))
    y_bp = frame["y_bad_path"].to_numpy(np.float64)
    def _bce(params):
        t, c = np.exp(params[0]), params[1]
        z = logit / t + c
        pz = 1 / (1 + np.exp(-z))
        pz = np.clip(pz, 1e-12, 1 - 1e-12)
        return -np.mean(y_bp * np.log(pz) + (1 - y_bp) * np.log(1 - pz))
    bce_before = _bce(np.array([0.0, 0.0]))
    res = minimize(_bce, np.zeros(2), method="Nelder-Mead", options={"maxiter": 2000})
    bp_t, bp_b = float(np.exp(res.x[0])), float(res.x[1])
    bce_after = _bce(res.x)
    if not np.isfinite([a, b, bp_t, bp_b, mse_after, bce_after]).all() or bp_t <= 0:
        print(f"FATAL: invalid path fit a={a} b={b} T={bp_t} c={bp_b}", file=sys.stderr)
        return 2
    if mse_after > mse_before or bce_after > bce_before:
        print(f"FATAL: path calibration did not improve (mse {mse_before:.3f}->{mse_after:.3f}, "
              f"bce {bce_before:.5f}->{bce_after:.5f})", file=sys.stderr)
        return 2
    # sign sanity (the cand#1 lesson): recal never fixes sign — refuse if the raw
    # correlation with the realized target is non-positive.
    corr = float(np.corrcoef(x_pq, y_pq)[0, 1])
    if not (corr > 0.0):
        print(f"FATAL: path_quality_pred correlates non-positively with realized "
              f"path_quality_bps (r={corr:.3f}) — a SIGN defect; fix recipe/retrain, "
              "do not calibrate over it.", file=sys.stderr)
        return 2
    payload = {
        "enabled": True,
        "path_quality_scale": float(a),
        "path_quality_shift": float(b),
        "bad_path_temperature": bp_t,
        "bad_path_bias": bp_b,
        "fitted_on_split": args.fit_split,
        "fitted_rows": int(len(frame)),
        "path_quality_mse_before": mse_before,
        "path_quality_mse_after": mse_after,
        "bad_path_bce_before": bce_before,
        "bad_path_bce_after": bce_after,
        "path_quality_corr_raw": corr,
        "predictions_parquet": str(pred_path),
        "predictions_sha256": _sha256_file(pred_path),
        "fitted_utc": datetime.now(timezone.utc).isoformat(),
        "vedtak": args.vedtak,
        "note": ("Post-hoc PATH-head calibration (2026-07-05 leg; cand#1 inversion "
                 "lesson). Installed by load_entry_v10_ctx_bundle via "
                 "set_path_calibration; audit == serve by construction."),
    }
    print(json.dumps({"fit": payload}, indent=1))
    if args.dry_run:
        print("DRY-RUN: not writing bundle metadata")
        return 0
    backup = bundle_dir / "bundle_metadata.pre_path_cal.json"
    shutil.copy(meta_path, backup)
    meta["path_calibration"] = payload
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"WROTE path_calibration -> {meta_path} (backup: {backup})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--predictions-parquet", required=True,
                    help="selective_edge_predictions.parquet produced from the SAME bundle, pre-calibration")
    ap.add_argument("--fit-split", default="val", help="held-out split to fit on (never 'train')")
    ap.add_argument("--model-name", default="candidate")
    ap.add_argument("--vedtak", required=True)
    ap.add_argument(
        "--heads",
        choices=("direction", "path"),
        default="direction",
        help=(
            "direction (default): temperature+bias on direction logits. "
            "path: affine on path_quality_pred (least-squares vs path_quality_bps) + "
            "Platt on bad_path_prob (vs y_bad_path), written to "
            "bundle_metadata['path_calibration'] (2026-07-05 path-leg; cand#1 "
            "inversion lesson — recal fixes MAGNITUDE, never SIGN; wrong-sign "
            "stays a slice-audit hard fail)."
        ),
    )
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
    if args.heads == "path":
        return _fit_path_calibration(args, bundle_dir, meta_path, meta)
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
