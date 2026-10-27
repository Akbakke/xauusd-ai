"""Signed midpoint-bps supervision for the existing 1/5/15 direction experiment.

No files, market readers, model construction or run authority live here. Scales
are fitted only on the caller's complete own TRAIN prefix, never later rows.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np
import torch

HORIZONS = (1, 5, 15)
MINUTE_NS = 60_000_000_000
SCHEMA = "gx1_direction_own_prefix_signed_bps_scale_v1"


def _require(ok, reason):
    if not ok:
        raise RuntimeError("NATIVE_DIRECTION_BPS_" + reason)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def require_direction_bps_scale(value):
    _require(isinstance(value, Mapping) and set(value) == {
        "schema_version", "horizons_minutes", "fit_partition", "prefix_rows", "cutoff_ns",
        "target_source", "parent_rows_sha256", "decision_ns_sha256", "eligible_targets_sha256",
        "counts", "sum_bps", "sumsq_bps", "rms_bps", "mean_bps", "contract_sha256"}, "SCALE_KEYS")
    plain = {k: v for k, v in value.items() if k != "contract_sha256"}
    _require(value["contract_sha256"] == _sha(plain), "SCALE_SEAL")
    _require(value["schema_version"] == SCHEMA and value["fit_partition"] == "train"
        and value["horizons_minutes"] == list(HORIZONS)
        and type(value["prefix_rows"]) is int and value["prefix_rows"] > 0
        and type(value["cutoff_ns"]) is int and 0 < value["cutoff_ns"] <= 2**63 - 1, "SCALE_SCOPE")
    ref = value["target_source"]
    _require(isinstance(ref, Mapping) and set(ref) == {"path", "sha256"}
        and type(ref["path"]) is str and ref["path"].startswith("/"), "TARGET_REFERENCE")
    for digest in [ref["sha256"], value["parent_rows_sha256"], value["decision_ns_sha256"],
                   value["eligible_targets_sha256"]]:
        _require(type(digest) is str and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest), "SCALE_DIGEST")
    for key in ("counts", "sum_bps", "sumsq_bps", "rms_bps", "mean_bps"):
        _require(type(value[key]) is list and len(value[key]) == 3, "SCALE_COORDINATES")
    n = np.asarray(value["counts"], np.float64)
    sums = np.asarray(value["sum_bps"], np.float64)
    ss = np.asarray(value["sumsq_bps"], np.float64)
    rms = np.asarray(value["rms_bps"], np.float64)
    means = np.asarray(value["mean_bps"], np.float64)
    _require(all(type(x) is int and 0 < x <= value["prefix_rows"] for x in value["counts"])
        and all(np.isfinite(x).all() for x in (sums, ss, rms, means))
        and np.all(ss > 0) and np.array_equal(rms, np.sqrt(ss / n))
        and np.array_equal(means, sums / n)
        and np.isfinite(rms.astype(np.float32)).all() and np.all(rms.astype(np.float32) > 0), "SCALE_VALUES")
    return json.loads(json.dumps(value, allow_nan=False))


def fit_direction_bps_scale(*, parent_rows, decision_ns, terminal_time_ns, observed,
                            sign, mid_return_bps, cutoff_ns, target_source, fit_partition):
    _require(fit_partition == "train", "TRAIN_ONLY")
    n = len(parent_rows)
    _require(type(cutoff_ns) is int and 0 < cutoff_ns <= 2**63 - 1, "CUTOFF")
    _require(isinstance(parent_rows, np.ndarray) and parent_rows.dtype == np.int64
        and parent_rows.shape == (n,) and n > 0 and np.all(parent_rows >= 0)
        and np.all(np.diff(parent_rows) > 0), "PREFIX_PARENTS")
    _require(isinstance(decision_ns, np.ndarray) and decision_ns.dtype == np.int64
        and decision_ns.shape == (n,) and np.all(decision_ns > 0)
        and np.all(np.diff(decision_ns) > 0) and np.all(decision_ns < cutoff_ns)
        and np.all(decision_ns <= 2**63 - 1 - 15 * MINUTE_NS), "PREFIX_CLOCK")
    _require(terminal_time_ns.dtype == np.int64 and terminal_time_ns.shape == (n, 3)
        and np.array_equal(terminal_time_ns, decision_ns[:, None]
            + np.asarray(HORIZONS, np.int64)[None, :] * MINUTE_NS), "EXACT_HORIZONS")
    _require(observed.dtype == np.bool_ and observed.shape == (n, 3)
        and sign.dtype == np.int8 and sign.shape == (n, 3)
        and mid_return_bps.dtype == np.float64 and mid_return_bps.shape == (n, 3)
        and np.isfinite(mid_return_bps).all() and np.all(mid_return_bps[~observed] == 0)
        and np.array_equal(np.sign(mid_return_bps).astype(np.int8), sign), "TARGET_DOMAIN")
    target = mid_return_bps.astype(np.float32)
    _require(np.isfinite(target).all() and np.array_equal(np.sign(target).astype(np.int8), sign),
        "FP32_TARGET_REPRESENTATION")
    eligible = observed & (terminal_time_ns < cutoff_ns)
    values = np.where(eligible, target.astype(np.float64), 0)
    counts = eligible.sum(axis=0, dtype=np.int64)
    sums = values.sum(axis=0, dtype=np.float64)
    ss = (values**2).sum(axis=0, dtype=np.float64)
    _require(np.all(counts > 0) and np.isfinite(ss).all() and np.all(ss > 0), "EMPTY_OR_ZERO_RMS")
    value = {"schema_version": SCHEMA, "horizons_minutes": list(HORIZONS), "fit_partition": "train",
        "prefix_rows": n, "cutoff_ns": cutoff_ns, "target_source": dict(target_source),
        "parent_rows_sha256": _array_sha(parent_rows.astype("<i8")),
        "decision_ns_sha256": _array_sha(decision_ns.astype("<i8")),
        "eligible_targets_sha256": _array_sha(values.astype("<f8")),
        "counts": counts.tolist(), "sum_bps": sums.tolist(), "sumsq_bps": ss.tolist(),
        "rms_bps": np.sqrt(ss / counts).tolist(), "mean_bps": (sums / counts).tolist()}
    value["contract_sha256"] = _sha(value)
    return require_direction_bps_scale(value)


def native_direction_bps_loss(logit_up, mean_return_bps, *, rms_bps, mid_return_bps,
                              sign, observed, decision_ns, terminal_time_ns, cutoff_ns):
    from gx1.contracts.entry_native_direction_research_v1 import (
        native_direction_binary_loss, native_direction_research_contract)
    direction = native_direction_binary_loss(logit_up, sign=sign, observed=observed,
        decision_ns=decision_ns, terminal_time_ns=terminal_time_ns, cutoff_ns=cutoff_ns)
    _require(isinstance(mean_return_bps, torch.Tensor) and mean_return_bps.dtype == torch.float32
        and mean_return_bps.shape == logit_up.shape and mean_return_bps.device == logit_up.device
        and bool(torch.isfinite(mean_return_bps).all()), "PREDICTION")
    _require(isinstance(rms_bps, torch.Tensor) and rms_bps.dtype == torch.float32
        and rms_bps.shape == (3,) and rms_bps.device == logit_up.device and not rms_bps.requires_grad
        and bool(torch.isfinite(rms_bps).all()) and bool((rms_bps > 0).all()), "RMS")
    _require(isinstance(mid_return_bps, torch.Tensor) and mid_return_bps.dtype == torch.float64
        and mid_return_bps.shape == logit_up.shape and mid_return_bps.device == logit_up.device
        and not mid_return_bps.requires_grad and bool(torch.isfinite(mid_return_bps).all())
        and bool((mid_return_bps[~observed] == 0).all())
        and torch.equal(torch.sign(mid_return_bps).to(torch.int8), sign), "TARGET")
    target = mid_return_bps.float()
    _require(bool(torch.isfinite(target).all()) and torch.equal(torch.sign(target).to(torch.int8), sign),
        "FP32_TARGET_REPRESENTATION")
    eligible = observed & (terminal_time_ns < cutoff_ns)
    counts = eligible.sum(dim=0)
    count = int(counts.sum())
    bps_loss = None
    raw_sse = normalized_sse = None
    if count:
        residual = mean_return_bps[eligible] - target[eligible]
        scales = rms_bps.expand_as(mean_return_bps)[eligible]
        normalized_sse = ((residual / scales)**2).sum()
        raw_sse = (residual.double()**2).sum()
        bps_loss = normalized_sse / count
        _require(bool(torch.isfinite(bps_loss)), "FINITE_LOSS")
    terms = [x for x in (direction["loss"], bps_loss) if x is not None]
    loss = sum(terms) if terms else None
    return {**direction, "loss": loss, "direction_loss": direction["loss"], "bps_loss": bps_loss,
        "update_allowed": bool(count), "bps_eligible_mask": eligible, "bps_eligible_cells": count,
        "bps_per_horizon_counts": counts.detach().cpu().tolist(),
        "bps_raw_sse": raw_sse, "bps_normalized_sse": normalized_sse,
        "research_objective_sha256": native_direction_research_contract(signed_bps=True)["contract_sha256"]}


def score_direction_bps(mean_return_bps, *, mid_return_bps, observed, terminal_time_ns,
                        cutoff_ns, scale):
    scale = require_direction_bps_scale(scale)
    _require(mean_return_bps.shape == mid_return_bps.shape == observed.shape == terminal_time_ns.shape
        and mean_return_bps.ndim == 2 and mean_return_bps.shape[1] == 3
        and np.isfinite(mean_return_bps).all() and np.isfinite(mid_return_bps).all()
        and observed.dtype == np.bool_ and terminal_time_ns.dtype == np.int64, "SCORE_INPUTS")
    mask = observed & (terminal_time_ns < cutoff_ns)
    rows = []
    for h, minute in enumerate(HORIZONS):
        target = mid_return_bps[mask[:, h], h].astype(np.float64)
        pred = mean_return_bps[mask[:, h], h].astype(np.float64)
        count = len(target)
        error = pred - target
        rows.append({"horizon_minutes": minute, "count": count,
            "sse_bps2": float(np.sum(error**2)), "absolute_error_sum_bps": float(np.sum(np.abs(error))),
            "mse_bps2": float(np.mean(error**2)) if count else None,
            "mae_bps": float(np.mean(np.abs(error))) if count else None,
            "prefix_scaled_mse": float(np.mean((error / scale["rms_bps"][h])**2)) if count else None,
            "mean_error_bps": float(np.mean(error)) if count else None,
            "zero_reference_sse_bps2": float(np.sum(target**2)),
            "own_prefix_mean_reference_sse_bps2": float(np.sum((target - scale["mean_bps"][h])**2))})
    return {"scope": "Gross midpoint conditional-mean error; no uncertainty, trading or economic claim.",
        "target_precision": "Original float64 saved bps for scoring; optimizer uses checked float32 targets.",
        "scale_contract_sha256": scale["contract_sha256"], "per_horizon": rows}
