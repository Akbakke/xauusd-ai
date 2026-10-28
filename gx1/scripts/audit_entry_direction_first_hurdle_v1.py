"""Fixed fold1 first-hurdle label and saved-direction rank audit.

Synthetic qualification is separate from the one explicitly admitted capped
TRAIN-development run. No model, optimizer, CONTROL, TEST or trading is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.contracts import entry_direction_block_rank_v1 as rank
from gx1.contracts.entry_causal_m1_first_positive_quote_v1 import (
    classify_first_quote_hurdle_direction, query_first_positive_quote,
)
from gx1.contracts.entry_causal_m1_outcomes_v1 import (
    DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT, build_dynamic_m1_fill_surface,
    prepare_dynamic_m1_prefix_index,
)
from gx1.contracts.entry_native_direction_context_report_v1 import CONTEXT_SPEC
from gx1.scripts import compare_entry_direction_saved_bps_v1 as saved
from gx1.scripts.compare_entry_direction_matched_exposure_v1 import _read_prediction

MINUTE = 60_000_000_000
DAY = 86_400_000_000_000
LATER_END = 1743465600000000000  # 2025-04-01 UTC, not CONTROL 2026
ROWS = 17288
FIRST_PARENT = 47602
HORIZONS = np.arange(1, 61, dtype=np.int64)
HURDLE_BPS = 10.0  # Previously used research hurdle, not a fitted trade rule.
REPLICATES = 10000
ALPHA = .05 / 2  # Fixed two clock-block schemes for the sole primary AUC.
STAGE = "direction_first_hurdle_oof_actual_20261028_001"
ALIGNMENT_STAGE = "direction_first_hurdle_endpoint_alignment_actual_20261028_001"
REFERENCE_STAGE = "direction_first_hurdle_reference_actual_20261028_001"
PREVIOUS = {"path": "/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2/DIRECTION_FIRST_HURDLE_OOF_ACTUAL_20261028_001/RESULT.json",
            "sha256": "bb46f1c5321ae77cf4853dd69dd4772629ea8a5217e4753285bc1314b39fa897"}
QUOTE = {"path": "/home/andre2/GX1_DATA/data/native_xau/XAU_M1_NATIVE_2009_20260701_PAIR_WEEKCLOSED_20260927/year=2025/part-000.parquet",
         "sha256": "1746d0c3c1bd13fa3d19c628fccea05dfe4715e681fcba9aa1b5a559961308dc",
         "rows": 353499, "offset": 5431388,
         "first": "2025-01-01T23:04:00+00:00", "last": "2025-12-31T21:59:00+00:00"}
TARGET = {"path": "/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_BASELINE_DATA_20261024_245/OUTPUT/PRICE_TARGETS.npz",
          "sha256": "9d4a5fb3c473b088aff5a5d5a176c199a64053b679229c14f125697b93af577d"}
PREDICTION = {"path": "/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_PREFIX_FIT_FOLD1_20261025_276/RUNTIME/prefix-fit/window-0002/OOF.npz",
              "sha256": "fd7d3177f1a4d588f235156a2dbc02d670137ab91862e779606bb932e0e4ef57"}
REFERENCE = {"path": "/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_EXPLICIT_PREVIOUS_RETURN_REFERENCE_20261024_259/OUTPUT/OOF_PREDICTIONS.npz",
             "sha256": "b42431e44ed531c4f5e225bf3480efacc3be7d698926c24038cd921b8c8e3bd1"}
REFERENCE_MEMBERS = ("parent_rows", "decision_ns", "terminal_time_ns", "observed", "sign", "fold_id",
                     "holdout_end_ns_exclusive", "saved_ET_FULL_probability_up")
TARGET_MEMBERS = ("parent_rows", "bar_start_ns", "dynamic_entry_m1_row", "decision_ns", "terminal_time_ns",
                  "elapsed_wall_seconds", "terminal_source_row", "observed", "mid_move", "mid_return_bps", "sign")
QUOTE_COLUMNS = ("bid_open", "bid_high", "bid_low", "ask_open", "ask_high", "ask_low")


def require(ok, reason):
    if not ok:
        raise RuntimeError("FIRST_HURDLE_AUDIT_" + reason)


def race_panel(index, fill, *, chunk_parents=128):
    """Query every 1..60 closed-minute horizon without a dense price path."""
    n = len(fill)
    require(n > 0 and type(chunk_parents) is int and 0 < chunk_parents <= 1024, "QUERY_SCOPE")
    outcome = np.zeros((n, 60), np.int8)
    observed = np.zeros((n, 60), np.bool_)
    reasons = np.zeros((n, 60), np.uint8)
    possibilities = np.zeros((n, 60), np.uint8)
    for left in range(0, n, chunk_parents):
        right = min(n, left + chunk_parents)
        entry_rows = np.repeat(np.arange(left, right, dtype=np.int64), 60)
        horizons = np.tile(HORIZONS, right - left)
        events = query_first_positive_quote(index, fill_surface=fill, query_entry_rows=entry_rows,
            horizon_minutes=horizons, event_basis="gross_quote_markout",
            endpoint_kind=DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT, favorable_bps=HURDLE_BPS)
        classified = classify_first_quote_hurdle_direction(events)
        shape = (right - left, 60)
        outcome[left:right] = classified["outcome"].reshape(shape)
        observed[left:right] = classified["outcome_observed"].reshape(shape)
        reasons[left:right] = classified["unknown_reason"].reshape(shape)
        allowed = classified["allowed_outcomes"]
        bits = (allowed[:, 0].astype(np.uint8) + 2 * allowed[:, 1].astype(np.uint8)
                + 4 * allowed[:, 2].astype(np.uint8))
        possibilities[left:right] = bits.reshape(shape)
    require(np.all((possibilities >= 1) & (possibilities <= 7))
            and np.array_equal(observed, np.isin(possibilities, (1, 2, 4)))
            and np.array_equal(observed, reasons == 0), "RACE_PANEL_INVARIANT")
    return {"outcome": outcome, "observed": observed,
            "reason": reasons, "possibilities": possibilities}


def race_five(index, fill):
    """Recompute only the predeclared five-minute first-hurdle outcomes."""
    n = len(fill)
    require(n > 0, "FIVE_SCOPE")
    events = query_first_positive_quote(index, fill_surface=fill,
        query_entry_rows=np.arange(n, dtype=np.int64),
        horizon_minutes=np.full(n, 5, dtype=np.int64), event_basis="gross_quote_markout",
        endpoint_kind=DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT, favorable_bps=HURDLE_BPS)
    return classify_first_quote_hurdle_direction(events)


def endpoint_alignment(race, endpoint_sign, endpoint_observed, logit_up_5):
    """Compare first-hit and endpoint signs on the same observed parents."""
    y, known = race["outcome"], race["outcome_observed"]
    require(isinstance(y, np.ndarray) and y.dtype == np.int8 and y.ndim == 1
            and isinstance(known, np.ndarray) and known.dtype == np.bool_ and known.shape == y.shape
            and isinstance(endpoint_sign, np.ndarray) and endpoint_sign.dtype == np.int8
            and endpoint_sign.shape == y.shape and np.isin(endpoint_sign, (-1, 0, 1)).all()
            and isinstance(endpoint_observed, np.ndarray) and endpoint_observed.dtype == np.bool_
            and endpoint_observed.shape == y.shape and isinstance(logit_up_5, np.ndarray)
            and logit_up_5.shape == y.shape and np.isfinite(logit_up_5).all(), "ALIGNMENT_INPUT")
    chosen = known & (y != 0)
    table = [[int((chosen & (y == direction) & endpoint_observed & (endpoint_sign == sign)).sum())
              for sign in (-1, 0, 1)] for direction in (1, -1)]
    missing = [int((chosen & (y == direction) & ~endpoint_observed).sum()) for direction in (1, -1)]
    same = chosen & endpoint_observed & (endpoint_sign == y)
    opposite = chosen & endpoint_observed & (endpoint_sign == -y)
    zero = chosen & endpoint_observed & (endpoint_sign == 0)
    require(int(chosen.sum()) == int(same.sum() + opposite.sum() + zero.sum() + sum(missing)),
            "ALIGNMENT_PARTITION")
    common = chosen & endpoint_observed & (endpoint_sign != 0)
    scores = np.ascontiguousarray(logit_up_5[common], dtype=np.float64)
    aucs = {}
    for name, signs in (("first_hurdle", y[common]), ("endpoint", endpoint_sign[common])):
        labels = np.ascontiguousarray((signs == 1).astype(np.int8))
        aucs[name] = saved.raw_rank_point(labels, scores, rank=rank, np=np)["auc"]
    return {"known_first_direction": int(chosen.sum()),
        "rows": ["long_first", "short_first"], "endpoint_columns": [-1, 0, 1],
        "first_by_endpoint_sign": table, "endpoint_missing_by_first": missing,
        "same_direction": int(same.sum()), "opposite_direction": int(opposite.sum()),
        "endpoint_zero": int(zero.sum()), "endpoint_missing": sum(missing),
        "common_observed_nonzero_endpoint": int(common.sum()), "same_score_auc_on_common_rows": aucs,
        "auc_scope": "descriptive only on realized first-hurdle-selected rows; not unconditional performance"}


def verify_previous_five(race, scores, previous):
    """Fail unless the prior fixed five-minute counts and AUC are reproduced."""
    y, known, why = race["outcome"], race["outcome_observed"], race["unknown_reason"]
    counts = {"long_first": int((known & (y == 1)).sum()),
              "short_first": int((known & (y == -1)).sum()),
              "neither_known": int((known & (y == 0)).sum()),
              "path_missing": int((why == 1).sum()),
              "entry_minute_unknown": int((why == 2).sum()),
              "same_minute_unknown": int((why == 3).sum()),
              "terminal_open_unknown": int((why == 4).sum())}
    chosen = known & (y != 0)
    labels = np.ascontiguousarray((y[chosen] == 1).astype(np.int8))
    z = np.ascontiguousarray(scores[chosen], dtype=np.float64)
    auc = saved.raw_rank_point(labels, z, rank=rank, np=np)["auc"]
    require(previous["parents"] == len(y) and previous["coverage"][4]["counts"] == counts
            and previous["primary_conditional_rank"]["known_direction_rows"] == int(chosen.sum())
            and previous["primary_conditional_rank"]["auc"] == auc, "PREVIOUS_FIVE_PARITY")
    return {"counts_exact": True, "known_rows": int(chosen.sum()), "auc_exact": True, "auc": auc}


def join_reference(prediction, reference):
    """Match the pre-existing ET252 score by parent, clock, outcome and fold."""
    require(set(reference) == set(REFERENCE_MEMBERS), "REFERENCE_MEMBERS")
    parents, wanted = reference["parent_rows"], prediction["parent_rows"]
    require(parents.dtype == np.int64 and parents.ndim == 1 and len(parents) > 0
            and np.all(np.diff(parents) == 1) and wanted.dtype == np.int64 and wanted.ndim == 1,
            "REFERENCE_PARENT_AXIS")
    positions = np.searchsorted(parents, wanted)
    require(np.all(positions < len(parents)) and np.array_equal(parents[positions], wanted), "REFERENCE_PARENT_JOIN")
    for key in ("decision_ns", "terminal_time_ns", "observed", "sign"):
        require(np.array_equal(reference[key][positions], prediction[key]), "REFERENCE_TARGET_JOIN:" + key)
    require(np.all(reference["fold_id"][positions] == 1)
            and np.all(reference["holdout_end_ns_exclusive"][positions] == LATER_END), "REFERENCE_FOLD")
    scores = reference["saved_ET_FULL_probability_up"]
    require(scores.dtype == np.float64 and scores.shape == (len(parents), 3)
            and np.isfinite(scores).all() and np.all((scores >= 0.) & (scores <= 1.)), "REFERENCE_SCORE")
    return np.ascontiguousarray(scores[positions, 1])


def compare_reference(race, native_score, reference_score, decision_ns, *, replicates=REPLICATES, minimum_blocks=8):
    """Same-row conditional AUCs and paired calendar-block intervals."""
    y, known = race["outcome"], race["outcome_observed"]
    require(y.ndim == 1 and known.shape == native_score.shape == reference_score.shape == decision_ns.shape == y.shape
            and known.dtype == np.bool_ and decision_ns.dtype == np.int64
            and np.isfinite(native_score).all() and np.isfinite(reference_score).all(), "REFERENCE_COMPARE_INPUT")
    chosen = known & (y != 0)
    labels = np.ascontiguousarray((y[chosen] == 1).astype(np.int8))
    native = np.ascontiguousarray(native_score[chosen], dtype=np.float64)
    ref = np.ascontiguousarray(reference_score[chosen], dtype=np.float64)
    require(len(labels) > 0 and np.unique(labels).size == 2, "REFERENCE_BOTH_CLASSES")
    native_auc = saved.raw_rank_point(labels, native, rank=rank, np=np)["auc"]
    reference_auc = saved.raw_rank_point(labels, ref, rank=rank, np=np)["auc"]
    days = decision_ns // DAY; intervals = []
    for name, values, seed in (("UTC_DAY", days, 20261027), ("UTC_WEEK_MONDAY", (days + 3) // 7, 20261028)):
        blocks, inverse = np.unique(values, return_inverse=True)
        if len(blocks) < minimum_blocks:
            intervals.append({"scheme": name, "blocks": len(blocks), "status": "INSUFFICIENT_BLOCKS"})
            continue
        weights = saved.block_weights(len(blocks), seed=seed, np=np, replicates=replicates)
        parent_blocks = inverse[chosen].astype(np.int64)
        native_draws = saved.raw_rank_replicates(labels, native, parent_blocks, weights, rank=rank, np=np)
        ref_draws = saved.raw_rank_replicates(labels, ref, parent_blocks, weights, rank=rank, np=np)
        intervals.append({"scheme": name, "blocks": len(blocks),
            "resampling_population": "all fixed parents before first-hurdle eligibility",
            **saved.percentile_interval(ref_draws - native_draws, np=np, alpha=ALPHA)})
    return {"known_first_direction": int(chosen.sum()), "known_long": int((labels == 1).sum()),
            "known_short": int((labels == 0).sum()), "native_auc": native_auc,
            "et252_auc": reference_auc, "et252_minus_native_auc": reference_auc - native_auc,
            "paired_intervals": intervals,
            "scope": "conditional on realized first-hurdle event and observable order; ET252 uses different model and inputs"}


def coverage_report(panel):
    """Keep missing paths and partial outcomes distinct for all horizons."""
    outcome, known, reason, possible = (panel[k] for k in ("outcome", "observed", "reason", "possibilities"))
    require(outcome.ndim == 2 and outcome.shape[1] == 60
            and all(v.shape == outcome.shape for v in (known, reason, possible)), "PANEL_SHAPE")
    result = []
    for j, horizon in enumerate(HORIZONS):
        y, ok, why, bits = outcome[:, j], known[:, j], reason[:, j], possible[:, j]
        counts = {"long_first": int((ok & (y == 1)).sum()),
                  "short_first": int((ok & (y == -1)).sum()),
                  "neither_known": int((ok & (y == 0)).sum()),
                  "path_missing": int((why == 1).sum()),
                  "entry_minute_unknown": int((why == 2).sum()),
                  "same_minute_unknown": int((why == 3).sum()),
                  "terminal_open_unknown": int((why == 4).sum())}
        require(sum(counts.values()) == len(y), "COVERAGE_PARTITION")
        result.append({"horizon_minutes": int(horizon), "parents": len(y), "counts": counts,
            "possible_sets_bit_order_long_short_neither": {str(i): int((bits == i).sum()) for i in range(1, 8)}})
    return result


def conditional_rank(panel, logit_up_5, decision_ns, *, replicates=REPLICATES, minimum_blocks=8):
    """Fixed 5min score AUC only among the observed first-long/first-short rows."""
    y = panel["outcome"][:, 4]
    mask = panel["observed"][:, 4] & (y != 0)
    require(logit_up_5.shape == decision_ns.shape == y.shape
            and np.isfinite(logit_up_5).all() and decision_ns.dtype == np.int64, "RANK_INPUT")
    labels = np.ascontiguousarray((y[mask] == 1).astype(np.int8))
    scores = np.ascontiguousarray(logit_up_5[mask], dtype=np.float64)
    result = {"horizon_minutes": 5, "parents": len(y), "known_direction_rows": int(mask.sum()),
              "known_long": int((labels == 1).sum()), "known_short": int((labels == 0).sum()),
              "selection": "conditioned on observable singleton first-long/first-short; outcome-dependent and not representative",
              "class_or_coverage_selected_after_results": False, "auc": None, "block_intervals": []}
    if not len(labels) or np.unique(labels).size < 2:
        result["status"] = "INSUFFICIENT_BOTH_CLASSES"
        return result
    point = saved.raw_rank_point(labels, scores, rank=rank, np=np)
    result["auc"] = point["auc"]
    days = decision_ns // DAY
    for name, block_values, seed in (("UTC_DAY", days, 20261027),
                                     ("UTC_WEEK_MONDAY", (days + 3) // 7, 20261028)):
        blocks, inverse = np.unique(block_values, return_inverse=True)
        if len(blocks) < minimum_blocks:
            result["block_intervals"].append({"scheme": name, "blocks": len(blocks), "status": "INSUFFICIENT_BLOCKS"})
            continue
        weights = saved.block_weights(len(blocks), seed=seed, np=np, replicates=replicates)
        draws = saved.raw_rank_replicates(labels, scores, inverse[mask].astype(np.int64), weights, rank=rank, np=np)
        interval = saved.percentile_interval(draws, np=np, alpha=ALPHA)
        result["block_intervals"].append({"scheme": name, "blocks": len(blocks),
            "resampling_population": "all fixed parents before first-hurdle eligibility", **interval})
    result["status"] = "CONDITIONAL_DESCRIPTIVE_REUSED_DEVELOPMENT"
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def verified_payload(ref):
    path = Path(ref["path"])
    require(path.is_absolute() and not path.is_symlink() and path.resolve(strict=True) == path, "PAYLOAD_PATH")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        before = os.fstat(handle.fileno()); digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        require(digest.hexdigest() == ref["sha256"], "PAYLOAD_SHA")
        handle.seek(0)
        yield handle
        after = os.fstat(handle.fileno())
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "PAYLOAD_CHANGED")


def bound_plan(path):
    plan_path = Path(path); require(plan_path.is_absolute() and not plan_path.is_symlink(), "PLAN_PATH")
    plan = json.loads(plan_path.read_text()); policy_path = Path("NEXT_RUN_POLICY.json")
    mode = plan.get("mode", "full")
    require(mode in ("full", "endpoint_alignment", "reference_comparison"), "MODE")
    stage_name = {"full": STAGE, "endpoint_alignment": ALIGNMENT_STAGE,
                  "reference_comparison": REFERENCE_STAGE}[mode]
    stage = json.loads(policy_path.read_text()).get(stage_name, {}); plan_sha = sha(plan_path)
    require(stage.get("execution_authorized") is True and stage.get("claimed_invocations") == 0
            and stage.get("max_invocations") == 1 and stage.get("relaunch_allowed") is False
            and stage.get("plan") == {"path": str(plan_path), "sha256": plan_sha}, "POLICY")
    schema = {"full": "gx1_first_hurdle_oof_audit_plan_v1",
              "endpoint_alignment": "gx1_first_hurdle_endpoint_alignment_plan_v1",
              "reference_comparison": "gx1_first_hurdle_reference_plan_v1"}[mode]
    horizons = HORIZONS.tolist() if mode == "full" else [5]
    require(plan.get("schema_version") == schema
            and plan.get("quote") == QUOTE and plan.get("target") == TARGET
            and plan.get("prediction") == PREDICTION and plan.get("parents") == ROWS
            and plan.get("hurdle_bps") == HURDLE_BPS and plan.get("horizons") == horizons
            and plan.get("test_control_exit_authorized") is False, "PLAN_SCOPE")
    if mode != "full":
        require(plan.get("previous_result") == PREVIOUS and sha(PREVIOUS["path"]) == PREVIOUS["sha256"],
                "PREVIOUS_RESULT_BINDING")
    if mode == "reference_comparison":
        require(plan.get("reference") == REFERENCE, "REFERENCE_BINDING")
    output = Path(plan["output_dir"])
    require(output.is_absolute() and output.is_dir() and not output.is_symlink(), "OUTPUT")
    claim = json.loads((output / "INVOCATION_CLAIM.json").read_text())
    require(claim.get("plan_sha256") == plan_sha and claim.get("policy_sha256") == sha(policy_path)
            and claim.get("invocation") == 1, "CLAIM")
    for owner in plan["source_owners"]:
        require(sha(owner["path"]) == owner["sha256"], "SOURCE_DRIFT")
    return plan, output


def run(plan_path):
    import pyarrow.parquet as pq
    plan, output = bound_plan(plan_path)
    prediction = _read_prediction(PREDICTION, context_spec=CONTEXT_SPEC)
    require(np.array_equal(prediction["parent_rows"], np.arange(FIRST_PARENT, FIRST_PARENT + ROWS, dtype=np.int64)),
            "FOLD_PARENT_RANGE")
    with verified_payload(TARGET) as handle:
        with np.load(handle, allow_pickle=False) as archive:
            require(archive.files == list(TARGET_MEMBERS), "TARGET_MEMBERS")
            target = {key: archive[key] for key in TARGET_MEMBERS}
    saved.join_targets(prediction, target, np=np)
    expected_entry_rows = target["dynamic_entry_m1_row"][prediction["parent_rows"]]
    with verified_payload(QUOTE) as handle:
        parquet = pq.ParquetFile(handle)
        clock = pd.DatetimeIndex(pd.to_datetime(parquet.read(columns=["time"]).column("time").to_pandas(), utc=True)).as_unit("ns")
        times = np.asarray(clock.asi8, dtype=np.int64)
        start = pd.Timestamp("2025-01-01T00:00:00Z").value
        end = pd.Timestamp("2026-01-01T00:00:00Z").value
        require(len(times) == QUOTE["rows"] and times[0] == pd.Timestamp(QUOTE["first"]).value
                and times[-1] == pd.Timestamp(QUOTE["last"]).value and np.all(np.diff(times) > 0)
                and np.all((times >= start) & (times < end)) and np.all(times % MINUTE == 0), "YEAR_CLOCK")
        keep = int(np.searchsorted(times, LATER_END, side="left"))
        table = parquet.read(columns=list(QUOTE_COLUMNS)).slice(0, keep)
        frame = pd.DataFrame({"time": clock[:keep], **{
            key: np.asarray(table.column(key).combine_chunks().to_numpy(), dtype=np.float64) for key in QUOTE_COLUMNS}})
    index = prepare_dynamic_m1_prefix_index(frame, split_end_exclusive=pd.Timestamp(LATER_END, tz="UTC"),
                                             source_parquet_sha256=QUOTE["sha256"], source_row_offset=QUOTE["offset"])
    fills = build_dynamic_m1_fill_surface(index, m5_decision_times=pd.to_datetime(prediction["bar_start_ns"], utc=True))
    require(fills["entry_fill_bound"].to_numpy(bool).all()
            and np.array_equal(fills["entry_m1_row"].to_numpy(np.int64), expected_entry_rows)
            and np.array_equal(pd.DatetimeIndex(fills["entry_decision_at"]).asi8, prediction["decision_ns"]),
            "ENTRY_JOIN")
    mode = plan.get("mode", "full")
    if mode != "full":
        race = race_five(index, fills)
        previous = json.loads(Path(PREVIOUS["path"]).read_text())
        replay = verify_previous_five(race, prediction["logit_up"][:, 1], previous)
        if mode == "endpoint_alignment":
            report = endpoint_alignment(race, prediction["sign"][:, 1], prediction["observed"][:, 1],
                                        prediction["logit_up"][:, 1])
            schema, report_key = "gx1_first_hurdle_endpoint_alignment_result_v1", "alignment"
        else:
            with verified_payload(REFERENCE) as handle:
                with np.load(handle, allow_pickle=False) as archive:
                    require(set(REFERENCE_MEMBERS).issubset(archive.files), "REFERENCE_ARCHIVE")
                    reference = {key: archive[key] for key in REFERENCE_MEMBERS}
            ref_score = join_reference(prediction, reference)
            report = compare_reference(race, prediction["logit_up"][:, 1], ref_score, prediction["decision_ns"])
            schema, report_key = "gx1_first_hurdle_reference_result_v1", "reference_comparison"
        result = {"schema_version": schema, "status": "COMPLETE", "parents": ROWS,
                  "horizon_minutes": 5, "favorable_bps": HURDLE_BPS, "previous_five_parity": replay,
                  report_key: report, "model_forwards": 0, "optimizer_steps": 0, "fits": 0,
                  "test_control_accessed": False,
                  "limits": ["One reused TRAIN-development fold and realized first-hurdle-selected parents.",
                             "Same-row AUCs do not establish causality, generalization or trading edge."]}
    else:
        panel = race_panel(index, fills)
        result = {"schema_version": "gx1_first_hurdle_oof_result_v1", "status": "COMPLETE",
                  "parents": ROWS, "horizons_minutes": HORIZONS.tolist(), "favorable_bps": HURDLE_BPS,
                  "event_basis": "gross_quote_markout", "endpoint_kind": "closed_m1_prefix",
                  "coverage": coverage_report(panel),
                  "primary_conditional_rank": conditional_rank(panel, prediction["logit_up"][:, 1], prediction["decision_ns"]),
                  "model_forwards": 0, "optimizer_steps": 0, "fits": 0,
                  "test_control_accessed": False, "combined_pretest_quote_opened": False,
                  "limits": ["One repeatedly reused TRAIN-development fold and an already used 10bps research hurdle.",
                             "Known-first-direction AUC conditions on outcome-dependent M1 observability; not unconditional trading edge.",
                             "M1 intrabar order and quote age remain unknown; no commission or slippage is included.",
                             "No model has been trained on this first-hurdle label; no action or holding rule is defined."]}
    for owner in plan["source_owners"]:
        require(sha(owner["path"]) == owner["sha256"], "SOURCE_DRIFT_AFTER")
    require(sha("NEXT_RUN_POLICY.json") == json.loads((output / "INVOCATION_CLAIM.json").read_text())["policy_sha256"],
            "POLICY_DRIFT_AFTER")
    with (output / "RESULT.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False); handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--plan", required=True); args = parser.parse_args()
    result = run(args.plan)
    summary = {"status": result["status"], "parents": result["parents"]}
    if "alignment" in result:
        summary["alignment"] = result["alignment"]
    elif "reference_comparison" in result:
        summary["reference_comparison"] = result["reference_comparison"]
    else:
        summary["primary_conditional_rank"] = result["primary_conditional_rank"]
    print(json.dumps(summary, indent=2))



if __name__ == "__main__":
    main()
