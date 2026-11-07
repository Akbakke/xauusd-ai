"""Read-only binding of corrected TRAIN endpoint masks to native model rows.

The returned arrays are targets for loss/scoring, never model forward inputs.
No target construction, fitting, TEST access, or launch authority lives here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from gx1.contracts.market_state_five_endpoint_contract_v1 import LEAD_MINUTES

MINUTE_NS = 60_000_000_000
M5_NS = 5 * MINUTE_NS
TRAIN_START_NS = 1714521600000000000  # 2024-05-01 00:00 UTC
FOLD_BOUNDARIES_NS = (
    1727740800000000000,  # 2024-10-01
    1735689600000000000,  # 2025-01-01
    1743465600000000000,  # 2025-04-01
    1751328000000000000,  # 2025-07-01
    1759276800000000000,  # 2025-10-01
    1767225600000000000,  # 2026-01-01
)
ROOT = "/home/andre2/src/GX1_CURRENT"
RUNS = "/home/andre2/GX1_RUNS"
REFS = {
    "mask": (f"{RUNS}/MARKET_STATE_LOCAL_SIDE_EXCEPTIONS_20261107_001/ROW_MASKS.npz",
             "2b063e35b97cc1dbdbb522d2b2876a39514b1145d8c59d8a2ab07561764f9edc"),
    "mask_result": (f"{RUNS}/MARKET_STATE_LOCAL_SIDE_EXCEPTIONS_20261107_001/RESULT.json",
                    "126aceee074dc223a60894545533d1f242275d8c4b4327acc67b048449cac7b8"),
    "mask_terminal": (f"{RUNS}/MARKET_STATE_LOCAL_SIDE_EXCEPTIONS_20261107_001/TERMINAL.json",
                      "457a9da5085d04362a5eeb76568a6a97cd68d424cef1efc6d4aee1332c2cb782"),
    "mask_plan": (f"{ROOT}/configs/research/MARKET_STATE_LOCAL_SIDE_EXCEPTIONS_20261107_001.json",
                  "5639382b63f2b56dcf3e263eff4d6a1408bb84555567f542b9bff4897cbcd0b7"),
    "selected_plan": (f"{ROOT}/configs/research/MARKET_STATE_SELECTED_ACTIVITY_MASKS_20261107_001.json",
                      "9b43bd30b3d3a9bb414923df547191200ca5b9c1e6993fc906cfdd283cd0f406"),
    "target_source_plan": (f"{ROOT}/configs/research/FULL_TRAIN_MARKET_STATE_ENDPOINT_CANDIDATE_20261107_001.json",
                           "c3c789f4e7ed3edba2880bfb9e51decc9efad779324351bbd19ac86d2a607ce2"),
    "model_axes": (f"{RUNS}/NATIVE_M5_DIRECTION_BASELINE_DATA_20261024_245/OUTPUT/PRICE_TARGETS.npz",
                   "9d4a5fb3c473b088aff5a5d5a176c199a64053b679229c14f125697b93af577d"),
    "model_plan": (f"{RUNS}/NATIVE_M5_DIRECTION_NATIVE_PREFIX_INPUT_BINDING_REPAIR_20261024_264/PLAN_SUPPORT.json",
                   "96ff82533042ab069292505b3a44315c13223a9b9d9e9ee669cd757e3f84c086"),
}
MASK_KEYS = frozenset(("issue_ns", "m5_source_row", "lead_minutes", "m1_source_row",
    "allowed_set", "activity_observed", "activity_active", "side_observed", "side_up",
    "singleton_observed"))
MODEL_KEYS = frozenset(("parent_rows", "bar_start_ns", "dynamic_entry_m1_row",
    "decision_ns", "terminal_time_ns", "elapsed_wall_seconds", "terminal_source_row",
    "observed", "mid_move", "mid_return_bps", "sign"))


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("FIVE_ENDPOINT_BINDING_" + reason)


def _ref(name: str) -> dict:
    path, sha256 = REFS[name]
    return {"path": path, "sha256": sha256}


def _checked_bytes(name: str, limit: int) -> bytes:
    path, expected = REFS[name]
    size = Path(path).stat().st_size
    _need(0 < size <= limit, name.upper() + "_SIZE")
    raw = Path(path).read_bytes()
    _need(len(raw) == size and hashlib.sha256(raw).hexdigest() == expected,
          name.upper() + "_HASH")
    return raw


def _json(name: str) -> dict:
    return json.loads(_checked_bytes(name, 4 * 1024 * 1024))


def _axis_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<i8").tobytes()).hexdigest()


def _array(archive, key: str, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    value = archive[key]
    _need(value.dtype == np.dtype(dtype) and value.shape == shape, key.upper() + "_SCHEMA")
    return np.array(value, copy=True, order="C")


@dataclass(frozen=True)
class BoundFiveEndpointTargets:
    parent_rows: np.ndarray
    decision_ns: np.ndarray
    allowed_set: np.ndarray
    m1_source_row: np.ndarray
    m5_source_row: np.ndarray
    unjoined_target_rows: int

    def gather(self, parent_rows: np.ndarray, decision_ns: np.ndarray, *,
               fold_id: int, partition: str) -> dict[str, np.ndarray]:
        """Gather target-only arrays in caller order; censor by the fold boundary."""
        _need(type(fold_id) is int and 1 <= fold_id <= 5
              and partition in ("fit", "score"), "FOLD")
        parents = np.asarray(parent_rows)
        issues = np.asarray(decision_ns)
        _need(parents.dtype == np.int64 and issues.dtype == np.int64
              and parents.ndim == issues.ndim == 1 and len(parents) > 0
              and len(parents) == len(issues), "QUERY_AXES")
        at = np.searchsorted(self.parent_rows, parents)
        _need(bool(np.all(at < len(self.parent_rows))), "QUERY_PARENT_RANGE")
        _need(np.array_equal(self.parent_rows[at], parents)
              and np.array_equal(self.decision_ns[at], issues), "QUERY_EXACT_CLOCK")
        train_end = FOLD_BOUNDARIES_NS[fold_id - 1]
        lower, upper = ((TRAIN_START_NS, train_end) if partition == "fit"
                        else (train_end, FOLD_BOUNDARIES_NS[fold_id]))
        _need(bool(np.all((issues >= lower) & (issues < upper))), "QUERY_PARTITION")
        validity = issues[:, None] + np.asarray(LEAD_MINUTES, np.int64)[None, :] * MINUTE_NS
        allowed = self.allowed_set[at].copy()
        eligible = (allowed != 7) & (self.m1_source_row[at] >= 0)
        mature = validity < upper
        return {"parent_rows": parents.copy(), "issue_ns": issues.copy(),
                "validity_ns": validity, "allowed": allowed,
                "eligible": eligible, "mature": mature}


def _bind_arrays(model, mask, *, expected_model_rows: int | None = None,
                 expected_target_rows: int | None = None) -> BoundFiveEndpointTargets:
    """Pure axis and semantic checks, also used with tiny synthetic fixtures."""
    _need(set(model.files) == MODEL_KEYS and set(mask.files) == MASK_KEYS, "NPZ_MEMBERS")
    n, m = len(model["parent_rows"]), len(mask["issue_ns"])
    _need(n > 0 and m >= n
          and (expected_model_rows is None or n == expected_model_rows)
          and (expected_target_rows is None or m == expected_target_rows), "ROW_COUNTS")
    parents = _array(model, "parent_rows", "int64", (n,))
    starts = _array(model, "bar_start_ns", "int64", (n,))
    decisions = _array(model, "decision_ns", "int64", (n,))
    issues = _array(mask, "issue_ns", "int64", (m,))
    source_m5 = _array(mask, "m5_source_row", "int64", (m,))
    leads = _array(mask, "lead_minutes", "int16", (5,))
    source_m1 = _array(mask, "m1_source_row", "int64", (m, 5))
    allowed = _array(mask, "allowed_set", "uint8", (m, 5))
    _need(np.array_equal(leads, LEAD_MINUTES)
          and bool(np.all(np.diff(parents) > 0))
          and bool(np.all(np.diff(decisions) > 0))
          and bool(np.all(np.diff(issues) > 0))
          and bool(np.all(np.diff(source_m5) > 0))
          and bool(np.all(starts + M5_NS == decisions))
          and bool(np.all((issues >= TRAIN_START_NS) & (issues < FOLD_BOUNDARIES_NS[-1])))
          and bool(np.all(source_m1 >= -1))
          and bool(np.isin(allowed, (1, 2, 4, 5, 7)).all()), "AXES_AND_BITS")
    at = np.searchsorted(issues, decisions)
    _need(bool(np.all(at < m)) and np.array_equal(issues[at], decisions), "EXACT_ISSUE_JOIN")
    _need(bool(np.all(source_m1[allowed != 7] >= 0)), "INFORMATIVE_SOURCE")
    expected_flags = {
        "activity_observed": np.isin(allowed, (1, 2, 4, 5)),
        "activity_active": np.isin(allowed, (1, 4, 5)),
        "side_observed": np.isin(allowed, (1, 4)),
        "side_up": allowed == 4,
        "singleton_observed": np.isin(allowed, (1, 2, 4)),
    }
    for name, expected in expected_flags.items():
        actual = _array(mask, name, "bool", (m, 5))
        _need(np.array_equal(actual, expected), name.upper() + "_PARITY")
    joined = [parents, decisions, allowed[at].copy(), source_m1[at].copy(), source_m5[at].copy()]
    for value in joined:
        value.setflags(write=False)
    return BoundFiveEndpointTargets(*joined, unjoined_target_rows=m - n)


def load_bound_five_endpoint_targets() -> BoundFiveEndpointTargets:
    """Load only pinned receipts and compact arrays; never open raw M1/TEST."""
    result, terminal = _json("mask_result"), _json("mask_terminal")
    corrected, selected = _json("mask_plan"), _json("selected_plan")
    target_source, model_plan = _json("target_source_plan"), _json("model_plan")
    _need(result.get("row_masks") == _ref("mask")
          and result.get("plan") == _ref("mask_plan")
          and result.get("decision") == "PASS_SEVEN_LOCAL_SIDE_DEMOTIONS_SOURCE_ONLY"
          and result.get("status") == "COMPLETE" and result.get("changed_cells") == 7
          and result.get("test_data_used") is False
          and terminal.get("result") == _ref("mask_result")
          and terminal.get("row_masks") == _ref("mask")
          and terminal.get("exit_code") == 0 and terminal.get("status") == "COMPLETE"
          and terminal.get("test_data_used") is False, "CORRECTED_RECEIPTS")
    _need(corrected.get("inputs", {}).get("selected", {}).get("plan") == _ref("selected_plan")
          and selected.get("inputs", {}).get("full_plan") == _ref("target_source_plan")
          and corrected.get("test_data_used") is False
          and selected.get("test_data_used") is False
          and target_source.get("test_data_used") is False
          and target_source.get("lead_minutes") == list(LEAD_MINUTES)
          and target_source.get("train_start_utc") == "2024-05-01T00:00:00+00:00"
          and target_source.get("train_end_utc_exclusive") == "2026-01-01T00:00:00+00:00",
          "TARGET_LINEAGE")
    axes = model_plan.get("axes", {})
    base = model_plan.get("base_lineage", {})
    m5 = target_source.get("m5", {}).get("source", {})
    _need(axes.get("source") == _ref("model_axes")
          and axes.get("rows") == 118249
          and axes.get("decision_rule") == "bar_start_ns_plus_300000000000"
          and base.get("m5_prebuilt_path") == m5.get("path")
          and base.get("m5_prebuilt_sha256") == m5.get("sha256"), "MODEL_SOURCE_LINEAGE")
    with np.load(io.BytesIO(_checked_bytes("model_axes", 32 * 1024 * 1024)), allow_pickle=False) as model:
        _need(_axis_hash(model["parent_rows"]) == axes.get("parent_rows_sha256")
              and _axis_hash(model["bar_start_ns"]) == axes.get("bar_start_ns_sha256"),
              "MODEL_AXES_HASH")
        with np.load(io.BytesIO(_checked_bytes("mask", 4 * 1024 * 1024)), allow_pickle=False) as mask:
            bound = _bind_arrays(model, mask, expected_model_rows=118249,
                                 expected_target_rows=118706)
    _need(bound.unjoined_target_rows == 457, "UNJOINED_TARGET_ROWS")
    return bound
