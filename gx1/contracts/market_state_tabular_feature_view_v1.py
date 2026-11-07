"""Causal current snapshot for a bounded, offline market-state tabular test.

The caller owns immutable TRAIN source receipts and model/target-axis binding.
This view reads no labels and fits no statistics. It omits ``ctx_cont`` only
because every one of its physical columns has an exact ``signal`` alias;
``take`` verifies the actual values before emitting a chunk.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np

from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.features import htf_features as htf
from gx1.features.entry_specialist_feature_groups_v1 import compile_compact_entry_execution_profile
from gx1.models.entry_v10 import entry_v10_input_normalization as native

TIMEFRAMES = ("M5", "M15", "H1", "H4", "D1")
BAR_NS = 300_000_000_000
MAX_CHUNK_ROWS = 2048


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError("MARKET_STATE_TABULAR_" + reason)


def _axis_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<i8").tobytes()).hexdigest()


class MarketStateTabularFeatureView:
    """One model row per closed M5 issue; no full feature matrix is cached."""

    def __init__(self, *, parent_rows: np.ndarray, bar_start_ns: np.ndarray,
                 decision_ns: np.ndarray, expected_axes: Mapping,
                 signal: np.ndarray, ctx_cont: np.ndarray, ctx_cat: np.ndarray,
                 multi_tf_sources: Mapping) -> None:
        physical = compile_compact_entry_execution_profile()["physical_field_names"]
        signal_names = list(physical["signal"])
        context_names = list(physical["ctx_cont"])
        cat_names = list(physical["ctx_cat"])
        mtf_names = list(physical["mtf"])
        n = len(parent_rows)
        _need(n > 0 and all(isinstance(a, np.ndarray) and a.dtype == np.int64
                            and a.shape == (n,) for a in (parent_rows, bar_start_ns, decision_ns)),
              "AXIS_SHAPE")
        _need(bool(np.all(np.diff(parent_rows) > 0))
              and bool(np.all(np.diff(decision_ns) > 0))
              and bool(np.all(bar_start_ns % BAR_NS == 0))
              and bool(np.array_equal(decision_ns, bar_start_ns + BAR_NS)), "DECISION_CLOCK")
        _need(isinstance(expected_axes, Mapping)
              and set(expected_axes) == {"source", "rows", "parent_rows_sha256",
                                         "bar_start_ns_sha256", "decision_ns_sha256"}
              and expected_axes["rows"] == n
              and expected_axes["parent_rows_sha256"] == _axis_hash(parent_rows)
              and expected_axes["bar_start_ns_sha256"] == _axis_hash(bar_start_ns)
              and expected_axes["decision_ns_sha256"] == _axis_hash(decision_ns),
              "AXIS_BINDING")
        _need(isinstance(signal, np.ndarray) and signal.dtype == np.float32
              and signal.shape == (n, len(signal_names))
              and isinstance(ctx_cont, np.ndarray) and ctx_cont.dtype == np.float32
              and ctx_cont.shape == (n, len(context_names))
              and isinstance(ctx_cat, np.ndarray) and ctx_cat.dtype == np.int64
              and ctx_cat.shape == (n, len(cat_names)), "CURRENT_SHAPE")
        _need(isinstance(multi_tf_sources, Mapping)
              and set(multi_tf_sources) == set(TIMEFRAMES), "FIVE_TF")
        alias_names = ["ctx_cont." + name for name in context_names]
        _need(len(signal_names) == len(set(signal_names))
              and len(context_names) == len(set(context_names))
              and len(alias_names) == len(set(alias_names))
              and all(name in signal_names for name in alias_names), "CONTEXT_ALIAS_NAMES")
        alias_indices = np.asarray([signal_names.index(name) for name in alias_names], dtype=np.int64)
        tf_values: dict[str, np.ndarray] = {}
        tf_tails: dict[str, np.ndarray] = {}
        for tf in TIMEFRAMES:
            frame = multi_tf_sources[tf]
            source_ns, values, warmup = native._extract_mtf_source(frame, tf=tf)
            _need(list(frame.columns) == mtf_names and values.shape[1] == len(mtf_names),
                  "TF_FIELDS:" + tf)
            shift_ns = int(htf.MULTI_TF_SHIFT[tf].value)
            _need(bool(np.all(decision_ns >= shift_ns))
                  and bool(np.all(source_ns <= np.iinfo(np.int64).max - shift_ns)),
                  "TF_CLOCK_RANGE:" + tf)
            tail = np.searchsorted(source_ns, decision_ns - shift_ns,
                                   side="right").astype(np.int64, copy=False) - 1
            _need(bool(np.all(tail >= warmup)) and bool(np.all(tail < len(source_ns)))
                  and bool(np.all(np.diff(tail) >= 0))
                  and bool(np.all(source_ns[tail] + shift_ns <= decision_ns))
                  and bool(np.all((tail + 1 == len(source_ns)) |
                                  (source_ns[np.minimum(tail + 1, len(source_ns) - 1)] + shift_ns > decision_ns))),
                  "TF_LAST_CLOSED:" + tf)
            tf_values[tf] = values
            tf_tails[tf] = tail
        names = (["signal." + name for name in signal_names]
                 + ["ctx_cat." + name for name in cat_names]
                 + ["mtf_" + tf.lower() + "." + name
                    for tf in TIMEFRAMES for name in mtf_names])
        _need(len(names) == len(set(names)), "OUTPUT_NAMES")
        categorical = []
        for name, domain in norm.SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS.items():
            _need(name in signal_names, "SIGNAL_CATEGORY")
            categorical.append({"index": signal_names.index(name), "name": "signal." + name,
                                "domain": list(domain)})
        for name, domain in norm.CTX_CAT_DOMAINS.items():
            _need(name in cat_names, "CONTEXT_CATEGORY")
            categorical.append({"index": len(signal_names) + cat_names.index(name),
                                "name": "ctx_cat." + name, "domain": list(domain)})
        for j, tf in enumerate(TIMEFRAMES):
            for name, domain in norm.MTF_SEMANTIC_CATEGORICAL_DOMAINS.items():
                _need(name in mtf_names, "TF_CATEGORY:" + tf)
                categorical.append({"index": len(signal_names) + len(cat_names)
                                    + j * len(mtf_names) + mtf_names.index(name),
                                    "name": "mtf_" + tf.lower() + "." + name,
                                    "domain": list(domain)})
        _need(len(categorical) == 7 and len({x["index"] for x in categorical}) == 7,
              "CATEGORICAL_LAYOUT")
        self.parent_rows = parent_rows
        self.decision_ns = decision_ns
        self.signal = signal
        self.ctx_cont = ctx_cont
        self.ctx_cat = ctx_cat
        self.alias_indices = alias_indices
        self.tf_values = tf_values
        self.tf_tails = tf_tails
        self.feature_names = tuple(names)
        self.categorical = tuple(categorical)
        self.offsets = {tf: len(signal_names) + len(cat_names) + j * len(mtf_names)
                        for j, tf in enumerate(TIMEFRAMES)}
        self.width = len(names)

    def take(self, model_positions: np.ndarray, *, max_rows: int) -> np.ndarray:
        """Emit only a caller-bounded chronological chunk of raw, causal values."""
        _need(type(max_rows) is int and max_rows > 0
              and isinstance(model_positions, np.ndarray)
              and model_positions.dtype == np.int64 and model_positions.ndim == 1
              and 0 < len(model_positions) <= min(max_rows, MAX_CHUNK_ROWS)
              and bool(np.all(model_positions >= 0))
              and bool(np.all(model_positions < len(self.parent_rows)))
              and bool(np.all(np.diff(model_positions) > 0)), "CHUNK_ROWS")
        snap = np.asarray(self.signal[model_positions], dtype=np.float32)
        context = np.asarray(self.ctx_cont[model_positions], dtype=np.float32)
        _need(bool(np.array_equal(np.ascontiguousarray(snap[:, self.alias_indices]).view("<u4"),
                                  np.ascontiguousarray(context).view("<u4"))),
              "ACTUAL_CONTEXT_ALIAS")
        out = np.empty((len(model_positions), self.width), dtype=np.float32)
        signal_end = self.signal.shape[1]
        out[:, :signal_end] = snap
        out[:, signal_end:signal_end + self.ctx_cat.shape[1]] = self.ctx_cat[model_positions]
        for tf in TIMEFRAMES:
            start = self.offsets[tf]
            values = self.tf_values[tf][self.tf_tails[tf][model_positions]]
            out[:, start:start + values.shape[1]] = values
        _need(not bool(np.isinf(out).any()), "INFINITE_VALUE")
        for item in self.categorical:
            values = out[:, item["index"]]
            _need(bool(np.isin(values, item["domain"]).all()),
                  "CATEGORICAL_DOMAIN:" + item["name"])
        return out
