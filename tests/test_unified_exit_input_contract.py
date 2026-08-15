from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_exit_feature_base_v1 import (
    EXIT_FEATURE_SEQUENCE_BARS,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.entry_exit_feature_surface_v1 import (
    ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.entry_decision_token_v1 import (
    build_entry_decision_token_snapshot,
)
from gx1.contracts.unified_exit_input_v1 import (
    build_unified_exit_input_envelope,
    require_unified_exit_input_envelope,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_PATH_ENVELOPE_SCHEMA_VERSION,
    canonical_closed_m1_bar,
    canonical_closed_m1_full_path_chain_sha256,
    canonical_closed_m1_path_sha256,
    require_unified_exit_path_envelope,
)


def _inputs() -> dict:
    decision_time = pd.Timestamp("2026-07-17T12:00:00Z")
    row = canonical_closed_m1_bar(
        m1_bar_ts=decision_time,
        complete=True,
        source_path="/immutable/unit/xau_m1.parquet",
        source_sha256="a" * 64,
        bid_open=3300.0,
        bid_high=3301.0,
        bid_low=3299.0,
        bid_close=3300.5,
        ask_open=3300.2,
        ask_high=3301.2,
        ask_low=3299.2,
        ask_close=3300.7,
        mid_open=3300.1,
        mid_high=3301.1,
        mid_low=3299.1,
        mid_close=3300.6,
        volume=10,
    )
    path = require_unified_exit_path_envelope(
        {
            "schema_version": UNIFIED_EXIT_PATH_ENVELOPE_SCHEMA_VERSION,
            "entry_fill_ts": decision_time.isoformat(),
            "first_full_m1_bar_ts": decision_time.isoformat(),
            "last_closed_m1_bar_ts": decision_time.isoformat(),
            "bars_in_trade": 1,
            "retained_path_length": 1,
            "path_rows": [row],
            "path_rows_sha256": canonical_closed_m1_path_sha256([row]),
            "full_path_chain_sha256": (
                canonical_closed_m1_full_path_chain_sha256([row])
            ),
        },
        context="UNIT_FULL_EXIT_INPUT",
    )
    signal = np.zeros(
        (EXIT_FEATURE_SEQUENCE_BARS, MODEL_NATIVE_SIGNAL_DIM),
        dtype=np.float32,
    )
    m1 = {
        "schema_version": ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION,
        "decision_time": decision_time.isoformat(),
        "sequence_first_time": (
            decision_time - pd.Timedelta(minutes=479)
        ).isoformat(),
        "sequence_last_time": decision_time.isoformat(),
        "dataset_run_id": "UNIT_DATASET",
        "pair_generation_id": "UNIT_PAIR",
        "feature_base_sha256": "b" * 64,
        "feature_manifest_sha256": "c" * 64,
        "feature_field_order_sha256": "d" * 64,
        "sequence_bars": EXIT_FEATURE_SEQUENCE_BARS,
        "signal": signal,
        "snap": signal[-1].copy(),
        "ctx_cont": np.zeros(MODEL_NATIVE_CTX_CONT_DIM, dtype=np.float32),
        "ctx_cat": np.zeros(MODEL_NATIVE_CTX_CAT_DIM, dtype=np.int64),
    }
    lengths = {timeframe: 2 for timeframe in EXIT_MTF_CONTEXT_TIMEFRAMES}
    mtf = {
        timeframe: np.zeros(
            (2, len(MULTI_TF_PER_BAR_FEATURES_V4)),
            dtype=np.float32,
        )
        for timeframe in EXIT_MTF_CONTEXT_TIMEFRAMES
    }
    entry_token = [float(index) / 128.0 for index in range(128)]
    entry_snapshot = {
        "decision_ts": decision_time.isoformat(),
        "model_direction_index": 0,
        "model_direction": "LONG",
        "entry_decision_representation": entry_token,
    }
    return {
        "decision_time": decision_time,
        "decision_identity": "unit-replay-id",
        "side": "long",
        "entry_bid": 3300.0,
        "entry_ask": 3300.2,
        "bundle_sha256": "e" * 64,
        "entry_snapshot": entry_snapshot,
        "entry_decision_token_snapshot": build_entry_decision_token_snapshot(
            token=entry_token,
            decision_time=decision_time,
            fill_time=decision_time,
            model_identity_kind="bundle_sha256",
            model_identity_sha256="e" * 64,
            input_normalization_sha256="9" * 64,
            contract_mode=MODEL_NATIVE_CONTRACT_MODE,
            model_direction_index=0,
            model_direction="LONG",
            side="long",
            entry_bid=3300.0,
            entry_ask=3300.2,
            trade_identity="unit-replay-id",
        ),
        "exit_path_envelope": path,
        "m1_feature_window": m1,
        "mtf_windows": mtf,
        "mtf_cache_binding": {
            "cache_identity_sha256": "f" * 64,
            "manifest_sha256": "1" * 64,
        },
        "per_tf_seq_lens": lengths,
    }


def test_full_exit_input_hash_changes_for_every_tensor_lane() -> None:
    source = _inputs()
    baseline = build_unified_exit_input_envelope(**source)

    m1_changed = _inputs()
    m1_changed["m1_feature_window"]["signal"][0, 0] = np.float32(1.0)
    assert (
        build_unified_exit_input_envelope(**m1_changed)["input_envelope_sha256"]
        != baseline["input_envelope_sha256"]
    )

    current_changed = _inputs()
    current_changed["m1_feature_window"]["signal"][-1, 1] = np.float32(2.0)
    current_changed["m1_feature_window"]["snap"][1] = np.float32(2.0)
    assert (
        build_unified_exit_input_envelope(**current_changed)["input_envelope_sha256"]
        != baseline["input_envelope_sha256"]
    )

    for timeframe in EXIT_MTF_CONTEXT_TIMEFRAMES:
        changed = _inputs()
        changed["mtf_windows"][timeframe][0, 0] = np.float32(3.0)
        observed = build_unified_exit_input_envelope(**changed)
        assert observed["input_envelope_sha256"] != baseline["input_envelope_sha256"]
        assert (
            observed["mtf_windows"][timeframe]["sha256"]
            != baseline["mtf_windows"][timeframe]["sha256"]
        )


def test_full_exit_input_binds_identity_clock_artifacts_and_self_hash() -> None:
    source = _inputs()
    envelope = build_unified_exit_input_envelope(**source)
    assert require_unified_exit_input_envelope(envelope) == envelope

    for path, replacement in (
        (("decision_identity",), "other-replay"),
        (("bundle_sha256",), "2" * 64),
        (
            ("m1_feature_window", "sequence_first_time"),
            "2026-07-17T04:00:00+00:00",
        ),
        (("m1_feature_window", "pair_generation_id"), "OTHER_PAIR"),
        (("mtf_cache_binding", "manifest_sha256"), "3" * 64),
    ):
        tampered = deepcopy(envelope)
        target = tampered
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        with pytest.raises(
            RuntimeError,
            match="HASH_MISMATCH|CONTENT_HASH_MISMATCH|TOKEN_BINDING_MISMATCH",
        ):
            require_unified_exit_input_envelope(tampered)

    clock_mismatch = _inputs()
    clock_mismatch["m1_feature_window"]["decision_time"] = (
        "2026-07-17T12:01:00+00:00"
    )
    with pytest.raises(RuntimeError, match="M1_.*CLOCK_INVALID|M1_CLOCK_MISMATCH"):
        build_unified_exit_input_envelope(**clock_mismatch)
