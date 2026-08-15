from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch

from gx1.contracts.entry_exit_feature_base_v1 import (
    EXIT_FEATURE_SEQUENCE_BARS,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.entry_exit_feature_surface_v1 import (
    ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.unified_exit_input_v1 import (
    build_unified_exit_input_envelope,
)
from gx1.contracts.entry_decision_token_v1 import (
    ENTRY_DECISION_TOKEN_KEY,
    build_entry_decision_token_snapshot,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4
from gx1.contracts.unified_exit_incremental_carry_v1 import (
    UNIFIED_EXIT_INCREMENTAL_CARRY_GENESIS_SHA256,
    build_unified_exit_incremental_carry_envelope,
)
from gx1.models.entry_v10.direction_decision_contract import (
    canonical_unified_evidence_sha256,
)


def unified_exit_input_fixture(
    *,
    entry_snapshot: Mapping[str, Any],
    exit_path_envelope: Mapping[str, Any],
    bundle_sha256: str,
    decision_identity: str,
    side: str,
    entry_bid: float,
    entry_ask: float,
    pair_generation_id: str = "UNIT_EXIT_INPUT_PAIR",
    entry_decision_token_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    decision_time = str(exit_path_envelope["last_closed_m1_bar_ts"])
    signal = np.zeros(
        (EXIT_FEATURE_SEQUENCE_BARS, MODEL_NATIVE_SIGNAL_DIM),
        dtype=np.float32,
    )
    m1 = {
        "schema_version": ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION,
        "decision_time": decision_time,
        "sequence_first_time": (
            pd.Timestamp(decision_time) - pd.Timedelta(minutes=479)
        ).isoformat(),
        "sequence_last_time": decision_time,
        "dataset_run_id": "UNIT_EXIT_INPUT_DATASET",
        "pair_generation_id": pair_generation_id,
        "feature_base_sha256": "1" * 64,
        "feature_manifest_sha256": "2" * 64,
        "feature_field_order_sha256": "3" * 64,
        "sequence_bars": EXIT_FEATURE_SEQUENCE_BARS,
        "signal": signal,
        "snap": signal[-1].copy(),
        "ctx_cont": np.zeros(MODEL_NATIVE_CTX_CONT_DIM, dtype=np.float32),
        "ctx_cat": np.zeros(MODEL_NATIVE_CTX_CAT_DIM, dtype=np.int64),
    }
    lengths = {timeframe: 2 for timeframe in EXIT_MTF_CONTEXT_TIMEFRAMES}
    mtf = {
        timeframe: np.zeros(
            (lengths[timeframe], len(MULTI_TF_PER_BAR_FEATURES_V4)),
            dtype=np.float32,
        )
        for timeframe in EXIT_MTF_CONTEXT_TIMEFRAMES
    }
    return build_unified_exit_input_envelope(
        decision_time=decision_time,
        decision_identity=decision_identity,
        side=side,
        entry_bid=entry_bid,
        entry_ask=entry_ask,
        bundle_sha256=bundle_sha256,
        entry_snapshot=entry_snapshot,
        entry_decision_token_snapshot=(
            dict(entry_decision_token_snapshot)
            if entry_decision_token_snapshot is not None
            else build_entry_decision_token_snapshot(
                token=entry_snapshot[ENTRY_DECISION_TOKEN_KEY],
                decision_time=entry_snapshot["decision_ts"],
                fill_time=exit_path_envelope["entry_fill_ts"],
                model_identity_kind="bundle_sha256",
                model_identity_sha256=bundle_sha256,
                input_normalization_sha256="6" * 64,
                contract_mode=MODEL_NATIVE_CONTRACT_MODE,
                model_direction_index=int(
                    entry_snapshot["model_direction_index"]
                ),
                model_direction=str(entry_snapshot["model_direction"]),
                side=side,
                entry_bid=entry_bid,
                entry_ask=entry_ask,
                trade_identity=decision_identity,
            )
        ),
        exit_path_envelope=exit_path_envelope,
        m1_feature_window=m1,
        mtf_windows=mtf,
        mtf_cache_binding={
            "cache_identity_sha256": "4" * 64,
            "manifest_sha256": "5" * 64,
        },
        per_tf_seq_lens=lengths,
    )


def unified_exit_carry_fixture(
    *,
    input_envelope: Mapping[str, Any],
    exit_path_envelope: Mapping[str, Any],
    previous_carry_envelope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    token = input_envelope["entry_decision_token_snapshot"]
    step = int(exit_path_envelope["bars_in_trade"])
    return build_unified_exit_incremental_carry_envelope(
        tensor_state={"fixture_hidden": torch.tensor([[[float(step)]]])},
        step_count=step,
        last_closed_m1_bar_ts=exit_path_envelope[
            "last_closed_m1_bar_ts"
        ],
        trade_identity=input_envelope["decision_identity"],
        side=input_envelope["side"],
        bundle_sha256=input_envelope["bundle_sha256"],
        input_normalization_sha256=token[
            "input_normalization_sha256"
        ],
        entry_token_snapshot_sha256=canonical_unified_evidence_sha256(token),
        full_path_chain_sha256=exit_path_envelope[
            "full_path_chain_sha256"
        ],
        input_envelope_sha256=input_envelope["input_envelope_sha256"],
        previous_carry_envelope_sha256=(
            UNIFIED_EXIT_INCREMENTAL_CARRY_GENESIS_SHA256
            if previous_carry_envelope is None
            else previous_carry_envelope["carry_envelope_sha256"]
        ),
        mtf_last_row_sha256=input_envelope["mtf_last_row_sha256"],
    )


__all__ = ("unified_exit_carry_fixture", "unified_exit_input_fixture")
