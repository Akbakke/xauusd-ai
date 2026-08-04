from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    build_candlestick_derived_layer,
    build_price_derived_layer,
)
from gx1.features.entry_support_resistance_memory_v1 import (
    SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS,
    build_entry_support_resistance_memory_layer,
)
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    _build_inline_seq_structure_extension,
)
from gx1.scripts.materialize_entry_exit_m1_feature_base_v1 import (
    _M1_BOUNDED_CAUSAL_OVERLAP_ROWS,
    _build_bounded_extension_chunk,
)


def _synthetic_enriched_frame(rows: int) -> pd.DataFrame:
    index = np.arange(rows, dtype=np.float32)
    close = 2_000.0 + 0.08 * index + 0.35 * np.sin(index * 0.31)
    open_ = close - 0.12 * np.cos(index * 0.17)
    columns: dict[str, object] = {
        "time": pd.date_range(
            "2026-01-01",
            periods=rows,
            freq="min",
            tz="UTC",
        ),
        "open": open_,
        "high": np.maximum(open_, close) + 0.3,
        "low": np.minimum(open_, close) - 0.3,
        "close": close,
        "atr": np.full(rows, 1.25, dtype=np.float32),
    }
    numeric_fields = [
        *MODEL_NATIVE_BASE_FIELDS,
        *MODEL_NATIVE_CTX_CONT_FIELDS,
    ]
    for offset, name in enumerate(dict.fromkeys(numeric_fields)):
        if name in columns:
            continue
        columns[name] = (
            0.2 * np.sin(index * np.float32(0.013 + offset * 0.0001))
            + 0.1 * np.cos(index * np.float32(0.021 + offset * 0.00007))
        ).astype(np.float32)
    for name in MODEL_NATIVE_CTX_CAT_FIELDS:
        columns[name] = np.zeros(rows, dtype=np.int64)
    return pd.DataFrame(columns)


def test_support_resistance_memory_state_is_exact_across_batches() -> None:
    rows = 97
    index = np.arange(rows, dtype=np.float32)
    matrix = np.column_stack(
        [
            0.25 * np.sin(index * np.float32(0.03 + column * 0.001))
            for column in range(len(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS))
        ]
    ).astype(np.float32)
    names = list(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS)

    expected, expected_names = build_entry_support_resistance_memory_layer(
        matrix,
        names,
    )
    state = None
    pieces: list[np.ndarray] = []
    for start, stop in ((0, 19), (19, 53), (53, 71), (71, rows)):
        piece, piece_names, state = (
            build_entry_support_resistance_memory_layer(
                matrix[start:stop],
                names,
                memory_state=state,
                return_memory_state=True,
            )
        )
        assert piece_names == expected_names
        pieces.append(piece)

    np.testing.assert_array_equal(np.concatenate(pieces), expected)


def test_bounded_owner_orchestration_matches_full_history_exactly(
    tmp_path: Path,
) -> None:
    frame = _synthetic_enriched_frame(89)
    source = tmp_path / "synthetic_enriched_m1.parquet"
    frame.to_parquet(source, index=False)
    requested = [
        *MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
        "ctx_cont.d1_trend_age_mature_flag_v3",
        "chart.geometry_channel_center_bias",
        "candle.pattern_close_location",
    ]

    expected, expected_names, expected_meta = (
        _build_inline_seq_structure_extension(
            frame,
            requested_features=requested,
            ctx_cont_names=list(MODEL_NATIVE_CTX_CONT_FIELDS),
            ctx_cat_names=list(MODEL_NATIVE_CTX_CAT_FIELDS),
            source_parquet=source,
            source_contract_label="causal_enriched_m1_frame_v1",
            base_signal_fields=list(MODEL_NATIVE_BASE_FIELDS),
        )
    )
    sample_times = frame[["time"]].copy()
    price, price_names = build_price_derived_layer(sample_times, source)
    candle, candle_names = build_candlestick_derived_layer(
        sample_times,
        source,
    )

    state = None
    observed: list[np.ndarray] = []
    observed_meta = None
    observed_names = None
    batch_rows = 23
    for start in range(0, len(frame), batch_rows):
        stop = min(start + batch_rows, len(frame))
        prefix = max(0, start - _M1_BOUNDED_CAUSAL_OVERLAP_ROWS)
        chunk, names, meta, state = _build_bounded_extension_chunk(
            frame.iloc[prefix:stop].reset_index(drop=True),
            source_parquet=source,
            requested_features=requested,
            ctx_cont_names=list(MODEL_NATIVE_CTX_CONT_FIELDS),
            ctx_cat_names=list(MODEL_NATIVE_CTX_CAT_FIELDS),
            base_signal_fields=list(MODEL_NATIVE_BASE_FIELDS),
            price_layer=price[prefix:stop],
            price_names=price_names,
            candle_layer=candle[prefix:stop],
            candle_names=candle_names,
            emit_offset=start - prefix,
            support_memory_state=state,
            source_contract_label="causal_enriched_m1_frame_v1",
        )
        observed.append(chunk)
        if observed_names is None:
            observed_names = names
            observed_meta = meta
        else:
            assert names == observed_names
            assert meta == observed_meta

    assert observed_names == expected_names
    assert observed_meta == expected_meta
    np.testing.assert_array_equal(np.concatenate(observed), expected)
