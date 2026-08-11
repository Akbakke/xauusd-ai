"""Shared synthetic V29 registry-constants payload for tests.

The V29 registry blocks require an explicit TRAIN-fitted constants payload
(rule 2a: no default exists in code).  Tests exercise the code paths on
synthetic frames, where a synthetic-execution payload proves only that the
code runs (rule 2c) — no production conclusion may be drawn from these
values.  The shapes/keys are validated by the real
``require_v29_registry_constants`` contract, so a schema drift still fails
loudly here.
"""
from __future__ import annotations

from gx1.features.htf_features import require_v29_registry_constants

# The per-TF window bars mirror the production architecture declaration and
# the entry seq len mirrors MODEL_NATIVE_SEQ_LEN; the tolerance/band values
# are synthetic-execution inputs only.
_SYNTHETIC_TEST_TOL_ATR = 1.0
_SYNTHETIC_TEST_BAND_ATR = 0.5


def synthetic_v29_registry_constants() -> dict:
    """Return a schema-valid synthetic-execution constants payload."""

    from gx1.contracts.entry_exit_production_architecture_v1 import (
        PRODUCTION_MTF_PER_TF_WINDOW_BARS,
    )
    from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SEQ_LEN

    timeframes = tuple(tf for tf, _bars in PRODUCTION_MTF_PER_TF_WINDOW_BARS)
    return require_v29_registry_constants(
        {
            "schema_version": "htf_v4_v29_registry_constants_v1",
            "level_tol_quantile_recipe_key": "level_registry_tol_quantile_q",
            "level_tol_quantile_q": 0.25,
            "declared_train_window_end": "synthetic-test-window",
            "level_tol_atr": {tf: _SYNTHETIC_TEST_TOL_ATR for tf in timeframes},
            "trendline_band_atr": {
                tf: _SYNTHETIC_TEST_BAND_ATR for tf in timeframes
            },
            "per_tf_seq_lens": dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS),
            "entry_m5": {
                "seq_len": MODEL_NATIVE_SEQ_LEN,
                "trendline_band_atr": _SYNTHETIC_TEST_BAND_ATR,
            },
            "provenance": {
                "evidence_class": "synthetic_execution_only",
                "note": (
                    "test payload; proves code runs, never a production value"
                ),
            },
        }
    )
