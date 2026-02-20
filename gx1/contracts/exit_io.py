#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single source of truth for exit IO contract (canonical: IOV3_CLEAN).

TRUTH/SMOKE path must import from here, not from exit_transformer_io_v1/v2/v3.
See gx1/docs/EXIT_IOV3_CLEAN.md.
"""

from __future__ import annotations

from gx1.contracts.exit_transformer_io_v3 import (
    EXIT_IO_FEATURE_COUNT,
    EXIT_IO_VERSION,
    FEATURE_DIM_V3,
    IOV2_DIM,
    MIN_ATR_BPS,
    ORDERED_EXIT_EXTRA_FIELDS_V3,
    ORDERED_EXIT_FEATURES_V3,
    V3_EXTRAS_DIM,
    assert_exit_io_v3_clean_in_truth,
    config_hash_v3,
    ordered_feature_names_v3,
    row_to_feature_vector_v3,
    validate_row_v3,
    validate_window_v3,
)

__all__ = [
    "EXIT_IO_VERSION",
    "EXIT_IO_FEATURE_COUNT",
    "ORDERED_EXIT_FEATURES_V3",
    "ORDERED_EXIT_EXTRA_FIELDS_V3",
    "IOV2_DIM",
    "V3_EXTRAS_DIM",
    "FEATURE_DIM_V3",
    "MIN_ATR_BPS",
    "config_hash_v3",
    "ordered_feature_names_v3",
    "validate_row_v3",
    "validate_window_v3",
    "row_to_feature_vector_v3",
    "assert_exit_io_v3_clean_in_truth",
]
