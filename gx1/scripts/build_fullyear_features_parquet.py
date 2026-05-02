#!/usr/bin/env python3
"""Compatibility shim for reserved-column prebuilt feature tests.

The full-year builder itself lives under ``gx1.scripts._legacy``.  The active
repo still keeps the reserved candle-column sanitizer importable here because
older tests and audits use this path as a schema guard.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def sanitize_feature_columns(df_features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    from gx1.runtime.column_collision_guard import RESERVED_CANDLE_COLUMNS, check_reserved_candle_columns

    reserved_found = check_reserved_candle_columns(df_features, context="prebuilt builder")
    metadata: dict[str, Any] = {"dropped_columns": [], "reserved_found": reserved_found}

    if reserved_found:
        if "CLOSE" in reserved_found:
            df_features = df_features.drop(columns=["CLOSE"], errors="ignore")
            metadata["dropped_columns"].append("CLOSE")
            reserved_found = [column for column in reserved_found if column != "CLOSE"]

        if reserved_found:
            raise RuntimeError(
                f"[PREBUILT_SCHEMA_FAIL] Prebuilt features contain reserved candle columns: {reserved_found}. "
                f"Reserved columns (case-insensitive): {sorted(RESERVED_CANDLE_COLUMNS)}. "
                "These must not appear in prebuilt parquet - they come from candles directly."
            )

    return df_features, metadata


def main() -> None:
    raise SystemExit(
        "gx1.scripts.build_fullyear_features_parquet is a compatibility shim for sanitize_feature_columns only; "
        "the full legacy builder is not an active pipeline entrypoint."
    )


if __name__ == "__main__":
    main()
