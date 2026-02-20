"""
I/O utilities for GX1 Engine.

Provides output guards to ensure proper file placement:
- Large artifacts (joblib/parquet/pt/pkl/sqlite/checkpoints) → GX1_DATA
- Small reports (json/md) → GX1_ENGINE/reports/
"""

from gx1.io.output_guard import (
    assert_output_path_allowed,
    get_gx1_data_root,
    is_large_artifact,
    is_small_report,
)

__all__ = [
    "assert_output_path_allowed",
    "get_gx1_data_root",
    "is_large_artifact",
    "is_small_report",
]
