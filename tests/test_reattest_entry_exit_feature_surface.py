from __future__ import annotations

from pathlib import Path

import pytest

from gx1.scripts import reattest_entry_exit_feature_surface_v1 as reattest


def test_reattest_requires_exact_signal_contract_equality() -> None:
    reattest._require_equivalent_signal_contract({"fields": ["a"]}, {"fields": ["a"]})
    with pytest.raises(
        RuntimeError, match="FEATURE_SURFACE_REATTEST_SIGNAL_CONTRACT_CHANGED"
    ):
        reattest._require_equivalent_signal_contract(
            {"fields": ["a"]}, {"fields": ["b"]}
        )


def test_copy_file_noreplace_preserves_bytes_and_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "output.parquet"
    source.write_bytes(b"immutable-surface-bytes")
    reattest._copy_file_noreplace(source, output)
    assert output.read_bytes() == source.read_bytes()
    with pytest.raises(RuntimeError, match="FEATURE_SURFACE_REATTEST_OUTPUT_EXISTS"):
        reattest._copy_file_noreplace(source, output)
