from __future__ import annotations

import pytest

from gx1.contracts.entry_run_lineage_v1 import (
    EntryRunLineageError,
    require_entry_run_id,
)


def test_entry_run_id_is_normalized_lineage_not_authority() -> None:
    assert (
        require_entry_run_id(" XAU_SEQ513_REBUILD_20260720_V2 ")
        == "XAU_SEQ513_REBUILD_20260720_V2"
    )


@pytest.mark.parametrize(
    "value",
    [None, "", "short", "TODO", "ENTRY_RUN_ID", "run identity with spaces"],
)
def test_entry_run_id_rejects_missing_placeholder_or_ambiguous_values(
    value: object,
) -> None:
    with pytest.raises(EntryRunLineageError):
        require_entry_run_id(value)
