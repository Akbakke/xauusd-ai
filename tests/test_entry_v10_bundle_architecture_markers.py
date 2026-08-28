"""The bundle loader must match the architecture that the trainer records."""

from __future__ import annotations

import pytest

import gx1.models.entry_v10.entry_v10_bundle as bundle


def test_active_architecture_marks_positional_encoding_and_retired_film() -> None:
    bundle._require_model_native_architecture_markers(
        {"enable_pos_enc": True, "enable_regime_film": False}
    )


@pytest.mark.parametrize(
    ("metadata", "error"),
    (
        ({"enable_pos_enc": False, "enable_regime_film": False}, "enable_pos_enc"),
        ({"enable_pos_enc": True, "enable_regime_film": True}, "RETIRED_REGIME_FILM"),
        ({"enable_pos_enc": True}, "RETIRED_REGIME_FILM"),
    ),
)
def test_architecture_markers_fail_closed_on_train_serve_split(
    metadata: dict[str, object], error: str
) -> None:
    with pytest.raises(RuntimeError, match=error):
        bundle._require_model_native_architecture_markers(metadata)
