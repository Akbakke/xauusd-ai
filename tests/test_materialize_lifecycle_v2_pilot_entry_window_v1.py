from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gx1.scripts.materialize_lifecycle_v2_pilot_entry_window_v1 import (
    materialize_pilot_entry_window,
)
from gx1.scripts.prepare_unified_exit_lifecycle_v2_pilot_v1 import (
    build_pilot_readiness,
)
from tests.test_prepare_unified_exit_lifecycle_v2_pilot_v1 import _source_fixture


def test_validate_no_publish_reads_exact_source_without_writing(tmp_path: Path) -> None:
    recipe, digest = _source_fixture(tmp_path / "source")
    pilot = tmp_path / "pilot"
    output = pilot / "ENTRY_WINDOW"

    report = materialize_pilot_entry_window(
        source_recipe_path=recipe,
        source_recipe_sha256=digest,
        pilot_root=pilot,
        output_dir=output,
        publish=False,
    )

    assert report["decision"] == "READY_TO_MATERIALIZE"
    assert report["published"] is False
    assert report["selection_bindings"]["train"]["selected_rows"] == 2
    assert report["selection_bindings"]["val"]["selected_rows"] == 2
    assert not pilot.exists()


def test_publish_is_atomic_and_receipt_reopens_exact_subset(tmp_path: Path) -> None:
    recipe, digest = _source_fixture(tmp_path / "source")
    pilot = tmp_path / "pilot"
    output = pilot / "ENTRY_WINDOW"

    report = materialize_pilot_entry_window(
        source_recipe_path=recipe,
        source_recipe_sha256=digest,
        pilot_root=pilot,
        output_dir=output,
        publish=True,
    )

    assert report["root"]["decision"] == "PASS"
    assert report["receipt"]["decision"] == "PASS"
    assert len(pd.read_parquet(output / "train.parquet")) == 2
    assert len(pd.read_parquet(output / "val.parquet")) == 2
    readiness = build_pilot_readiness(
        source_recipe_path=recipe,
        source_recipe_sha256=digest,
        pilot_root=pilot,
        entry_window_adoption_receipt=output
        / "ENTRY_WINDOW_ADOPTION_RECEIPT.json",
    )
    assert readiness["missing_or_blocked_stages"][0] == "train_economics"

    (output / "train.parquet").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="ARTIFACT_BINDING_INVALID"):
        build_pilot_readiness(
            source_recipe_path=recipe,
            source_recipe_sha256=digest,
            pilot_root=pilot,
            entry_window_adoption_receipt=output
            / "ENTRY_WINDOW_ADOPTION_RECEIPT.json",
        )


def test_output_must_be_exact_pilot_entry_directory(tmp_path: Path) -> None:
    recipe, digest = _source_fixture(tmp_path / "source")
    with pytest.raises(RuntimeError, match="OUTPUT_PATH_INVALID"):
        materialize_pilot_entry_window(
            source_recipe_path=recipe,
            source_recipe_sha256=digest,
            pilot_root=tmp_path / "pilot",
            output_dir=tmp_path / "wrong",
            publish=False,
        )
