from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts.materialize_lifecycle_v2_pilot_entry_window_v1 import (
    materialize_pilot_entry_window,
)
from gx1.scripts.validate_lifecycle_v2_pilot_child_view_v1 import (
    publish_pilot_child_view_admission,
    validate_pilot_child_view,
)
from gx1.scripts.prepare_unified_exit_lifecycle_v2_pilot_v1 import (
    build_pilot_readiness,
)
from tests.test_prepare_unified_exit_lifecycle_v2_pilot_v1 import _source_fixture


def _published(tmp_path: Path) -> tuple[Path, str, Path, Path]:
    recipe, digest = _source_fixture(tmp_path / "source")
    pilot = tmp_path / "pilot"
    materialize_pilot_entry_window(
        source_recipe_path=recipe,
        source_recipe_sha256=digest,
        pilot_root=pilot,
        output_dir=pilot / "ENTRY_WINDOW",
        publish=True,
    )
    return recipe, digest, pilot, pilot / "ENTRY_WINDOW" / "ENTRY_WINDOW_ROOT.json"


def _parent_admission(recipe: Path) -> dict[str, object]:
    source = json.loads(recipe.read_text(encoding="utf-8"))["artifact_bindings"]
    return {
        "root_manifest_path": Path(
            source["unified_exit_lifecycle_manifest"]["path"]
        ),
        "root_manifest_sha256": source["unified_exit_lifecycle_manifest"]["sha256"],
    }


def test_child_view_witness_binds_parent_child_and_exact_clocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, digest, pilot, root = _published(tmp_path)
    monkeypatch.setattr(
        "gx1.scripts.validate_lifecycle_v2_pilot_child_view_v1."
        "_require_full_v1_admission",
        lambda **_kwargs: _parent_admission(recipe),
    )

    witness = validate_pilot_child_view(
        source_recipe_path=recipe,
        source_recipe_sha256=digest,
        pilot_root=pilot,
        child_root_path=root,
    )

    assert witness["decision"] == "PASS"
    assert witness["parent_dataset_run_id"] == "SOURCE_RUN"
    assert witness["child_dataset_run_id"].endswith("__PILOT_20250601_20260630")
    assert witness["splits"]["train"]["rows"] == 2
    assert witness["splits"]["val"]["rows"] == 2
    assert witness["test_accessed"] is False
    published = publish_pilot_child_view_admission(
        source_recipe_path=recipe,
        source_recipe_sha256=digest,
        pilot_root=pilot,
        child_root_path=root,
        output_path=pilot / "ADMISSION" / "CHILD_VIEW_ADMISSION.json",
    )
    witness_path = Path(published["path"])
    readiness = build_pilot_readiness(
        source_recipe_path=recipe,
        source_recipe_sha256=digest,
        pilot_root=pilot,
        entry_window_adoption_receipt=pilot
        / "ENTRY_WINDOW"
        / "ENTRY_WINDOW_ADOPTION_RECEIPT.json",
        child_view_admission=witness_path,
    )
    assert readiness["missing_or_blocked_stages"][0] == "train_economics"
    with pytest.raises(FileExistsError):
        publish_pilot_child_view_admission(
            source_recipe_path=recipe,
            source_recipe_sha256=digest,
            pilot_root=pilot,
            child_root_path=root,
            output_path=witness_path,
        )


def test_child_view_rejects_swapped_child_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, digest, pilot, root = _published(tmp_path)
    monkeypatch.setattr(
        "gx1.scripts.validate_lifecycle_v2_pilot_child_view_v1."
        "_require_full_v1_admission",
        lambda **_kwargs: _parent_admission(recipe),
    )
    (pilot / "ENTRY_WINDOW" / "train.parquet").write_bytes(b"swapped")

    with pytest.raises(RuntimeError, match="SPLIT_HASH_INVALID"):
        validate_pilot_child_view(
            source_recipe_path=recipe,
            source_recipe_sha256=digest,
            pilot_root=pilot,
            child_root_path=root,
        )


def test_child_view_rejects_wrong_parent_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, digest, pilot, root = _published(tmp_path)
    monkeypatch.setattr(
        "gx1.scripts.validate_lifecycle_v2_pilot_child_view_v1."
        "_require_full_v1_admission",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("parent rejected")),
    )
    with pytest.raises(RuntimeError, match="parent rejected"):
        validate_pilot_child_view(
            source_recipe_path=recipe,
            source_recipe_sha256=digest,
            pilot_root=pilot,
            child_root_path=root,
        )
