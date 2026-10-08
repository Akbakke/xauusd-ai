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


def test_child_admission_publication_preserves_failed_staging(tmp_path, monkeypatch):
    from gx1.scripts import validate_lifecycle_v2_pilot_child_view_v1 as owner
    witness = {"decision": "PASS", "test_accessed": False}
    monkeypatch.setattr(owner, "validate_pilot_child_view", lambda **_: witness)
    pilot = tmp_path / "pilot"
    output = pilot / "ADMISSION" / "CHILD_VIEW_ADMISSION.json"
    original = owner._publish_file_noreplace
    def collide(source, destination):
        destination.write_text("keep")
        return original(source, destination)
    monkeypatch.setattr(owner, "_publish_file_noreplace", collide)
    with pytest.raises(RuntimeError, match="already exists"):
        owner.publish_pilot_child_view_admission(
            source_recipe_path=tmp_path / "unused", source_recipe_sha256="a" * 64,
            pilot_root=pilot, child_root_path=tmp_path / "unused", output_path=output,
        )
    assert output.read_text() == "keep"
    stages = list(output.parent.glob(".CHILD_VIEW_ADMISSION.json.*.tmp"))
    assert len(stages) == 1
    assert json.loads(stages[0].read_text()) == witness


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
    with pytest.raises(RuntimeError, match="already exists"):
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


def _published_frozen_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from gx1.contracts.unified_exit_lifecycle_v1 import UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION
    from gx1.scripts.prepare_unified_exit_lifecycle_v2_pilot_v1 import _canonical_sha256
    from tests.test_prepare_unified_exit_lifecycle_v2_pilot_v1 import _m1_rebinding_fixture, _write_json, _file_sha

    path, recipe, manifests, _, _ = _m1_rebinding_fixture(tmp_path / "source")
    source = recipe["artifact_bindings"]
    lifecycle_path = Path(source["unified_exit_lifecycle_manifest"]["path"])
    original = manifests["train"]["extra"]["unified_exit_lifecycle"]["m1_authority"]
    _write_json(lifecycle_path, {
        "schema_version": UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION,
        "decision": "PASS", "entry_run_id": "SOURCE_RUN",
        "m1_authority": original, "m1_authority_sha256": _canonical_sha256(original),
        "splits": {split: {
            "entry_dataset_path": source[f"{split}_parquet"]["path"],
            "entry_dataset_sha256": source[f"{split}_parquet"]["sha256"],
        } for split in ("train", "val")},
    })
    source["unified_exit_lifecycle_manifest"]["sha256"] = _file_sha(lifecycle_path)
    _write_json(path, recipe)
    # TEST-seal admission has separate real-metadata and negative tests.
    # This fixture isolates the subsequent Entry/calendar producer chain.
    monkeypatch.setattr(
        "gx1.scripts.prepare_unified_exit_lifecycle_v2_pilot_v1._entry_test_guard_lineage",
        lambda *args, **kwargs: {"synthetic_admission": True},
    )
    def old_m1_must_not_open(**kwargs):
        raise AssertionError("Old M1/lifecycle state admission must not be reused")
    monkeypatch.setattr(
        "gx1.scripts.validate_lifecycle_v2_pilot_child_view_v1._require_full_v1_admission",
        old_m1_must_not_open,
    )
    pilot = tmp_path / "pilot"
    report = materialize_pilot_entry_window(
        source_recipe_path=path, source_recipe_sha256=_file_sha(path),
        pilot_root=pilot, output_dir=pilot / "ENTRY_WINDOW", publish=True,
    )
    witness = validate_pilot_child_view(
        source_recipe_path=path, source_recipe_sha256=_file_sha(path),
        pilot_root=pilot, child_root_path=pilot / "ENTRY_WINDOW/ENTRY_WINDOW_ROOT.json",
    )
    return path, pilot, report, witness


def test_frozen_child_adopts_original_feature_bytes_without_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.scripts.validate_lifecycle_v2_pilot_child_view_v1 import require_pilot_child_calendar

    path, pilot, report, witness = _published_frozen_source(tmp_path, monkeypatch)
    recipe = json.loads(path.read_text())
    for split in ("train", "val"):
        assert report["root"]["splits"][split]["parquet"] == recipe["artifact_bindings"][f"{split}_parquet"]
        assert not (pilot / f"ENTRY_WINDOW/{split}.parquet").exists()
    assert witness["parent_admission_scope"] == "frozen_entry_bytes_only_no_parent_m1_states"
    windows = require_pilot_child_calendar(witness)
    assert windows["train"]["end_utc_exclusive"] == "2025-06-01T00:00:00+00:00"
    assert windows["val"]["start_utc"] == windows["train"]["end_utc_exclusive"]
    assert witness["splits"]["train"]["rows"] == 2
    assert witness["splits"]["val"]["rows"] == 2


@pytest.mark.parametrize("change", ["window", "rows", "clock", "parquet", "design_hash", "scope"])
def test_frozen_child_calendar_rejects_resealed_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    from gx1.scripts.validate_lifecycle_v2_pilot_child_view_v1 import require_pilot_child_calendar
    from gx1.scripts.prepare_unified_exit_lifecycle_v2_pilot_v1 import _canonical_sha256

    _, _, _, witness = _published_frozen_source(tmp_path, monkeypatch)
    if change == "window":
        witness["windows"]["train"]["end_utc_exclusive"] = "2026-06-01T00:00:00+00:00"
    elif change == "rows":
        witness["splits"]["val"]["rows"] = 5508
    elif change == "clock":
        witness["splits"]["val"]["clock_sha256"] = "0" * 64
    elif change == "parquet":
        witness["splits"]["val"]["parquet_path"] = witness["splits"]["train"]["parquet_path"]
    elif change == "design_hash":
        witness["chronological_learning_design"]["sha256"] = "0" * 64
    else:
        witness["parent_admission_scope"] = "reuse_old_m1_states"
    witness["witness_sha256"] = _canonical_sha256({k:v for k,v in witness.items() if k != "witness_sha256"})
    with pytest.raises(RuntimeError, match="PILOT_CHILD_CALENDAR"):
        require_pilot_child_calendar(witness)


def test_frozen_child_m1_views_keep_separate_clocks_and_feed_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    import pandas as pd
    from gx1.scripts.materialize_unified_exit_pilot_m1_views_v1 import build_views
    from gx1.scripts import materialize_unified_exit_pilot_summary_fit_v1 as summary
    from gx1.scripts.prepare_unified_exit_lifecycle_v2_pilot_v1 import _canonical_sha256
    from tests.test_prepare_unified_exit_lifecycle_v2_pilot_v1 import _write_json, _file_sha

    _, pilot, _, witness = _published_frozen_source(tmp_path, monkeypatch)
    entry_times = pd.DatetimeIndex(pd.to_datetime([
        "2011-06-01T00:00:00Z", "2025-05-30T12:55:00Z",
        "2025-06-01T22:00:00Z", "2026-06-30T14:55:00Z",
    ]))
    times = pd.DatetimeIndex(np.concatenate([
        pd.date_range(t-pd.Timedelta(minutes=480), t+pd.Timedelta(minutes=120), freq="min").asi8
        for t in entry_times
    ]), tz="UTC")
    parent = tmp_path / "complete-m1.parquet"
    pd.DataFrame({"time": times}).to_parquet(parent, index=False)
    manifest = tmp_path / "complete-m1.json"
    _write_json(manifest, {
        "output_parquet": str(parent), "output_parquet_sha256": _file_sha(parent),
        "quote_complete_m1": True, "test_accessed": False,
    })
    witness["m1_source_binding"] = {
        "parquet_path": str(parent), "parquet_sha256": _file_sha(parent),
        "manifest_path": str(manifest), "manifest_sha256": _file_sha(manifest),
    }
    witness["witness_sha256"] = _canonical_sha256({k:v for k,v in witness.items() if k != "witness_sha256"})
    admission = tmp_path / "admission.json"
    _write_json(admission, witness)
    output = pilot / "M1_VIEWS"
    build_views(
        child_admission_path=admission, parent_m1_path=parent,
        parent_m1_manifest_path=manifest, output_root=output, publish=True,
    )
    train = pd.read_parquet(output / "train.m1.parquet")["time"]
    val = pd.read_parquet(output / "val.m1.parquet")["time"]
    assert train.max() < pd.Timestamp("2025-06-01T00:00:00Z")
    assert val.max() < pd.Timestamp("2026-07-01T00:00:00Z")
    tm = json.loads((output / "train.manifest.json").read_text())
    vm = json.loads((output / "val.manifest.json").read_text())
    assert tm["fit_window_end_utc_exclusive"] == vm["fit_window_start_utc"]
    assert tm["context_rows_excluded_from_policy_fit"] is True
    assert vm["required_local_history_rows"] == 480

    class ReachedSummaryPriceRead(Exception):
        pass
    def stop_before_outcomes(*args, **kwargs):
        raise ReachedSummaryPriceRead
    monkeypatch.setattr(summary.pq, "read_table", stop_before_outcomes)
    with pytest.raises(ReachedSummaryPriceRead):
        summary.materialize(
            split="val", child_admission_path=admission,
            m1_path=output/"val.m1.parquet", m1_manifest_path=output/"val.manifest.json",
            closure_path=tmp_path/"unused", output_dir=tmp_path/"unused-output", publish=False,
        )
    vm["right_censor_time_utc_exclusive"] = "2026-09-01T00:00:00+00:00"
    _write_json(output/"val.manifest.json", vm)
    with pytest.raises(RuntimeError, match="PILOT_SUMMARY_SOURCE_INVALID"):
        summary.materialize(
            split="val", child_admission_path=admission,
            m1_path=output/"val.m1.parquet", m1_manifest_path=output/"val.manifest.json",
            closure_path=tmp_path/"unused", output_dir=tmp_path/"unused-output", publish=False,
        )


def test_frozen_parent_entry_rejects_wrong_bound_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.scripts.validate_lifecycle_v2_pilot_child_view_v1 import _require_parent_entry_admission
    from tests.test_prepare_unified_exit_lifecycle_v2_pilot_v1 import _write_json, _file_sha

    path, pilot, _, _ = _published_frozen_source(tmp_path, monkeypatch)
    plan = build_pilot_readiness(
        source_recipe_path=path, source_recipe_sha256=_file_sha(path), pilot_root=pilot,
    )
    binding = plan["source_bindings"]["unified_exit_lifecycle_manifest"]
    root = Path(binding["path"])
    value = json.loads(root.read_text())
    value["splits"]["val"]["entry_dataset_sha256"] = "0"*64
    _write_json(root, value)
    binding["sha256"] = _file_sha(root)
    with pytest.raises(RuntimeError, match="PILOT_CHILD_PARENT_ENTRY_LINEAGE_INVALID"):
        _require_parent_entry_admission(plan)


def test_compact_producer_uses_frozen_calendar_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd
    from gx1.scripts import materialize_unified_exit_lifecycle_v2_pilot_child_v1 as owner

    path, pilot, _, witness = _published_frozen_source(tmp_path, monkeypatch)
    monkeypatch.setattr(owner, "validate_pilot_child_view", lambda **kwargs: witness)
    m1 = witness["m1_source_binding"]
    monkeypatch.setattr(owner, "_validate_m1_source", lambda *args: (
        pd.DatetimeIndex(pd.to_datetime(["2011-06-01T00:05:00Z"])), {},
        m1["manifest_sha256"], m1["parquet_sha256"],
    ))
    authority = tmp_path/"economics.json"
    authority.write_text("{}")
    observed = {}
    def build(**kwargs):
        observed[kwargs["split"]] = kwargs["split_end"]
        if kwargs["split"] == "val":
            raise owner._EconomicAuthorityBlocked("stop after verified endpoints")
        return pd.DataFrame(), {}
    monkeypatch.setattr(owner, "_build_child_split", build)
    report = owner.materialize_pilot_child_lifecycle_v2(
        source_recipe_path=path, source_recipe_sha256="0"*64,
        pilot_root=pilot, child_root_path=pilot/"ENTRY_WINDOW/ENTRY_WINDOW_ROOT.json",
        output_dir=pilot/"LIFECYCLE_V2",
        m1_source_path=Path(m1["parquet_path"]), m1_source_manifest_path=Path(m1["manifest_path"]),
        train_economic_authority_path=authority, val_economic_authority_path=authority,
        publish=False,
    )
    assert report["decision"] == "BLOCKED"
    assert observed == {
        "train": "2025-06-01T00:00:00+00:00",
        "val": "2026-07-01T00:00:00+00:00",
    }
