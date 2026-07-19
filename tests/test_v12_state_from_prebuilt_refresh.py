from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_async_refresh_aborts_cv3_swap_when_mtf_refresh_fails() -> None:
    text = (REPO / "gx1/execution/v12_state_from_prebuilt.py").read_text(encoding="utf-8")

    assert "new-cv3/stale-mtf split-brain" in text
    assert "keeping stale" not in text


def test_async_refresh_reaugments_when_base28_advances() -> None:
    text = (REPO / "gx1/execution/v12_state_from_prebuilt.py").read_text(encoding="utf-8")

    assert "cv3_advanced or b28_advanced" in text


def test_active_prebuilt_augmentation_has_no_alternate_or_skip_path() -> None:
    text = (REPO / "gx1/execution/v12_state_from_prebuilt.py").read_text(encoding="utf-8")

    assert "sequential-fallback" not in text
    assert "augment skipped" not in text
    assert "falling back to on-disk cache" not in text
    assert "refusing a second transform path" in text
