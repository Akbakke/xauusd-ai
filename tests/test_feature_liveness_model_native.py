from __future__ import annotations

from pathlib import Path

import numpy as np

from gx1.audit import feature_liveness
from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    ordered_model_native_signal_fields,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields


def _signal_names() -> list[str]:
    selected = canonical_model_native_selected_fields(
        remainder_prefix="session_regime.feature_liveness_fixture"
    )
    return list(ordered_model_native_signal_fields(selected))


def _live(rows: int, width: int) -> np.ndarray:
    row = np.arange(rows, dtype=np.float64)[:, None]
    column = np.arange(width, dtype=np.float64)[None, :] / max(width, 1)
    return row + column


def _batch(*, rows: int = 8) -> tuple[dict, list[str], list[str]]:
    signal_names = _signal_names()
    ctx_cont_names = [f"ctx_cont.fixture_{index:03d}" for index in range(142)]
    batch = {
        "seq_x": np.stack(
            [_live(rows, len(signal_names)), _live(rows, len(signal_names)) + 1.0],
            axis=1,
        ),
        "snap_x": _live(rows, len(signal_names)),
        "ctx_cont": _live(rows, len(ctx_cont_names)),
        # Presence triggers the integrity helper, which these unit tests stub so
        # they can isolate the authoritative signal/context contract.
        "seq_m5": np.ones((1, 2, 1), dtype=np.float64),
    }
    return batch, signal_names, ctx_cont_names


def _stub_multi_tf(monkeypatch, seen: dict | None = None) -> None:
    def check(_seq, *, allow_known_dead):
        if seen is not None:
            seen["allow_known_dead"] = allow_known_dead
        return {
            "missing": [],
            "new_dead": [],
            "duplicate": [],
            "atr_by_tf": {"M5": 1.0, "D1": 2.0},
        }

    monkeypatch.setattr(
        feature_liveness,
        "check_multi_tf_integrity",
        check,
    )


def test_retired_bridge_fields_have_no_constant_exemption() -> None:
    assert set(FORBIDDEN_LEGACY_BRIDGE_FIELDS).isdisjoint(
        feature_liveness.KNOWN_ALLOWED_DEAD
    )

    dead = feature_liveness._dead_cols(
        np.zeros((4, len(FORBIDDEN_LEGACY_BRIDGE_FIELDS)), dtype=np.float64),
        FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    )

    assert len(dead) == len(FORBIDDEN_LEGACY_BRIDGE_FIELDS)
    assert all(any(field in row for row in dead) for field in FORBIDDEN_LEGACY_BRIDGE_FIELDS)


def test_exact_model_native_seq513_can_pass_only_when_finite_and_live(monkeypatch) -> None:
    seen: dict[str, bool] = {}
    _stub_multi_tf(monkeypatch, seen)
    batch, signal_names, ctx_cont_names = _batch()

    report = feature_liveness.assert_v10_batch_liveness(
        batch,
        snap_names=signal_names,
        ctx_cont_names=ctx_cont_names,
        raise_on_fail=False,
    )

    assert report == {
        "ok": True,
        "authoritative": True,
        "contract": "model_native_seq513",
        "issues": [],
        "multi_tf_atr": {"M5": 1.0, "D1": 2.0},
    }
    assert seen == {"allow_known_dead": False}


def test_model_native_seq513_does_not_inherit_legacy_constant_allowlist(monkeypatch) -> None:
    _stub_multi_tf(monkeypatch)
    batch, signal_names, ctx_cont_names = _batch()
    smc_choch_index = signal_names.index("smc_choch")
    assert "smc_choch" in feature_liveness.KNOWN_ALLOWED_DEAD
    batch["snap_x"][:, smc_choch_index] = 0.0

    report = feature_liveness.assert_v10_batch_liveness(
        batch,
        snap_names=signal_names,
        ctx_cont_names=ctx_cont_names,
        raise_on_fail=False,
    )

    assert report["ok"] is False
    assert any("signal:smc_choch (std=" in issue for issue in report["issues"])


def test_model_native_seq513_rejects_nonfinite_input(monkeypatch) -> None:
    _stub_multi_tf(monkeypatch)
    batch, signal_names, ctx_cont_names = _batch()
    batch["snap_x"][0, 0] = np.nan

    report = feature_liveness.assert_v10_batch_liveness(
        batch,
        snap_names=signal_names,
        ctx_cont_names=ctx_cont_names,
        raise_on_fail=False,
    )

    assert report["ok"] is False
    assert any("nonfinite=1" in issue for issue in report["issues"])


def test_retired_520_surface_cannot_pass_even_when_values_vary(monkeypatch) -> None:
    _stub_multi_tf(monkeypatch)
    batch, signal_names, ctx_cont_names = _batch()
    legacy_names = list(FORBIDDEN_LEGACY_BRIDGE_FIELDS) + signal_names
    batch["snap_x"] = _live(8, len(legacy_names))

    report = feature_liveness.assert_v10_batch_liveness(
        batch,
        snap_names=legacy_names,
        ctx_cont_names=ctx_cont_names,
        raise_on_fail=False,
    )

    assert report["ok"] is False
    assert any("width=520 expected=513" in issue for issue in report["issues"])
    assert any("forbidden legacy bridge fields" in issue for issue in report["issues"])


def test_missing_model_native_surface_or_names_is_a_hard_fail(monkeypatch) -> None:
    _stub_multi_tf(monkeypatch)
    batch, _, ctx_cont_names = _batch()
    del batch["snap_x"]
    del batch["seq_x"]

    report = feature_liveness.assert_v10_batch_liveness(
        batch,
        snap_names=None,
        ctx_cont_names=ctx_cont_names,
        raise_on_fail=False,
    )

    assert report["ok"] is False
    assert "signal: exact ordered field names missing" in report["issues"]
    assert "signal_sequence: seq_x surface missing" in report["issues"]
    assert "signal: snap_x surface missing" in report["issues"]


def test_trainer_cannot_slice_bridge_or_soft_skip_post_export_liveness() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"
    ).read_text(encoding="utf-8")
    block = source[source.index("# ── ALWAYS-RUN exact model-native feature-liveness audit") :]
    block = block[: block.index("\ndef main(")]

    assert '"seq_x": np.asarray(_live_ds._np_seq[_sample_idx]' in block
    assert "_snap_names = _snap_names[7:]" not in block
    assert '"snap_x"][:, 7:]' not in block
    assert "audit skipped (non-fatal)" not in block
    assert "[FEATURE_LIVENESS_AUDIT_UNAVAILABLE]" in block
