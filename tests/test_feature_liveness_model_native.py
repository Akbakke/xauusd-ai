from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

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


def test_feature_liveness_cli_uses_current_exact_dataset_constructor() -> None:
    tree = ast.parse(inspect.getsource(feature_liveness._main))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "EntryV10CtxDataset"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg for keyword in calls[0].keywords}
    assert keywords == {
        "parquet_path",
        "seq_len",
        "m5_prebuilt_path",
        "multi_tf_seq_len",
        "per_tf_seq_lens",
    }


# ── Population escalation: a sample cannot rule on deadness ────────────────────
# The magnitudes pinned below are the ones measured on the real V26 TRAIN
# population (369,303 rows) on 2026-07-26, not invented fixtures:
#   d1_regime_changed_flag_v3                     std 1.5347e-02  nunique 2
#   session_vol_spread_breakout_readiness         std 5.5970e-05  nunique 137,844
# Both were reported dead by the 1024-row sample gate while being alive.
V26_SPARSE_IMPULSE = (1.5347e-02, 2)
V26_SMALL_MAGNITUDE_SCORE = (5.5970e-05, 137844)


def _flat_sample(monkeypatch, field: str) -> tuple[dict, list[str], list[str]]:
    """A batch whose ``field`` is constant on the sample, as the real gate saw it."""
    _stub_multi_tf(monkeypatch)
    batch, signal_names, ctx_cont_names = _batch()
    batch["snap_x"][:, signal_names.index(field)] = 0.0
    return batch, signal_names, ctx_cont_names


def test_sparse_impulse_flag_is_alive_on_its_full_population(monkeypatch) -> None:
    field = "smc_choch"
    batch, signal_names, ctx_cont_names = _flat_sample(monkeypatch, field)

    report = feature_liveness.assert_v10_batch_liveness(
        batch,
        snap_names=signal_names,
        ctx_cont_names=ctx_cont_names,
        raise_on_fail=False,
        population_stats=lambda surface, name: (
            V26_SPARSE_IMPULSE if name == field else None
        ),
    )

    assert report["ok"] is True, report["issues"]


def test_small_magnitude_varying_score_is_alive_on_its_full_population(monkeypatch) -> None:
    field = "smc_choch"
    batch, signal_names, ctx_cont_names = _flat_sample(monkeypatch, field)
    assert V26_SMALL_MAGNITUDE_SCORE[0] < feature_liveness.DEAD_STD

    report = feature_liveness.assert_v10_batch_liveness(
        batch,
        snap_names=signal_names,
        ctx_cont_names=ctx_cont_names,
        raise_on_fail=False,
        population_stats=lambda surface, name: (
            V26_SMALL_MAGNITUDE_SCORE if name == field else None
        ),
    )

    assert report["ok"] is True, report["issues"]


def test_truly_constant_field_stays_dead_under_escalation(monkeypatch) -> None:
    field = "smc_choch"
    batch, signal_names, ctx_cont_names = _flat_sample(monkeypatch, field)

    report = feature_liveness.assert_v10_batch_liveness(
        batch,
        snap_names=signal_names,
        ctx_cont_names=ctx_cont_names,
        raise_on_fail=False,
        population_stats=lambda surface, name: (0.0, 1),
    )

    assert report["ok"] is False
    assert any(
        f"signal:{field}" in issue
        and "population_std=0.0e+00" in issue
        and "population_nunique=1" in issue
        for issue in report["issues"]
    ), report["issues"]


def test_escalation_absent_by_default_so_the_sample_verdict_stands(monkeypatch) -> None:
    field = "smc_choch"
    batch, signal_names, ctx_cont_names = _flat_sample(monkeypatch, field)

    report = feature_liveness.assert_v10_batch_liveness(
        batch,
        snap_names=signal_names,
        ctx_cont_names=ctx_cont_names,
        raise_on_fail=False,
    )

    assert report["ok"] is False
    assert any(f"signal:{field} (std=" in issue for issue in report["issues"])


def test_unresolvable_field_keeps_the_sample_verdict(monkeypatch) -> None:
    field = "smc_choch"
    batch, signal_names, ctx_cont_names = _flat_sample(monkeypatch, field)

    report = feature_liveness.assert_v10_batch_liveness(
        batch,
        snap_names=signal_names,
        ctx_cont_names=ctx_cont_names,
        raise_on_fail=False,
        population_stats=lambda surface, name: None,
    )

    assert report["ok"] is False
    assert any(f"signal:{field} (std=" in issue for issue in report["issues"])


def test_each_surface_is_escalated_separately(monkeypatch) -> None:
    """The gate asks per surface, so a caller can answer each one deliberately."""
    field = "smc_choch"
    _stub_multi_tf(monkeypatch)
    batch, signal_names, ctx_cont_names = _batch()
    index = signal_names.index(field)
    batch["snap_x"][:, index] = 0.0
    batch["seq_x"][:, :, index] = 0.0
    asked: list[tuple[str, str]] = []

    def stats(surface: str, name: str):
        asked.append((surface, name))
        return V26_SPARSE_IMPULSE if name == field else None

    report = feature_liveness.assert_v10_batch_liveness(
        batch,
        snap_names=signal_names,
        ctx_cont_names=ctx_cont_names,
        raise_on_fail=False,
        population_stats=stats,
    )

    assert report["ok"] is True, report["issues"]
    assert ("signal", field) in asked
    assert ("signal_sequence", field) in asked


def test_population_verdict_needs_no_third_constant() -> None:
    """Both halves of the verdict reuse a constant this owner already defines."""
    assert feature_liveness._population_alive(feature_liveness.DEAD_STD, 1) is True
    assert (
        feature_liveness._population_alive(
            0.0, feature_liveness.LIVE_TAIL_REF_MIN_NUNIQUE
        )
        is True
    )
    assert (
        feature_liveness._population_alive(
            feature_liveness.DEAD_STD / 2.0,
            feature_liveness.LIVE_TAIL_REF_MIN_NUNIQUE - 1,
        )
        is False
    )


# ── Per-timeframe lookback windows must be declared, never defaulted ───────────
def test_multi_tf_windows_are_caller_declared_not_wrapper_defaults() -> None:
    """Rule 14: how far back each timeframe reaches is decision-affecting.

    It was hardcoded in two places that silently disagreed 6x - the smoke
    wrapper pinned 16/16/16/8/8 (D1 = 8 days) while the candidate wrapper pinned
    96/96/96/48/30 - and a third ladder sat behind the GX1_MTF_TAPERED
    environment variable, which is exactly the ambient path rule 14 forbids.
    """
    repo = Path(__file__).resolve().parents[1]
    trainer = (repo / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py").read_text()

    assert "GX1_MTF_TAPERED" not in trainer.replace("former GX1_MTF_TAPERED", "")
    for timeframe in ("m5", "m15", "h1", "h4", "d1"):
        assert f'"--per-tf-seq-len-{timeframe}"' in trainer

    for wrapper_name in (
        "run_entry_model_native_seq513_smoke_train.sh",
        "run_entry_model_native_seq513_candidate_train.sh",
    ):
        wrapper = (repo / "scripts" / wrapper_name).read_text()
        assert "--multi-tf-seq-len 16" not in wrapper
        assert "--multi-tf-seq-len 96" not in wrapper
        for variable in (
            "MULTI_TF_SEQ_LEN",
            "PER_TF_SEQ_LEN_M5",
            "PER_TF_SEQ_LEN_M15",
            "PER_TF_SEQ_LEN_H1",
            "PER_TF_SEQ_LEN_H4",
            "PER_TF_SEQ_LEN_D1",
        ):
            assert f'"${variable}"' in wrapper, (wrapper_name, variable)
            assert variable in wrapper.split("; do")[0], (wrapper_name, variable)


def test_prior_match_tolerance_cannot_demand_less_than_sampling_noise() -> None:
    """The prior-match terms compare against the CURRENT BATCH's label rates.

    A rate from n samples has standard error up to sqrt(0.25/n), so a tolerance
    below that trains the model to chase the batch's own sampling noise. At the
    bound batch size of 64 the floor is 0.0625 against a declared 0.02, and at
    the slice minimum of 8 rows it is 0.1768 - nearly nine times the declared
    tolerance.
    """
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    assert trainer._batch_rate_sampling_floor(64) == pytest.approx(0.0625)
    assert trainer._batch_rate_sampling_floor(8) == pytest.approx(0.176776, rel=1e-5)
    assert trainer._batch_rate_sampling_floor(0) == 0.0
    # Monotone: more evidence permits a tighter demand.
    floors = [trainer._batch_rate_sampling_floor(n) for n in (8, 16, 32, 64, 256)]
    assert floors == sorted(floors, reverse=True)


def test_every_declared_timeframe_window_reaches_the_trainer() -> None:
    """A CLI window that never reaches the call site silently falls back.

    Measured on 2026-07-28: the launcher declared M5=16 and M15=64, the trainer
    logged M5=96 M15=96, because the argparse and the function signature carried
    the new parameters while the call site still passed only h4 and d1. The
    fallback is silent by design (0 means "use the global"), so nothing failed
    closed. Pin that every timeframe is threaded end to end.
    """
    repo = Path(__file__).resolve().parents[1]
    trainer = (repo / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py").read_text()

    for timeframe in ("m5", "m15", "h1", "h4", "d1"):
        assert f'"--per-tf-seq-len-{timeframe}"' in trainer, timeframe
        assert f"per_tf_seq_len_{timeframe}: int = 0," in trainer, timeframe
        assert (
            f"per_tf_seq_len_{timeframe}=int(args.per_tf_seq_len_{timeframe}),"
            in trainer
        ), f"{timeframe} is declared but never passed to the trainer"


def test_smoke_audit_measures_edge_and_gates_only_on_validity() -> None:
    """User vedtak 2026-07-28: the unreachable ambition bars are gone.

    The smoke bundle audit used to fail a bundle below 0.90 direction accuracy,
    0.90 balanced accuracy, 0.98 trade precision and 0.95 per-class precision,
    applied over every row of val and test. The majority baseline is 0.3858 and
    the best figure ever measured on this substrate is 0.4021, so no bundle could
    pass. What remains is the honest bar the audit already had - beat the majority
    baseline, which is computed from the split's own label rates - plus the
    support floors that make a measurement valid at all.
    """
    repo = Path(__file__).resolve().parents[1]
    audit = (repo / "gx1/scripts/audit_entry_foundation_smoke_bundle_v1.py").read_text()

    # The honest, data-derived gate stays.
    assert "if accuracy <= majority:" in audit
    assert "does not beat majority" in audit

    # Validity gates stay.
    assert "below required support={required_trade_rows}" in audit
    assert "does not contain all three label classes" in audit
    assert "does not emit all LONG/SHORT/FLAT classes" in audit

    # Ambition gates are gone: no failure is raised against these bars.
    for retired in (
        "below {MIN_DIRECTION_ACCURACY",
        "below {MIN_BALANCED_ACCURACY",
        "below {required_trade_precision",
        "below {MIN_CLASS_PRECISION",
        "below {required_class_wilson_lower",
    ):
        assert retired not in audit, retired

    # And nothing reports an unenforced bound as if it were a minimum.
    for stale in (
        '"minimum_trade_direction_precision"',
        '"minimum_trade_precision_wilson_lower"',
        '"minimum_class_precision_wilson_lower"',
    ):
        assert stale not in audit, stale


def test_model_seq_lens_read_the_same_resolution_as_the_data() -> None:
    """The model's declared per-TF shapes must come from the one resolution.

    Measured on 2026-07-28: after the windows reached the Dataset, V21 still died
    with SEQ_M5_LEN_MISMATCH got=16 expected=96, because the model was still
    constructed with m5_seq_len=multi_tf_seq_len and h1_seq_len=multi_tf_seq_len.
    The data followed the declaration and the model did not. The mismatch failed
    closed - the shape guard caught it - but a second derivation is a second
    truth, so both construction sites and the bundle metadata now read
    _effective_tf_lens.
    """
    repo = Path(__file__).resolve().parents[1]
    trainer = (repo / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py").read_text()

    for timeframe in ("m5", "m15", "h1", "h4", "d1"):
        assert f"{timeframe}_seq_len=multi_tf_seq_len" not in trainer, timeframe
        assert f'"{timeframe}_seq_len": int(multi_tf_seq_len)' not in trainer, timeframe
        # and the resolved local is what is passed
        assert f"{timeframe}_seq_len=_{timeframe}_len," in trainer, timeframe
        assert f'"{timeframe}_seq_len": int(_{timeframe}_len)' in trainer, timeframe

    # All five resolved locals come from the single mapping.
    for timeframe in ("M5", "M15", "H1", "H4", "D1"):
        assert f'_{timeframe.lower()}_len = int(_effective_tf_lens["{timeframe}"])' in trainer
