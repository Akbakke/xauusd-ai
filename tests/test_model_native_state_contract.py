from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_model_native_state_v2 import (
    MODEL_NATIVE_HISTORY_MODE,
    MODEL_NATIVE_STATE_SCHEMA_VERSION,
    RETIRED_RANK_STATE_FIELDS,
    STALE_STATE_CONTRACT_FIELDS,
    validate_state_contract_metadata_v2,
)
from gx1.execution.v12_model_native_state_live import (
    ModelNativeStateBuilder,
    ModelNativeStateContract,
    _ENTRY_SESSION_CONT_DOMAINS,
    _require_model_native_entry_context_frame,
    compute_htf_ctx_full_frame,
)
from gx1.execution.v12_ctx_augment_live import _add_session_features
from tests.model_native_signal_support import canonical_model_native_selected_fields


def test_model_native_full_frame_helpers_require_explicit_state_contract() -> None:
    frame = pd.DataFrame(index=pd.date_range("2026-07-08T18:00:00Z", periods=2, freq="5min"))

    with pytest.raises(RuntimeError, match="explicit model-native state contract required"):
        compute_htf_ctx_full_frame(frame)


def test_htf_ctx_computes_full_prefix_before_history_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HTF block is computed over the complete causal prefix, then sliced.

    Starting the computation at feature_history_start would discard prior D1
    transitions and make the carried HTF distances frame-dependent.
    """
    from gx1.execution import v12_ctx_augment_live as live_ctx

    contract = _early_validation_builder(tmp_path).state_contract
    index = pd.date_range("2025-11-30T23:30:00Z", periods=20, freq="5min")
    close = 2_000.0 + np.arange(len(index), dtype=np.float64) * 0.1
    cv3 = pd.DataFrame(
        {
            "open": close - 0.02,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
        },
        index=index,
    )

    # The exact columns compute_htf_ctx_full_frame requires from the HTF owner.
    htf_cols = (
        "D1_dist_from_ema200_atr",
        "d1_dist_change_1bar_atr_v4",
        "h4_mid_ema50_dist_atr_canon_v2",
    )
    observed: dict[str, int] = {}

    def fake_htf(work: pd.DataFrame, m5: pd.DataFrame) -> None:
        observed["htf_rows"] = len(work)
        assert len(work) == len(m5) == len(cv3)
        for name in htf_cols:
            work[name] = np.arange(len(work), dtype=np.float64)

    monkeypatch.setattr(live_ctx, "_add_htf_features", fake_htf)

    result = compute_htf_ctx_full_frame(cv3, contract)
    first_position = int(index.searchsorted(contract.feature_history_start_utc))

    assert observed == {"htf_rows": len(cv3)}
    assert list(result.columns) == list(htf_cols)
    assert result.index[0] == index[first_position]
    # Values were stamped over the full prefix, so the first retained row
    # carries its absolute position, not zero.
    assert first_position > 0
    for name in htf_cols:
        assert result.iloc[0][name] == first_position


def test_model_native_state_builder_requires_explicit_contracts() -> None:
    with pytest.raises(TypeError, match="state_contract"):
        ModelNativeStateBuilder(ordered_signal_names=[])


def _early_validation_builder(tmp_path: Path) -> ModelNativeStateBuilder:
    from tests.volatility_squeeze_test_support import (
        make_volatility_squeeze_artifact_set,
    )
    signal_contract = model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.state_contract_fixture"
        )
    )
    state_contract = ModelNativeStateContract(
        feature_history_start_utc=pd.Timestamp("2025-12-01T00:00:00Z"),
        raw={},
    )
    return ModelNativeStateBuilder(
        ordered_signal_names=list(signal_contract["fields"]),
        state_contract=state_contract,
        signal_contract=signal_contract,
        volatility_squeeze_artifacts=(
            make_volatility_squeeze_artifact_set(tmp_path)
        ),
    )


def _valid_entry_context_frame() -> pd.DataFrame:
    times = pd.date_range("2026-07-01T08:00:00Z", periods=96, freq="5min")
    frame = pd.DataFrame(index=times)
    _add_session_features(frame)
    frame.insert(0, "time", times)
    return frame.reset_index(drop=True)


def test_entry_context_boundary_requires_exactly_the_contract_categoricals() -> None:
    """ctx_cat is owned by the signal contract; the boundary may not add its own.

    The retired percentile categoricals (vol_regime_id, atr_bucket,
    spread_bucket, H4_trend_sign_cat) must not reappear as required Entry
    context: their evidence is carried continuously in the per-TF lanes.
    """
    assert tuple(MODEL_NATIVE_CTX_CAT_FIELDS) == ("session_id",)
    retired = {
        "vol_regime_id",
        "atr_bucket",
        "spread_bucket",
        "H4_trend_sign_cat",
        "session_tradable",
    }
    required = set(MODEL_NATIVE_CTX_CAT_FIELDS) | set(_ENTRY_SESSION_CONT_DOMAINS)
    assert not (required & retired)


def test_session_context_switches_at_m5_decision_boundary_without_extra_lag() -> None:
    labels = pd.date_range(
        "2026-07-01T06:50:00Z",
        periods=3,
        freq="5min",
    )
    frame = pd.DataFrame(index=labels)
    _add_session_features(frame)

    assert frame["session_id"].tolist() == [0, 1, 1]
    assert frame["session_change_flag"].tolist() == [1, 1, 0]
    assert frame["minutes_since_session_open"].tolist() == [535.0, 0.0, 5.0]
    # V30 wave 2 (2026-08-18): `is_ASIA` and `minutes_to_next_session_boundary`
    # left MODEL_NATIVE_CTX_CONT_SESSION_FIELDS and their producer went with
    # them; the boundary clock is still pinned by the two survivors above.
    assert "is_ASIA" not in frame.columns
    assert "minutes_to_next_session_boundary" not in frame.columns


def test_session_context_is_append_stable() -> None:
    labels = pd.date_range(
        "2026-07-01T06:30:00Z",
        periods=12,
        freq="5min",
    )
    prefix = pd.DataFrame(index=labels[:8])
    full = pd.DataFrame(index=labels)
    _add_session_features(prefix)
    _add_session_features(full)
    pd.testing.assert_frame_equal(prefix, full.iloc[:8])


def test_model_native_entry_context_accepts_exact_categorical_session_frame() -> None:
    _require_model_native_entry_context_frame(
        _valid_entry_context_frame(),
        context="test",
    )


@pytest.mark.parametrize(
    "missing_field",
    [
        "session_id",
        "minutes_since_session_open",
        "session_change_flag",
    ],
)
def test_model_native_entry_boundary_rejects_missing_context_before_feature_build(
    tmp_path: Path,
    missing_field: str,
) -> None:
    frame = _valid_entry_context_frame().drop(columns=[missing_field])
    builder = _early_validation_builder(tmp_path)

    with pytest.raises(
        RuntimeError,
        match=r"MODEL_NATIVE_ENTRY_CONTEXT_NO_DIRECTION.*missing categorical/session fields",
    ):
        builder.build_states(frame, [frame["time"].iloc[-1]])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("session_id", np.nan, "missing/non-finite"),
        ("session_id", 4, "outside semantic domain"),
        ("session_id", -1, "outside semantic domain"),
        ("session_id", 1.5, "non-integral category"),
        ("session_change_flag", np.nan, "missing/non-finite"),
        ("session_change_flag", 2, "outside semantic domain"),
        ("session_change_flag", 0.5, "non-integral flag"),
        ("minutes_since_session_open", np.nan, "missing/non-finite"),
        ("minutes_since_session_open", np.inf, "missing/non-finite"),
        ("minutes_since_session_open", -1, "disagrees with UTC-derived"),
        ("session_change_flag", 0, "disagrees with UTC-derived"),
    ],
)
def test_model_native_entry_boundary_rejects_invalid_context_without_coercion(
    tmp_path: Path,
    field: str,
    value: float,
    message: str,
) -> None:
    frame = _valid_entry_context_frame()
    frame[field] = frame[field].astype(np.float64)
    frame.loc[frame.index[-1], field] = value
    builder = _early_validation_builder(tmp_path)

    with pytest.raises(
        RuntimeError,
        match=rf"MODEL_NATIVE_ENTRY_CONTEXT_NO_DIRECTION.*{field}.*{message}",
    ):
        builder.build_states(frame, [frame["time"].iloc[-1]])


def test_model_native_entry_boundary_never_turns_unknown_session_into_asia(
    tmp_path: Path,
) -> None:
    frame = _valid_entry_context_frame()
    # The final bar becomes available at 16:00 UTC (US). Zero is in-domain, but accepting it would be
    # exactly the retired unknown-session -> ASIA soft fallback.
    frame.loc[frame.index[-1], "session_id"] = 0
    builder = _early_validation_builder(tmp_path)

    with pytest.raises(
        RuntimeError,
        match=r"session_id disagrees.*ASIA fallback forbidden",
    ):
        builder.build_states(frame, [frame["time"].iloc[-1]])


@pytest.mark.parametrize(
    ("times", "message"),
    [
        (["2026-07-01T00:00:00Z", pd.NaT], "missing/invalid timestamps"),
        (
            ["2026-07-01T00:00:00Z", "2026-07-01T00:00:00Z"],
            "timestamps are not unique",
        ),
        (
            ["2026-07-01T00:05:00Z", "2026-07-01T00:00:00Z"],
            "not strictly chronological",
        ),
    ],
)
def test_model_native_state_rejects_invalid_time_before_feature_build(
    tmp_path: Path, times: list[object], message: str
) -> None:
    builder = _early_validation_builder(tmp_path)
    with pytest.raises(RuntimeError, match=message):
        builder.prepare_frame(pd.DataFrame({"time": times}))


def _state_metadata() -> dict:
    """The exact live state-contract surface: common history, no rank artifact."""
    return {
        "schema_version": MODEL_NATIVE_STATE_SCHEMA_VERSION,
        "feature_history_start_utc": "2026-05-20T23:00:00Z",
        "feature_history_mode": MODEL_NATIVE_HISTORY_MODE,
        "split_reset_allowed": False,
        "runtime_rule_free": True,
        "entry_run_id": "MODEL_NATIVE_STATE_CONTRACT_PYTEST",
    }


def test_model_native_state_contract_accepts_the_exact_live_surface() -> None:
    contract = ModelNativeStateContract.from_metadata(_state_metadata())

    assert contract.feature_history_start_utc == pd.Timestamp("2026-05-20T23:00:00Z")
    assert contract.raw["feature_history_mode"] == MODEL_NATIVE_HISTORY_MODE
    assert contract.as_report()["schema_version"] == MODEL_NATIVE_STATE_SCHEMA_VERSION


@pytest.mark.parametrize("retired_field", sorted(RETIRED_RANK_STATE_FIELDS))
def test_model_native_state_contract_rejects_every_retired_rank_field(
    retired_field: str,
) -> None:
    """The retired TRAIN-rank state surface must fail closed, never pass through.

    A stale bundle carrying any rank-reference key is evidence that it was
    built against the retired fixed top-k ranking subsystem; admitting it
    would silently serve a contract the code no longer implements.
    """
    raw = _state_metadata()
    raw[retired_field] = "any-value"

    with pytest.raises(
        RuntimeError,
        match=rf"STATE_RETIRED_FIELDS_FORBIDDEN.*{retired_field}",
    ):
        ModelNativeStateContract.from_metadata(raw)


@pytest.mark.parametrize("stale_field", sorted(STALE_STATE_CONTRACT_FIELDS))
def test_model_native_state_contract_rejects_every_stale_field(
    stale_field: str,
) -> None:
    raw = _state_metadata()
    raw[stale_field] = "any-value"

    with pytest.raises(
        RuntimeError,
        match=rf"STATE_RETIRED_FIELDS_FORBIDDEN.*{stale_field}",
    ):
        validate_state_contract_metadata_v2(raw)


def test_retired_rank_state_fields_cover_the_whole_rank_reference_surface() -> None:
    """Every retired key names the rank subsystem; none collide with live keys."""
    assert RETIRED_RANK_STATE_FIELDS
    assert all(
        "rank" in name for name in RETIRED_RANK_STATE_FIELDS
    ), sorted(RETIRED_RANK_STATE_FIELDS)
    assert not (RETIRED_RANK_STATE_FIELDS & set(_state_metadata()))
    assert not (RETIRED_RANK_STATE_FIELDS & STALE_STATE_CONTRACT_FIELDS)


@pytest.mark.parametrize(
    "retired_schema",
    ["model_native_state_contract_v1", "model_native_state_contract_v2"],
)
def test_model_native_state_contract_rejects_retired_schema_and_stale_fields(
    retired_schema: str,
) -> None:
    raw = _state_metadata()
    raw["schema_version"] = retired_schema
    raw["frame_anchor_utc"] = "2026-05-21T00:00:00Z"

    with pytest.raises(RuntimeError, match="STATE_SCHEMA_INVALID"):
        ModelNativeStateContract.from_metadata(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("feature_history_mode", "per_split_reset", "STATE_HISTORY_MODE_INVALID"),
        ("split_reset_allowed", True, "STATE_SPLIT_RESET_FORBIDDEN"),
        ("runtime_rule_free", False, "STATE_RUNTIME_RULE_FREE_REQUIRED"),
        ("entry_run_id", "  ", "STATE_EXPLICIT_RUN_ID_MISSING"),
        ("feature_history_start_utc", "not-a-time", "STATE_TIMESTAMP_INVALID"),
    ],
)
def test_model_native_state_contract_fails_closed_on_invalid_live_field(
    field: str, value: object, message: str
) -> None:
    raw = _state_metadata()
    raw[field] = value

    with pytest.raises(RuntimeError, match=message):
        ModelNativeStateContract.from_metadata(raw)


def test_model_native_state_contract_requires_every_live_field() -> None:
    for field in sorted(_state_metadata()):
        if field == "schema_version":
            continue
        raw = _state_metadata()
        raw.pop(field)
        with pytest.raises(RuntimeError, match="STATE_FIELDS_MISSING|STATE_"):
            ModelNativeStateContract.from_metadata(raw)
