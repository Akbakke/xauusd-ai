from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_oanda_mutation_transport_failure_is_never_retried() -> None:
    import requests

    from gx1.execution.oanda_client import OandaAPIError, OandaClient

    class _FailingSession:
        headers: dict[str, str] = {}

        def __init__(self) -> None:
            self.calls = 0

        def request(self, **_kwargs: object) -> object:
            self.calls += 1
            raise requests.Timeout("response outcome unknown")

    session = _FailingSession()
    client = object.__new__(OandaClient)
    client.base_url = "https://api-fxpractice.oanda.com/v3"
    client.timeout = 1.0
    client.session = session

    with pytest.raises(OandaAPIError, match="outcome unknown"):
        client._request(
            "POST",
            "/accounts/test/orders",
            json={"order": {}},
            max_retries=3,
        )
    assert session.calls == 1


def test_oanda_client_binds_idempotency_to_client_extensions() -> None:
    from gx1.execution.oanda_client import OandaClient

    observed: dict[str, object] = {}
    client = object.__new__(OandaClient)
    client.account_id = "practice-account"

    def _request(
        method: str,
        path: str,
        *,
        json: dict[str, object],
    ) -> dict[str, object]:
        observed.update(
            {"method": method, "path": path, "json": json}
        )
        return {"orderCreateTransaction": {}}

    client._request = _request
    client.create_market_order(
        "XAU_USD",
        2,
        client_order_id="gx1-idempotent-order",
    )

    order = observed["json"]["order"]
    assert "clientOrderID" not in order
    assert order["clientExtensions"] == {
        "id": "gx1-idempotent-order",
        "tag": "GX1_V12",
    }


def test_runner_strict_credentials_reject_invalid_environment_at_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.execution.oanda_credentials import load_oanda_credentials

    monkeypatch.setenv("OANDA_ENV", "practcie")
    monkeypatch.setenv("OANDA_API_TOKEN", "unit-token")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "unit-account")

    with pytest.raises(ValueError, match="Invalid OANDA_ENV"):
        load_oanda_credentials(
            prod_baseline=True,
            require_live_latch=True,
        )


def _runtime_sizing_authority_for_broker_fact_tests():
    import json

    from gx1.contracts.entry_model_native_sizing_authority_v1 import (
        ValidatedLearnedSizingAuthority,
    )

    return ValidatedLearnedSizingAuthority(
        authority_json="{}",
        adoption_json="{}",
        calibration_json=json.dumps(
            {
                "instrument_constraints": {
                    "instrument": "XAU_USD",
                    "account_currency": "USD",
                    "quote_currency": "USD",
                    "unit_step": 1,
                    "minimum_order_units": 1,
                    "maximum_gross_xau_units": 1000,
                    "margin_rate": 0.05,
                }
            }
        ),
        proof_json="{}",
        joint_proof_json="{}",
        candidate_bundle_authority_json="{}",
        content_hash_key=(),
        file_stats=(),
    )


def _broker_fact_client(
    *,
    hedging_enabled: bool,
    transaction_ids: tuple[str, str, str],
    trades: list[dict] | None = None,
):
    account_tx, instrument_tx, exposure_tx = transaction_ids
    return SimpleNamespace(
        get_account_summary=lambda: {
            "account": {
                "currency": "USD",
                "hedgingEnabled": hedging_enabled,
                "NAV": "10000",
                "balance": "10000",
                "marginAvailable": "1000",
                "marginUsed": "0",
            },
            "lastTransactionID": account_tx,
        },
        get_account_instruments=lambda _instruments: {
            "instruments": [
                {
                    "name": "XAU_USD",
                    "tradeUnitsPrecision": 0,
                    "minimumTradeSize": "1",
                    "maximumOrderUnits": "100000",
                    "marginRate": "0.05",
                }
            ],
            "lastTransactionID": instrument_tx,
        },
        get_open_trades=lambda: {
            "trades": [] if trades is None else trades,
            "lastTransactionID": exposure_tx,
        },
    )


def test_raw_base28_frame_contains_only_exact_native_m1_identity() -> None:
    from gx1.execution import v12_canonical_incremental as incremental

    timestamp = pd.Timestamp("2026-07-16T12:04:00Z")
    m1 = pd.DataFrame(
        {
            column: [2400.0 + offset]
            for offset, column in enumerate(
                incremental.M1_MARKET_IDENTITY_COLUMNS
            )
        },
        index=pd.DatetimeIndex([timestamp]),
    )

    m1["stale_context"] = 999.0
    frame = incremental._build_raw_base28_owned_frame(m1)

    assert tuple(frame.columns) == incremental.RAW_BASE28_COLUMNS
    pd.testing.assert_frame_equal(
        frame,
        m1.loc[:, list(incremental.RAW_BASE28_COLUMNS)].rename_axis("time"),
    )


def test_raw_base28_rejects_missing_native_m1_field() -> None:
    from gx1.execution import v12_canonical_incremental as incremental

    timestamp = pd.Timestamp("2026-07-16T12:04:00Z")
    m1 = pd.DataFrame(
        {
            column: [2400.0 + offset]
            for offset, column in enumerate(incremental.RAW_BASE28_COLUMNS)
            if column != "ask_close"
        },
        index=pd.DatetimeIndex([timestamp]),
    )

    with pytest.raises(RuntimeError, match="RAW_BASE28_M1_FIELDS_MISSING"):
        incremental._build_raw_base28_owned_frame(m1)


@pytest.mark.parametrize(
    ("volume", "error_code"),
    [
        (0.0, "PLUS5_VOLUME_INVALID"),
        (10.5, "PLUS5_VOLUME_INVALID"),
        (np.nan, "PLUS5_SOURCE_NONFINITE"),
    ],
)
def test_plus5_rejects_unobserved_volume_instead_of_using_one(
    volume: float,
    error_code: str,
) -> None:
    from gx1.execution import v12_canonical_incremental as incremental

    frame = pd.DataFrame(
        {
            "open": [2400.0, 2400.5],
            "high": [2401.0, 2401.5],
            "low": [2399.0, 2399.5],
            "close": [2400.5, 2401.0],
            "volume": [10.0, volume],
        }
    )

    with pytest.raises(RuntimeError, match=error_code):
        incremental._compute_plus5_features(frame)


def test_plus5_rejects_missing_volume_source() -> None:
    from gx1.execution import v12_canonical_incremental as incremental

    frame = pd.DataFrame(
        {
            "open": [2400.0],
            "high": [2401.0],
            "low": [2399.0],
            "close": [2400.5],
        }
    )

    with pytest.raises(RuntimeError, match="PLUS5_SOURCE_MISSING"):
        incremental._compute_plus5_features(frame)


def test_plus5_serve_owner_rejects_zero_volume_instead_of_using_one() -> None:
    from gx1.execution.v12_state_from_prebuilt import PrebuiltStateLoader

    frame = pd.DataFrame(
        {
            "open": [2400.0, 2400.5],
            "high": [2401.0, 2401.5],
            "low": [2399.0, 2399.5],
            "close": [2400.5, 2401.0],
            "volume": [10.0, 0.0],
        }
    )

    with pytest.raises(RuntimeError, match="PLUS5_VOLUME_INVALID"):
        PrebuiltStateLoader()._augment_cv3_with_v1_legacy(frame)


def test_plus5_build_and_serve_delegate_to_identical_formula_owner() -> None:
    from gx1.execution import v12_canonical_incremental as incremental
    from gx1.execution.v12_state_from_prebuilt import PrebuiltStateLoader
    from gx1.features.basic_v1 import PLUS5_FEATURES

    n = 64
    close = 2400.0 + np.linspace(0.0, 3.0, n)
    frame = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": (10 + np.arange(n) % 91).astype(np.float64),
        }
    )

    built = incremental._compute_plus5_features(frame)
    served = PrebuiltStateLoader()._augment_cv3_with_v1_legacy(frame)

    pd.testing.assert_frame_equal(
        built[list(PLUS5_FEATURES)],
        served[list(PLUS5_FEATURES)],
    )


def _collector_frame(times: list[str]) -> pd.DataFrame:
    parsed = pd.to_datetime(times, utc=True)
    rows: list[dict[str, object]] = []
    for position, timestamp in enumerate(parsed):
        middle = 3300.0 + position
        rows.append(
            {
                "time": timestamp,
                "open": middle,
                "high": middle + 1.0,
                "low": middle - 1.0,
                "close": middle + 0.25,
                "bid_open": middle - 0.1,
                "bid_high": middle + 0.9,
                "bid_low": middle - 1.1,
                "bid_close": middle + 0.15,
                "ask_open": middle + 0.1,
                "ask_high": middle + 1.1,
                "ask_low": middle - 0.9,
                "ask_close": middle + 0.35,
                "volume": 10 + position,
            }
        )
    return pd.DataFrame(rows)


def test_oanda_client_requires_literal_candle_completion_flag() -> None:
    from gx1.execution.oanda_client import OandaAPIError, OandaClient

    client = object.__new__(OandaClient)
    client._request = lambda *args, **kwargs: {
        "instrument": "XAU_USD",
        "granularity": "M1",
        "candles": [
            {
                "time": "2026-07-29T15:00:00Z",
                "mid": {
                    "o": "3300",
                    "h": "3301",
                    "l": "3299",
                    "c": "3300.5",
                },
                "volume": 10,
            }
        ]
    }
    with pytest.raises(OandaAPIError, match="completion flag missing or invalid"):
        client.get_candles("XAU_USD", "M1", count=1)


def test_oanda_client_rejects_non_object_response_as_latchable_contract_error() -> None:
    from gx1.execution.oanda_client import OandaDataContractError, OandaClient

    client = object.__new__(OandaClient)
    client._request = lambda *args, **kwargs: []
    with pytest.raises(
        OandaDataContractError,
        match="response root is not an object",
    ) as caught:
        client.get_candles("XAU_USD", "M1", count=1)
    assert len(caught.value.evidence["source_response_sha256"]) == 64


@pytest.mark.parametrize(
    ("from_ts", "to_ts", "match"),
    [
        (pd.Timestamp("2026-07-29T15:00:00Z"), None, "provided together"),
        (
            pd.Timestamp("2026-07-29T17:00:00+02:00"),
            pd.Timestamp("2026-07-29T17:01:00+02:00"),
            "explicitly UTC",
        ),
        (
            pd.Timestamp("2026-07-29T15:00:01Z"),
            pd.Timestamp("2026-07-29T15:01:00Z"),
            "granularity-aligned",
        ),
        (
            pd.Timestamp("2026-07-29T15:01:00Z"),
            pd.Timestamp("2026-07-29T15:00:00Z"),
            "increasing",
        ),
    ],
)
def test_oanda_client_requires_exact_half_open_utc_request_interval(
    from_ts: pd.Timestamp,
    to_ts: pd.Timestamp | None,
    match: str,
) -> None:
    from gx1.execution.oanda_client import OandaDataContractError, OandaClient

    client = object.__new__(OandaClient)
    client._request = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("invalid interval must fail before network access")
    )
    with pytest.raises(OandaDataContractError, match=match):
        client.get_candles(
            "XAU_USD",
            "M1",
            from_ts=from_ts,
            to_ts=to_ts,
        )


def test_oanda_client_requires_literal_mid_bid_ask_components() -> None:
    from gx1.execution.oanda_client import OandaAPIError, OandaClient

    client = object.__new__(OandaClient)
    client._request = lambda *args, **kwargs: {
        "instrument": "XAU_USD",
        "granularity": "M1",
        "candles": [
            {
                "complete": True,
                "time": "2026-07-29T15:00:00Z",
                "mid": {
                    "o": "3300",
                    "h": "3301",
                    "l": "3299",
                    "c": "3300.5",
                },
                "volume": 10,
            }
        ]
    }
    with pytest.raises(OandaAPIError, match="literal M/B/A"):
        client.get_candles("XAU_USD", "M1", count=1)


def _oanda_candle(
    timestamp: str,
    *,
    complete: bool = True,
) -> dict[str, object]:
    return {
        "complete": complete,
        "time": timestamp,
        "mid": {"o": "3300", "h": "3301", "l": "3299", "c": "3300.5"},
        "bid": {
            "o": "3299.9",
            "h": "3300.9",
            "l": "3298.9",
            "c": "3300.4",
        },
        "ask": {
            "o": "3300.1",
            "h": "3301.1",
            "l": "3299.1",
            "c": "3300.6",
        },
        "volume": 10,
    }


def test_oanda_client_rejects_out_of_interval_candle_instead_of_dropping_it() -> None:
    from gx1.execution.oanda_client import OandaDataContractError, OandaClient

    client = object.__new__(OandaClient)
    client._request = lambda *args, **kwargs: {
        "instrument": "XAU_USD",
        "granularity": "M1",
        "candles": [_oanda_candle("2026-07-29T15:01:00Z")],
    }
    with pytest.raises(OandaDataContractError, match="outside the requested"):
        client.get_candles(
            "XAU_USD",
            "M1",
            from_ts=pd.Timestamp("2026-07-29T15:00:00Z"),
            to_ts=pd.Timestamp("2026-07-29T15:01:00Z"),
        )


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-07-29T15:00:00",
        "2026-07-29 15:00:00Z",
        "2026-07-29T17:00:00+02:00",
        1785337200,
    ],
)
def test_oanda_client_requires_explicit_utc_rfc3339_response_time(
    timestamp: object,
) -> None:
    from gx1.execution.oanda_client import OandaDataContractError, OandaClient

    client = object.__new__(OandaClient)
    client._request = lambda *args, **kwargs: {
        "instrument": "XAU_USD",
        "granularity": "M1",
        "candles": [_oanda_candle(timestamp)],
    }
    with pytest.raises(
        OandaDataContractError,
        match="explicit UTC RFC3339",
    ):
        client.get_candles("XAU_USD", "M1", count=1)


@pytest.mark.parametrize(
    ("response_instrument", "response_granularity"),
    [("GBP_USD", "M1"), ("XAU_USD", "H4")],
)
def test_oanda_client_rejects_response_instrument_or_timeframe_mismatch(
    response_instrument: str,
    response_granularity: str,
) -> None:
    from gx1.execution.oanda_client import OandaDataContractError, OandaClient

    client = object.__new__(OandaClient)
    client._request = lambda *args, **kwargs: {
        "instrument": response_instrument,
        "granularity": response_granularity,
        "candles": [_oanda_candle("2026-07-29T15:00:00Z")],
    }
    with pytest.raises(OandaDataContractError, match="mismatch"):
        client.get_candles("XAU_USD", "M1", count=1)


def test_oanda_client_rejects_off_grid_or_duplicate_response_without_repair() -> None:
    from gx1.execution.oanda_client import OandaDataContractError, OandaClient

    client = object.__new__(OandaClient)
    payload = {
        "instrument": "XAU_USD",
        "granularity": "M1",
        "candles": [_oanda_candle("2026-07-29T15:00:59Z")],
    }
    client._request = lambda *args, **kwargs: payload
    with pytest.raises(OandaDataContractError, match="exactly granularity-aligned"):
        client.get_candles("XAU_USD", "M1", count=1)

    payload["candles"] = [
        _oanda_candle("2026-07-29T15:00:00Z"),
        _oanda_candle("2026-07-29T15:00:00Z"),
    ]
    with pytest.raises(OandaDataContractError, match="order/uniqueness"):
        client.get_candles("XAU_USD", "M1", count=2)


def test_oanda_client_rejects_invalid_geometry_and_noninteger_volume() -> None:
    from gx1.execution.oanda_client import OandaDataContractError, OandaClient

    client = object.__new__(OandaClient)
    candle = _oanda_candle("2026-07-29T15:00:00Z")
    candle["volume"] = -1.5
    client._request = lambda *args, **kwargs: {
        "instrument": "XAU_USD",
        "granularity": "M1",
        "candles": [candle],
    }
    with pytest.raises(OandaDataContractError, match="non-negative integer"):
        client.get_candles("XAU_USD", "M1", count=1)

    candle["volume"] = 10
    candle["ask"] = {
        "o": "3299.0",
        "h": "3300.0",
        "l": "3298.0",
        "c": "3299.5",
    }
    with pytest.raises(OandaDataContractError, match="BID_ASK_GEOMETRY_INVALID"):
        client.get_candles("XAU_USD", "M1", count=1)
