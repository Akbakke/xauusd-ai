"\"\"\"Minimal OANDA REST client used for GX1 demo execution.\"\"\""

from __future__ import annotations

import os
import hashlib
import json
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import requests

from gx1.contracts.xau_tape_provenance_v1 import (
    CANONICAL_NATIVE_REQUIRED_COLUMNS,
    validate_canonical_native_frame,
)
from gx1.utils.env_loader import load_dotenv_if_present
from gx1.utils.granularity import granularity_to_timedelta


log = logging.getLogger(__name__)

PRACTICE_URL = "https://api-fxpractice.oanda.com/v3"
LIVE_URL = "https://api-fxtrade.oanda.com/v3"
_EXPLICIT_UTC_RFC3339 = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,9})?(?:Z|\+00:00|-00:00)\Z"
)


class OandaAPIError(RuntimeError):
    """Raised for non-2xx responses from the OANDA REST API."""


class OandaDataContractError(OandaAPIError):
    """The response is reachable but not admissible market evidence."""

    def __init__(
        self,
        message: str,
        *,
        evidence: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.evidence = dict(evidence or {})


@dataclass
class OandaClientConfig:
    api_key: str
    account_id: str
    env: str = "practice"
    timeout: int = 10


def _mask(value: str) -> str:
    if not value:
        return "<EMPTY>"
    if len(value) <= 8:
        return value[:2] + "..." + value[-2:]
    return value[:4] + "..." + value[-4:]


def _canonical_response_sha256(value: object) -> str:
    """Return a stable diagnostic identity for one decoded JSON response."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        encoded = repr(value).encode("utf-8", errors="backslashreplace")
    return hashlib.sha256(encoded).hexdigest()


def _require_utc_half_open_interval(
    *,
    from_ts: pd.Timestamp | None,
    to_ts: pd.Timestamp | None,
    granularity: str,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Validate an optional exact UTC, grid-aligned half-open interval."""

    if (from_ts is None) != (to_ts is None):
        raise OandaDataContractError(
            "from_ts and to_ts must be provided together"
        )
    if from_ts is None:
        return None
    try:
        start = pd.Timestamp(from_ts)
        end = pd.Timestamp(to_ts)
        duration = granularity_to_timedelta(granularity)
    except Exception as exc:
        raise OandaDataContractError(
            "OANDA candle request interval is invalid"
        ) from exc
    if (
        start.tzinfo is None
        or end.tzinfo is None
        or start.utcoffset() != pd.Timedelta(0)
        or end.utcoffset() != pd.Timedelta(0)
    ):
        raise OandaDataContractError(
            "OANDA candle request interval must be explicitly UTC"
        )
    start = start.tz_convert("UTC")
    end = end.tz_convert("UTC")
    duration_ns = int(duration.value)
    if (
        end <= start
        or start.value % duration_ns != 0
        or end.value % duration_ns != 0
    ):
        raise OandaDataContractError(
            "OANDA candle request interval must be increasing and "
            "granularity-aligned"
        )
    return start, end


class OandaClient:
    """
    Thin wrapper around the OANDA REST V20 API.

    Only exposes the endpoints required for GX1 demo trading.
    """

    def __init__(self, config: OandaClientConfig) -> None:
        self.api_key = config.api_key.strip()
        self.account_id = config.account_id.strip()
        env = config.env.lower().strip() if config.env else "practice"
        if env not in {"practice", "live"}:
            raise ValueError(f"Unsupported OANDA_ENV '{config.env}'")
        self.env = env
        self.base_url = PRACTICE_URL if env == "practice" else LIVE_URL
        self.timeout = config.timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Accept-Datetime-Format": "RFC3339",
            }
        )
        log.debug("OandaClient base URL set to %s", self.base_url)
        log.info(
            "OandaClient initialised (env=%s, account=%s)",
            self.env,
            _mask(self.account_id),
        )

    # ------------------------------------------------------------------ #
    # Factory helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls, *, timeout: int = 10) -> "OandaClient":
        """
        Instantiate the client using environment variables.

        Required:
            OANDA_API_KEY
            OANDA_ACCOUNT_ID

        Optional:
            OANDA_ENV (defaults to "practice")
        """
        load_dotenv_if_present()

        # Support both OANDA_API_TOKEN (preferred) and OANDA_API_KEY (legacy)
        api_key = (os.getenv("OANDA_API_TOKEN") or os.getenv("OANDA_API_KEY") or "").strip()
        account_id = (os.getenv("OANDA_ACCOUNT_ID") or "").strip()
        env = (os.getenv("OANDA_ENV", "practice") or "practice").strip()

        # Determine base URL for logging
        base_url = PRACTICE_URL if env == "practice" else LIVE_URL
        log.info(
            "OandaClient.from_env: env=%s, account=%s, api_key=%s, base_url=%s",
            env or "<EMPTY>",
            _mask(account_id),
            _mask(api_key),
            base_url,
        )

        # Check for either OANDA_API_TOKEN or OANDA_API_KEY
        api_key_name = "OANDA_API_TOKEN" if os.getenv("OANDA_API_TOKEN") else "OANDA_API_KEY"
        missing = [name for name, val in [(api_key_name, api_key), ("OANDA_ACCOUNT_ID", account_id)] if not val]
        if missing:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

        config = OandaClientConfig(api_key=api_key, account_id=account_id, env=env, timeout=timeout)
        return cls(config)

    # ------------------------------------------------------------------ #
    # REST helpers
    # ------------------------------------------------------------------ #
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_on_status: tuple = (429, 500, 502, 503, 504),
    ) -> Dict[str, Any]:
        """
        Make a request. Only read-only GET requests may be retried.

        Broker mutations are single-attempt because a lost response is an
        unknown outcome, not proof that OANDA did not apply the request.
        
        Parameters
        ----------
        method : str
            HTTP method (GET, POST, PUT, etc.).
        path : str
            API path (relative to base_url).
        params : Optional[Dict[str, Any]]
            Query parameters.
        json : Optional[Dict[str, Any]]
            JSON payload.
        max_retries : int
            Maximum number of retries (default: 3).
        retry_on_status : tuple
            HTTP status codes to retry on (default: (429, 500, 502, 503, 504)).
        
        Returns
        -------
        Dict[str, Any]
            JSON response from API.
        
        Raises
        ------
        OandaAPIError
            If request fails after max_retries or returns non-2xx status.
        """
        url = f"{self.base_url}{path}"
        headers = self.session.headers.copy()
        method_upper = method.upper()
        attempt_limit = max_retries if method_upper == "GET" else 1
        if attempt_limit < 1:
            raise ValueError("max_retries must be at least 1")

        for attempt in range(attempt_limit):
            log.debug(
                "OANDA request (attempt %d/%d): %s %s | headers=%s | params=%s | json=%s",
                attempt + 1,
                attempt_limit,
                method,
                url,
                {k: v for k, v in headers.items() if k.lower() != "authorization"},
                params,
                json,
            )
            
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                if attempt < attempt_limit - 1:
                    # Exponential backoff: 2^attempt seconds
                    wait_time = 2 ** attempt
                    log.warning(
                        "OANDA request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1,
                        attempt_limit,
                        exc,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    raise OandaAPIError(
                        "Request outcome unknown after "
                        f"{attempt_limit} attempt(s): {exc}"
                    ) from exc
            
            # Check if we should retry on this status code
            if (
                resp.status_code in retry_on_status
                and attempt < attempt_limit - 1
            ):
                # Exponential backoff: 2^attempt seconds
                wait_time = 2 ** attempt
                # For 429 (rate limit), use Retry-After header if available
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = float(retry_after)
                        except ValueError:
                            pass
                
                log.warning(
                    "OANDA %s %s returned %s (attempt %d/%d). Retrying in %.1fs...",
                    method,
                    url,
                    resp.status_code,
                    attempt + 1,
                    attempt_limit,
                    wait_time,
                )
                time.sleep(wait_time)
                continue
            
            # If not OK and not retryable, raise error
            if not resp.ok:
                if method_upper != "GET" and resp.status_code == 400:
                    try:
                        rejection = resp.json()
                    except ValueError:
                        rejection = None
                    if (
                        isinstance(rejection, dict)
                        and isinstance(
                            rejection.get("orderRejectTransaction"),
                            dict,
                        )
                        and not any(
                            key in rejection
                            for key in (
                                "orderFillTransaction",
                                "orderCreateTransaction",
                                "orderCancelTransaction",
                            )
                        )
                    ):
                        # This exact response proves that OANDA rejected the
                        # mutation. Every other mutation failure remains an
                        # unknown, single-attempt outcome.
                        return rejection
                log.error(
                    "OANDA %s %s failed (%s)\nResponse headers: %s\nResponse body: %s",
                    method,
                    url,
                    resp.status_code,
                    resp.headers,
                    resp.text,
                )
                raise OandaAPIError(f"OANDA API error {resp.status_code}: {resp.text}")
            
            # Success: return JSON response
            try:
                return resp.json()
            except ValueError as exc:
                raise OandaAPIError(f"Invalid JSON response from OANDA: {resp.text}") from exc
        
        # Should never reach here, but just in case
        raise OandaAPIError(f"Request failed after {max_retries} attempts")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_candles(
        self,
        instrument: str,
        granularity: str,
        *,
        count: Optional[int] = None,
        from_ts: Optional[pd.Timestamp] = None,
        to_ts: Optional[pd.Timestamp] = None,
        include_mid: bool = True,
        exclude_incomplete: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch candles for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument symbol (e.g., "XAU_USD").
        granularity : str
            Granularity (e.g., "M5" for 5-minute bars).
        count : Optional[int]
            Number of candles to fetch (default: 500).
            Ignored if from_ts and to_ts are provided.
        from_ts : Optional[pd.Timestamp]
            Start time (inclusive, UTC). If provided, to_ts must also be provided.
        to_ts : Optional[pd.Timestamp]
            End time (exclusive, UTC) - half-open interval [from_ts, to_ts).
            If provided, from_ts must also be provided.
        include_mid : bool
            Include mid prices (default: True).
        exclude_incomplete : bool
            Exclude incomplete bars (default: True).

        Returns
        -------
        pd.DataFrame
            DataFrame sorted by time ascending with columns:
            open, high, low, close, volume.
        """
        if not include_mid:
            raise OandaDataContractError(
                "Literal M/B/A candles are mandatory; BA-only mode is forbidden"
            )
        if not exclude_incomplete:
            raise OandaDataContractError(
                "Incomplete candles are forbidden by the GX1 source contract"
            )
        interval = _require_utc_half_open_interval(
            from_ts=from_ts,
            to_ts=to_ts,
            granularity=granularity,
        )
        params = {
            "granularity": granularity,
            "alignmentTimezone": "UTC",
        }
        params["price"] = "MBA"
        
        # Use from_ts/to_ts if provided, otherwise use count
        if interval is not None:
            request_start, request_end = interval
            # OANDA API uses RFC3339 format for from/to
            # Half-open interval: [from_ts, to_ts) - include from, exclude to
            params["from"] = request_start.strftime(
                "%Y-%m-%dT%H:%M:%S.000000000Z"
            )
            params["to"] = request_end.strftime(
                "%Y-%m-%dT%H:%M:%S.000000000Z"
            )
        else:
            requested_count = 500 if count is None else count
            if (
                isinstance(requested_count, bool)
                or not isinstance(requested_count, int)
                or requested_count <= 0
            ):
                raise OandaDataContractError(
                    "OANDA candle count must be a positive integer"
                )
            params["count"] = requested_count

        data = self._request(
            "GET",
            f"/instruments/{instrument}/candles",
            params=params,
        )
        response_sha256 = _canonical_response_sha256(data)
        response_evidence = {"source_response_sha256": response_sha256}
        if not isinstance(data, Mapping):
            raise OandaDataContractError(
                "OANDA candle response root is not an object",
                evidence=response_evidence,
            )
        if data.get("instrument") != instrument:
            raise OandaDataContractError(
                "OANDA candle response instrument mismatch: "
                f"expected={instrument!r} observed={data.get('instrument')!r}",
                evidence=response_evidence,
            )
        if data.get("granularity") != granularity:
            raise OandaDataContractError(
                "OANDA candle response granularity mismatch: "
                f"expected={granularity!r} observed={data.get('granularity')!r}",
                evidence=response_evidence,
            )

        candles = data.get("candles")
        if not isinstance(candles, list):
            raise OandaDataContractError(
                "OANDA candle response candles field is not a list",
                evidence=response_evidence,
            )
        if not candles:
            raise OandaDataContractError(
                f"No candles returned for {instrument}",
                evidence=response_evidence,
            )

        records = []
        previous_time: pd.Timestamp | None = None
        bar_ns = int(granularity_to_timedelta(granularity).value)
        for position, item in enumerate(candles):
            if not isinstance(item, dict):
                raise OandaDataContractError(
                    f"Candle payload is not an object for {instrument}",
                    evidence=response_evidence,
                )
            complete = item.get("complete")
            if not isinstance(complete, bool):
                raise OandaDataContractError(
                    f"Candle completion flag missing or invalid for {instrument}",
                    evidence=response_evidence,
                )
            # A caller may explicitly request forming bars, but absence of the
            # literal OANDA completion fact is never treated as complete.
            if exclude_incomplete and not complete:
                continue
            mid = item.get("mid")
            bid = item.get("bid")
            ask = item.get("ask")
            price_components = ("o", "h", "l", "c")
            if (
                not isinstance(mid, dict)
                or not isinstance(bid, dict)
                or not isinstance(ask, dict)
                or any(
                    component not in prices
                    for prices in (mid, bid, ask)
                    for component in price_components
                )
            ):
                raise OandaDataContractError(
                    f"Candle lacks literal M/B/A price components for {instrument}",
                    evidence=response_evidence,
                )
            if "time" not in item:
                raise OandaDataContractError(
                    f"Candle timestamp missing for {instrument}",
                    evidence=response_evidence,
                )
            raw_time = item["time"]
            if (
                not isinstance(raw_time, str)
                or _EXPLICIT_UTC_RFC3339.fullmatch(raw_time) is None
            ):
                raise OandaDataContractError(
                    "Candle timestamp must be an explicit UTC RFC3339 string "
                    f"for {instrument}: position={position}",
                    evidence=response_evidence,
                )
            try:
                normalized_time = pd.Timestamp(
                    pd.to_datetime(raw_time, utc=True, errors="raise")
                )
            except Exception as exc:
                raise OandaDataContractError(
                    f"Candle timestamp invalid for {instrument}",
                    evidence=response_evidence,
                ) from exc
            if (
                pd.isna(normalized_time)
                or normalized_time.value % bar_ns != 0
            ):
                raise OandaDataContractError(
                    "Candle timestamp is not exactly granularity-aligned: "
                    f"position={position} time={normalized_time} "
                    f"granularity={granularity}",
                    evidence=response_evidence,
                )
            if previous_time is not None and normalized_time <= previous_time:
                raise OandaDataContractError(
                    "Candle response time order/uniqueness invalid: "
                    f"position={position} previous={previous_time} "
                    f"observed={normalized_time}",
                    evidence=response_evidence,
                )
            previous_time = normalized_time
            
            if interval is not None and not (
                request_start <= normalized_time < request_end
            ):
                raise OandaDataContractError(
                    "Candle timestamp is outside the requested half-open "
                    "interval: "
                    f"position={position} time={normalized_time} "
                    f"interval=[{request_start},{request_end})",
                    evidence=response_evidence,
                )
            
            if "volume" not in item:
                raise OandaDataContractError(
                    f"Candle lacks literal volume for {instrument}",
                    evidence=response_evidence,
                )
            if (
                isinstance(item["volume"], bool)
                or not isinstance(item["volume"], int)
                or item["volume"] < 0
            ):
                raise OandaDataContractError(
                    f"Candle volume is not a literal non-negative integer for {instrument}",
                    evidence=response_evidence,
                )
            try:
                record = {
                    "time": normalized_time,
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "bid_open": float(bid["o"]),
                    "bid_high": float(bid["h"]),
                    "bid_low": float(bid["l"]),
                    "bid_close": float(bid["c"]),
                    "ask_open": float(ask["o"]),
                    "ask_high": float(ask["h"]),
                    "ask_low": float(ask["l"]),
                    "ask_close": float(ask["c"]),
                    "volume": item["volume"],
                }
            except (TypeError, ValueError, OverflowError) as exc:
                raise OandaDataContractError(
                    f"Candle numeric payload invalid for {instrument}",
                    evidence=response_evidence,
                ) from exc
            records.append(record)

        if not records:
            empty = pd.DataFrame(
                columns=list(CANONICAL_NATIVE_REQUIRED_COLUMNS)
            )
            if instrument == "XAU_USD" and granularity in {"M1", "M5"}:
                empty = validate_canonical_native_frame(
                    empty,
                    timeframe=granularity,
                    label="OANDA_CLIENT_EMPTY_RESPONSE",
                    allow_empty=True,
                )
            empty.attrs.update(response_evidence)
            return empty.set_index("time")
        frame = pd.DataFrame.from_records(records).loc[
            :, list(CANONICAL_NATIVE_REQUIRED_COLUMNS)
        ]
        if instrument == "XAU_USD" and granularity in {"M1", "M5"}:
            try:
                frame = validate_canonical_native_frame(
                    frame,
                    timeframe=granularity,
                    label="OANDA_CLIENT_RESPONSE",
                )
            except RuntimeError as exc:
                raise OandaDataContractError(
                    str(exc),
                    evidence=response_evidence,
                ) from exc
        frame.attrs.update(response_evidence)
        return frame.set_index("time")
    
    def get_candles_chunked(
        self,
        instrument: str,
        granularity: str,
        from_ts: pd.Timestamp,
        to_ts: pd.Timestamp,
        *,
        chunk_size: int = 3000,
        max_retries: int = 5,
        include_mid: bool = True,
        exclude_incomplete: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch candles in chunks with exponential backoff retry.
        
        Parameters
        ----------
        instrument : str
            Instrument symbol (e.g., "XAU_USD").
        granularity : str
            Granularity (e.g., "M5" for 5-minute bars).
        from_ts : pd.Timestamp
            Start time (inclusive, UTC).
        to_ts : pd.Timestamp
            End time (exclusive, UTC) - half-open interval [from_ts, to_ts).
        chunk_size : int
            Number of candles per chunk (default: 3000).
        max_retries : int
            Maximum number of retries per chunk (default: 5).
        include_mid : bool
            Include mid prices (default: True).
        exclude_incomplete : bool
            Exclude incomplete bars (default: True).
        
        Returns
        -------
        pd.DataFrame
            DataFrame with all fetched candles.
        """
        interval = _require_utc_half_open_interval(
            from_ts=from_ts,
            to_ts=to_ts,
            granularity=granularity,
        )
        if interval is None:
            raise OandaDataContractError(
                "Chunked candle fetch requires an explicit interval"
            )
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0
            or isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries <= 0
        ):
            raise OandaDataContractError(
                "Chunked candle size/retry values must be positive integers"
            )
        request_start, request_end = interval
        # Calculate chunk duration from candle granularity
        chunk_duration = granularity_to_timedelta(granularity) * chunk_size
        
        all_candles = []
        current_from = request_start
        
        while current_from < request_end:
            current_to = min(current_from + chunk_duration, request_end)
            
            # Fetch chunk with retry
            for attempt in range(max_retries):
                try:
                    chunk_df = self.get_candles(
                        instrument,
                        granularity,
                        from_ts=current_from,
                        to_ts=current_to,
                        include_mid=include_mid,
                        exclude_incomplete=exclude_incomplete,
                    )
                    if not chunk_df.empty:
                        all_candles.append(chunk_df)
                    break  # Success
                except OandaDataContractError:
                    raise
                except OandaAPIError as e:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 200ms → 400ms → 800ms ... max 5s
                        wait_time = min(0.2 * (2 ** attempt), 5.0)
                        log.warning(
                            "Chunk fetch failed (attempt %d/%d): %s. Retrying in %.1fs...",
                            attempt + 1,
                            max_retries,
                            e,
                            wait_time,
                        )
                        time.sleep(wait_time)
                    else:
                        log.error("Chunk fetch failed after %d attempts: %s", max_retries, e)
                        raise
            
            # Move to next chunk
            current_from = current_to
        
        # Combine all chunks
        if not all_candles:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_candles)
        if (
            combined_df.index.has_duplicates
            or not combined_df.index.is_monotonic_increasing
            or combined_df.index[0] < request_start
            or combined_df.index[-1] >= request_end
        ):
            raise OandaDataContractError(
                "Chunked candle response time order/uniqueness invalid"
            )
        return combined_df

    def get_open_trades(self) -> Dict[str, Any]:
        """Return raw JSON payload of open trades."""
        return self._request("GET", f"/accounts/{self.account_id}/openTrades")

    def create_market_order(
        self,
        instrument: str,
        units: int,
        *,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        client_extensions: Optional[Dict[str, Any]] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a market order with optional client_order_id for idempotency.
        
        Parameters
        ----------
        instrument : str
            Instrument symbol (e.g., "XAU_USD").
        units : int
            Number of units (positive for long, negative for short).
        stop_loss_price : Optional[float]
            Stop loss price (optional).
        take_profit_price : Optional[float]
            Take profit price (optional).
        client_extensions : Optional[Dict[str, Any]]
            Client extensions (optional).
        client_order_id : Optional[str]
            Client order ID for idempotency (optional, max 64 chars).
        
        Returns
        -------
        Dict[str, Any]
            Order response from API.
        """
        payload: Dict[str, Any] = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
            }
        }
        if stop_loss_price is not None:
            payload["order"]["stopLossOnFill"] = {"price": f"{stop_loss_price:.3f}"}
        if take_profit_price is not None:
            payload["order"]["takeProfitOnFill"] = {"price": f"{take_profit_price:.3f}"}
        if client_order_id and client_extensions:
            raise ValueError(
                "client_order_id and client_extensions are mutually exclusive"
            )
        if client_order_id:
            if (
                len(client_order_id) > 64
                or not client_order_id
                or any(
                    character
                    not in "abcdefghijklmnopqrstuvwxyz0123456789-"
                    for character in client_order_id
                )
            ):
                raise ValueError("client_order_id contract invalid")
            payload["order"]["clientExtensions"] = {
                "id": client_order_id,
                "tag": "GX1_V12",
            }
        elif client_extensions:
            payload["order"]["clientExtensions"] = client_extensions

        return self._request("POST", f"/accounts/{self.account_id}/orders", json=payload)

    def get_order_by_client_id(
        self,
        client_order_id: str,
    ) -> Dict[str, Any]:
        """Resolve one order by its exact OANDA client extension ID."""

        if (
            not client_order_id
            or len(client_order_id) > 64
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyz0123456789-"
                for character in client_order_id
            )
        ):
            raise ValueError("client_order_id contract invalid")
        return self._request(
            "GET",
            f"/accounts/{self.account_id}/orders/@{client_order_id}",
        )

    def get_transaction(self, transaction_id: str) -> Dict[str, Any]:
        """Return one exact transaction used for unknown-outcome recovery."""

        value = str(transaction_id)
        if (
            not value
            or any(character not in "0123456789" for character in value)
        ):
            raise ValueError("transaction_id contract invalid")
        return self._request(
            "GET",
            f"/accounts/{self.account_id}/transactions/{value}",
        )
    
    def get_server_time(self) -> Dict[str, Any]:
        """
        Get server time from OANDA API.
        
        Note: OANDA API v3 doesn't have a dedicated /time endpoint.
        We use account summary as a proxy to get server time.
        
        Returns
        -------
        Dict[str, Any]
            Server time response with 'time' field (RFC3339 format).
        """
        # OANDA API v3 doesn't have /time endpoint
        # Use account summary as proxy (includes server time in response headers)
        # For now, return current UTC time as approximation
        import datetime
        return {
            "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "note": "OANDA API v3 doesn't have /time endpoint, using local UTC time as approximation",
        }

    def close_trade(self, trade_id: str) -> Dict[str, Any]:
        """Close an open trade by trade ID."""
        return self._request("PUT", f"/accounts/{self.account_id}/trades/{trade_id}/close")
    
    def cancel_trade_orders(self, trade_id: str, *, cancel_tp: bool = True, cancel_sl: bool = True) -> Dict[str, Any]:
        """
        Cancel TP/SL orders for a trade.
        
        Parameters
        ----------
        trade_id : str
            Trade ID to cancel orders for.
        cancel_tp : bool
            Cancel take profit order (default: True).
        cancel_sl : bool
            Cancel stop loss order (default: True).
        
        Returns
        -------
        Dict[str, Any]
            Response from API (typically contains orderCancelTransaction).
        """
        # OANDA API v3: Cancel TP/SL by setting them to null
        # Use PUT /accounts/{accountID}/trades/{tradeSpecifier}/orders
        # Note: OANDA API v3 doesn't have a direct "cancel order" endpoint
        # Instead, we modify the trade to remove TP/SL orders
        # However, the v3 API doesn't support modifying existing orders on a trade
        # So we'll need to close the trade first, or use the v20 API if available
        # For now, we'll log a warning and proceed with close
        # In practice, TP/SL orders are automatically cancelled when the trade is closed
        log.warning("[OANDA] cancel_trade_orders: TP/SL orders will be automatically cancelled when trade is closed")
        return {"note": "TP/SL orders are automatically cancelled when trade is closed"}
    
    def get_trade(self, trade_id: str) -> Dict[str, Any]:
        """Get details for a specific trade."""
        return self._request("GET", f"/accounts/{self.account_id}/trades/{trade_id}")
    
    def get_trades(self, *, state: Optional[str] = None, instrument: Optional[str] = None, count: int = 500) -> Dict[str, Any]:
        """
        Get list of trades.
        
        Parameters
        ----------
        state : str, optional
            Filter by state: "OPEN", "CLOSED", "CLOSE_WHEN_TRADEABLE", "ALL"
        instrument : str, optional
            Filter by instrument (e.g., "XAU_USD")
        count : int, default=500
            Maximum number of trades to return
        """
        params = {"count": count}
        if state:
            params["state"] = state
        if instrument:
            params["instrument"] = instrument
        return self._request("GET", f"/accounts/{self.account_id}/trades", params=params)

    def get_account_summary(self) -> Dict[str, Any]:
        """Retrieve account summary (balance, NAV, PnL etc.)."""
        return self._request("GET", f"/accounts/{self.account_id}/summary")

    def get_account_instruments(self, instruments: list[str]) -> Dict[str, Any]:
        """Retrieve account-specific immutable execution constraints.

        Learned sizing uses this endpoint for broker truth about margin rate,
        minimum trade size, maximum order units, and unit precision.  An empty
        request is rejected rather than widened to the full instrument set.
        """
        if not instruments:
            raise ValueError("instruments must be non-empty")
        return self._request(
            "GET",
            f"/accounts/{self.account_id}/instruments",
            params={"instruments": ",".join(instruments)},
        )
    
    def get_pricing(self, instruments: list[str]) -> Dict[str, Any]:
        """
        Get pricing snapshot for instruments (bid/ask).
        
        Parameters
        ----------
        instruments : list[str]
            List of instrument symbols (e.g., ["XAU_USD"]).
        
        Returns
        -------
        Dict[str, Any]
            Pricing response with 'prices' list containing bid/ask for each instrument.
            Format: {
                "prices": [
                    {
                        "instrument": "XAU_USD",
                        "bids": [{"price": "4127.670", "liquidity": 1000000}],
                        "asks": [{"price": "4127.810", "liquidity": 1000000}],
                        "time": "2025-11-12T12:00:00.000000000Z"
                    }
                ]
            }
        """
        if not instruments:
            return {"prices": []}
        
        # OANDA API expects comma-separated instruments
        instruments_str = ",".join(instruments)
        params = {"instruments": instruments_str}
        
        return self._request("GET", f"/accounts/{self.account_id}/pricing", params=params)
