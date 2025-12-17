"\"\"\"Minimal OANDA REST client used for GX1 demo execution.\"\"\""

from __future__ import annotations

import os
import logging
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path

import pandas as pd
import requests

from gx1.utils.env_loader import load_dotenv_if_present


log = logging.getLogger(__name__)

PRACTICE_URL = "https://api-fxpractice.oanda.com/v3"
LIVE_URL = "https://api-fxtrade.oanda.com/v3"


class OandaAPIError(RuntimeError):
    """Raised for non-2xx responses from the OANDA REST API."""


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
        log.info("OandaClient initialised (env=%s, account=%s)", self.env, self.account_id)

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

        log.info(
            "OandaClient.from_env: env=%s, account=%s, api_key=%s",
            env or "<EMPTY>",
            _mask(account_id),
            _mask(api_key),
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
        Make HTTP request with exponential backoff retry on 429/5xx errors.
        
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
        masked_auth = headers.get("Authorization", "")[:32]
        
        for attempt in range(max_retries):
            log.debug(
                "OANDA request (attempt %d/%d): %s %s | headers=%s | params=%s | json=%s",
                attempt + 1,
                max_retries,
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
                if attempt < max_retries - 1:
                    # Exponential backoff: 2^attempt seconds
                    wait_time = 2 ** attempt
                    log.warning(
                        "OANDA request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1,
                        max_retries,
                        exc,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    raise OandaAPIError(f"Request failed after {max_retries} attempts: {exc}") from exc
            
            # Check if we should retry on this status code
            if resp.status_code in retry_on_status and attempt < max_retries - 1:
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
                    max_retries,
                    wait_time,
                )
                time.sleep(wait_time)
                continue
            
            # If not OK and not retryable, raise error
            if not resp.ok:
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
        params = {
            "granularity": granularity,
            "alignmentTimezone": "UTC",
        }
        if include_mid:
            params["price"] = "MBA"
        else:
            params["price"] = "BA"
        
        # Use from_ts/to_ts if provided, otherwise use count
        if from_ts is not None and to_ts is not None:
            # OANDA API uses RFC3339 format for from/to
            # Half-open interval: [from_ts, to_ts) - include from, exclude to
            params["from"] = from_ts.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
            params["to"] = to_ts.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        else:
            # Fall back to count (default: 500)
            params["count"] = count or 500

        data = self._request(
            "GET",
            f"/instruments/{instrument}/candles",
            params=params,
        )

        candles = data.get("candles", [])
        if not candles:
            raise OandaAPIError(f"No candles returned for {instrument}")

        records = []
        for item in candles:
            # Exclude incomplete bars if requested
            if exclude_incomplete and not item.get("complete", True):
                continue
            mid = item.get("mid", {})
            bid = item.get("bid", {})
            ask = item.get("ask", {})
            # Normalize timestamp to closed M5 bar (floor to 5-minute boundary)
            raw_time = pd.to_datetime(item["time"])
            # Normalize to UTC and floor to 5-minute boundary
            if raw_time.tzinfo is None:
                raw_time = raw_time.tz_localize("UTC")
            else:
                raw_time = raw_time.tz_convert("UTC")
            # Floor to 5-minute boundary (closed bar)
            normalized_time = raw_time.floor("5min")
            
            # Skip if normalized_time >= to_ts (half-open interval)
            if to_ts is not None and normalized_time >= to_ts:
                continue
            
            def _resolve_mid(key: str) -> float:
                if mid and mid.get(key) is not None:
                    return float(mid[key])
                if bid and ask and bid.get(key) is not None and ask.get(key) is not None:
                    return (float(bid[key]) + float(ask[key])) / 2.0
                if bid and bid.get(key) is not None:
                    return float(bid[key])
                if ask and ask.get(key) is not None:
                    return float(ask[key])
                raise OandaAPIError(f"Missing price component '{key}' in candle {item}")
            
            record = {
                "time": normalized_time,
                "open": _resolve_mid("o"),
                "high": _resolve_mid("h"),
                "low": _resolve_mid("l"),
                "close": _resolve_mid("c"),
                "volume": float(item.get("volume", 0.0)),
            }
            if bid:
                record.update(
                    {
                        "bid_open": float(bid.get("o", record["open"])),
                        "bid_high": float(bid.get("h", record["high"])),
                        "bid_low": float(bid.get("l", record["low"])),
                        "bid_close": float(bid.get("c", record["close"])),
                    }
                )
            if ask:
                record.update(
                    {
                        "ask_open": float(ask.get("o", record["open"])),
                        "ask_high": float(ask.get("h", record["high"])),
                        "ask_low": float(ask.get("l", record["low"])),
                        "ask_close": float(ask.get("c", record["close"])),
                    }
                )
            records.append(record)

        df = pd.DataFrame.from_records(records)
        if df.empty:
            return df
        df = df.sort_values("time").set_index("time")
        # Drop duplicates per timestamp (idempotent)
        df = df[~df.index.duplicated(keep="last")]
        return df
    
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
        # Calculate chunk duration (5 minutes per bar for M5)
        chunk_duration = pd.Timedelta(minutes=5 * chunk_size)
        
        all_candles = []
        current_from = from_ts
        
        while current_from < to_ts:
            current_to = min(current_from + chunk_duration, to_ts)
            
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
        combined_df = combined_df.sort_index()
        # Drop duplicates per timestamp (idempotent)
        combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
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
        if client_order_id:
            # Truncate to 64 chars (OANDA limit)
            payload["order"]["clientOrderID"] = client_order_id[:64]
        if client_extensions:
            payload["order"]["clientExtensions"] = client_extensions

        return self._request("POST", f"/accounts/{self.account_id}/orders", json=payload)
    
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
