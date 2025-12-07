from __future__ import annotations

from typing import Any, Dict, Optional

from gx1.execution.oanda_client import OandaClient


class BrokerClient:
    """Thin wrapper around the OandaClient to keep GX1DemoRunner slimmer."""

    def __init__(self, client: Optional[OandaClient] = None) -> None:
        self._client = client or OandaClient.from_env()

    def get_open_trades(self) -> Dict[str, Any]:
        return self._client.get_open_trades()

    def get_candles_chunked(self, *args: Any, **kwargs: Any) -> Any:
        return self._client.get_candles_chunked(*args, **kwargs)

    def get_trade(self, trade_id: str) -> Dict[str, Any]:
        return self._client.get_trade(trade_id)

    def close_trade(self, trade_id: str, **kwargs: Any) -> Dict[str, Any]:
        return self._client.close_trade(trade_id, **kwargs)

    def create_market_order(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self._client.create_market_order(*args, **kwargs)

    def get_server_time(self) -> Dict[str, Any]:
        return self._client.get_server_time()

    def get_candles(self, *args: Any, **kwargs: Any) -> Any:
        return self._client.get_candles(*args, **kwargs)
