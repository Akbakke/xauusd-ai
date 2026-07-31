from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_active_runtime_source_has_no_decision_state_substitution() -> None:
    root = Path(__file__).resolve().parents[1]
    pipeline_source = (root / "gx1/execution/v12_pipeline.py").read_text(encoding="utf-8")
    runner_source = (root / "gx1/execution/v12_paper_runner.py").read_text(encoding="utf-8")
    exit_start = pipeline_source.index("    def make_exit_decision(")
    active_exit = pipeline_source[exit_start:]

    for forbidden in (
        "using zero fallback",
        "Use latest available bar as fallback",
        '"error": "no_canonical_data"',
        "trade.update_bar(bid=bid",
    ):
        assert forbidden not in active_exit

    assert "m1_close = (bid + ask) / 2.0" not in runner_source
    assert ".read_parquet(_p" not in runner_source
    assert 'fill_price = float(order_result.get("fill_price") or 0.0)' not in runner_source
    assert 'float(t.get("currentUnits", 0) or 0)' not in runner_source
    assert 'get_open_trades().get("trades", [])' not in runner_source
    assert "fill_price - spread_abs" not in runner_source
    assert "fill_price + spread_abs" not in runner_source
    assert "except ExitDecisionUnavailable as exc:" in runner_source
    assert '"exit_decision": None' in runner_source
    assert "if exit_decision_unavailable:" in runner_source
    assert (
        "FILLED_STATE_UNAVAILABLE_RECONCILIATION_REQUIRED"
        in runner_source
    )
    assert "BROKER_CLOSE_OUTCOME_UNRESOLVED" in runner_source
    assert "submit_broker_close_with_durable_intent" in runner_source
    assert runner_source.count("attempt_close_trade(") == 3
    assert "BROKER_RECONCILIATION_REQUIRED" in runner_source
    assert 'exit_action = exit_decision["action"]' in runner_source
    assert 'exit_decision.get("action_id")' not in runner_source


def test_missing_trade_id_fails_closed_without_counter_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        runner,
        "attempt_market_entry",
        lambda _client, side, units: calls.append((side, units)) or {"status": "filled"},
    )
    trade = SimpleNamespace(trade_id=None, side="long", units=7)

    result = runner.attempt_close_trade(object(), trade)

    assert result["status"] == "missing_trade_id"
    assert result["trade_id"] is None
    assert calls == []


def test_empty_trade_id_fails_closed_without_counter_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        runner,
        "attempt_market_entry",
        lambda _client, side, units: calls.append((side, units)) or {"status": "filled"},
    )
    trade = SimpleNamespace(trade_id="", side="short", units=3)

    result = runner.attempt_close_trade(object(), trade)

    assert result["status"] == "missing_trade_id"
    assert result["trade_id"] is None
    assert calls == []


def _close_fill_response(
    *,
    trade_id: str = "trade-7",
    units: str = "-7",
) -> dict[str, object]:
    return {
        "orderFillTransaction": {
            "id": "close-tx-1",
            "orderID": "close-order-1",
            "instrument": "XAU_USD",
            "time": "2026-07-31T10:00:00.125000000Z",
            "price": "3300.25",
            "pl": "4.50",
            "units": units,
            "tradesClosed": [
                {
                    "tradeID": trade_id,
                    "units": units,
                    "realizedPL": "4.50",
                }
            ],
        }
    }


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        ({}, "order_fill_transaction_missing"),
        (
            _close_fill_response(trade_id="wrong-trade"),
            "closed_trade_id_mismatch",
        ),
        (
            _close_fill_response(units="-3"),
            "closed_units_not_exact_all",
        ),
        (
            _close_fill_response(units="7"),
            "closed_units_not_exact_all",
        ),
    ],
)
def test_close_requires_exact_full_trade_fill(
    response: dict[str, object],
    reason: str,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    client = SimpleNamespace(close_trade=lambda _trade_id: response)
    trade = SimpleNamespace(
        trade_id="trade-7",
        side="long",
        units=7,
    )

    result = runner.attempt_close_trade(client, trade)

    assert result["status"] == "close_fill_mismatch"
    assert reason in result["reason"]


def test_close_accepts_only_exact_reconciled_trade_fill() -> None:
    from gx1.execution import v12_paper_runner as runner

    client = SimpleNamespace(
        close_trade=lambda _trade_id: _close_fill_response()
    )
    trade = SimpleNamespace(
        trade_id="trade-7",
        side="long",
        units=7,
    )

    result = runner.attempt_close_trade(client, trade)

    assert result["status"] == "closed"
    assert result["trade_id"] == "trade-7"
    assert result["closed_signed_units"] == -7
    assert result["fill_price"] == pytest.approx(3300.25)


def test_close_reject_with_fill_evidence_is_ambiguous_not_terminal() -> None:
    from gx1.execution import v12_paper_runner as runner

    response = _close_fill_response()
    response["orderRejectTransaction"] = {
        "id": "reject-and-fill",
        "rejectReason": "UNKNOWN",
    }
    client = SimpleNamespace(
        close_trade=lambda _trade_id: response
    )
    trade = SimpleNamespace(
        trade_id="trade-7",
        side="long",
        units=7,
    )

    result = runner.attempt_close_trade(client, trade)

    assert result["status"] == "ambiguous_response"


@pytest.mark.parametrize(
    ("dry_run", "shadow_only", "mode", "error"),
    [
        (
            True,
            False,
            "learned_broker_fill",
            "TRADE_STATE_EXECUTION_MODE_MISMATCH",
        ),
        (
            False,
            False,
            "learned_virtual_dry_run",
            "TRADE_STATE_EXECUTION_MODE_MISMATCH",
        ),
        (
            True,
            True,
            "learned_broker_fill",
            "SHADOW_RUNNER_PERSISTED_TRADE_STATE_PRESENT",
        ),
    ],
)
def test_runner_rejects_cross_mode_persisted_trade_state(
    dry_run: bool,
    shadow_only: bool,
    mode: str,
    error: str,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    trade = SimpleNamespace(
        trade_id="mode-bound-trade",
        sizing_execution_evidence={"mode": mode},
    )

    with pytest.raises(RuntimeError, match=error):
        runner.require_runner_trade_state_mode(
            [trade],
            dry_run=dry_run,
            shadow_only=shadow_only,
        )


def test_runner_accepts_only_matching_persisted_trade_state_mode() -> None:
    from gx1.execution import v12_paper_runner as runner

    virtual = SimpleNamespace(
        trade_id="virtual",
        sizing_execution_evidence={"mode": "learned_virtual_dry_run"},
    )
    broker = SimpleNamespace(
        trade_id="broker",
        sizing_execution_evidence={"mode": "learned_broker_fill"},
    )
    runner.require_runner_trade_state_mode(
        [virtual],
        dry_run=True,
        shadow_only=False,
    )
    runner.require_runner_trade_state_mode(
        [broker],
        dry_run=False,
        shadow_only=False,
    )
    runner.require_runner_trade_state_mode(
        [],
        dry_run=True,
        shadow_only=True,
    )


def _broker_account_binding(
    digest_character: str = "a",
) -> dict[str, str]:
    return {
        "schema_version": "gx1_trade_state_broker_account_binding_v1",
        "environment": "practice",
        "account_id_sha256": digest_character * 64,
    }


def test_runner_rejects_persisted_trade_from_other_broker_account() -> None:
    from gx1.execution import v12_paper_runner as runner

    trade = SimpleNamespace(
        trade_id="account-bound-trade",
        broker_account_binding=_broker_account_binding("a"),
    )
    with pytest.raises(
        RuntimeError,
        match="TRADE_STATE_BROKER_ACCOUNT_MISMATCH",
    ):
        runner.require_runner_broker_account_binding(
            [trade],
            broker_account_binding=_broker_account_binding("b"),
            dry_run=False,
            shadow_only=False,
        )
    runner.require_runner_broker_account_binding(
        [trade],
        broker_account_binding=_broker_account_binding("a"),
        dry_run=False,
        shadow_only=False,
    )


def test_runner_singleton_lock_blocks_second_state_writer(
    tmp_path: Path,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    lock_path = tmp_path / "runner.lock"
    first = runner.acquire_runner_singleton_lock(lock_path)
    try:
        with pytest.raises(
            RuntimeError,
            match="RUNNER_SINGLETON_ALREADY_ACTIVE",
        ):
            runner.acquire_runner_singleton_lock(lock_path)
    finally:
        first.close()
    replacement = runner.acquire_runner_singleton_lock(lock_path)
    replacement.close()


class _CloseIntentTrade:
    trade_id = "trade-7"
    side = "long"
    units = 7
    last_exit_decision = {
        "action": "EXIT_NOW",
        "decision_ts": "2026-07-31T10:00:00+00:00",
    }
    broker_account_binding = _broker_account_binding("a")

    def __init__(self, journal_path: Path | None = None) -> None:
        self.deleted = False
        self.journal_path = journal_path

    def to_dict(self) -> dict[str, object]:
        return {
            "trade_id": self.trade_id,
            "side": self.side,
            "units": self.units,
            "last_exit_decision": self.last_exit_decision,
            "broker_account_binding": self.broker_account_binding,
        }

    def delete_state_file(self, _directory: Path) -> None:
        if self.journal_path is not None:
            assert self.journal_path.is_file()
        self.deleted = True


def test_close_unknown_outcome_is_reconciled_by_get_without_retry(
    tmp_path: Path,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    unresolved = tmp_path / "unresolved"
    resolved = tmp_path / "resolved"
    journal_path = tmp_path / "close_recovery.jsonl"
    trade = _CloseIntentTrade(journal_path)

    class _LostResponseClient:
        def __init__(self) -> None:
            self.close_calls = 0

        def close_trade(self, _trade_id: str) -> dict[str, object]:
            assert len(list(unresolved.glob("*.json"))) == 1
            self.close_calls += 1
            raise TimeoutError("lost close response")

    lost_client = _LostResponseClient()
    intent, intent_path, result = (
        runner.submit_broker_close_with_durable_intent(
            lost_client,
            trade,
            broker_account_binding=_broker_account_binding("a"),
            intent_root=unresolved,
            resolved_root=resolved,
        )
    )
    assert intent["broker_account_binding"] == (
        _broker_account_binding("a")
    )
    assert result["status"] == "api_error"
    assert lost_client.close_calls == 1
    assert intent_path.is_file()

    loaded_intent = runner.load_broker_close_intent(intent_path)
    assert loaded_intent == intent
    assert loaded_intent["expected_close_signed_units"] == -7

    transaction = _close_fill_response()["orderFillTransaction"]

    class _ReconciliationClient:
        close_calls = 0

        @staticmethod
        def get_trade(_trade_id: str) -> dict[str, object]:
            return {
                "trade": {
                    "id": "trade-7",
                    "instrument": "XAU_USD",
                    "state": "CLOSED",
                    "closingTransactionIDs": ["close-tx-1"],
                }
            }

        @staticmethod
        def get_transaction(
            _transaction_id: str,
        ) -> dict[str, object]:
            return {"transaction": transaction}

    reconciliation_client = _ReconciliationClient()
    recovered = runner.reconcile_unresolved_broker_close_intents(
        reconciliation_client,
        open_trades=[trade],
        dry_run=False,
        broker_account_binding=_broker_account_binding("a"),
        journal_path=journal_path,
        intent_root=unresolved,
        resolved_root=resolved,
    )

    assert recovered == []
    assert trade.deleted is True
    assert reconciliation_client.close_calls == 0
    assert not intent_path.exists()
    assert (resolved / intent_path.name).is_file()
    assert "BROKER_CLOSE_INTENT_TERMINAL_CLOSED" in (
        journal_path.read_text(encoding="utf-8")
    )


def test_close_unknown_outcome_never_retries_while_trade_is_open(
    tmp_path: Path,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    unresolved = tmp_path / "unresolved"
    trade = _CloseIntentTrade()
    intent = runner.build_broker_close_intent(
        trade,
        broker_account_binding=_broker_account_binding("a"),
        created_utc=pd.Timestamp("2026-07-31T10:00:00Z"),
    )
    intent_path = runner.persist_broker_close_intent(
        intent,
        intent_root=unresolved,
        resolved_root=tmp_path / "resolved",
    )
    client = SimpleNamespace(
        get_trade=lambda _trade_id: {
            "trade": {
                "id": "trade-7",
                "instrument": "XAU_USD",
                "state": "OPEN",
            }
        }
    )

    with pytest.raises(
        RuntimeError,
        match="OUTCOME_UNRESOLVED_NO_RETRY",
    ):
        runner.reconcile_unresolved_broker_close_intents(
            client,
            open_trades=[trade],
            dry_run=False,
            broker_account_binding=_broker_account_binding("a"),
            journal_path=tmp_path / "journal.jsonl",
            intent_root=unresolved,
            resolved_root=tmp_path / "resolved",
        )
    assert intent_path.is_file()
    assert trade.deleted is False


def test_same_exposure_blocks_second_intent_with_different_exit_snapshot(
    tmp_path: Path,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    unresolved = tmp_path / "unresolved"
    resolved = tmp_path / "resolved"
    first_trade = _CloseIntentTrade()
    first_trade.last_exit_decision = {
        "action": "EXIT_NOW",
        "decision_ts": "2026-07-31T10:00:00+00:00",
    }
    first_intent = runner.build_broker_close_intent(
        first_trade,
        broker_account_binding=_broker_account_binding("a"),
        created_utc=pd.Timestamp("2026-07-31T10:00:00Z"),
    )
    first_path = runner.persist_broker_close_intent(
        first_intent,
        intent_root=unresolved,
        resolved_root=resolved,
    )
    stale_trade = _CloseIntentTrade()
    stale_trade.last_exit_decision = {
        "action": "EXIT_NOW",
        "decision_ts": "2026-07-31T10:01:00+00:00",
    }
    second_intent = runner.build_broker_close_intent(
        stale_trade,
        broker_account_binding=_broker_account_binding("a"),
        created_utc=pd.Timestamp("2026-07-31T10:01:00Z"),
    )

    assert (
        first_intent["close_intent_id"]
        != second_intent["close_intent_id"]
    )
    assert runner._broker_close_exposure_filename(
        first_intent
    ) == runner._broker_close_exposure_filename(second_intent)
    with pytest.raises(
        RuntimeError,
        match="ALREADY_EXISTS_RECONCILIATION_REQUIRED",
    ):
        runner.persist_broker_close_intent(
            second_intent,
            intent_root=unresolved,
            resolved_root=resolved,
        )
    assert first_path.is_file()
    assert len(list(unresolved.glob("*.json"))) == 1


def test_close_recovery_survives_crash_after_state_delete(
    tmp_path: Path,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    unresolved = tmp_path / "unresolved"
    resolved = tmp_path / "resolved"
    trade = _CloseIntentTrade()
    intent = runner.build_broker_close_intent(
        trade,
        broker_account_binding=_broker_account_binding("a"),
        created_utc=pd.Timestamp("2026-07-31T10:00:00Z"),
    )
    intent_path = runner.persist_broker_close_intent(
        intent,
        intent_root=unresolved,
        resolved_root=resolved,
    )
    transaction = _close_fill_response()["orderFillTransaction"]
    client = SimpleNamespace(
        get_trade=lambda _trade_id: {
            "trade": {
                "id": "trade-7",
                "instrument": "XAU_USD",
                "state": "CLOSED",
                "closingTransactionIDs": ["close-tx-1"],
            }
        },
        get_transaction=lambda _transaction_id: {
            "transaction": transaction
        },
    )

    recovered = runner.reconcile_unresolved_broker_close_intents(
        client,
        open_trades=[],
        dry_run=False,
        broker_account_binding=_broker_account_binding("a"),
        journal_path=tmp_path / "recovery.jsonl",
        intent_root=unresolved,
        resolved_root=resolved,
    )

    assert recovered == []
    assert not intent_path.exists()
    assert (resolved / intent_path.name).is_file()


def test_resolved_close_tombstone_blocks_delayed_second_runner(
    tmp_path: Path,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    unresolved = tmp_path / "unresolved"
    resolved = tmp_path / "resolved"
    journal_path = tmp_path / "finalized.jsonl"
    trade = _CloseIntentTrade(journal_path)

    class _Client:
        def __init__(self) -> None:
            self.close_calls = 0

        def close_trade(self, _trade_id: str) -> dict[str, object]:
            self.close_calls += 1
            return _close_fill_response()

    client = _Client()
    intent, intent_path, close_result = (
        runner.submit_broker_close_with_durable_intent(
            client,
            trade,
            broker_account_binding=_broker_account_binding("a"),
            intent_root=unresolved,
            resolved_root=resolved,
        )
    )
    runner.finalize_broker_close_intent(
        intent=intent,
        intent_path=intent_path,
        close_result=close_result,
        trade=trade,
        journal_path=journal_path,
        unresolved_root=unresolved,
        resolved_root=resolved,
    )

    assert client.close_calls == 1
    assert (resolved / intent_path.name).is_file()
    trade.last_exit_decision = {
        "action": "EXIT_NOW",
        "decision_ts": "2026-07-31T10:01:00+00:00",
    }
    with pytest.raises(
        RuntimeError,
        match="ALREADY_RESOLVED_NO_REPLAY",
    ):
        runner.submit_broker_close_with_durable_intent(
            client,
            trade,
            broker_account_binding=_broker_account_binding("a"),
            intent_root=unresolved,
            resolved_root=resolved,
        )
    assert client.close_calls == 1


def test_known_close_rejection_archives_no_mutation_and_allows_restart(
    tmp_path: Path,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    unresolved = tmp_path / "unresolved"
    resolved = tmp_path / "resolved"
    rejected = tmp_path / "rejected"
    journal_path = tmp_path / "rejected.jsonl"
    trade = _CloseIntentTrade()

    class _RejectingClient:
        def __init__(self) -> None:
            self.close_calls = 0

        def close_trade(self, _trade_id: str) -> dict[str, object]:
            self.close_calls += 1
            return {
                "orderRejectTransaction": {
                    "id": f"reject-{self.close_calls}",
                    "rejectReason": "MARKET_HALTED",
                }
            }

    client = _RejectingClient()
    intent, intent_path, close_result = (
        runner.submit_broker_close_with_durable_intent(
            client,
            trade,
            broker_account_binding=_broker_account_binding("a"),
            intent_root=unresolved,
            resolved_root=resolved,
        )
    )
    archive = runner.finalize_broker_close_rejection(
        intent=intent,
        intent_path=intent_path,
        close_result=close_result,
        trade=trade,
        journal_path=journal_path,
        unresolved_root=unresolved,
        resolved_root=resolved,
        rejected_root=rejected,
    )

    assert archive.is_file()
    assert not intent_path.exists()
    assert trade.deleted is False
    assert "BROKER_CLOSE_INTENT_REJECTED_NO_MUTATION" in (
        journal_path.read_text(encoding="utf-8")
    )
    _, retry_path, retry_result = (
        runner.submit_broker_close_with_durable_intent(
            client,
            trade,
            broker_account_binding=_broker_account_binding("a"),
            intent_root=unresolved,
            resolved_root=resolved,
        )
    )
    assert retry_path.is_file()
    assert retry_result["status"] == "rejected"
    assert client.close_calls == 2


def test_close_mutation_lock_blocks_concurrent_intent_publication(
    tmp_path: Path,
) -> None:
    import threading

    from gx1.execution import v12_paper_runner as runner

    unresolved = tmp_path / "unresolved"
    resolved = tmp_path / "resolved"
    trade = _CloseIntentTrade()
    intent = runner.build_broker_close_intent(
        trade,
        broker_account_binding=_broker_account_binding("a"),
        created_utc=pd.Timestamp("2026-07-31T10:00:00Z"),
    )
    finished = threading.Event()
    failures: list[BaseException] = []

    def publish() -> None:
        try:
            runner.persist_broker_close_intent(
                intent,
                intent_root=unresolved,
                resolved_root=resolved,
            )
        except BaseException as exc:  # pragma: no cover - assertion below
            failures.append(exc)
        finally:
            finished.set()

    with runner.broker_close_mutation_lock(
        intent_root=unresolved,
        resolved_root=resolved,
    ):
        thread = threading.Thread(target=publish)
        thread.start()
        assert finished.wait(0.1) is False
    thread.join(timeout=2.0)

    assert finished.is_set()
    assert failures == []
    assert len(list(unresolved.glob("*.json"))) == 1


def test_close_terminal_journal_idempotency_key_deduplicates_restart(
    tmp_path: Path,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    path = tmp_path / "journal.jsonl"
    event = {
        "event": "BROKER_CLOSE_INTENT_TERMINAL_CLOSED",
        "idempotency_key": "broker-close-terminal:gx1-close-proof",
        "trade_id": "trade-7",
    }
    runner.log_journal_event(path, event)
    runner.log_journal_event(path, event)

    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(
        RuntimeError,
        match="JOURNAL_IDEMPOTENCY_PAYLOAD_MISMATCH",
    ):
        runner.log_journal_event(
            path,
            {
                **event,
                "trade_id": "different-trade",
            },
        )


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


def test_oanda_explicit_http_400_order_rejection_reaches_runner() -> None:
    from gx1.execution import v12_paper_runner as runner
    from gx1.execution.oanda_client import OandaClient

    rejection = {
        "orderRejectTransaction": {
            "id": "reject-1",
            "rejectReason": "MARKET_HALTED",
        },
        "lastTransactionID": "reject-1",
    }

    class _Response:
        status_code = 400
        ok = False
        headers: dict[str, str] = {}
        text = "explicit order rejection"

        @staticmethod
        def json() -> dict[str, object]:
            return rejection

    class _Session:
        headers: dict[str, str] = {}

        def __init__(self) -> None:
            self.calls = 0

        def request(self, **_kwargs: object) -> _Response:
            self.calls += 1
            return _Response()

    session = _Session()
    client = object.__new__(OandaClient)
    client.account_id = "practice-account"
    client.base_url = "https://api-fxpractice.oanda.com/v3"
    client.timeout = 1.0
    client.session = session

    result = runner.attempt_market_entry(
        client,
        "long",
        units=2,
        client_order_id="gx1-known-rejection",
    )

    assert session.calls == 1
    assert result == {
        "status": "rejected",
        "reason": "MARKET_HALTED",
        "client_order_id": "gx1-known-rejection",
        "raw": rejection,
    }


@pytest.mark.parametrize(
    ("status_code", "payload"),
    [
        (400, {"errorMessage": "malformed order"}),
        (
            500,
            {
                "orderRejectTransaction": {
                    "rejectReason": "INTERNAL_SERVER_ERROR"
                }
            },
        ),
    ],
)
def test_other_oanda_mutation_http_failures_stay_unknown_and_single_attempt(
    status_code: int,
    payload: dict[str, object],
) -> None:
    from gx1.execution import v12_paper_runner as runner
    from gx1.execution.oanda_client import OandaClient

    class _Response:
        ok = False
        headers: dict[str, str] = {}
        text = "mutation failure"

        def __init__(self) -> None:
            self.status_code = status_code

        @staticmethod
        def json() -> dict[str, object]:
            return payload

    class _Session:
        headers: dict[str, str] = {}

        def __init__(self) -> None:
            self.calls = 0

        def request(self, **_kwargs: object) -> _Response:
            self.calls += 1
            return _Response()

    session = _Session()
    client = object.__new__(OandaClient)
    client.account_id = "practice-account"
    client.base_url = "https://api-fxpractice.oanda.com/v3"
    client.timeout = 1.0
    client.session = session

    result = runner.attempt_market_entry(
        client,
        "long",
        units=2,
        client_order_id="gx1-unknown-http-failure",
    )

    assert session.calls == 1
    assert result["status"] == "unknown_outcome"
    assert f"API error {status_code}" in result["reason"]


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


def test_broker_entry_intent_is_durable_no_replace_and_blocks_restart(
    tmp_path: Path,
) -> None:
    from datetime import datetime, timezone

    from gx1.execution import v12_paper_runner as runner

    unresolved = tmp_path / "unresolved"
    resolved = tmp_path / "resolved"
    intent = runner.build_broker_entry_intent(
        side="long",
        units=3,
        decision_snapshot={
            "decision_ts": "2026-07-31T10:00:00+00:00",
            "model_direction": "LONG",
        },
        sizing_application={"units": 3},
        model_bundle_binding={
            "bundle_sha256": "b" * 64,
        },
        entry_source_pair_binding={
            "pair_generation_id": "c" * 64,
            "pair_manifest_sha256": "d" * 64,
        },
        broker_account_binding={
            "schema_version": "gx1_trade_state_broker_account_binding_v1",
            "environment": "practice",
            "account_id_sha256": "a" * 64,
        },
        launch_lease={
            "launch_state_sha256": "e" * 64,
            "artifact_registry_sha256": "f" * 64,
        },
        created_utc=datetime(2026, 7, 31, 10, 0, tzinfo=timezone.utc),
    )

    path = runner.persist_broker_entry_intent(
        intent,
        intent_root=unresolved,
    )
    assert path.is_file()
    with pytest.raises(
        RuntimeError,
        match="ALREADY_EXISTS_RECONCILIATION_REQUIRED",
    ):
        runner.persist_broker_entry_intent(
            intent,
            intent_root=unresolved,
        )
    with pytest.raises(RuntimeError, match="UNRESOLVED"):
        runner.require_no_unresolved_broker_entry_intents(
            intent_root=unresolved,
        )

    resolved_path = runner.resolve_broker_entry_intent(
        path,
        unresolved_root=unresolved,
        resolved_root=resolved,
    )
    assert resolved_path.is_file()
    assert not path.exists()
    runner.require_no_unresolved_broker_entry_intents(
        intent_root=unresolved,
    )


def test_broker_entry_intent_identity_binds_artifact_registry_lease(
    tmp_path: Path,
) -> None:
    from datetime import datetime, timezone

    from gx1.execution import v12_paper_runner as runner

    def build(
        registry_sha256: str,
        account_digest_character: str = "a",
    ) -> dict[str, object]:
        return runner.build_broker_entry_intent(
            side="long",
            units=3,
            decision_snapshot={
                "decision_ts": "2026-07-31T10:00:00+00:00",
                "model_direction": "LONG",
            },
            sizing_application={"units": 3},
            model_bundle_binding={"bundle_sha256": "b" * 64},
            entry_source_pair_binding={
                "pair_generation_id": "c" * 64,
                "pair_manifest_sha256": "d" * 64,
            },
            broker_account_binding={
                "schema_version": "gx1_trade_state_broker_account_binding_v1",
                "environment": "practice",
                "account_id_sha256": account_digest_character * 64,
            },
            launch_lease={
                "launch_state_sha256": "e" * 64,
                "artifact_registry_sha256": registry_sha256,
            },
            created_utc=datetime(2026, 7, 31, 10, 0, tzinfo=timezone.utc),
        )

    first = build("f" * 64)
    replacement = build("0" * 64)
    other_account = build("f" * 64, "b")
    assert first["identity_sha256"] != replacement["identity_sha256"]
    assert first["client_order_id"] != replacement["client_order_id"]
    assert first["identity_sha256"] != other_account["identity_sha256"]
    assert first["client_order_id"] != other_account["client_order_id"]

    path = runner.persist_broker_entry_intent(
        first,
        intent_root=tmp_path,
    )
    assert runner.load_broker_entry_intent(path) == first
    tampered = dict(first)
    tampered["launch_lease"] = {
        **first["launch_lease"],
        "artifact_registry_sha256": "0" * 64,
    }
    path.write_bytes(runner._canonical_json_bytes(tampered) + b"\n")

    with pytest.raises(RuntimeError, match="IDENTITY_HASH_MISMATCH"):
        runner.load_broker_entry_intent(path)


def test_entry_transport_unknown_outcome_keeps_client_identity() -> None:
    from gx1.execution import v12_paper_runner as runner

    client = SimpleNamespace(
        create_market_order=lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(TimeoutError("lost response"))
        )
    )
    result = runner.attempt_market_entry(
        client,
        "long",
        units=2,
        client_order_id="gx1-unknown-outcome",
    )

    assert result["status"] == "unknown_outcome"
    assert result["client_order_id"] == "gx1-unknown-outcome"


def test_unknown_entry_outcome_reconciles_exact_fill_before_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime, timezone

    from gx1.execution import v12_paper_runner as runner

    unresolved = tmp_path / "unresolved"
    resolved = tmp_path / "resolved"
    intent = runner.build_broker_entry_intent(
        side="long",
        units=2,
        decision_snapshot={
            "decision_ts": "2026-07-31T10:00:00+00:00",
            "model_direction": "LONG",
        },
        sizing_application={"units": 2},
        model_bundle_binding={"bundle_sha256": "b" * 64},
        entry_source_pair_binding={
            "pair_generation_id": "c" * 64,
            "pair_manifest_sha256": "d" * 64,
        },
        broker_account_binding={
            "schema_version": "gx1_trade_state_broker_account_binding_v1",
            "environment": "practice",
            "account_id_sha256": "a" * 64,
        },
        launch_lease={
            "launch_state_sha256": "e" * 64,
            "artifact_registry_sha256": "f" * 64,
        },
        created_utc=datetime(2026, 7, 31, 10, 0, tzinfo=timezone.utc),
    )
    path = runner.persist_broker_entry_intent(
        intent,
        intent_root=unresolved,
    )
    client_order_id = intent["client_order_id"]
    fill_time = "2026-07-31T10:00:00.125000000Z"
    transaction = {
        "id": "123",
        "orderID": "122",
        "time": fill_time,
        "instrument": "XAU_USD",
        "units": "2",
        "fullVWAP": "3300.25",
        "tradeOpened": {
            "tradeID": "trade-recovered",
            "units": "2",
            "price": "3300.25",
        },
        "fullPrice": {
            "time": fill_time,
            "bids": [{"price": "3300.00", "liquidity": "100"}],
            "asks": [{"price": "3300.20", "liquidity": "100"}],
            "closeoutBid": "3299.90",
            "closeoutAsk": "3300.30",
        },
        "clientExtensions": {"id": client_order_id},
    }
    client = SimpleNamespace(
        get_order_by_client_id=lambda _client_id: {
            "order": {
                "state": "FILLED",
                "fillingTransactionID": "123",
                "clientExtensions": {"id": client_order_id},
            }
        },
        get_transaction=lambda _transaction_id: {
            "transaction": transaction
        },
        get_open_trades=lambda: {
            "trades": [
                {
                    "id": "trade-recovered",
                    "instrument": "XAU_USD",
                    "currentUnits": "2",
                    "clientExtensions": {"id": client_order_id},
                }
            ]
        },
    )

    class _RecoveredTrade:
        trade_id = "trade-recovered"
        model_bundle_binding = intent["model_bundle_binding"]
        entry_source_pair_binding = intent[
            "entry_source_pair_binding"
        ]
        broker_account_binding = intent["broker_account_binding"]
        v10_snapshot = intent["decision_snapshot"]

        def save(self, _directory: Path) -> None:
            return None

    class _TradeState:
        @classmethod
        def open(cls, **_kwargs: object) -> _RecoveredTrade:
            return _RecoveredTrade()

    monkeypatch.setattr(runner, "TradeState", _TradeState, raising=False)
    recovered = runner.reconcile_unresolved_broker_entry_intents(
        client,
        open_trades=[],
        dry_run=False,
        broker_account_binding=intent["broker_account_binding"],
        journal_path=tmp_path / "recovery.jsonl",
        intent_root=unresolved,
        resolved_root=resolved,
    )

    assert [trade.trade_id for trade in recovered] == [
        "trade-recovered"
    ]
    assert not path.exists()
    assert (resolved / path.name).is_file()
    assert (tmp_path / "recovery.jsonl").is_file()


def test_live_practice_boundary_and_runner_latch_are_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    launch_source = (
        root / "scripts/launch_live_practice.sh"
    ).read_text(encoding="utf-8")
    runner_source = (
        root / "gx1/execution/v12_paper_runner.py"
    ).read_text(encoding="utf-8")

    assert (
        'if [[ "${OANDA_ENV:-practice}" != "practice" ]]; then'
        in launch_source
    )
    assert "requires OANDA_ENV=practice exactly" in launch_source
    assert (
        "require_live_latch=not (args.dry_run or args.shadow_only)"
        in runner_source
    )
    assert "prod_baseline=True" in runner_source
    assert runner_source.index("load_runner_open_trades(") < (
        runner_source.index("load_oanda_credentials(", 10_000)
    )


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


def test_runtime_launch_lease_rejects_replacement_and_in_check_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.execution import v12_paper_runner as runner
    from gx1.execution import v12_smart_entry_live as smart_live
    from gx1_guards import artifacts

    state_path = tmp_path / "launch.json"
    registry_path = tmp_path / "registry.json"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    state_path.write_text('{"decision":"ALLOW"}\n', encoding="utf-8")
    registry_path.write_text('{"active":{}}\n', encoding="utf-8")
    monkeypatch.setattr(artifacts, "XAU_DIRECTION_LAUNCH_CONTRACT", state_path)
    monkeypatch.setattr(artifacts, "SELECTION_CONTRACT", registry_path)
    monkeypatch.setattr(smart_live, "assert_smart_serving_gate", lambda: {})
    monkeypatch.setattr(
        artifacts,
        "load_decision_entry",
        lambda _role: {
            "path": bundle,
            "xau_direction_launch_state": {
                "accepted_via_vedtak": {
                    "event_sha256": "a" * 64,
                    "vedtak_id": "UNIT_RUNTIME_LEASE",
                }
            },
        },
    )

    lease = runner.require_runtime_entry_launch_lease()
    assert lease["artifact_registry_sha256"] == runner._sha256_regular_file(
        registry_path,
        label="artifact registry",
    )
    assert "registry_sha256" not in lease
    state_path.write_text('{"decision":"BLOCK"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="replaced or revoked"):
        runner.require_runtime_entry_launch_lease(expected_lease=lease)

    def mutate_during_check() -> dict:
        registry_path.write_text('{"changed":true}\n', encoding="utf-8")
        return {}

    monkeypatch.setattr(
        smart_live,
        "assert_smart_serving_gate",
        mutate_during_check,
    )
    with pytest.raises(RuntimeError, match="changed during lease"):
        runner.require_runtime_entry_launch_lease()


def test_filled_order_with_missing_price_is_explicitly_incomplete_not_zero() -> None:
    from gx1.execution import v12_paper_runner as runner

    client = SimpleNamespace(
        create_market_order=lambda *_args, **_kwargs: {
            "orderFillTransaction": {
                "id": "tx-1",
                "orderID": "order-1",
                "units": "2",
                "tradeOpened": {"tradeID": "trade-1", "units": "2"},
            }
        }
    )

    result = runner.attempt_market_entry(
        client,
        "long",
        units=2,
        client_order_id="gx1-test-missing-price",
    )

    assert result["status"] == "filled"
    assert result["trade_id"] == "trade-1"
    assert result["fill_price"] is None
    assert result["fill_price_pair_exact"] is False


def test_filled_order_uses_exact_full_price_pair_not_polling_quote() -> None:
    from gx1.execution import v12_paper_runner as runner

    fill_time = "2026-07-16T12:00:17.125000000Z"
    client = SimpleNamespace(
        create_market_order=lambda *_args, **_kwargs: {
            "orderFillTransaction": {
                "id": "tx-exact",
                "orderID": "order-exact",
                "time": fill_time,
                "units": "2",
                "fullVWAP": "2400.25",
                "tradeOpened": {
                    "tradeID": "trade-exact",
                    "units": "2",
                    "price": "2400.25",
                },
                "fullPrice": {
                    "time": fill_time,
                    "bids": [
                        {"price": "2400.00", "liquidity": "10"},
                        {"price": "2399.95", "liquidity": "100"},
                    ],
                    "asks": [{"price": "2400.20", "liquidity": "100"}],
                    "closeoutBid": "2399.90",
                    "closeoutAsk": "2400.30",
                },
            }
        }
    )

    result = runner.attempt_market_entry(
        client,
        "long",
        units=2,
        client_order_id="gx1-test-full-price",
    )

    assert result["status"] == "filled"
    assert result["fill_price_pair_exact"] is True
    assert result["fill_price"] == pytest.approx(2400.25)
    assert result["fill_bid"] == pytest.approx(2400.00)
    assert result["fill_ask"] == pytest.approx(2400.25)
    assert pd.Timestamp(result["fill_time"]) == pd.Timestamp(fill_time)


def test_filled_order_units_must_exactly_match_requested_learned_units() -> None:
    from gx1.execution import v12_paper_runner as runner

    client = SimpleNamespace(
        create_market_order=lambda *_args, **_kwargs: {
            "orderFillTransaction": {
                "id": "tx-2",
                "orderID": "order-2",
                "units": "1",
                "price": "2400.2",
                "tradeOpened": {"tradeID": "trade-2", "units": "1"},
            }
        }
    )

    result = runner.attempt_market_entry(
        client,
        "long",
        units=2,
        client_order_id="gx1-test-units",
    )

    assert result["status"] == "filled_units_mismatch"
    assert result["requested_signed_units"] == 2
    assert result["filled_signed_units"] == 1
    assert result["fill_units_exact"] is False


def test_mixed_netting_fill_is_never_accepted_as_new_trade_state() -> None:
    from gx1.execution import v12_paper_runner as runner

    client = SimpleNamespace(
        create_market_order=lambda *_args, **_kwargs: {
            "orderFillTransaction": {
                "id": "tx-mixed",
                "orderID": "order-mixed",
                "units": "5",
                "price": "2400.2",
                "tradeOpened": {"tradeID": "trade-new", "units": "2"},
                "tradesClosed": [{"tradeID": "trade-old", "units": "-3"}],
            }
        }
    )

    result = runner.attempt_market_entry(
        client,
        "long",
        units=5,
        client_order_id="gx1-test-netting",
    )

    assert result["status"] == "filled_structure_mismatch"
    assert result["fill_units_exact"] is True
    assert result["pure_trade_open"] is False
    assert result["trade_id"] == "trade-new"


def test_fill_without_trade_opened_never_infers_trade_identity() -> None:
    from gx1.execution import v12_paper_runner as runner

    client = SimpleNamespace(
        create_market_order=lambda *_args, **_kwargs: {
            "orderFillTransaction": {
                "id": "tx-close-only",
                "orderID": "order-close-only",
                "units": "2",
                "tradesClosed": [{"tradeID": "old-trade", "units": "-2"}],
            }
        }
    )

    result = runner.attempt_market_entry(
        client,
        "long",
        units=2,
        client_order_id="gx1-test-no-inferred-trade-id",
    )

    assert result["status"] == "filled_structure_mismatch"
    assert result["trade_id"] is None


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


def test_live_sizing_requires_one_coherent_hedging_broker_snapshot() -> None:
    from gx1.execution import v12_paper_runner as runner

    constraints = runner.learned_sizing_runtime_constraints(
        _broker_fact_client(
            hedging_enabled=True,
            transaction_ids=("9001", "9001", "9001"),
        ),
        bid=2400.0,
        ask=2400.2,
        validated_authority=_runtime_sizing_authority_for_broker_fact_tests(),
    )

    assert constraints["account_last_transaction_id"] == "9001"
    assert constraints["instrument_last_transaction_id"] == "9001"
    assert constraints["exposure_last_transaction_id"] == "9001"


@pytest.mark.parametrize(
    ("hedging_enabled", "transaction_ids", "match"),
    [
        (False, ("9001", "9001", "9001"), "hedgingEnabled=true"),
        (True, ("9001", "9002", "9001"), "different lastTransactionID"),
    ],
)
def test_live_sizing_rejects_netting_or_torn_broker_snapshot(
    hedging_enabled: bool,
    transaction_ids: tuple[str, str, str],
    match: str,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    with pytest.raises(RuntimeError, match=match):
        runner.learned_sizing_runtime_constraints(
            _broker_fact_client(
                hedging_enabled=hedging_enabled,
                transaction_ids=transaction_ids,
            ),
            bid=2400.0,
            ask=2400.2,
            validated_authority=_runtime_sizing_authority_for_broker_fact_tests(),
        )


def test_live_entry_reconciles_exact_broker_and_local_xau_trade_ids() -> None:
    from gx1.execution import v12_paper_runner as runner

    empty_client = _broker_fact_client(
        hedging_enabled=True,
        transaction_ids=("9001", "9001", "9001"),
    )
    assert runner.require_broker_xau_trade_reconciliation(
        empty_client,
        local_open_trades=[],
        max_trades=1,
        expected_exposure_transaction_id="9001",
    ) == ()

    broker_trade = {
        "id": "77",
        "instrument": "XAU_USD",
        "currentUnits": "3",
    }
    orphan_client = _broker_fact_client(
        hedging_enabled=True,
        transaction_ids=("9001", "9001", "9001"),
        trades=[broker_trade],
    )
    with pytest.raises(RuntimeError, match="broker/local XAU trade identity mismatch"):
        runner.require_broker_xau_trade_reconciliation(
            orphan_client,
            local_open_trades=[],
            max_trades=1,
            expected_exposure_transaction_id="9001",
        )
    with pytest.raises(RuntimeError, match="at the admitted cap"):
        runner.require_broker_xau_trade_reconciliation(
            orphan_client,
            local_open_trades=[SimpleNamespace(trade_id="77")],
            max_trades=1,
            expected_exposure_transaction_id="9001",
        )
    with pytest.raises(RuntimeError, match="exposure changed"):
        runner.require_broker_xau_trade_reconciliation(
            empty_client,
            local_open_trades=[],
            max_trades=1,
            expected_exposure_transaction_id="9000",
        )


@pytest.mark.parametrize(
    "quote",
    [
        {"bids": [{"price": "2400.0"}], "asks": [{"price": "2400.2"}]},
        {
            "time": "2026-07-16T12:00:00Z",
            "bids": [{"price": "2400.0"}],
            "asks": [{"price": "2399.9"}],
        },
    ],
)
def test_quote_missing_time_or_valid_bid_ask_contract_fails_closed(quote: dict) -> None:
    from gx1.execution import v12_paper_runner as runner

    client = SimpleNamespace(get_pricing=lambda _instruments: {"prices": [quote]})

    with pytest.raises(ValueError):
        runner.get_current_spread_bps(
            client,
            now_utc=pd.Timestamp("2026-07-16T12:00:30Z").to_pydatetime(),
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
            "volume": np.linspace(10.0, 100.0, n),
        }
    )

    built = incremental._compute_plus5_features(frame)
    served = PrebuiltStateLoader()._augment_cv3_with_v1_legacy(frame)

    pd.testing.assert_frame_equal(
        built[list(PLUS5_FEATURES)],
        served[list(PLUS5_FEATURES)],
    )


def test_collector_cli_rejects_introspection_before_credentials_or_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.execution import v12_oanda_data_collector as collector

    monkeypatch.setattr(
        collector,
        "load_oanda_credentials",
        lambda: (_ for _ in ()).throw(
            AssertionError("credentials must not load")
        ),
    )
    monkeypatch.setenv("GX1_COLLECTOR_POLL_SECONDS", "not-an-integer")
    with pytest.raises(SystemExit) as help_exit:
        collector.main(["--help"])
    assert help_exit.value.code == 0
    with pytest.raises(SystemExit) as invalid_exit:
        collector.main(["--unexpected"])
    assert invalid_exit.value.code == 2


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


def test_collector_rejects_conflicting_completed_bar_overlap() -> None:
    from gx1.execution import v12_oanda_data_collector as collector

    existing = _collector_frame(["2026-07-29T15:00:00Z"])
    identical = existing.copy()
    merged = collector._merge_and_dedupe(existing, identical)
    pd.testing.assert_frame_equal(merged, existing)

    conflicting = identical.copy()
    conflicting.loc[0, "close"] = 3300.5
    conflicting.attrs["source_response_sha256"] = "a" * 64
    with pytest.raises(
        RuntimeError,
        match="COLLECTOR_COMPLETED_BAR_CONFLICT",
    ) as caught:
        collector._merge_and_dedupe(existing, conflicting)
    evidence = caught.value.evidence
    assert evidence["source_response_sha256"] == "a" * 64
    conflict = evidence["completed_bar_conflicts"][0]
    assert conflict["time_utc"] == "2026-07-29T15:00:00+00:00"
    assert conflict["existing"][0]["row_sha256"]
    assert conflict["incoming"][0]["row_sha256"]
    assert (
        conflict["existing"][0]["row_sha256"]
        != conflict["incoming"][0]["row_sha256"]
    )


def test_collector_atomic_write_preserves_previous_snapshot_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.execution import v12_oanda_data_collector as collector

    out_path = tmp_path / "xauusd_m1_20260729.parquet"
    _collector_frame(["2026-07-29T15:00:00Z"]).to_parquet(
        out_path,
        index=False,
    )
    previous = out_path.read_bytes()

    def _partial_then_fail(
        self: pd.DataFrame,
        path,
        *,
        index: bool,
    ) -> None:
        del self, index
        path.write(b"partial")
        raise OSError("simulated parquet failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _partial_then_fail)
    with pytest.raises(OSError, match="simulated parquet failure"):
        collector._write_parquet_atomic(
            _collector_frame(["2026-07-29T15:01:00Z"]),
            out_path,
        )

    assert out_path.read_bytes() == previous
    assert not list(tmp_path.glob(f".{out_path.name}.*.tmp"))


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


def test_collector_partitions_by_candle_utc_date_not_process_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.execution import v12_oanda_data_collector as collector

    monkeypatch.setattr(collector, "OUT_DIR", tmp_path)
    frame = _collector_frame(
        [
            "2026-07-28T23:59:00Z",
            "2026-07-29T00:00:00Z",
        ]
    )
    written = collector._persist_collected_batch(frame)

    assert [path.name for path in written] == [
        "xauusd_m1_20260728.parquet",
        "xauusd_m1_20260729.parquet",
    ]
    assert pd.read_parquet(written[0])["time"].tolist() == [frame["time"].iloc[0]]
    assert pd.read_parquet(written[1])["time"].tolist() == [frame["time"].iloc[1]]
