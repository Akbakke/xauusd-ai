"""Minimal closed-M1 path state for canonical offline unified-Exit replay."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    require_model_native_exit_replay_entry_time,
    require_model_native_runtime_head_evidence,
)
from gx1.models.entry_v10.direction_decision_contract import (
    CLOSED_M1_PATH_SCHEMA_VERSION,
    UNIFIED_EXIT_MAX_PATH_BARS,
    UNIFIED_EXIT_PATH_CHAIN_GENESIS_SHA256,
    UNIFIED_EXIT_PATH_ENVELOPE_SCHEMA_VERSION,
    canonical_closed_m1_bar,
    canonical_closed_m1_path_sha256,
    extend_closed_m1_path_chain_sha256,
    require_unified_exit_output,
    require_unified_exit_path_envelope,
)
from gx1.contracts.unified_exit_input_v1 import (
    require_unified_exit_input_envelope,
)
from gx1.contracts.entry_decision_token_v1 import (
    ENTRY_DECISION_TOKEN_KEY,
    build_entry_decision_token_snapshot,
    require_entry_decision_token_snapshot,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
)


_LONG = "long"
_SHORT = "short"
_SIDES = (_LONG, _SHORT)
_UNIT_NORMALIZATION_CONTRACT = "unit_normalized_direction_exit_research_v1"


def first_full_closed_m1_bar_ts(entry_fill_ts: pd.Timestamp) -> pd.Timestamp:
    """Return the first M1 bar whose full interval starts at/after Entry."""

    parsed = pd.Timestamp(entry_fill_ts)
    if (
        pd.isna(parsed)
        or parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset().total_seconds() != 0.0
    ):
        raise ValueError("entry fill timestamp must be timezone-aware UTC")
    return parsed.tz_convert("UTC").ceil("min")


def _finite_positive_price(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite positive price")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite positive price") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} must be a finite positive price")
    return parsed


@dataclass
class UnifiedExitPathState:
    """One unit-normalized, non-executable Entry-to-Exit replay path."""

    replay_id: str
    entry_fill_ts: pd.Timestamp
    side: str
    entry_bid: float
    entry_ask: float
    entry_snapshot: dict[str, Any]
    entry_decision_token_snapshot: dict[str, Any]
    normalization_contract: str
    bars_in_trade: int = 0
    last_processed_m1_ts: pd.Timestamp | None = None
    current_bid: float = 0.0
    current_ask: float = 0.0
    current_pnl_bps: float = 0.0
    closed_m1_path: list[dict[str, Any]] = field(default_factory=list)
    full_path_chain_sha256: str = UNIFIED_EXIT_PATH_CHAIN_GENESIS_SHA256
    last_exit_decision: dict[str, Any] | None = None
    last_exit_input_envelope: dict[str, Any] | None = None

    @classmethod
    def open_unit_normalized_research(
        cls,
        *,
        entry_ts: pd.Timestamp,
        side: str,
        entry_bid: float,
        entry_ask: float,
        v10_snapshot: dict[str, Any],
        replay_id: str,
        normalization_contract: str,
        model_identity_kind: str,
        model_identity_sha256: str,
        input_normalization_sha256: str,
        contract_mode: str,
    ) -> "UnifiedExitPathState":
        """Open the sole non-executable state admitted by offline replay."""

        if normalization_contract != _UNIT_NORMALIZATION_CONTRACT:
            raise ValueError("exact unit-normalized research contract is required")
        if contract_mode != MODEL_NATIVE_CONTRACT_MODE:
            raise ValueError("offline replay token contract mode is stale")
        if side not in _SIDES:
            raise ValueError(f"side must be one of {_SIDES}")
        if (
            not isinstance(replay_id, str)
            or not replay_id
            or replay_id.strip() != replay_id
            or "\x00" in replay_id
        ):
            raise ValueError("offline replay identity must be an exact non-empty string")

        bid = _finite_positive_price(entry_bid, label="entry_bid")
        ask = _finite_positive_price(entry_ask, label="entry_ask")
        if ask <= bid:
            raise ValueError("entry_ask must exceed entry_bid")
        first_full_closed_m1_bar_ts(entry_ts)
        raw_entry = pd.Timestamp(entry_ts).tz_convert("UTC")

        snapshot = require_model_native_runtime_head_evidence(
            v10_snapshot,
            context="UNIFIED_EXIT_PATH_STATE_OPEN",
        )
        require_model_native_exit_replay_entry_time(
            snapshot,
            raw_entry,
            context="UNIFIED_EXIT_PATH_STATE_OPEN",
        )
        expected_side = {"LONG": _LONG, "SHORT": _SHORT}.get(
            snapshot["model_direction"]
        )
        if expected_side != side:
            raise ValueError("offline replay side differs from model direction")

        token_snapshot = build_entry_decision_token_snapshot(
            token=snapshot[ENTRY_DECISION_TOKEN_KEY],
            decision_time=snapshot["decision_ts"],
            fill_time=raw_entry,
            model_identity_kind=model_identity_kind,
            model_identity_sha256=model_identity_sha256,
            input_normalization_sha256=input_normalization_sha256,
            contract_mode=contract_mode,
            model_direction_index=int(snapshot["model_direction_index"]),
            model_direction=str(snapshot["model_direction"]),
            side=side,
            entry_bid=bid,
            entry_ask=ask,
            trade_identity=replay_id,
        )

        return cls(
            replay_id=replay_id,
            entry_fill_ts=raw_entry,
            side=side,
            entry_bid=bid,
            entry_ask=ask,
            entry_snapshot=deepcopy(snapshot),
            entry_decision_token_snapshot=token_snapshot,
            normalization_contract=normalization_contract,
            current_bid=bid,
            current_ask=ask,
        )

    def clone_for_exit_decision(self) -> "UnifiedExitPathState":
        """Return a detached state for one transactional M1/model transition."""

        return deepcopy(self)

    def _pnl_bps(self, bid: float, ask: float) -> float:
        if self.side == _LONG:
            return (bid - self.entry_ask) / self.entry_ask * 10_000.0
        return (self.entry_bid - ask) / self.entry_bid * 10_000.0

    def update_bar(
        self,
        *,
        schema_version: str,
        time: str,
        complete: bool,
        source_path: str,
        source_sha256: str,
        bid_open: float,
        bid_high: float,
        bid_low: float,
        bid_close: float,
        ask_open: float,
        ask_high: float,
        ask_low: float,
        ask_close: float,
        mid_open: float,
        mid_high: float,
        mid_low: float,
        mid_close: float,
        volume: int,
    ) -> None:
        """Stage one literal, complete, source-bound closed M1 row."""

        if schema_version != CLOSED_M1_PATH_SCHEMA_VERSION:
            raise ValueError("closed M1 path schema_version mismatch")
        canonical_bar = canonical_closed_m1_bar(
            m1_bar_ts=pd.Timestamp(time),
            complete=complete,
            source_path=source_path,
            source_sha256=source_sha256,
            bid_open=bid_open,
            bid_high=bid_high,
            bid_low=bid_low,
            bid_close=bid_close,
            ask_open=ask_open,
            ask_high=ask_high,
            ask_low=ask_low,
            ask_close=ask_close,
            mid_open=mid_open,
            mid_high=mid_high,
            mid_low=mid_low,
            mid_close=mid_close,
            volume=volume,
        )
        observed_ts = pd.Timestamp(canonical_bar["time"])
        if self.last_processed_m1_ts is None:
            expected_first = first_full_closed_m1_bar_ts(self.entry_fill_ts)
            if observed_ts != expected_first:
                raise ValueError(
                    "first closed M1 row must be the first full bar after Entry"
                )
        elif observed_ts <= self.last_processed_m1_ts:
            raise ValueError("closed M1 row clock must move strictly forward")

        next_bid = float(canonical_bar["bid_close"])
        next_ask = float(canonical_bar["ask_close"])
        next_pnl = float(self._pnl_bps(next_bid, next_ask))
        self.full_path_chain_sha256 = extend_closed_m1_path_chain_sha256(
            self.full_path_chain_sha256,
            canonical_bar,
        )
        self.closed_m1_path.append(canonical_bar)
        if len(self.closed_m1_path) > UNIFIED_EXIT_MAX_PATH_BARS:
            self.closed_m1_path.pop(0)
        self.bars_in_trade += 1
        self.last_processed_m1_ts = observed_ts
        self.current_bid = next_bid
        self.current_ask = next_ask
        self.current_pnl_bps = next_pnl

    def build_closed_m1_path_evidence(self) -> dict[str, Any]:
        """Build and validate the exact content-hashed closed-M1 prefix."""

        rows = deepcopy(self.closed_m1_path)
        if (
            not rows
            or self.last_processed_m1_ts is None
            or len(rows) != min(self.bars_in_trade, UNIFIED_EXIT_MAX_PATH_BARS)
        ):
            raise ValueError("closed M1 path/bar count mismatch")
        envelope = {
            "schema_version": UNIFIED_EXIT_PATH_ENVELOPE_SCHEMA_VERSION,
            "entry_fill_ts": self.entry_fill_ts.isoformat(),
            "first_full_m1_bar_ts": first_full_closed_m1_bar_ts(
                self.entry_fill_ts
            ).isoformat(),
            "last_closed_m1_bar_ts": self.last_processed_m1_ts.isoformat(),
            "bars_in_trade": self.bars_in_trade,
            "retained_path_length": len(rows),
            "path_rows": rows,
            "path_rows_sha256": canonical_closed_m1_path_sha256(rows),
            "full_path_chain_sha256": self.full_path_chain_sha256,
        }
        return require_unified_exit_path_envelope(
            envelope,
            context="UNIFIED_EXIT_PATH_STATE",
        )

    def bind_unified_exit_decision(
        self,
        decision: dict[str, Any],
        *,
        expected_bundle_sha256: str,
        exit_input_envelope: dict[str, Any],
    ) -> None:
        """Content-bind one same-bundle model decision to this path prefix."""

        snapshot = require_model_native_runtime_head_evidence(
            self.entry_snapshot,
            context="UNIFIED_EXIT_PATH_STATE_BIND",
        )
        envelope = self.build_closed_m1_path_evidence()
        input_envelope = require_unified_exit_input_envelope(
            exit_input_envelope
        )
        if (
            input_envelope["decision_identity"] != self.replay_id
            or input_envelope["decision_time"]
            != self.last_processed_m1_ts.isoformat()
            or input_envelope["side"] != self.side
            or float(input_envelope["entry_bid"]) != self.entry_bid
            or float(input_envelope["entry_ask"]) != self.entry_ask
            or input_envelope["entry_decision_token_snapshot"]
            != self.entry_decision_token_snapshot
        ):
            raise ValueError("Exit input envelope differs from replay state")
        validated = require_unified_exit_output(
            decision,
            context="UNIFIED_EXIT_PATH_STATE_BIND",
            expected_bundle_sha256=expected_bundle_sha256,
            entry_snapshot=snapshot,
            exit_path_envelope=envelope,
            exit_input_envelope=input_envelope,
        )
        self.last_exit_decision = deepcopy(validated)
        self.last_exit_input_envelope = deepcopy(input_envelope)

    def commit_complete_exit_bar(
        self,
        staged: "UnifiedExitPathState",
        *,
        expected_bundle_sha256: str,
    ) -> None:
        """Commit exactly one validated staged M1/model transition."""

        if not isinstance(staged, UnifiedExitPathState):
            raise TypeError("staged Exit state has the wrong type")
        immutable_identity = (
            "replay_id",
            "entry_fill_ts",
            "side",
            "entry_bid",
            "entry_ask",
            "entry_snapshot",
            "entry_decision_token_snapshot",
            "normalization_contract",
        )
        if any(
            getattr(staged, field_name) != getattr(self, field_name)
            for field_name in immutable_identity
        ):
            raise ValueError("staged Exit state changed offline replay identity")
        if (
            staged.bars_in_trade != self.bars_in_trade + 1
            or not staged.closed_m1_path
            or staged.closed_m1_path
            != (self.closed_m1_path + [staged.closed_m1_path[-1]])[
                -UNIFIED_EXIT_MAX_PATH_BARS:
            ]
        ):
            raise ValueError("staged Exit state must contain exactly one new M1 bar")
        if (
            staged.last_exit_decision is None
            or staged.last_exit_input_envelope is None
        ):
            raise ValueError("staged Exit state lacks its model decision/input")

        snapshot = require_model_native_runtime_head_evidence(
            staged.entry_snapshot,
            context="UNIFIED_EXIT_PATH_STATE_COMMIT",
        )
        envelope = staged.build_closed_m1_path_evidence()
        input_envelope = require_unified_exit_input_envelope(
            staged.last_exit_input_envelope
        )
        require_entry_decision_token_snapshot(
            staged.entry_decision_token_snapshot
        )
        require_unified_exit_output(
            staged.last_exit_decision,
            context="UNIFIED_EXIT_PATH_STATE_COMMIT",
            expected_bundle_sha256=expected_bundle_sha256,
            entry_snapshot=snapshot,
            exit_path_envelope=envelope,
            exit_input_envelope=input_envelope,
        )
        self.__dict__.clear()
        self.__dict__.update(deepcopy(staged.__dict__))


__all__ = ("UnifiedExitPathState", "first_full_closed_m1_bar_ts")
