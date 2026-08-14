#!/usr/bin/env python3
"""GX1 XAUUSD model-native paper/live execution runner.

Entry direction is the calibrated model-native XAU model's exact
LONG/SHORT/FLAT argmax.
The runner may fail closed for execution safety, but it has no session, trend,
utility, confidence, sizing, or threshold rule that can rewrite model direction.

Modus operandi:
    1. Wait for next M1 candle close (poll OANDA every 5-10s).
    2. Capture exact bid/ask spread as model/runtime evidence.
    3. Make the contract-bound model-direction decision via V12Pipeline.
    4. If model direction is LONG/SHORT and execution safety admits it, place a
       learned, proof-bound integer-unit market order via OANDA.
    5. Catch MARKET_ORDER_REJECT_TRANSACTION; log reason + spread + time.
    6. If trade open: same-bundle per-closed-M1 Exit argmax; close on EXIT_NOW.
    7. Log everything to daily journal for replay/comparison vs Phase 6 baseline.

Run (live demo on OANDA practice):
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 \\
        gx1/execution/v12_paper_runner.py --max-trades 1

We trade year-round, all sessions (Asia included): session, structure, trend,
liquidity, volatility, momentum, price action, path quality, and utility are
model inputs/evidence, never post-model direction overrides.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import logging
import math
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.contracts.entry_model_native_runtime_evidence_v1 import (  # noqa: E402
    MODEL_NATIVE_MAX_ENTRY_SIGNAL_LATENCY_SEC,
    MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS,
    MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
    MODEL_NATIVE_RUNTIME_POLICY,
    RETIRED_RUNTIME_EVIDENCE_FRAGMENTS,
    require_model_native_entry_time,
    require_model_native_runtime_evidence,
)
from gx1.contracts.entry_model_native_sizing_authority_v1 import (  # noqa: E402
    MODEL_NATIVE_SIZING_MODE_LEARNED,
    ModelNativeSizingUnavailable,
    ValidatedLearnedSizingAuthority,
    apply_model_native_sizing,
    prepare_model_native_sizing_authority,
    require_model_native_sizing_authority_contract,
)
from gx1.models.entry_v10.direction_decision_contract import (  # noqa: E402
    MODEL_DIRECTION_ACTION_BY_INDEX,
    MODEL_DIRECTION_ACTION_ID_BY_INDEX,
    MODEL_DIRECTION_EXECUTION_SIDE_BY_INDEX,
    MODEL_DIRECTION_INDEX_BY_NAME,
    MODEL_DIRECTION_SELECTION_MODE,
)
from gx1.time.session_detector import SESSION_ORDER  # noqa: E402
from gx1.contracts.entry_model_native_signal_v1 import (  # noqa: E402
    MODEL_NATIVE_CONTRACT_MODE,
)

ENV_FILE = REPO_ROOT / ".env"
if ENV_FILE.is_file():
    with ENV_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)

LOG = logging.getLogger("v12_paper")
INSTRUMENT = "XAU_USD"
_RUNTIME_DEPS_LOADED = False


def _load_runtime_dependencies() -> None:
    """Import artifact-touching runtime dependencies only after the Entry guard."""
    global _RUNTIME_DEPS_LOADED
    global OandaClient, OandaClientConfig, load_oanda_credentials
    global EntryDecisionUnavailable, ExitDecisionUnavailable, V12Pipeline, TradeState, TradeJournal
    global TRADE_STATE_MODEL_BUNDLE_BINDING_SCHEMA_VERSION
    global TRADE_STATE_SOURCE_PAIR_BINDING_SCHEMA_VERSION
    global TRADE_STATE_BROKER_ACCOUNT_BINDING_SCHEMA_VERSION
    global build_trade_broker_account_binding
    global require_trade_broker_account_binding
    if _RUNTIME_DEPS_LOADED:
        return

    from gx1.execution.oanda_client import OandaClient as _OandaClient
    from gx1.execution.oanda_client import OandaClientConfig as _OandaClientConfig
    from gx1.execution.oanda_credentials import load_oanda_credentials as _load_oanda_credentials
    from gx1.execution.v12_pipeline import (
        EntryDecisionUnavailable as _EntryDecisionUnavailable,
        ExitDecisionUnavailable as _ExitDecisionUnavailable,
        V12Pipeline as _V12Pipeline,
    )
    from gx1.execution.v12_trade_state import (
        TRADE_STATE_BROKER_ACCOUNT_BINDING_SCHEMA_VERSION as _BROKER_ACCOUNT_BINDING_SCHEMA,
        TRADE_STATE_MODEL_BUNDLE_BINDING_SCHEMA_VERSION as _BUNDLE_BINDING_SCHEMA,
        TRADE_STATE_SOURCE_PAIR_BINDING_SCHEMA_VERSION as _PAIR_BINDING_SCHEMA,
        TradeState as _TradeState,
        build_trade_broker_account_binding as _build_broker_account_binding,
        require_trade_broker_account_binding as _require_broker_account_binding,
    )
    from gx1.monitoring.trade_journal import TradeJournal as _TradeJournal

    OandaClient = _OandaClient
    OandaClientConfig = _OandaClientConfig
    load_oanda_credentials = _load_oanda_credentials
    EntryDecisionUnavailable = _EntryDecisionUnavailable
    ExitDecisionUnavailable = _ExitDecisionUnavailable
    V12Pipeline = _V12Pipeline
    TradeState = _TradeState
    TRADE_STATE_MODEL_BUNDLE_BINDING_SCHEMA_VERSION = _BUNDLE_BINDING_SCHEMA
    TRADE_STATE_SOURCE_PAIR_BINDING_SCHEMA_VERSION = _PAIR_BINDING_SCHEMA
    TRADE_STATE_BROKER_ACCOUNT_BINDING_SCHEMA_VERSION = (
        _BROKER_ACCOUNT_BINDING_SCHEMA
    )
    build_trade_broker_account_binding = _build_broker_account_binding
    require_trade_broker_account_binding = _require_broker_account_binding
    TradeJournal = _TradeJournal
    _RUNTIME_DEPS_LOADED = True

# These legacy variables used to arm live-only direction gates or post-model
# sizing. Their code paths are deleted. Presence is rejected at startup so a
# stale shell/.env cannot silently imply an operating point that no longer exists.
RETIRED_ENTRY_OVERRIDE_ENV = (
    "GX1_PURE_PHASE6",
    "GX1_SKIP_ASIA",
    "GX1_ADAPTIVE_MIN_ADV_ATR_MULT",
    "GX1_ADAPTIVE_MIN_ADV_FLOOR_BPS",
    "GX1_POSITION_CONFIDENCE_MIN",
    "GX1_SIZING_MODE",
    "GX1_SIZING_MAX_MULT",
    "GX1_SIZING_MIN_MULT",
    "GX1_SIZING_CONV_LO",
    "GX1_SIZING_CONV_HI",
    "GX1_SIZING_CONV_SRC",
    "GX1_SIZING_MARGIN_POW",
    "GX1_SIZING_MARGIN_REF",
    "GX1_SIZING_ATR_REF_BPS",
    "GX1_SIZING_ATR_FLOOR_BPS",
    "GX1_DYNAMIC_SIZING",
    "GX1_POSITION_SIZE_MULTIPLIER",
    "GX1_POSITION_SIZE_FROM_MODEL",
    "GX1_REGIME_SIZE_MULTIPLIER",
    "GX1_SESSION_SIZE_MULTIPLIER",
    "GX1_TREND_SIZE_MULTIPLIER",
    "GX1_UTILITY_SIZE_MULTIPLIER",
    "GX1_CONVICTION_SIZE_MULTIPLIER",
    "GX1_REGIME_RECAL",
    "GX1_MAX_ENTRY_DECISION_LATENCY_SEC",
    "GX1_SMART_CTX_MTF_INCREMENTAL",
    "GX1_SMART_CTX_MTF_WARMUP_M5",
    "GX1_MODEL_NATIVE_CTX_MTF_WARMUP_M5",
    "GX1_SMART_PARITY_GATE_MAX_AGE_HOURS",
    "GX1_SMART_PARITY_GATE_MAX_CUTOFF_LAG_HOURS",
    "GX1_SMART_DIRECTION_AUDIT_MAX_AGE_HOURS",
    "GX1_SMART_CTX_MAX_STALENESS_M5",
)
RETIRED_ENTRY_SIZING_ENV_PREFIXES = (
    "GX1_SIZING_",
    "GX1_DYNAMIC_SIZING",
    "GX1_POSITION_SIZE_",
    "GX1_REGIME_SIZE_",
    "GX1_SESSION_SIZE_",
    "GX1_TREND_SIZE_",
    "GX1_UTILITY_SIZE_",
    "GX1_CONVICTION_SIZE_",
)

JOURNAL_DIR = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")
TRADE_STATE_FILE = JOURNAL_DIR / "open_trade_state.json"  # LEGACY single-trade marker (migrated on startup)
TRADE_STATE_DIR = JOURNAL_DIR / "open_trades"             # one JSON file per open virtual trade
ENTRY_INTENT_DIR = TRADE_STATE_DIR / "entry_intents"
RESOLVED_ENTRY_INTENT_DIR = TRADE_STATE_DIR / "resolved_entry_intents"
BROKER_ENTRY_INTENT_SCHEMA_VERSION = "gx1_broker_entry_intent_v2"
CLOSE_INTENT_DIR = TRADE_STATE_DIR / "close_intents"
RESOLVED_CLOSE_INTENT_DIR = TRADE_STATE_DIR / "resolved_close_intents"
REJECTED_CLOSE_INTENT_DIR = TRADE_STATE_DIR / "rejected_close_intents"
BROKER_CLOSE_INTENT_SCHEMA_VERSION = "gx1_broker_close_intent_v1"
BROKER_ACCOUNT_BINDING_SCHEMA_VERSION = (
    "gx1_trade_state_broker_account_binding_v1"
)
BROKER_CLOSE_MUTATION_LOCK_FILENAME = ".broker_close_mutation.lock"
RUNNER_SINGLETON_LOCK_FILE = TRADE_STATE_DIR / ".v12_runner.lock"
TRADE_ALERTS_FILE = Path("/home/andre2/TRADES_ALERTS.txt")  # easy-to-tail alerts file


def write_trade_alert(line: str) -> None:
    """Append a one-line alert to TRADES_ALERTS.txt (for `tail -f` monitoring)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        with TRADE_ALERTS_FILE.open("a") as f:
            f.write(f"[{ts}] {line}\n")
    except Exception:
        pass   # alerts file is best-effort; never crash the runner over it


def acquire_runner_singleton_lock(
    lock_path: Path = RUNNER_SINGLETON_LOCK_FILE,
):
    """Own the persisted-state writer lease for this runner process."""

    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if (
        path.parent.is_symlink()
        or not path.parent.is_dir()
        or path.is_symlink()
    ):
        raise RuntimeError("RUNNER_SINGLETON_LOCK_PATH_INVALID")
    handle = path.open("a+b")
    try:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("RUNNER_SINGLETON_LOCK_PATH_INVALID")
        fcntl.flock(
            handle.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            "RUNNER_SINGLETON_ALREADY_ACTIVE"
        ) from exc
    except Exception:
        handle.close()
        raise
    return handle


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _require_broker_account_binding_record(
    value: object,
) -> dict[str, str]:
    if (
        not isinstance(value, dict)
        or set(value)
        != {"schema_version", "environment", "account_id_sha256"}
        or value.get("schema_version")
        != BROKER_ACCOUNT_BINDING_SCHEMA_VERSION
        or value.get("environment") not in {"practice", "live"}
    ):
        raise RuntimeError("BROKER_ACCOUNT_BINDING_SCHEMA_INVALID")
    digest = value.get("account_id_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest.lower() != digest
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError("BROKER_ACCOUNT_BINDING_HASH_INVALID")
    return {
        "schema_version": BROKER_ACCOUNT_BINDING_SCHEMA_VERSION,
        "environment": value["environment"],
        "account_id_sha256": digest,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_broker_entry_intent(
    *,
    side: str,
    units: int,
    decision_snapshot: dict[str, Any],
    sizing_application: dict[str, Any],
    model_bundle_binding: dict[str, Any],
    entry_source_pair_binding: dict[str, Any],
    broker_account_binding: dict[str, str],
    launch_lease: dict[str, str],
    created_utc: datetime,
) -> dict[str, Any]:
    """Build one idempotent broker intent before any order mutation."""

    if (
        side not in {"long", "short"}
        or isinstance(units, bool)
        or not isinstance(units, int)
        or units <= 0
    ):
        raise RuntimeError("BROKER_ENTRY_INTENT_SIDE_OR_UNITS_INVALID")
    created = pd.Timestamp(created_utc)
    if (
        pd.isna(created)
        or created.tzinfo is None
        or created.utcoffset() is None
        or created.utcoffset().total_seconds() != 0.0
    ):
        raise RuntimeError("BROKER_ENTRY_INTENT_TIME_INVALID")
    signed_units = units if side == "long" else -units
    validated_broker_account_binding = (
        _require_broker_account_binding_record(
            broker_account_binding
        )
    )
    identity = {
        "instrument": INSTRUMENT,
        "side": side,
        "signed_units": signed_units,
        "decision_ts": decision_snapshot.get("decision_ts"),
        "model_direction": decision_snapshot.get("model_direction"),
        "model_bundle_sha256": model_bundle_binding.get(
            "bundle_sha256"
        ),
        "model_bundle_binding_sha256": hashlib.sha256(
            _canonical_json_bytes(model_bundle_binding)
        ).hexdigest(),
        "pair_generation_id": entry_source_pair_binding.get(
            "pair_generation_id"
        ),
        "pair_manifest_sha256": entry_source_pair_binding.get(
            "pair_manifest_sha256"
        ),
        "sizing_application_sha256": hashlib.sha256(
            _canonical_json_bytes(sizing_application)
        ).hexdigest(),
        "launch_state_sha256": launch_lease.get(
            "launch_state_sha256"
        ),
        "artifact_registry_sha256": launch_lease.get(
            "artifact_registry_sha256"
        ),
        "broker_account_binding": validated_broker_account_binding,
    }
    client_order_id = (
        "gx1-" + hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()[:40]
    )
    return {
        "schema_version": BROKER_ENTRY_INTENT_SCHEMA_VERSION,
        "created_utc": created.tz_convert("UTC").isoformat(),
        "client_order_id": client_order_id,
        "instrument": INSTRUMENT,
        "side": side,
        "signed_units": signed_units,
        "decision_snapshot": dict(decision_snapshot),
        "sizing_application": dict(sizing_application),
        "model_bundle_binding": dict(model_bundle_binding),
        "entry_source_pair_binding": dict(entry_source_pair_binding),
        "broker_account_binding": validated_broker_account_binding,
        "launch_lease": dict(launch_lease),
        "identity_sha256": hashlib.sha256(
            _canonical_json_bytes(identity)
        ).hexdigest(),
    }


def persist_broker_entry_intent(
    intent: dict[str, Any],
    *,
    intent_root: Path = ENTRY_INTENT_DIR,
) -> Path:
    """Durably create one no-replace intent before the broker call."""

    client_order_id = str(intent.get("client_order_id") or "")
    if (
        intent.get("schema_version") != BROKER_ENTRY_INTENT_SCHEMA_VERSION
        or not client_order_id.startswith("gx1-")
    ):
        raise RuntimeError("BROKER_ENTRY_INTENT_SCHEMA_INVALID")
    root = Path(intent_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{client_order_id}.json"
    encoded = _canonical_json_bytes(intent) + b"\n"
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise RuntimeError(
            "BROKER_ENTRY_INTENT_ALREADY_EXISTS_RECONCILIATION_REQUIRED"
        ) from exc
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short broker Entry intent write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(root)
    return path


def resolve_broker_entry_intent(
    intent_path: Path,
    *,
    unresolved_root: Path = ENTRY_INTENT_DIR,
    resolved_root: Path = RESOLVED_ENTRY_INTENT_DIR,
) -> Path:
    """Move a known-outcome intent out of the unresolved authority set."""

    source = Path(intent_path)
    if source.parent.resolve() != Path(unresolved_root).resolve():
        raise RuntimeError("BROKER_ENTRY_INTENT_ROOT_MISMATCH")
    resolved_root = Path(resolved_root)
    resolved_root.mkdir(parents=True, exist_ok=True)
    if resolved_root.is_symlink() or not resolved_root.is_dir():
        raise RuntimeError("BROKER_ENTRY_RESOLVED_ROOT_INVALID")
    target = resolved_root / source.name
    if target.exists():
        raise RuntimeError("BROKER_ENTRY_INTENT_RESOLUTION_EXISTS")
    os.replace(source, target)
    _fsync_directory(source.parent)
    _fsync_directory(resolved_root)
    return target


def build_broker_close_intent(
    trade: TradeState,
    *,
    broker_account_binding: dict[str, str],
    created_utc: datetime,
) -> dict[str, Any]:
    """Freeze one exact EXIT_NOW mutation before calling the broker."""

    trade_id = getattr(trade, "trade_id", None)
    side = getattr(trade, "side", None)
    units = getattr(trade, "units", None)
    exit_decision = getattr(trade, "last_exit_decision", None)
    if (
        not isinstance(trade_id, str)
        or not trade_id
        or trade_id.strip() != trade_id
        or side not in {"long", "short"}
        or isinstance(units, bool)
        or not isinstance(units, int)
        or units <= 0
        or not isinstance(exit_decision, dict)
        or exit_decision.get("action") != "EXIT_NOW"
    ):
        raise RuntimeError("BROKER_CLOSE_INTENT_TRADE_STATE_INVALID")
    created = pd.Timestamp(created_utc)
    if (
        pd.isna(created)
        or created.tzinfo is None
        or created.utcoffset() is None
        or created.utcoffset().total_seconds() != 0.0
    ):
        raise RuntimeError("BROKER_CLOSE_INTENT_TIME_INVALID")
    account_binding = _require_broker_account_binding_record(
        broker_account_binding
    )
    trade_binding = _require_broker_account_binding_record(
        getattr(trade, "broker_account_binding", None)
    )
    if account_binding != trade_binding:
        raise RuntimeError("BROKER_CLOSE_INTENT_ACCOUNT_MISMATCH")
    trade_state_sha256 = hashlib.sha256(
        _canonical_json_bytes(trade.to_dict())
    ).hexdigest()
    exit_decision_sha256 = hashlib.sha256(
        _canonical_json_bytes(exit_decision)
    ).hexdigest()
    exposure_identity = {
        "instrument": INSTRUMENT,
        "trade_id": trade_id,
        "broker_account_binding": account_binding,
    }
    exposure_identity_sha256 = hashlib.sha256(
        _canonical_json_bytes(exposure_identity)
    ).hexdigest()
    identity = {
        **exposure_identity,
        "side": side,
        "expected_close_signed_units": (
            -units if side == "long" else units
        ),
        "exposure_identity_sha256": exposure_identity_sha256,
        "trade_state_sha256": trade_state_sha256,
        "exit_decision_sha256": exit_decision_sha256,
    }
    identity_sha256 = hashlib.sha256(
        _canonical_json_bytes(identity)
    ).hexdigest()
    return {
        "schema_version": BROKER_CLOSE_INTENT_SCHEMA_VERSION,
        "created_utc": created.tz_convert("UTC").isoformat(),
        **identity,
        "identity_sha256": identity_sha256,
        "close_intent_id": f"gx1-close-{identity_sha256[:40]}",
    }


def _broker_close_exposure_filename(
    intent: dict[str, Any],
) -> str:
    exposure_identity_sha256 = str(
        intent.get("exposure_identity_sha256") or ""
    )
    if (
        len(exposure_identity_sha256) != 64
        or exposure_identity_sha256.lower()
        != exposure_identity_sha256
        or any(
            character not in "0123456789abcdef"
            for character in exposure_identity_sha256
        )
    ):
        raise RuntimeError(
            "BROKER_CLOSE_EXPOSURE_IDENTITY_HASH_INVALID"
        )
    return (
        f"gx1-close-exposure-{exposure_identity_sha256}.json"
    )


@contextmanager
def broker_close_mutation_lock(
    *,
    intent_root: Path = CLOSE_INTENT_DIR,
    resolved_root: Path = RESOLVED_CLOSE_INTENT_DIR,
):
    """Serialize close intent creation, mutation and tombstone publication."""

    unresolved_parent = Path(intent_root).parent
    resolved_parent = Path(resolved_root).parent
    unresolved_parent.mkdir(parents=True, exist_ok=True)
    resolved_parent.mkdir(parents=True, exist_ok=True)
    if (
        unresolved_parent.is_symlink()
        or resolved_parent.is_symlink()
        or not unresolved_parent.is_dir()
        or not resolved_parent.is_dir()
    ):
        raise RuntimeError("BROKER_CLOSE_MUTATION_LOCK_ROOT_INVALID")
    if unresolved_parent.resolve() != resolved_parent.resolve():
        raise RuntimeError("BROKER_CLOSE_INTENT_PARENT_ROOT_MISMATCH")
    lock_path = (
        unresolved_parent / BROKER_CLOSE_MUTATION_LOCK_FILENAME
    )
    if lock_path.is_symlink():
        raise RuntimeError("BROKER_CLOSE_MUTATION_LOCK_PATH_INVALID")
    with lock_path.open("a+b") as lock_handle:
        if lock_path.is_symlink() or not lock_path.is_file():
            raise RuntimeError(
                "BROKER_CLOSE_MUTATION_LOCK_PATH_INVALID"
            )
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _persist_broker_close_intent_locked(
    intent: dict[str, Any],
    *,
    intent_root: Path,
    resolved_root: Path,
) -> Path:
    """Create one intent while the shared close lock is held."""

    intent_id = str(intent.get("close_intent_id") or "")
    if (
        intent.get("schema_version") != BROKER_CLOSE_INTENT_SCHEMA_VERSION
        or not intent_id.startswith("gx1-close-")
    ):
        raise RuntimeError("BROKER_CLOSE_INTENT_SCHEMA_INVALID")
    root = Path(intent_root)
    resolved_root = Path(resolved_root)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("BROKER_CLOSE_INTENT_ROOT_INVALID")
    if resolved_root.is_symlink():
        raise RuntimeError("BROKER_CLOSE_RESOLVED_ROOT_INVALID")
    path = root / _broker_close_exposure_filename(intent)
    tombstone = resolved_root / path.name
    if tombstone.exists() or tombstone.is_symlink():
        raise RuntimeError(
            "BROKER_CLOSE_INTENT_ALREADY_RESOLVED_NO_REPLAY"
        )
    encoded = _canonical_json_bytes(intent) + b"\n"
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise RuntimeError(
            "BROKER_CLOSE_INTENT_ALREADY_EXISTS_RECONCILIATION_REQUIRED"
        ) from exc
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short broker close intent write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(root)
    return path


def persist_broker_close_intent(
    intent: dict[str, Any],
    *,
    intent_root: Path = CLOSE_INTENT_DIR,
    resolved_root: Path = RESOLVED_CLOSE_INTENT_DIR,
) -> Path:
    """Durably create one globally no-replay close intent."""

    with broker_close_mutation_lock(
        intent_root=intent_root,
        resolved_root=resolved_root,
    ):
        return _persist_broker_close_intent_locked(
            intent,
            intent_root=intent_root,
            resolved_root=resolved_root,
        )


def load_broker_close_intent(path: Path) -> dict[str, Any]:
    """Strict-load a durable close intent without filling evidence."""

    expected_fields = {
        "schema_version",
        "created_utc",
        "instrument",
        "trade_id",
        "side",
        "expected_close_signed_units",
        "broker_account_binding",
        "exposure_identity_sha256",
        "trade_state_sha256",
        "exit_decision_sha256",
        "identity_sha256",
        "close_intent_id",
    }
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise RuntimeError("BROKER_CLOSE_INTENT_FILE_INVALID")
    raw = source.read_bytes()
    try:
        intent = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("BROKER_CLOSE_INTENT_JSON_INVALID") from exc
    if (
        not isinstance(intent, dict)
        or set(intent) != expected_fields
        or raw != _canonical_json_bytes(intent) + b"\n"
        or intent["schema_version"] != BROKER_CLOSE_INTENT_SCHEMA_VERSION
        or intent["instrument"] != INSTRUMENT
        or intent["side"] not in {"long", "short"}
        or not isinstance(intent["trade_id"], str)
        or not intent["trade_id"]
        or intent["trade_id"].strip() != intent["trade_id"]
        or isinstance(intent["expected_close_signed_units"], bool)
        or not isinstance(intent["expected_close_signed_units"], int)
        or intent["expected_close_signed_units"] == 0
        or (
            intent["expected_close_signed_units"] < 0
        )
        != (intent["side"] == "long")
        or source.name != _broker_close_exposure_filename(intent)
    ):
        raise RuntimeError("BROKER_CLOSE_INTENT_CONTRACT_INVALID")
    account_binding = _require_broker_account_binding_record(
        intent["broker_account_binding"]
    )
    identity = {
        "instrument": INSTRUMENT,
        "trade_id": intent["trade_id"],
        "side": intent["side"],
        "expected_close_signed_units": intent[
            "expected_close_signed_units"
        ],
        "broker_account_binding": account_binding,
        "exposure_identity_sha256": intent[
            "exposure_identity_sha256"
        ],
        "trade_state_sha256": intent["trade_state_sha256"],
        "exit_decision_sha256": intent["exit_decision_sha256"],
    }
    for field in (
        "exposure_identity_sha256",
        "trade_state_sha256",
        "exit_decision_sha256",
        "identity_sha256",
    ):
        digest = intent[field]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest.lower() != digest
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise RuntimeError("BROKER_CLOSE_INTENT_HASH_INVALID")
    identity_sha256 = hashlib.sha256(
        _canonical_json_bytes(identity)
    ).hexdigest()
    exposure_identity_sha256 = hashlib.sha256(
        _canonical_json_bytes(
            {
                "instrument": INSTRUMENT,
                "trade_id": intent["trade_id"],
                "broker_account_binding": account_binding,
            }
        )
    ).hexdigest()
    if (
        intent["exposure_identity_sha256"]
        != exposure_identity_sha256
        or
        intent["identity_sha256"] != identity_sha256
        or intent["close_intent_id"]
        != f"gx1-close-{identity_sha256[:40]}"
    ):
        raise RuntimeError("BROKER_CLOSE_INTENT_IDENTITY_HASH_MISMATCH")
    return intent


def _resolve_broker_close_intent_locked(
    intent_path: Path,
    *,
    unresolved_root: Path,
    resolved_root: Path,
) -> Path:
    """Publish a permanent resolved tombstone while holding the close lock."""

    source = Path(intent_path)
    if source.parent.resolve() != Path(unresolved_root).resolve():
        raise RuntimeError("BROKER_CLOSE_INTENT_ROOT_MISMATCH")
    resolved_root = Path(resolved_root)
    resolved_root.mkdir(parents=True, exist_ok=True)
    if resolved_root.is_symlink() or not resolved_root.is_dir():
        raise RuntimeError("BROKER_CLOSE_RESOLVED_ROOT_INVALID")
    target = resolved_root / source.name
    if target.exists() or target.is_symlink():
        raise RuntimeError("BROKER_CLOSE_INTENT_RESOLUTION_EXISTS")
    os.replace(source, target)
    _fsync_directory(source.parent)
    _fsync_directory(resolved_root)
    return target


def resolve_broker_close_intent(
    intent_path: Path,
    *,
    unresolved_root: Path = CLOSE_INTENT_DIR,
    resolved_root: Path = RESOLVED_CLOSE_INTENT_DIR,
) -> Path:
    """Move a proven close to its permanent no-replay tombstone."""

    with broker_close_mutation_lock(
        intent_root=unresolved_root,
        resolved_root=resolved_root,
    ):
        return _resolve_broker_close_intent_locked(
            intent_path,
            unresolved_root=unresolved_root,
            resolved_root=resolved_root,
        )


def finalize_broker_close_rejection(
    *,
    intent: dict[str, Any],
    intent_path: Path,
    close_result: dict[str, Any],
    trade: TradeState,
    journal_path: Path,
    unresolved_root: Path = CLOSE_INTENT_DIR,
    resolved_root: Path = RESOLVED_CLOSE_INTENT_DIR,
    rejected_root: Path = REJECTED_CLOSE_INTENT_DIR,
) -> Path:
    """Archive a proven no-mutation rejection while retaining exposure."""

    if (
        close_result.get("status") != "rejected"
        or close_result.get("trade_id") != intent.get("trade_id")
        or getattr(trade, "trade_id", None) != intent.get("trade_id")
    ):
        raise RuntimeError("BROKER_CLOSE_REJECTION_PROOF_INVALID")
    validated = load_broker_close_intent(intent_path)
    if validated != intent:
        raise RuntimeError("BROKER_CLOSE_REJECTION_INTENT_MISMATCH")
    intent_sha256 = hashlib.sha256(
        intent_path.read_bytes()
    ).hexdigest()
    log_journal_event(
        journal_path,
        {
            "event": "BROKER_CLOSE_INTENT_REJECTED_NO_MUTATION",
            "idempotency_key": (
                f"broker-close-rejected:{intent_sha256}"
            ),
            "close_intent_id": intent["close_intent_id"],
            "close_intent_sha256": intent_sha256,
            "trade_id": intent["trade_id"],
            "broker_account_binding": intent[
                "broker_account_binding"
            ],
            "close_result": close_result,
        },
    )
    rejected_root = Path(rejected_root)
    if rejected_root.parent.resolve() != Path(
        unresolved_root
    ).parent.resolve():
        raise RuntimeError("BROKER_CLOSE_REJECTED_ROOT_MISMATCH")
    with broker_close_mutation_lock(
        intent_root=unresolved_root,
        resolved_root=resolved_root,
    ):
        if load_broker_close_intent(intent_path) != intent:
            raise RuntimeError(
                "BROKER_CLOSE_REJECTION_CHANGED_BEFORE_ARCHIVE"
            )
        rejected_root.mkdir(parents=True, exist_ok=True)
        if rejected_root.is_symlink() or not rejected_root.is_dir():
            raise RuntimeError(
                "BROKER_CLOSE_REJECTED_ROOT_INVALID"
            )
        target = rejected_root / (
            f"{intent_path.stem}.{intent_sha256[:16]}.json"
        )
        if target.exists() or target.is_symlink():
            raise RuntimeError(
                "BROKER_CLOSE_REJECTION_ARCHIVE_EXISTS"
            )
        os.replace(intent_path, target)
        _fsync_directory(Path(unresolved_root))
        _fsync_directory(rejected_root)
    return target


def require_no_unresolved_broker_entry_intents(
    *,
    intent_root: Path = ENTRY_INTENT_DIR,
) -> None:
    """Block startup until every unknown broker outcome is reconciled."""

    root = Path(intent_root)
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("BROKER_ENTRY_INTENT_ROOT_INVALID")
    unresolved = sorted(root.glob("*.json"))
    if unresolved:
        raise RuntimeError(
            "BROKER_ENTRY_INTENT_UNRESOLVED_RECONCILIATION_REQUIRED: "
            + ",".join(path.name for path in unresolved)
        )


def load_broker_entry_intent(path: Path) -> dict[str, Any]:
    """Strict-load one unresolved intent without filling any field."""

    expected_fields = {
        "schema_version",
        "created_utc",
        "client_order_id",
        "instrument",
        "side",
        "signed_units",
        "decision_snapshot",
        "sizing_application",
        "model_bundle_binding",
        "entry_source_pair_binding",
        "broker_account_binding",
        "launch_lease",
        "identity_sha256",
    }
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise RuntimeError("BROKER_ENTRY_INTENT_FILE_INVALID")
    raw = source.read_bytes()
    try:
        intent = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("BROKER_ENTRY_INTENT_JSON_INVALID") from exc
    if (
        not isinstance(intent, dict)
        or set(intent) != expected_fields
        or raw != _canonical_json_bytes(intent) + b"\n"
        or intent["schema_version"] != BROKER_ENTRY_INTENT_SCHEMA_VERSION
        or intent["instrument"] != INSTRUMENT
        or intent["side"] not in {"long", "short"}
        or not isinstance(intent["signed_units"], int)
        or isinstance(intent["signed_units"], bool)
        or intent["signed_units"] == 0
        or (intent["signed_units"] > 0)
        != (intent["side"] == "long")
        or source.name != f"{intent['client_order_id']}.json"
    ):
        raise RuntimeError("BROKER_ENTRY_INTENT_CONTRACT_INVALID")
    for field in (
        "decision_snapshot",
        "sizing_application",
        "model_bundle_binding",
        "entry_source_pair_binding",
        "launch_lease",
    ):
        if not isinstance(intent[field], dict) or not intent[field]:
            raise RuntimeError(
                f"BROKER_ENTRY_INTENT_{field.upper()}_INVALID"
            )
    validated_broker_account_binding = (
        _require_broker_account_binding_record(
            intent["broker_account_binding"]
        )
    )
    identity = {
        "instrument": INSTRUMENT,
        "side": intent["side"],
        "signed_units": intent["signed_units"],
        "decision_ts": intent["decision_snapshot"].get(
            "decision_ts"
        ),
        "model_direction": intent["decision_snapshot"].get(
            "model_direction"
        ),
        "model_bundle_sha256": intent[
            "model_bundle_binding"
        ].get("bundle_sha256"),
        "model_bundle_binding_sha256": hashlib.sha256(
            _canonical_json_bytes(intent["model_bundle_binding"])
        ).hexdigest(),
        "pair_generation_id": intent[
            "entry_source_pair_binding"
        ].get("pair_generation_id"),
        "pair_manifest_sha256": intent[
            "entry_source_pair_binding"
        ].get("pair_manifest_sha256"),
        "sizing_application_sha256": hashlib.sha256(
            _canonical_json_bytes(intent["sizing_application"])
        ).hexdigest(),
        "launch_state_sha256": intent["launch_lease"].get(
            "launch_state_sha256"
        ),
        "artifact_registry_sha256": intent["launch_lease"].get(
            "artifact_registry_sha256"
        ),
        "broker_account_binding": validated_broker_account_binding,
    }
    identity_sha256 = hashlib.sha256(
        _canonical_json_bytes(identity)
    ).hexdigest()
    if (
        intent["identity_sha256"] != identity_sha256
        or intent["client_order_id"]
        != f"gx1-{identity_sha256[:40]}"
    ):
        raise RuntimeError(
            "BROKER_ENTRY_INTENT_IDENTITY_HASH_MISMATCH"
        )
    return intent


def reconcile_unresolved_broker_entry_intents(
    client: OandaClient,
    *,
    open_trades: list[TradeState],
    dry_run: bool,
    broker_account_binding: dict[str, str],
    journal_path: Path,
    intent_root: Path = ENTRY_INTENT_DIR,
    resolved_root: Path = RESOLVED_ENTRY_INTENT_DIR,
) -> list[TradeState]:
    """Recover an accepted lost response by client ID, or fail closed."""

    root = Path(intent_root)
    if not root.exists():
        return open_trades
    paths = sorted(root.glob("*.json"))
    if not paths:
        return open_trades
    if dry_run:
        raise RuntimeError(
            "BROKER_ENTRY_INTENT_PRESENT_IN_DRY_RUN"
        )
    recovered = list(open_trades)
    expected_account_binding = _require_broker_account_binding_record(
        broker_account_binding
    )
    for path in paths:
        intent = load_broker_entry_intent(path)
        if intent["broker_account_binding"] != expected_account_binding:
            raise RuntimeError(
                "BROKER_ENTRY_INTENT_ACCOUNT_MISMATCH"
            )
        client_order_id = intent["client_order_id"]
        try:
            order_payload = client.get_order_by_client_id(
                client_order_id
            )
        except Exception as exc:
            raise RuntimeError(
                "BROKER_ENTRY_INTENT_ORDER_RECONCILIATION_FAILED: "
                f"{client_order_id}: {exc}"
            ) from exc
        order = (
            order_payload.get("order")
            if isinstance(order_payload, dict)
            else None
        )
        if (
            not isinstance(order, dict)
            or order.get("clientExtensions", {}).get("id")
            != client_order_id
        ):
            raise RuntimeError(
                "BROKER_ENTRY_INTENT_ORDER_IDENTITY_MISMATCH"
            )
        state = order.get("state")
        if state == "CANCELLED":
            open_payload = client.get_open_trades()
            broker_trades = (
                open_payload.get("trades")
                if isinstance(open_payload, dict)
                else None
            )
            if not isinstance(broker_trades, list) or any(
                isinstance(trade, dict)
                and trade.get("clientExtensions", {}).get("id")
                == client_order_id
                for trade in broker_trades
            ):
                raise RuntimeError(
                    "BROKER_ENTRY_INTENT_CANCELLED_OPEN_TRADE_MISMATCH"
                )
            log_journal_event(
                journal_path,
                {
                    "event": "BROKER_ENTRY_INTENT_RECONCILED_CANCELLED",
                    "client_order_id": client_order_id,
                    "intent_sha256": hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest(),
                    "order": order,
                },
            )
            resolve_broker_entry_intent(
                path,
                unresolved_root=root,
                resolved_root=resolved_root,
            )
            continue
        if state != "FILLED":
            raise RuntimeError(
                "BROKER_ENTRY_INTENT_ORDER_NOT_TERMINAL: "
                f"{client_order_id} state={state!r}"
            )
        transaction_id = str(order.get("fillingTransactionID") or "")
        try:
            transaction_payload = client.get_transaction(
                transaction_id
            )
        except Exception as exc:
            raise RuntimeError(
                "BROKER_ENTRY_INTENT_FILL_RECONCILIATION_FAILED: "
                f"{client_order_id}: {exc}"
            ) from exc
        transaction = (
            transaction_payload.get("transaction")
            if isinstance(transaction_payload, dict)
            else None
        )
        if not isinstance(transaction, dict):
            raise RuntimeError(
                "BROKER_ENTRY_INTENT_FILL_TRANSACTION_MISSING"
            )
        response = {"orderFillTransaction": transaction}
        replay_client = SimpleNamespace(
            create_market_order=(
                lambda *_args, **_kwargs: response
            )
        )
        units = abs(int(intent["signed_units"]))
        parsed = attempt_market_entry(
            replay_client,
            intent["side"],
            units,
            client_order_id=client_order_id,
        )
        if (
            parsed.get("status") != "filled"
            or parsed.get("fill_price_pair_exact") is not True
            or str(parsed.get("oanda_transaction_id") or "")
            != transaction_id
        ):
            raise RuntimeError(
                "BROKER_ENTRY_INTENT_FILL_CONTRACT_MISMATCH"
            )
        open_payload = client.get_open_trades()
        broker_trades = (
            open_payload.get("trades")
            if isinstance(open_payload, dict)
            else None
        )
        if not isinstance(broker_trades, list):
            raise RuntimeError(
                "BROKER_ENTRY_INTENT_OPEN_TRADES_INVALID"
            )
        matching = [
            trade
            for trade in broker_trades
            if isinstance(trade, dict)
            and trade.get("id") == parsed.get("trade_id")
            and trade.get("instrument") == INSTRUMENT
            and trade.get("clientExtensions", {}).get("id")
            == client_order_id
        ]
        if len(matching) != 1:
            raise RuntimeError(
                "BROKER_ENTRY_INTENT_OPEN_TRADE_IDENTITY_MISMATCH"
            )
        try:
            broker_units_float = float(matching[0]["currentUnits"])
            broker_units = int(broker_units_float)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "BROKER_ENTRY_INTENT_OPEN_TRADE_UNITS_INVALID"
            ) from exc
        if (
            not math.isfinite(broker_units_float)
            or broker_units_float != broker_units
            or broker_units != intent["signed_units"]
        ):
            raise RuntimeError(
                "BROKER_ENTRY_INTENT_OPEN_TRADE_UNITS_MISMATCH"
            )
        existing = [
            trade
            for trade in recovered
            if trade.trade_id == parsed["trade_id"]
        ]
        if existing:
            if (
                len(existing) != 1
                or existing[0].model_bundle_binding
                != intent["model_bundle_binding"]
                or existing[0].entry_source_pair_binding
                != intent["entry_source_pair_binding"]
                or existing[0].broker_account_binding
                != expected_account_binding
                or existing[0].v10_snapshot
                != intent["decision_snapshot"]
            ):
                raise RuntimeError(
                    "BROKER_ENTRY_INTENT_LOCAL_STATE_MISMATCH"
                )
        else:
            fill_ts = pd.Timestamp(parsed["fill_time"])
            fill_bid = _float_or_none(parsed["fill_bid"])
            fill_ask = _float_or_none(parsed["fill_ask"])
            if (
                fill_bid is None
                or fill_ask is None
                or fill_ask <= fill_bid
            ):
                raise RuntimeError(
                    "BROKER_ENTRY_INTENT_FILL_PRICE_PAIR_INVALID"
                )
            recovered_trade = TradeState.open(
                entry_ts=fill_ts,
                side=intent["side"],
                entry_bid=fill_bid,
                entry_ask=fill_ask,
                v10_snapshot=intent["decision_snapshot"],
                trade_id=str(parsed["trade_id"]),
                units=units,
                sizing_application=intent["sizing_application"],
                fill_transaction_id=transaction_id,
                execution_mode="learned_broker_fill",
                model_bundle_binding=intent[
                    "model_bundle_binding"
                ],
                entry_source_pair_binding=intent[
                    "entry_source_pair_binding"
                ],
                broker_account_binding=expected_account_binding,
            )
            recovered_trade.save(TRADE_STATE_DIR)
            recovered.append(recovered_trade)
        log_journal_event(
            journal_path,
            {
                "event": "BROKER_ENTRY_INTENT_RECONCILED_FILLED",
                "client_order_id": client_order_id,
                "trade_id": parsed["trade_id"],
                "transaction_id": transaction_id,
                "intent_sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
            },
        )
        resolve_broker_entry_intent(
            path,
            unresolved_root=root,
            resolved_root=resolved_root,
        )
    return recovered


def _fmt_optional_float(value: Any, spec: str) -> str:
    parsed = _float_or_none(value)
    return "NA" if parsed is None else format(parsed, spec)

DEFAULT_POLL_SECONDS = 10              # how often to check for new M1 close
DEFAULT_QUOTE_MAX_AGE_SEC = 90.0       # treat quote as stale (market closed/halted) if older
DEFAULT_MAX_TRADES = 1                 # max concurrent virtual trades held simultaneously
_EXECUTABLE_SNAPSHOT_ONLY_FIELDS = frozenset(
    {
        "runtime_evidence_schema_version",
        "model_policy",
        "atr_bps",
    }
)
MODEL_NATIVE_EXECUTABLE_DECISION_REQUIRED_FIELDS = frozenset(
    (
        MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS
        - _EXECUTABLE_SNAPSHOT_ONLY_FIELDS
    )
    | MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS
    | {
        "action",
        "action_id",
        "edge_score",
        "selection_score_mode",
        "selection_score",
        "session",
        "v10_path_quality_pred",
        "v10_mfe_pred_at_entry",
        "v10_tradable_prob",
        "v10_bad_path_prob",
        "_v10_snapshot",
        "policy",
        "stub",
        "entry_signal_latency_min",
        "entry_signal_latency_cap_sec",
        "entry_signal_stale",
        "context_refresh_in_flight",
        "context_mtf_incremental",
        "entry_source_pair_generation_id",
        "entry_source_pair_manifest_sha256",
    }
)


def _same_runtime_value(left: Any, right: Any) -> bool:
    """Exact envelope parity without allowing array-style truth ambiguity."""

    try:
        result = left == right
    except Exception:
        return False
    return result if isinstance(result, bool) else False


# Launch, broker-authority and live-tail admission were removed with the
# offline scope freeze (GX1_RULES.md "Frozen scope"). The paper runner keeps
# its journaling and executable-decision contracts, but every route that would
# claim launch authority now fails closed by construction instead of resolving
# a decision registry that no longer exists.
_LAUNCH_SCOPE_FROZEN = (
    "launch, broker and live-tail admission are outside the frozen offline "
    "scope; no launch authority can be granted from this checkout"
)


class _LaunchScopeFrozen(RuntimeError):
    """Raised whenever a retired launch-authority route is entered."""


@contextlib.contextmanager
def broker_entry_authority_mutation_lease():
    raise _LaunchScopeFrozen(_LAUNCH_SCOPE_FROZEN)
    yield  # pragma: no cover - unreachable, keeps the contextmanager shape


def require_runtime_entry_launch_lease(*_args: Any, **_kwargs: Any) -> dict[str, str]:
    raise _LaunchScopeFrozen(_LAUNCH_SCOPE_FROZEN)


def require_runtime_new_entry_live_tail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    raise _LaunchScopeFrozen(_LAUNCH_SCOPE_FROZEN)


def enforce_entry_next_edge_runner_guard(*_args: Any, **_kwargs: Any) -> dict[str, str]:
    raise _LaunchScopeFrozen(_LAUNCH_SCOPE_FROZEN)


def require_executable_model_native_entry_decision(
    decision: dict[str, Any],
    entry_time: Any,
) -> dict[str, Any]:
    """Validate the complete live decision envelope before any order is sent.

    SmartEntry validates its pure model snapshot first; the pipeline then adds
    the complete timing evidence.  This runner boundary validates both again,
    requires an exact outer schema, and proves that the action/policy/timing
    envelope is identical to the frozen snapshot consumed by TradeState and the
    journal.
    """

    if not isinstance(decision, dict) or not decision:
        raise RuntimeError("[RUNNER_MODEL_NATIVE_DECISION_INVALID] missing decision")
    retired = sorted(
        key
        for key in decision
        if any(fragment in str(key).lower() for fragment in RETIRED_RUNTIME_EVIDENCE_FRAGMENTS)
    )
    if retired:
        raise RuntimeError(
            "[RUNNER_MODEL_NATIVE_DECISION_INVALID] retired fields=" + repr(retired)
        )
    observed_fields = set(decision)
    if observed_fields != set(MODEL_NATIVE_EXECUTABLE_DECISION_REQUIRED_FIELDS):
        raise RuntimeError(
            "[RUNNER_MODEL_NATIVE_DECISION_INVALID] exact schema mismatch "
            f"missing={sorted(MODEL_NATIVE_EXECUTABLE_DECISION_REQUIRED_FIELDS - observed_fields)} "
            f"unexpected={sorted(observed_fields - MODEL_NATIVE_EXECUTABLE_DECISION_REQUIRED_FIELDS)}"
        )
    snapshot_raw = decision.get("_v10_snapshot")
    if not isinstance(snapshot_raw, dict):
        raise RuntimeError(
            "[RUNNER_MODEL_NATIVE_DECISION_INVALID] _v10_snapshot missing"
        )
    snapshot = require_model_native_runtime_evidence(
        snapshot_raw,
        context="V12_PAPER_RUNNER_PRE_ORDER",
    )
    if not MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS.issubset(snapshot):
        raise RuntimeError(
            "[RUNNER_MODEL_NATIVE_DECISION_INVALID] complete timing evidence missing"
        )
    require_model_native_entry_time(
        snapshot,
        entry_time,
        context="V12_PAPER_RUNNER_PRE_ORDER",
    )
    if (
        decision.get("policy") != MODEL_NATIVE_RUNTIME_POLICY
        or snapshot["model_policy"] != decision["policy"]
    ):
        raise RuntimeError(
            "[RUNNER_MODEL_NATIVE_DECISION_INVALID] model policy mismatch"
        )

    shared_fields = (
        MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS
        - _EXECUTABLE_SNAPSHOT_ONLY_FIELDS
    ) | MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS
    mismatched = sorted(
        field
        for field in shared_fields
        if not _same_runtime_value(decision[field], snapshot[field])
    )
    if mismatched:
        raise RuntimeError(
            "[RUNNER_MODEL_NATIVE_DECISION_INVALID] snapshot parity mismatch "
            f"fields={mismatched}"
        )

    direction_index = snapshot["model_direction_index"]
    expected_action = MODEL_DIRECTION_ACTION_BY_INDEX[direction_index]
    expected_action_id = MODEL_DIRECTION_ACTION_ID_BY_INDEX[direction_index]
    if decision["action"] != expected_action or decision["action_id"] != expected_action_id:
        raise RuntimeError(
            "[RUNNER_MODEL_NATIVE_DECISION_INVALID] action/direction parity mismatch"
        )
    if decision["selection_score_mode"] != MODEL_DIRECTION_SELECTION_MODE:
        raise RuntimeError(
            "[RUNNER_MODEL_NATIVE_DECISION_INVALID] selection mode mismatch"
        )
    for field in (
        "entry_source_pair_generation_id",
        "entry_source_pair_manifest_sha256",
    ):
        value = decision[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise RuntimeError(
                "[RUNNER_MODEL_NATIVE_DECISION_INVALID] "
                f"{field} is not an exact lowercase sha256 identity"
            )
    entry_q = snapshot["entry_action_q_bps"]
    expected_edge = max(entry_q[0], entry_q[1]) - entry_q[2]
    expected_selection = entry_q[direction_index]
    for field, expected in (
        ("edge_score", expected_edge),
        ("selection_score", expected_selection),
    ):
        observed = _float_or_none(decision[field])
        if observed is None or not math.isclose(
            observed,
            float(expected),
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise RuntimeError(
                f"[RUNNER_MODEL_NATIVE_DECISION_INVALID] {field} parity mismatch"
            )
    latency_sec = float(snapshot["entry_signal_latency_sec"])
    latency_min = _float_or_none(decision["entry_signal_latency_min"])
    latency_cap = _float_or_none(decision["entry_signal_latency_cap_sec"])
    if (
        latency_min is None
        or not math.isclose(latency_min, latency_sec / 60.0, rel_tol=1e-6, abs_tol=1e-7)
        or latency_cap != MODEL_NATIVE_MAX_ENTRY_SIGNAL_LATENCY_SEC
        or latency_sec > MODEL_NATIVE_MAX_ENTRY_SIGNAL_LATENCY_SEC
        or decision["entry_signal_stale"] is not False
    ):
        raise RuntimeError(
            "[RUNNER_MODEL_NATIVE_DECISION_INVALID] latency contract mismatch"
        )
    if decision["stub"] is not False:
        raise RuntimeError("[RUNNER_MODEL_NATIVE_DECISION_INVALID] stub decision forbidden")
    if decision["session"] not in SESSION_ORDER:
        raise RuntimeError("[RUNNER_MODEL_NATIVE_DECISION_INVALID] session missing")
    return snapshot


class StaleQuoteError(RuntimeError):
    """Raised when OANDA returns a quote older than max_age_sec.

    Happens when market is closed (e.g. weekend) — OANDA keeps serving the
    last close-of-week quote until Sunday Sydney open. Without this guard the
    paper-runner would log thousands of fake events with stale spreads.
    """
    def __init__(self, age_sec: float, quote_time: str):
        super().__init__(f"Quote is {age_sec:.0f}s old (quote_time={quote_time}) — market likely closed")
        self.age_sec = age_sec
        self.quote_time = quote_time


# ── Pre-trade spread + session checks ─────────────────────────────────────


def get_current_spread_bps(client: OandaClient,
                            *, max_age_sec: float = DEFAULT_QUOTE_MAX_AGE_SEC,
                            now_utc: datetime | None = None,
                            ) -> tuple[float, float, float]:
    """Returns (spread_bps, bid, ask). Raises StaleQuoteError if quote is older
    than max_age_sec (market closed). Raises ValueError on invalid bid."""
    pricing = client.get_pricing([INSTRUMENT])
    quote = pricing["prices"][0]
    quote_time_str = str(quote.get("time") or "").strip()
    if not quote_time_str:
        raise ValueError("quote_time_missing")
    quote_time = pd.to_datetime(quote_time_str, utc=True, errors="raise")
    now = pd.Timestamp(now_utc) if now_utc is not None else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    age_sec = (now - quote_time).total_seconds()
    if age_sec > max_age_sec:
        raise StaleQuoteError(age_sec, quote_time_str)
    bid = float(quote["bids"][0]["price"])
    ask = float(quote["asks"][0]["price"])
    if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0.0 or ask <= bid:
        raise ValueError(f"invalid_quote_prices bid={bid} ask={ask}")
    spread_bps = (ask - bid) / bid * 10000.0
    return spread_bps, bid, ask


# ── V12 decision (wired in sesjon 1-5) ────────────────────────────────────


def make_v12_decision(pipeline: V12Pipeline, now_minute: datetime,
                      bid: float, ask: float) -> dict[str, Any]:
    """Run the model-native SMART LONG/SHORT/FLAT entry stack.

    Returns the model argmax action plus diagnostics and timestamp. Pipeline
    caches state so each newly closed M5 row is decided exactly once.

    Portfolio/exposure state is execution admission, not a hidden parallel
    direction input. The model receives only its explicit hashed input contract.
    """
    return pipeline.make_entry_decision(pd.Timestamp(now_minute), bid, ask)


def learned_sizing_runtime_constraints(
    client: OandaClient,
    *,
    bid: float,
    ask: float,
    validated_authority: ValidatedLearnedSizingAuthority,
) -> dict[str, Any]:
    """Read fresh broker-truth account, instrument, and XAU exposure facts.

    These are facts, not knobs: the learned calibration rejects any value that
    differs from its immutable instrument contract.  Missing, fractional, or
    ambiguous broker fields raise and therefore authorize no order.
    """

    parsed_bid = _float_or_none(bid)
    parsed_ask = _float_or_none(ask)
    if (
        parsed_bid is None
        or parsed_ask is None
        or parsed_bid <= 0.0
        or parsed_ask < parsed_bid
    ):
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] invalid live XAU quote"
        )
    if not isinstance(validated_authority, ValidatedLearnedSizingAuthority):
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] startup-validated authority missing"
        )
    instrument_contract = validated_authority.calibration["instrument_constraints"]
    account_payload = client.get_account_summary()
    account_observed_utc = datetime.now(timezone.utc).isoformat()
    account = account_payload.get("account") if isinstance(account_payload, dict) else None
    if not isinstance(account, dict):
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] account summary missing account"
        )
    if account.get("hedgingEnabled") is not True:
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] account must prove hedgingEnabled=true"
        )
    instrument_payload = client.get_account_instruments([INSTRUMENT])
    instrument_observed_utc = datetime.now(timezone.utc).isoformat()
    rows = (
        instrument_payload.get("instruments")
        if isinstance(instrument_payload, dict)
        else None
    )
    matches = [
        row
        for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, dict) and row.get("name") == INSTRUMENT
    ]
    if len(matches) != 1:
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] exact XAU instrument row missing"
        )
    instrument = matches[0]
    exposure_payload = client.get_open_trades()
    exposure_observed_utc = datetime.now(timezone.utc).isoformat()
    broker_trades = (
        exposure_payload.get("trades")
        if isinstance(exposure_payload, dict)
        else None
    )
    if not isinstance(broker_trades, list):
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] open-trades exposure missing"
        )
    try:
        precision = int(instrument["tradeUnitsPrecision"])
        minimum_units_float = float(instrument["minimumTradeSize"])
        broker_maximum_units_float = float(instrument["maximumOrderUnits"])
        margin_rate = float(instrument["marginRate"])
        equity = float(account["NAV"])
        balance = float(account["balance"])
        margin_available = float(account["marginAvailable"])
        margin_used = float(account["marginUsed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] broker sizing field invalid"
        ) from exc
    if precision != 0:
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] XAU units must be integer precision"
        )
    minimum_units = int(minimum_units_float)
    broker_maximum_units = int(broker_maximum_units_float)
    if (
        minimum_units_float != minimum_units
        or broker_maximum_units_float != broker_maximum_units
        or not all(
            math.isfinite(value)
            for value in (
                margin_rate,
                equity,
                balance,
                margin_available,
                margin_used,
                minimum_units_float,
                broker_maximum_units_float,
            )
        )
    ):
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] non-exact broker unit constraints"
        )
    current_xau_abs_units = 0
    for row in broker_trades:
        if not isinstance(row, dict) or row.get("instrument") != INSTRUMENT:
            continue
        try:
            units_float = float(row["currentUnits"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelNativeSizingUnavailable(
                "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] XAU exposure units invalid"
            ) from exc
        units = int(units_float)
        if not math.isfinite(units_float) or units_float != units:
            raise ModelNativeSizingUnavailable(
                "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] XAU exposure units not exact integer"
            )
        current_xau_abs_units += abs(units)
    maximum_gross_units = int(instrument_contract["maximum_gross_xau_units"])
    if broker_maximum_units < maximum_gross_units:
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] broker maximum is below immutable gross cap"
        )
    if (
        account.get("currency") != instrument_contract["account_currency"]
        or margin_rate != instrument_contract["margin_rate"]
        or minimum_units != instrument_contract["minimum_order_units"]
    ):
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] broker facts differ from adopted instrument contract"
        )
    transaction_ids: dict[str, str] = {}
    for name, payload in (
        ("account", account_payload),
        ("instrument", instrument_payload),
        ("exposure", exposure_payload),
    ):
        raw = payload.get("lastTransactionID") if isinstance(payload, dict) else None
        if not isinstance(raw, str) or not raw.strip():
            raise ModelNativeSizingUnavailable(
                f"[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] {name} lastTransactionID missing"
            )
        transaction_ids[name] = raw.strip()
    if len(set(transaction_ids.values())) != 1:
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] broker facts span different lastTransactionID snapshots"
        )
    decision_utc = datetime.now(timezone.utc).isoformat()
    account_floating_drawdown_bps = max(
        0.0, (balance - equity) / balance * 10_000.0
    ) if balance > 0.0 else float("nan")
    return {
        "instrument": INSTRUMENT,
        "account_currency": account.get("currency"),
        "account_equity": equity,
        "account_balance": balance,
        "account_floating_drawdown_bps": account_floating_drawdown_bps,
        "margin_available": margin_available,
        "margin_used": margin_used,
        "mark_price": (parsed_bid + parsed_ask) / 2.0,
        "margin_rate": margin_rate,
        "unit_step": 1,
        "minimum_order_units": minimum_units,
        "maximum_gross_xau_units": maximum_gross_units,
        "current_xau_abs_units": current_xau_abs_units,
        "sizing_decision_utc": decision_utc,
        "account_observed_utc": account_observed_utc,
        "instrument_observed_utc": instrument_observed_utc,
        "exposure_observed_utc": exposure_observed_utc,
        "account_last_transaction_id": transaction_ids["account"],
        "instrument_last_transaction_id": transaction_ids["instrument"],
        "exposure_last_transaction_id": transaction_ids["exposure"],
        "fact_provenance_mode": "broker_live",
    }


def require_broker_xau_trade_reconciliation(
    client: OandaClient,
    *,
    local_open_trades: list[TradeState],
    max_trades: int,
    expected_exposure_transaction_id: str,
) -> tuple[str, ...]:
    """Require exact broker/local XAU trade identity immediately before order."""

    if isinstance(max_trades, bool) or not isinstance(max_trades, int) or max_trades != 1:
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] only max_trades=1 is proven"
        )
    payload = client.get_open_trades()
    transaction_id = (
        payload.get("lastTransactionID") if isinstance(payload, dict) else None
    )
    rows = payload.get("trades") if isinstance(payload, dict) else None
    if (
        not isinstance(transaction_id, str)
        or not transaction_id.strip()
        or transaction_id.strip() != str(expected_exposure_transaction_id).strip()
        or not isinstance(rows, list)
    ):
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] broker exposure changed "
            "between sizing and order admission"
        )
    broker_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("instrument") != INSTRUMENT:
            continue
        trade_id = str(row.get("id") or "").strip()
        try:
            units_float = float(row["currentUnits"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelNativeSizingUnavailable(
                "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] broker XAU trade is malformed"
            ) from exc
        if (
            not trade_id
            or not math.isfinite(units_float)
            or units_float == 0.0
            or units_float != int(units_float)
        ):
            raise ModelNativeSizingUnavailable(
                "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] broker XAU trade identity "
                "or units are invalid"
            )
        broker_ids.append(trade_id)
    if len(broker_ids) != len(set(broker_ids)):
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] duplicate broker XAU trade ID"
        )
    local_ids: list[str] = []
    for trade in local_open_trades:
        trade_id = str(getattr(trade, "trade_id", "") or "").strip()
        if not trade_id:
            raise ModelNativeSizingUnavailable(
                "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] local open trade lacks "
                "broker trade identity"
            )
        local_ids.append(trade_id)
    if len(local_ids) != len(set(local_ids)) or set(local_ids) != set(broker_ids):
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] broker/local XAU trade "
            "identity mismatch"
        )
    if len(broker_ids) >= max_trades:
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] broker XAU trade count is "
            "at the admitted cap"
        )
    return tuple(sorted(broker_ids))


def make_v12_exit_decision(pipeline: V12Pipeline, trade: TradeState,
                            now_minute: datetime, bid: float, ask: float,
                            *,
                            on_bar_committed: Callable[[TradeState], None],
                            ) -> dict[str, Any]:
    """Query the same contract-admitted model bundle that owned Entry."""
    return pipeline.make_exit_decision(
        trade,
        pd.Timestamp(now_minute),
        bid,
        ask,
        on_bar_committed=on_bar_committed,
    )


def journal_v12_exit_decision(journal: Any, trade: TradeState) -> None:
    """Durably reconcile the decision bound to the latest persisted M1 bar."""

    decision = trade.last_exit_decision
    if not isinstance(decision, dict):
        raise RuntimeError(
            "latest complete Exit bar has no bound model decision"
        )
    if not trade.trade_id:
        raise RuntimeError(
            "unified Exit journal requires broker trade identity"
        )
    journal.log_v12_bar_decision(
        trade_id=trade.trade_id,
        timestamp=trade.last_processed_m1_ts.isoformat(),
        bars_in_trade=trade.bars_in_trade,
        bid=trade.current_bid,
        ask=trade.current_ask,
        current_pnl_bps=trade.current_pnl_bps,
        cum_mfe_bps=trade.cum_mfe_bps,
        cum_mae_bps=trade.cum_mae_bps,
        bars_since_mfe_peak=trade.bars_since_mfe_peak,
        executable_range_bps=trade.last_executable_range_bps,
        exit_action=decision["action"],
        exit_action_index=decision["exit_action_index"],
        exit_action_q_bps=decision["exit_action_q_bps"],
        exit_action_valid_mask=decision["exit_action_valid_mask"],
        exit_decision_source=decision["decision_source"],
        bundle_sha256=decision["bundle_sha256"],
        entry_snapshot_sha256=decision["entry_snapshot_sha256"],
        exit_path_envelope_sha256=decision[
            "exit_path_envelope_sha256"
        ],
        exit_input_envelope_sha256=decision[
            "exit_input_envelope_sha256"
        ],
        output_evidence_sha256=decision["output_evidence_sha256"],
        exit_path_envelope=trade.build_closed_m1_path_evidence(),
        exit_input_envelope=decision["exit_input_envelope"],
        exit_incremental_carry_envelope=decision[
            "exit_incremental_carry_envelope"
        ],
    )


def persist_and_journal_v12_exit_bar(
    journal: Any,
    trade: TradeState,
    *,
    state_directory: Path = TRADE_STATE_DIR,
) -> None:
    """Durably persist and journal one atomically completed M1/Exit pair."""

    directory = Path(state_directory)
    directory.mkdir(parents=True, exist_ok=True)
    trade.save(directory)
    journal_v12_exit_decision(journal, trade)


# ── Order execution + reject handling ────────────────────────────────────


def _exact_order_fill_price_pair(
    fill: dict[str, Any],
    *,
    side: str,
) -> tuple[float, float, float, pd.Timestamp] | None:
    """Return exact executable entry pair from OANDA OrderFillTransaction.

    ``fullPrice`` is the account-specific ClientPrice in effect at fill time.
    The opened trade price/fullVWAP owns the executed side; the best literal
    liquidity bucket owns the opposite side. Polling quotes and closeout prices
    are never substituted.
    """

    trade_opened = fill.get("tradeOpened")
    full_price = fill.get("fullPrice")
    if not isinstance(trade_opened, dict) or not isinstance(full_price, dict):
        return None
    opened_price = _float_or_none(trade_opened.get("price"))
    full_vwap = _float_or_none(fill.get("fullVWAP"))
    fill_time_raw = fill.get("time")
    price_time_raw = full_price.get("time")
    if opened_price is None or full_vwap is None:
        return None
    try:
        fill_time = pd.Timestamp(fill_time_raw)
        price_time = pd.Timestamp(price_time_raw)
    except (TypeError, ValueError):
        return None
    if (
        pd.isna(fill_time)
        or pd.isna(price_time)
        or fill_time.tzinfo is None
        or price_time.tzinfo is None
        or fill_time.utcoffset() is None
        or price_time.utcoffset() is None
        or fill_time.utcoffset().total_seconds() != 0.0
        or price_time.utcoffset().total_seconds() != 0.0
    ):
        return None
    fill_time = fill_time.tz_convert("UTC")
    price_time = price_time.tz_convert("UTC")
    if fill_time != price_time or not math.isclose(
        opened_price,
        full_vwap,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        return None

    bids_raw = full_price.get("bids")
    asks_raw = full_price.get("asks")
    if not isinstance(bids_raw, list) or not bids_raw:
        return None
    if not isinstance(asks_raw, list) or not asks_raw:
        return None
    bids = [
        _float_or_none(bucket.get("price"))
        for bucket in bids_raw
        if isinstance(bucket, dict)
    ]
    asks = [
        _float_or_none(bucket.get("price"))
        for bucket in asks_raw
        if isinstance(bucket, dict)
    ]
    if (
        len(bids) != len(bids_raw)
        or len(asks) != len(asks_raw)
        or any(value is None or value <= 0.0 for value in (*bids, *asks))
    ):
        return None
    best_bid = max(float(value) for value in bids if value is not None)
    best_ask = min(float(value) for value in asks if value is not None)
    if best_ask <= best_bid:
        return None
    if side == "long":
        entry_bid, entry_ask = best_bid, opened_price
    elif side == "short":
        entry_bid, entry_ask = opened_price, best_ask
    else:
        return None
    if entry_ask <= entry_bid:
        return None
    return entry_bid, entry_ask, opened_price, fill_time


def attempt_market_entry(
    client: OandaClient,
    side: str,
    units: int,
    *,
    client_order_id: str,
) -> dict[str, Any]:
    """Submit market order. Returns dict with status + reason if rejected.

    OANDA returns MARKET_ORDER_REJECT_TRANSACTION on rejection. Common reasons:
      MARKET_HALTED, INSUFFICIENT_LIQUIDITY, INSTRUMENT_HALTED,
      ACCOUNT_NOT_TRADEABLE, MARGIN_RATE_INVALID, PRICE_PRECISION_EXCEEDED.
    """
    if side not in ("long", "short"):
        return {"status": "skipped", "reason": f"invalid_side {side}"}
    if isinstance(units, bool) or not isinstance(units, int) or units <= 0:
        return {
            "status": "skipped",
            "reason": "units must be an exact positive learned-sizing integer",
        }
    if (
        not isinstance(client_order_id, str)
        or not client_order_id.startswith("gx1-")
        or len(client_order_id) > 64
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in client_order_id
        )
    ):
        return {
            "status": "skipped",
            "reason": "client_order_id contract invalid",
        }
    signed_units = units if side == "long" else -units

    try:
        response = client.create_market_order(
            INSTRUMENT, units=signed_units, client_order_id=client_order_id,
        )
    except Exception as exc:
        LOG.error("OANDA order outcome unknown: %s", exc)
        return {
            "status": "unknown_outcome",
            "reason": str(exc),
            "client_order_id": client_order_id,
        }

    # Parse response: OANDA returns {orderCreateTransaction, orderFillTransaction} on success.
    # On failure it returns one of: orderRejectTransaction (instrument/price errors) or
    # orderCancelTransaction (e.g. INSUFFICIENT_MARGIN cancels post-creation).
    if (
        isinstance(response, dict)
        and (
            "orderRejectTransaction" in response
            or "orderCancelTransaction" in response
        )
    ):
        reject = (response.get("orderRejectTransaction")
                  or response.get("orderCancelTransaction") or {})
        reason = (reject.get("rejectReason")
                  or reject.get("reason") or "UNKNOWN")
        LOG.warning(f"REJECTED side={side} reason={reason}  cid={client_order_id}")
        return {"status": "rejected", "reason": reason, "client_order_id": client_order_id,
                 "raw": response}
    if (
        not isinstance(response, dict)
        or "orderFillTransaction" not in response
    ):
        return {
            "status": "unknown_outcome",
            "reason": "response lacks explicit reject/cancel or fill",
            "client_order_id": client_order_id,
            "raw": response,
        }

    fill = response["orderFillTransaction"]
    # Only a literal newly opened trade owns Entry state.  A reduced/closed
    # trade or the fill transaction is not an admissible identity substitute.
    # Missing tradeOpened therefore remains unresolved and must fail closed at
    # the caller's reconciliation boundary.
    trade_opened = fill.get("tradeOpened")
    trade_id = (
        trade_opened.get("tradeID")
        if isinstance(trade_opened, dict)
        else None
    )
    raw_fill_units = fill.get("units")
    try:
        fill_units_float = float(raw_fill_units)
        signed_fill_units = int(fill_units_float)
    except (TypeError, ValueError, OverflowError):
        signed_fill_units = None
    fill_units_exact = bool(
        signed_fill_units is not None
        and math.isfinite(fill_units_float)
        and fill_units_float == signed_fill_units
        and signed_fill_units == signed_units
    )
    raw_opened_units = (
        trade_opened.get("units") if isinstance(trade_opened, dict) else None
    )
    try:
        opened_units_float = float(raw_opened_units)
        signed_opened_units = int(opened_units_float)
    except (TypeError, ValueError, OverflowError):
        signed_opened_units = None
    pure_trade_open = bool(
        isinstance(trade_opened, dict)
        and isinstance(trade_opened.get("tradeID"), str)
        and trade_opened["tradeID"].strip()
        and signed_opened_units is not None
        and math.isfinite(opened_units_float)
        and opened_units_float == signed_opened_units
        and signed_opened_units == signed_units
        and not fill.get("tradesClosed")
        and not fill.get("tradeReduced")
    )
    if not trade_id:
        # The 2026-05-20 phantom-LONG incident demonstrated that netted legs
        # cannot identify a newly opened trade.  Do not infer an identity from
        # any other response field.
        LOG.warning(
            "FILL identity unresolved: literal tradeOpened.tradeID is missing; "
            "no trade identity inferred from reduced/closed legs or transaction id. "
            f"Raw orderFillTransaction keys: {sorted(fill.keys())}"
        )
    status = (
        "filled"
        if fill_units_exact and pure_trade_open
        else "filled_structure_mismatch"
        if fill_units_exact
        else "filled_units_mismatch"
    )
    if status != "filled":
        LOG.error(
            "FILL CONTRACT MISMATCH status=%s side=%s requested=%s fill=%r "
            "tradeOpened=%r closed=%r reduced=%r trade_id=%s",
            status,
            side,
            signed_units,
            raw_fill_units,
            raw_opened_units,
            fill.get("tradesClosed"),
            fill.get("tradeReduced"),
            trade_id,
        )
    else:
        LOG.info(
            "FILLED side=%s units=%s price=%s trade_id=%s",
            side,
            signed_fill_units,
            fill.get("price"),
            trade_id,
        )
    exact_fill_pair = _exact_order_fill_price_pair(fill, side=side)
    return {"status": status,
             "fill_price": (
                 exact_fill_pair[2] if exact_fill_pair is not None else None
             ),
             "fill_bid": (
                 exact_fill_pair[0] if exact_fill_pair is not None else None
             ),
             "fill_ask": (
                 exact_fill_pair[1] if exact_fill_pair is not None else None
             ),
             "fill_price_pair_exact": exact_fill_pair is not None,
             "requested_signed_units": signed_units,
             "filled_signed_units": signed_fill_units,
             "trade_opened_signed_units": signed_opened_units,
             "fill_units_exact": fill_units_exact,
             "pure_trade_open": pure_trade_open,
             "fill_time": (
                 exact_fill_pair[3].isoformat()
                 if exact_fill_pair is not None
                 else fill.get("time")
             ),
             "oanda_transaction_id": fill.get("id"),
             "oanda_order_id": fill.get("orderID"),
             "trade_id": trade_id,
             "client_order_id": client_order_id,
             "raw": response}


def attempt_close_trade(client: OandaClient, trade: TradeState) -> dict[str, Any]:
    """Close a specific virtual trade by its OANDA tradeID.

    Works in both NETTING and HEDGING account modes: OANDA's
    PUT /accounts/{id}/trades/{tradeID}/close endpoint closes only the units
    associated with that tradeID, not the full netted position.

    Missing ``trade_id`` is an unresolved broker-identity failure. Sending a
    counter-direction market order is forbidden because a hedging account may
    open a second trade instead of reducing the original exposure.
    """
    if not trade.trade_id:
        LOG.error("close_trade: trade has no OANDA tradeID; refusing counter-order")
        return {
            "status": "missing_trade_id",
            "trade_id": None,
            "reason": "broker trade identity is required for fail-closed close",
        }
    try:
        response = client.close_trade(trade.trade_id)
        if (
            isinstance(response, dict)
            and isinstance(
                response.get("orderRejectTransaction"),
                dict,
            )
        ):
            if any(
                key in response
                for key in (
                    "orderFillTransaction",
                    "orderCreateTransaction",
                    "orderCancelTransaction",
                )
            ):
                return {
                    "status": "ambiguous_response",
                    "trade_id": trade.trade_id,
                    "reason": (
                        "reject response also contains mutation evidence"
                    ),
                    "raw": response,
                }
            rejection = response["orderRejectTransaction"]
            return {
                "status": "rejected",
                "trade_id": trade.trade_id,
                "reason": str(
                    rejection.get("rejectReason")
                    or rejection.get("reason")
                    or "UNKNOWN"
                ),
                "raw": response,
            }
        fill = (
            response.get("orderFillTransaction")
            if isinstance(response, dict)
            else None
        )
        failures: list[str] = []
        if not isinstance(fill, dict):
            failures.append("order_fill_transaction_missing")
            fill = {}
        transaction_id = fill.get("id")
        order_id = fill.get("orderID")
        if not isinstance(transaction_id, str) or not transaction_id.strip():
            failures.append("fill_transaction_id_invalid")
        if not isinstance(order_id, str) or not order_id.strip():
            failures.append("fill_order_id_invalid")
        if fill.get("instrument") != INSTRUMENT:
            failures.append("fill_instrument_mismatch")
        try:
            fill_time = pd.Timestamp(fill.get("time"))
        except (TypeError, ValueError):
            fill_time = pd.NaT
        if (
            pd.isna(fill_time)
            or fill_time.tzinfo is None
            or fill_time.utcoffset() is None
            or fill_time.utcoffset().total_seconds() != 0.0
        ):
            failures.append("fill_time_invalid")
        else:
            fill_time = fill_time.tz_convert("UTC")
        fill_price = _float_or_none(fill.get("price"))
        if fill_price is None or fill_price <= 0.0:
            failures.append("fill_price_invalid")
        realized_pl = _float_or_none(fill.get("pl"))
        if realized_pl is None:
            failures.append("fill_realized_pl_invalid")

        closed_rows = fill.get("tradesClosed")
        if not isinstance(closed_rows, list) or len(closed_rows) != 1:
            failures.append("trades_closed_exactly_one_required")
            closed = {}
        else:
            closed = (
                closed_rows[0]
                if isinstance(closed_rows[0], dict)
                else {}
            )
            if not closed:
                failures.append("trades_closed_row_invalid")
        if closed.get("tradeID") != trade.trade_id:
            failures.append("closed_trade_id_mismatch")

        def _exact_signed_integer(raw: object) -> int | None:
            try:
                numeric = float(raw)
                integer = int(numeric)
            except (TypeError, ValueError, OverflowError):
                return None
            if (
                not math.isfinite(numeric)
                or numeric != integer
                or integer == 0
            ):
                return None
            return integer

        fill_units = _exact_signed_integer(fill.get("units"))
        closed_units = _exact_signed_integer(closed.get("units"))
        expected_units = getattr(trade, "units", None)
        if (
            isinstance(expected_units, bool)
            or not isinstance(expected_units, int)
            or expected_units <= 0
        ):
            failures.append("trade_state_units_invalid")
        elif (
            fill_units is None
            or closed_units is None
            or fill_units != closed_units
            or fill_units
            != (
                -expected_units
                if getattr(trade, "side", None) == "long"
                else expected_units
                if getattr(trade, "side", None) == "short"
                else 0
            )
        ):
            failures.append("closed_units_not_exact_all")
        if fill.get("tradeOpened"):
            failures.append("close_fill_opened_trade")
        if fill.get("tradeReduced"):
            failures.append("close_fill_reduced_trade")
        if failures:
            LOG.error(
                "CLOSE FILL CONTRACT MISMATCH trade_id=%s failures=%s",
                trade.trade_id,
                failures,
            )
            return {
                "status": "close_fill_mismatch",
                "trade_id": trade.trade_id,
                "reason": ",".join(failures),
                "raw": response,
            }
        LOG.info(
            "CLOSED trade_id=%s units=%s price=%s pl=%s",
            trade.trade_id,
            fill_units,
            fill_price,
            realized_pl,
        )
        return {
            "status": "closed",
            "trade_id": trade.trade_id,
            "fill_price": fill_price,
            "realized_pl": realized_pl,
            "fill_time": fill_time.isoformat(),
            "oanda_transaction_id": transaction_id,
            "oanda_order_id": order_id,
            "closed_signed_units": closed_units,
            "raw": response,
        }
    except Exception as exc:
        LOG.error(f"OANDA close_trade({trade.trade_id}) failed: {exc}")
        return {"status": "api_error", "trade_id": trade.trade_id, "reason": str(exc)}


def submit_broker_close_with_durable_intent(
    client: OandaClient,
    trade: TradeState,
    *,
    broker_account_binding: dict[str, str],
    intent_root: Path = CLOSE_INTENT_DIR,
    resolved_root: Path = RESOLVED_CLOSE_INTENT_DIR,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Persist exact EXIT_NOW authority, then issue one close mutation."""

    intent = build_broker_close_intent(
        trade,
        broker_account_binding=broker_account_binding,
        created_utc=datetime.now(timezone.utc),
    )
    with broker_close_mutation_lock(
        intent_root=intent_root,
        resolved_root=resolved_root,
    ):
        intent_path = _persist_broker_close_intent_locked(
            intent,
            intent_root=intent_root,
            resolved_root=resolved_root,
        )
        result = attempt_close_trade(client, trade)
    return intent, intent_path, result


def _broker_close_terminal_journal_event(
    *,
    intent: dict[str, Any],
    intent_path: Path,
    close_result: dict[str, Any],
) -> dict[str, Any]:
    """Return the same canonical close proof for direct and restart paths."""

    return {
        "event": "BROKER_CLOSE_INTENT_TERMINAL_CLOSED",
        "idempotency_key": (
            f"broker-close-terminal:{intent['close_intent_id']}"
        ),
        "close_intent_id": intent["close_intent_id"],
        "close_intent_sha256": hashlib.sha256(
            intent_path.read_bytes()
        ).hexdigest(),
        "trade_id": intent["trade_id"],
        "broker_account_binding": intent[
            "broker_account_binding"
        ],
        "close_proof": {
            field: close_result[field]
            for field in (
                "status",
                "trade_id",
                "fill_price",
                "realized_pl",
                "fill_time",
                "oanda_transaction_id",
                "oanda_order_id",
                "closed_signed_units",
            )
        },
    }


def finalize_broker_close_intent(
    *,
    intent: dict[str, Any],
    intent_path: Path,
    close_result: dict[str, Any],
    trade: TradeState,
    journal_path: Path,
    unresolved_root: Path = CLOSE_INTENT_DIR,
    resolved_root: Path = RESOLVED_CLOSE_INTENT_DIR,
) -> None:
    """Durably record exact closure before resolving intent and state."""

    validated_intent = load_broker_close_intent(intent_path)
    if validated_intent != intent:
        raise RuntimeError("BROKER_CLOSE_INTENT_MEMORY_DISK_MISMATCH")
    if (
        close_result.get("status") != "closed"
        or close_result.get("trade_id") != intent["trade_id"]
        or close_result.get("closed_signed_units")
        != intent["expected_close_signed_units"]
    ):
        raise RuntimeError("BROKER_CLOSE_FINALIZATION_PROOF_INVALID")
    log_journal_event(
        journal_path,
        _broker_close_terminal_journal_event(
            intent=intent,
            intent_path=intent_path,
            close_result=close_result,
        ),
    )
    with broker_close_mutation_lock(
        intent_root=unresolved_root,
        resolved_root=resolved_root,
    ):
        if load_broker_close_intent(intent_path) != intent:
            raise RuntimeError(
                "BROKER_CLOSE_INTENT_CHANGED_BEFORE_FINALIZE"
            )
        trade.delete_state_file(TRADE_STATE_DIR)
        _resolve_broker_close_intent_locked(
            intent_path,
            unresolved_root=unresolved_root,
            resolved_root=resolved_root,
        )


def reconcile_unresolved_broker_close_intents(
    client: OandaClient,
    *,
    open_trades: list[TradeState],
    dry_run: bool,
    broker_account_binding: dict[str, str],
    journal_path: Path,
    intent_root: Path = CLOSE_INTENT_DIR,
    resolved_root: Path = RESOLVED_CLOSE_INTENT_DIR,
) -> list[TradeState]:
    """Recover a lost close response without ever retrying the mutation."""

    root = Path(intent_root)
    if not root.exists():
        return open_trades
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("BROKER_CLOSE_INTENT_ROOT_INVALID")
    paths = sorted(root.glob("*.json"))
    if not paths:
        return open_trades
    if dry_run:
        raise RuntimeError("BROKER_CLOSE_INTENT_PRESENT_IN_DRY_RUN")
    expected_account_binding = _require_broker_account_binding_record(
        broker_account_binding
    )
    recovered = list(open_trades)
    by_trade_id = {
        trade.trade_id: trade
        for trade in recovered
        if isinstance(trade.trade_id, str) and trade.trade_id
    }
    if len(by_trade_id) != len(recovered):
        raise RuntimeError("BROKER_CLOSE_LOCAL_TRADE_IDENTITY_INVALID")
    for path in paths:
        intent = load_broker_close_intent(path)
        if intent["broker_account_binding"] != expected_account_binding:
            raise RuntimeError("BROKER_CLOSE_INTENT_ACCOUNT_MISMATCH")
        persisted_trade = by_trade_id.get(intent["trade_id"])
        if persisted_trade is not None:
            if (
                persisted_trade.broker_account_binding
                != expected_account_binding
                or persisted_trade.side != intent["side"]
                or (
                    -persisted_trade.units
                    if persisted_trade.side == "long"
                    else persisted_trade.units
                )
                != intent["expected_close_signed_units"]
                or hashlib.sha256(
                    _canonical_json_bytes(
                        persisted_trade.to_dict()
                    )
                ).hexdigest()
                != intent["trade_state_sha256"]
            ):
                raise RuntimeError(
                    "BROKER_CLOSE_INTENT_LOCAL_STATE_MISMATCH"
                )
            trade = persisted_trade
        else:
            trade = SimpleNamespace(
                trade_id=intent["trade_id"],
                side=intent["side"],
                units=abs(intent["expected_close_signed_units"]),
            )
        try:
            trade_payload = client.get_trade(intent["trade_id"])
        except Exception as exc:
            raise RuntimeError(
                "BROKER_CLOSE_INTENT_TRADE_RECONCILIATION_FAILED"
            ) from exc
        broker_trade = (
            trade_payload.get("trade")
            if isinstance(trade_payload, dict)
            else None
        )
        if (
            not isinstance(broker_trade, dict)
            or broker_trade.get("id") != intent["trade_id"]
            or broker_trade.get("instrument") != INSTRUMENT
        ):
            raise RuntimeError(
                "BROKER_CLOSE_INTENT_BROKER_TRADE_IDENTITY_MISMATCH"
            )
        if broker_trade.get("state") != "CLOSED":
            raise RuntimeError(
                "BROKER_CLOSE_INTENT_OUTCOME_UNRESOLVED_NO_RETRY: "
                f"state={broker_trade.get('state')!r}"
            )
        closing_ids = broker_trade.get("closingTransactionIDs")
        if (
            not isinstance(closing_ids, list)
            or len(closing_ids) != 1
            or not isinstance(closing_ids[0], str)
            or not closing_ids[0]
        ):
            raise RuntimeError(
                "BROKER_CLOSE_INTENT_CLOSING_TRANSACTION_ID_INVALID"
            )
        transaction_id = closing_ids[0]
        try:
            transaction_payload = client.get_transaction(transaction_id)
        except Exception as exc:
            raise RuntimeError(
                "BROKER_CLOSE_INTENT_TRANSACTION_RECONCILIATION_FAILED"
            ) from exc
        transaction = (
            transaction_payload.get("transaction")
            if isinstance(transaction_payload, dict)
            else None
        )
        if not isinstance(transaction, dict):
            raise RuntimeError(
                "BROKER_CLOSE_INTENT_TRANSACTION_MISSING"
            )
        replay_client = SimpleNamespace(
            close_trade=lambda *_args, **_kwargs: {
                "orderFillTransaction": transaction
            }
        )
        close_result = attempt_close_trade(replay_client, trade)
        if (
            close_result.get("status") != "closed"
            or close_result.get("oanda_transaction_id")
            != transaction_id
            or close_result.get("closed_signed_units")
            != intent["expected_close_signed_units"]
        ):
            raise RuntimeError(
                "BROKER_CLOSE_INTENT_TRANSACTION_CONTRACT_MISMATCH"
            )
        log_journal_event(
            journal_path,
            _broker_close_terminal_journal_event(
                intent=intent,
                intent_path=path,
                close_result=close_result,
            ),
        )
        with broker_close_mutation_lock(
            intent_root=root,
            resolved_root=resolved_root,
        ):
            if load_broker_close_intent(path) != intent:
                raise RuntimeError(
                    "BROKER_CLOSE_INTENT_CHANGED_BEFORE_RECONCILE"
                )
            if persisted_trade is not None:
                persisted_trade.delete_state_file(TRADE_STATE_DIR)
                recovered.remove(persisted_trade)
                del by_trade_id[intent["trade_id"]]
            _resolve_broker_close_intent_locked(
                path,
                unresolved_root=root,
                resolved_root=resolved_root,
            )
    return recovered


def require_runner_trade_state_mode(
    open_trades: list[TradeState],
    *,
    dry_run: bool,
    shadow_only: bool,
) -> None:
    """Reject persisted exposure owned by a different execution mode."""

    if shadow_only:
        if open_trades:
            raise RuntimeError(
                "SHADOW_RUNNER_PERSISTED_TRADE_STATE_PRESENT"
            )
        return
    expected_mode = (
        "learned_virtual_dry_run"
        if dry_run
        else "learned_broker_fill"
    )
    for trade in open_trades:
        evidence = getattr(
            trade,
            "sizing_execution_evidence",
            None,
        )
        observed_mode = (
            evidence.get("mode")
            if isinstance(evidence, dict)
            else None
        )
        if observed_mode != expected_mode:
            raise RuntimeError(
                "TRADE_STATE_EXECUTION_MODE_MISMATCH: "
                f"trade_id={getattr(trade, 'trade_id', None)!r} "
                f"observed={observed_mode!r} expected={expected_mode!r}"
            )


def require_runner_broker_account_binding(
    open_trades: list[TradeState],
    *,
    broker_account_binding: dict[str, str],
    dry_run: bool,
    shadow_only: bool,
) -> None:
    """Reject persisted exposure owned by another OANDA authority."""

    expected = _require_broker_account_binding_record(
        broker_account_binding
    )
    for trade in open_trades:
        observed = getattr(trade, "broker_account_binding", None)
        if dry_run or shadow_only:
            if observed is not None:
                raise RuntimeError(
                    "NON_BROKER_TRADE_STATE_HAS_ACCOUNT_BINDING"
                )
        elif observed != expected:
            raise RuntimeError(
                "TRADE_STATE_BROKER_ACCOUNT_MISMATCH: "
                f"trade_id={getattr(trade, 'trade_id', None)!r}"
            )


def runtime_trade_immutable_bindings(
    pipeline: V12Pipeline,
    decision: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze the exact model bundle and source pair that opened a trade."""

    model = getattr(pipeline, "smart_entry", None)
    bundle_dir_raw = getattr(model, "bundle_dir", None)
    bundle_sha256 = str(
        getattr(model, "_bundle_sha256", "") or ""
    )
    operating_point = getattr(model, "operating_point", None)
    model_metadata = getattr(model, "_meta", None)
    if bundle_dir_raw is None:
        raise RuntimeError("TRADE_MODEL_BUNDLE_PATH_MISSING")
    bundle_dir = Path(bundle_dir_raw).expanduser().resolve(strict=True)
    if (
        len(bundle_sha256) != 64
        or bundle_sha256.lower() != bundle_sha256
        or any(
            character not in "0123456789abcdef"
            for character in bundle_sha256
        )
        or not isinstance(operating_point, dict)
        or not isinstance(model_metadata, dict)
    ):
        raise RuntimeError("TRADE_MODEL_BUNDLE_IDENTITY_INVALID")
    input_normalization = model_metadata.get("input_normalization")
    input_normalization_sha256 = (
        input_normalization.get("contract_sha256")
        if isinstance(input_normalization, dict)
        else None
    )
    contract_mode = model_metadata.get("contract_mode")
    if (
        not isinstance(input_normalization_sha256, str)
        or len(input_normalization_sha256) != 64
        or input_normalization_sha256.lower()
        != input_normalization_sha256
        or any(
            character not in "0123456789abcdef"
            for character in input_normalization_sha256
        )
        or contract_mode != MODEL_NATIVE_CONTRACT_MODE
    ):
        raise RuntimeError("TRADE_MODEL_TOKEN_BINDING_INVALID")
    pair_generation_id = str(
        decision.get("entry_source_pair_generation_id") or ""
    )
    pair_manifest_sha256 = str(
        decision.get("entry_source_pair_manifest_sha256") or ""
    )
    for label, value in (
        ("PAIR_GENERATION", pair_generation_id),
        ("PAIR_MANIFEST", pair_manifest_sha256),
    ):
        if (
            len(value) != 64
            or value.lower() != value
            or any(
                character not in "0123456789abcdef"
                for character in value
            )
        ):
            raise RuntimeError(f"TRADE_SOURCE_{label}_INVALID")
    return (
        {
            "schema_version": (
                TRADE_STATE_MODEL_BUNDLE_BINDING_SCHEMA_VERSION
            ),
            "bundle_dir": str(bundle_dir),
            "bundle_sha256": bundle_sha256,
            "input_normalization_sha256": input_normalization_sha256,
            "contract_mode": contract_mode,
            "operating_point": dict(operating_point),
        },
        {
            "schema_version": (
                TRADE_STATE_SOURCE_PAIR_BINDING_SCHEMA_VERSION
            ),
            "pair_generation_id": pair_generation_id,
            "pair_manifest_sha256": pair_manifest_sha256,
        },
    )


def submit_market_entry_under_authority_lease(
    client: OandaClient,
    *,
    side: str,
    units: int,
    decision_snapshot: dict[str, Any],
    sizing_application: dict[str, Any],
    model_bundle_binding: dict[str, Any],
    entry_source_pair_binding: dict[str, Any],
    broker_account_binding: dict[str, str],
    launch_lease: dict[str, str],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Revalidate exact authority, persist intent, then mutate once."""

    with broker_entry_authority_mutation_lease():
        require_runtime_entry_launch_lease(
            expected_lease=launch_lease
        )
        require_runtime_new_entry_live_tail(
            expected_pair_generation_id=entry_source_pair_binding[
                "pair_generation_id"
            ],
            expected_pair_manifest_sha256=entry_source_pair_binding[
                "pair_manifest_sha256"
            ],
        )
        intent = build_broker_entry_intent(
            side=side,
            units=units,
            decision_snapshot=decision_snapshot,
            sizing_application=sizing_application,
            model_bundle_binding=model_bundle_binding,
            entry_source_pair_binding=entry_source_pair_binding,
            broker_account_binding=broker_account_binding,
            launch_lease=launch_lease,
            created_utc=datetime.now(timezone.utc),
        )
        intent_path = persist_broker_entry_intent(intent)
        result = attempt_market_entry(
            client,
            side,
            units=units,
            client_order_id=intent["client_order_id"],
        )
    return intent, intent_path, result


def load_runner_open_trades(
    *,
    dry_run: bool,
    shadow_only: bool,
) -> list[TradeState]:
    """Read and mode-bind persisted state before credentials or broker setup."""

    if TRADE_STATE_FILE.is_file():
        raise RuntimeError(
            "LEGACY_TRADE_STATE_LOCATION_REQUIRES_OFFLINE_RECONCILIATION"
        )
    open_trades = TradeState.load_all(TRADE_STATE_DIR)
    require_runner_trade_state_mode(
        open_trades,
        dry_run=dry_run,
        shadow_only=shadow_only,
    )
    return open_trades


# ── Journal — all decisions + outcomes for daily replay ──────────────────


def log_journal_event(journal_path: Path, event: dict[str, Any]) -> None:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(event)
    record["logged_at_utc"] = datetime.now(timezone.utc).isoformat()
    encoded = (json.dumps(record, default=str) + "\n").encode("utf-8")
    fd = os.open(
        journal_path,
        os.O_RDWR | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        idempotency_key = record.get("idempotency_key")
        if idempotency_key is not None:
            if (
                not isinstance(idempotency_key, str)
                or not idempotency_key
            ):
                raise RuntimeError(
                    "JOURNAL_IDEMPOTENCY_KEY_INVALID"
                )
            os.lseek(fd, 0, os.SEEK_SET)
            existing = bytearray()
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                existing.extend(chunk)
            for line in bytes(existing).splitlines():
                try:
                    prior = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "JOURNAL_EXISTING_JSON_INVALID"
                    ) from exc
                if (
                    isinstance(prior, dict)
                    and prior.get("idempotency_key")
                    == idempotency_key
                ):
                    prior_payload = dict(prior)
                    current_payload = dict(record)
                    prior_payload.pop("logged_at_utc", None)
                    current_payload.pop("logged_at_utc", None)
                    if prior_payload != current_payload:
                        raise RuntimeError(
                            "JOURNAL_IDEMPOTENCY_PAYLOAD_MISMATCH"
                        )
                    return
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"short journal write: {journal_path}")
            view = view[written:]
        os.fsync(fd)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def daily_journal_path(suffix: str = "") -> Path:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    suf = f"_{suffix}" if suffix else ""
    return JOURNAL_DIR / f"v12_paper_journal_{today}{suf}.jsonl"


def assert_no_retired_entry_overrides() -> None:
    """Reject every legacy post-model entry knob, including disabled values."""
    present = sorted(
        name
        for name in os.environ
        if name in RETIRED_ENTRY_OVERRIDE_ENV
        or any(
            name.startswith(prefix)
            for prefix in RETIRED_ENTRY_SIZING_ENV_PREFIXES
        )
    )
    if present:
        raise SystemExit(
            "[SMART_GATE] retired entry override environment variables are forbidden; "
            "remove them instead of relying on disabled/no-op values: " + ", ".join(present)
        )


def _sha256_regular_file(path: Path, *, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"[SMART_GATE] {label} is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ── Main loop ────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="GX1 XAUUSD model-direction runner")
    p.add_argument("--max-trades", type=int, default=DEFAULT_MAX_TRADES,
                   help="Max concurrent virtual trades held simultaneously (default: 1)")
    p.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    p.add_argument("--dry-run", action="store_true",
                   help="Don't actually send orders — just log what would happen (shadow mode)")
    p.add_argument("--shadow-only", action="store_true",
                   help="Observation-only mode: requires --dry-run, logs decisions, never opens virtual trades or writes trade state.")
    p.add_argument("--journal-suffix", type=str, default="",
                   help="Suffix for journal filename (e.g. 'live' or 'shadow') to allow parallel runners")
    args = p.parse_args()
    if args.shadow_only and not args.dry_run:
        raise SystemExit("--shadow-only requires --dry-run")
    runner_singleton_handle = acquire_runner_singleton_lock()
    persisted_state_hint = (
        TRADE_STATE_FILE.is_file()
        or (
            TRADE_STATE_DIR.is_dir()
            and any(TRADE_STATE_DIR.glob("open_trade_*.json"))
        )
        or (
            ENTRY_INTENT_DIR.is_dir()
            and any(ENTRY_INTENT_DIR.glob("*.json"))
        )
        or (
            CLOSE_INTENT_DIR.is_dir()
            and any(CLOSE_INTENT_DIR.glob("*.json"))
        )
    )
    startup_launch_lease = (
        None
        if persisted_state_hint
        else enforce_entry_next_edge_runner_guard(args)
    )

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")
    LOG.info(
        "runner singleton lease acquired fd=%s",
        runner_singleton_handle.fileno(),
    )
    _load_runtime_dependencies()

    open_trades = load_runner_open_trades(
        dry_run=args.dry_run,
        shadow_only=args.shadow_only,
    )
    creds = load_oanda_credentials(
        prod_baseline=True,
        require_live_latch=not (args.dry_run or args.shadow_only),
    )
    if (
        TRADE_STATE_BROKER_ACCOUNT_BINDING_SCHEMA_VERSION
        != BROKER_ACCOUNT_BINDING_SCHEMA_VERSION
    ):
        raise RuntimeError("BROKER_ACCOUNT_BINDING_SCHEMA_OWNER_MISMATCH")
    broker_account_binding = build_trade_broker_account_binding(
        environment=creds.env,
        account_id=creds.account_id,
    )
    require_runner_broker_account_binding(
        open_trades,
        broker_account_binding=broker_account_binding,
        dry_run=args.dry_run,
        shadow_only=args.shadow_only,
    )
    client = OandaClient(OandaClientConfig(api_key=creds.api_token,
                                            account_id=creds.account_id,
                                            env=creds.env))
    open_trades = reconcile_unresolved_broker_entry_intents(
        client,
        open_trades=open_trades,
        dry_run=args.dry_run,
        broker_account_binding=broker_account_binding,
        journal_path=daily_journal_path(args.journal_suffix),
    )
    open_trades = reconcile_unresolved_broker_close_intents(
        client,
        open_trades=open_trades,
        dry_run=args.dry_run,
        broker_account_binding=broker_account_binding,
        journal_path=daily_journal_path(args.journal_suffix),
    )
    require_runner_trade_state_mode(
        open_trades,
        dry_run=args.dry_run,
        shadow_only=args.shadow_only,
    )
    require_runner_broker_account_binding(
        open_trades,
        broker_account_binding=broker_account_binding,
        dry_run=args.dry_run,
        shadow_only=args.shadow_only,
    )
    recovery_only = bool(open_trades)
    if not recovery_only and startup_launch_lease is None:
        startup_launch_lease = enforce_entry_next_edge_runner_guard(args)
    LOG.info(f"V12 paper runner starting  env={creds.env}  account={creds.account_id}")
    LOG.info(
        "  max_trades=%s dry_run=%s; spread is model evidence",
        args.max_trades,
        args.dry_run,
    )
    LOG.info("  entry: exact model direction; immutable learned calibrated sizing")

    if recovery_only:
        serialized_bindings = {
            json.dumps(
                trade.model_bundle_binding,
                sort_keys=True,
                separators=(",", ":"),
            )
            for trade in open_trades
        }
        if len(serialized_bindings) != 1:
            raise RuntimeError(
                "EXIT_RECOVERY_MULTIPLE_MODEL_BUNDLES_FORBIDDEN"
            )
        LOG.warning(
            "loading immutable Exit-only recovery bundle; "
            "new Entry is disabled until process restart"
        )
        pipeline = V12Pipeline.load_exit_recovery(open_trades[0])
    else:
        LOG.info("loading unified Entry+Exit V12Pipeline...")
        pipeline = V12Pipeline.load_default()
    if pipeline.smart_entry is None:
        raise SystemExit("model-native learned sizing requires loaded smart_entry")
    runtime_sizing_authority = require_model_native_sizing_authority_contract(
        getattr(pipeline.smart_entry, "_sizing_authority", None),
        context="V12_PAPER_RUNNER_STARTUP",
        required_mode=MODEL_NATIVE_SIZING_MODE_LEARNED,
    )
    validated_sizing_authority = prepare_model_native_sizing_authority(
        runtime_sizing_authority,
        context="V12_PAPER_RUNNER_STARTUP",
    )
    LOG.info(
        "✓ unified model loaded mode=%s",
        "immutable_exit_recovery" if recovery_only else "entry_exit",
    )

    # Structured trade journal — per-trade JSON + aggregate index CSV.
    # Captures: exact model-native entry evidence, applied learned sizing,
    # per-bar exit decisions, execution events (OANDA fills),
    # exit_summary (realized PnL, MFE/MAE peaks).
    journal = TradeJournal(
        run_dir=JOURNAL_DIR,
        run_tag=f"v12_paper_{args.journal_suffix or 'live'}_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        header={"runner": "v12_paper_runner",
                "env": creds.env,
                "max_trades": args.max_trades,
                "dry_run": args.dry_run,
                "shadow_only": args.shadow_only,
                "entry_direction_mode": "model_direction_argmax",
                "entry_units_mode": MODEL_NATIVE_SIZING_MODE_LEARNED,
                "sizing_authority_contract": runtime_sizing_authority},
    )
    LOG.info(f"  trade journal: {journal.journal_dir}")

    last_decision_minute = None
    consecutive_errors = 0
    last_stale_log_minute = None

    # Persisted state was loaded and mode-bound before credentials/broker setup.
    if args.shadow_only:
        LOG.warning(
            "[SHADOW_ONLY] no persisted trade state admitted; "
            "no orders or virtual trades will be opened"
        )
    if open_trades:
        for t in open_trades:
            LOG.info(f"resumed open trade {t.trade_id or '(no-id)'}: "
                      f"side={t.side} bars={t.bars_in_trade} "
                      f"entry_ts={t.entry_ts}  pnl={t.current_pnl_bps:+.1f} bps")
    else:
        LOG.info(f"no open trades in {TRADE_STATE_DIR} — starting fresh")

    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            current_minute = now_utc.replace(second=0, microsecond=0)

            # Decide once per M1
            if last_decision_minute == current_minute:
                time.sleep(args.poll_seconds)
                continue

            try:
                spread_bps, bid, ask = get_current_spread_bps(client, now_utc=now_utc)
            except StaleQuoteError as exc:
                # Market closed (weekend/holiday) — OANDA serves last close-of-week quote.
                # Skip silently; log once per hour to confirm daemon alive without polluting journal.
                if last_stale_log_minute is None or (current_minute - last_stale_log_minute).total_seconds() >= 3600:
                    LOG.info(f"stale quote ({exc.age_sec:.0f}s old, market closed) — pausing journal writes")
                    last_stale_log_minute = current_minute
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue
            event = {
                "ts_utc": current_minute.isoformat(),
                "bid": bid, "ask": ask, "spread_bps": spread_bps,
                "n_open_trades": len(open_trades),
                "max_trades": args.max_trades,
                "has_open_trade": len(open_trades) > 0,   # back-compat for analytics
            }

            # ── EXIT branch: iterate all open trades, evaluate each per M1 ──
            # The pipeline owns the unique exact-closed-M1 contract.  The runner
            # must not manufacture that state from a tick midpoint/latest-row read.
            per_trade_records: list[dict[str, Any]] = []
            survivors: list[TradeState] = []
            exit_decision_unavailable = False
            for trade in open_trades:
                try:
                    prior_exit_decision = trade.last_exit_decision
                    if isinstance(prior_exit_decision, dict):
                        journal_v12_exit_decision(journal, trade)
                    if (
                        isinstance(prior_exit_decision, dict)
                        and prior_exit_decision.get("action") == "EXIT_NOW"
                    ):
                        exit_decision = dict(prior_exit_decision)
                    else:
                        exit_decision = make_v12_exit_decision(
                            pipeline, trade, current_minute, bid, ask,
                            on_bar_committed=lambda committed_trade: (
                                persist_and_journal_v12_exit_bar(
                                    journal,
                                    committed_trade,
                                )
                            ),
                        )
                except ExitDecisionUnavailable as exc:
                    exit_decision_unavailable = True
                    record = {
                        "trade_id": trade.trade_id,
                        "side": trade.side,
                        "entry_ts": trade.entry_ts.isoformat(),
                        "bars_in_trade": trade.bars_in_trade,
                        "units": trade.units,
                        "pnl_bps": trade.current_pnl_bps,
                        "peak_bps": trade.cum_mfe_bps,
                        "mae_bps": trade.cum_mae_bps,
                        "exit_decision": None,
                        "exit_decision_unavailable_reason": exc.reason,
                        "exit_decision_unavailable_evidence": exc.evidence,
                        "order_status": "EXIT_MODEL_DECISION_UNAVAILABLE",
                    }
                    LOG.error(
                        "[EXIT_MODEL_DECISION_UNAVAILABLE] trade_id=%s reason=%s evidence=%s",
                        trade.trade_id,
                        exc.reason,
                        exc.evidence,
                    )
                    if trade.trade_id:
                        journal.log(
                            event_type="EXIT_MODEL_DECISION_UNAVAILABLE",
                            trade_id=trade.trade_id,
                            payload={
                                "timestamp": current_minute.isoformat(),
                                "reason": exc.reason,
                                "evidence": exc.evidence,
                                "bars_in_trade": int(trade.bars_in_trade),
                                "current_pnl_bps": float(trade.current_pnl_bps),
                            },
                        )

                    trade.save(TRADE_STATE_DIR)
                    survivors.append(trade)
                    per_trade_records.append(record)
                    continue
                record = {
                    "trade_id": trade.trade_id,
                    "side": trade.side,
                    "entry_ts": trade.entry_ts.isoformat(),
                    "bars_in_trade": trade.bars_in_trade,
                    "units": trade.units,
                    "pnl_bps": trade.current_pnl_bps,
                    "peak_bps": trade.cum_mfe_bps,
                    "mae_bps": trade.cum_mae_bps,
                    "exit_decision": exit_decision,
                }

                # Persist the complete staged M1 transition and its exact model
                # decision before any broker close or state deletion. A journal
                # failure is blocking; an exposure may never mutate without
                # durable decision evidence.
                trade.save(TRADE_STATE_DIR)
                if trade.trade_id:
                    journal_v12_exit_decision(journal, trade)

                exit_action = exit_decision["action"]
                if exit_action == "EXIT_NOW":
                    record["order_status"] = "EXIT_NOW"
                    write_trade_alert(
                        f"EXIT  trade_id={trade.trade_id}  side={trade.side}  "
                        f"bars={trade.bars_in_trade}  "
                        f"pnl={trade.current_pnl_bps:+.1f} bps  peak={trade.cum_mfe_bps:+.1f}  "
                        f"mae={trade.cum_mae_bps:+.1f}  "
                        f"source={exit_decision['decision_source']}"
                    )
                    if args.dry_run:
                        LOG.info(f"[DRY] EXIT_NOW trade_id={trade.trade_id} after {trade.bars_in_trade} bars  "
                                  f"pnl={trade.current_pnl_bps:+.1f} bps  side={trade.side}")
                    else:
                        (
                            close_intent,
                            close_intent_path,
                            close_result,
                        ) = submit_broker_close_with_durable_intent(
                            client,
                            trade,
                            broker_account_binding=broker_account_binding,
                        )
                        record["broker_close_intent"] = {
                            "path": str(close_intent_path),
                            "sha256": hashlib.sha256(
                                close_intent_path.read_bytes()
                            ).hexdigest(),
                            "close_intent_id": close_intent[
                                "close_intent_id"
                            ],
                        }
                        record["close_order_details"] = close_result
                        if close_result.get("status") == "rejected":
                            rejection_archive = (
                                finalize_broker_close_rejection(
                                    intent=close_intent,
                                    intent_path=close_intent_path,
                                    close_result=close_result,
                                    trade=trade,
                                    journal_path=daily_journal_path(
                                        args.journal_suffix
                                    ),
                                )
                            )
                            record["order_status"] = (
                                "EXIT_CLOSE_REJECTED_NO_MUTATION"
                            )
                            record["broker_close_rejection_archive"] = (
                                str(rejection_archive)
                            )
                            raise SystemExit(
                                "[BROKER_CLOSE_REJECTED_NO_MUTATION] "
                                f"broker rejected trade {trade.trade_id}; "
                                "state is retained and restart is required"
                            )
                        if close_result.get("status") != "closed":
                            record["order_status"] = (
                                "EXIT_CLOSE_OUTCOME_UNRESOLVED"
                            )
                            log_journal_event(
                                daily_journal_path(
                                    args.journal_suffix
                                ),
                                {
                                    "event": (
                                        "BROKER_CLOSE_OUTCOME_UNRESOLVED"
                                    ),
                                    "trade_id": trade.trade_id,
                                    "broker_close_intent": record[
                                        "broker_close_intent"
                                    ],
                                    "close_result": close_result,
                                },
                            )
                            raise SystemExit(
                                "[BROKER_CLOSE_OUTCOME_UNRESOLVED] "
                                f"durable intent {close_intent_path} "
                                "requires restart reconciliation; "
                                "the close mutation will not be retried"
                            )
                elif exit_action != "HOLD":
                    raise RuntimeError(
                        "UNIFIED_EXIT_ACTION_NOT_EXHAUSTIVE: "
                        f"{exit_action!r}"
                    )
                else:
                    record["order_status"] = "HOLDING_TRADE"
                    trade.save(TRADE_STATE_DIR)
                    survivors.append(trade)

                # ── TradeJournal: terminal execution summary ──
                if trade.trade_id:
                    if record.get("order_status") == "EXIT_NOW":
                        close_result = record.get("close_order_details") or {}
                        exit_price = (
                            bid if trade.side == "long" else ask
                        ) if args.dry_run else float(
                            close_result["fill_price"]
                        )
                        realized_pnl = (
                            0.0
                            if args.dry_run
                            else float(close_result["realized_pl"])
                        )
                        if not args.dry_run:
                            journal.log_oanda_trade_update(
                                trade_id=trade.trade_id,
                                event_type="TRADE_CLOSED_OANDA",
                                oanda_trade_id=trade.trade_id,
                                oanda_transaction_id=close_result[
                                    "oanda_transaction_id"
                                ],
                                price=exit_price,
                                units=trade.units,
                                pl=realized_pnl,
                                ts_oanda=close_result["fill_time"],
                            )
                        journal.log_exit_summary(
                            trade_id=trade.trade_id,
                            exit_time=(
                                current_minute.isoformat()
                                if args.dry_run
                                else close_result["fill_time"]
                            ),
                            exit_price=exit_price,
                            exit_bid=bid, exit_ask=ask,
                            exit_spread_bps=spread_bps,
                            exit_reason=record["order_status"],
                            realized_pnl_bps=float(trade.current_pnl_bps),
                            max_mfe_bps=float(trade.cum_mfe_bps),
                            max_mae_bps=float(trade.cum_mae_bps),
                            intratrade_drawdown_bps=float(trade.cum_mfe_bps - trade.current_pnl_bps),
                        )
                        if args.dry_run:
                            trade.delete_state_file(TRADE_STATE_DIR)
                        else:
                            finalize_broker_close_intent(
                                intent=close_intent,
                                intent_path=close_intent_path,
                                close_result=close_result,
                                trade=trade,
                                journal_path=daily_journal_path(
                                    args.journal_suffix
                                ),
                            )
                per_trade_records.append(record)
            open_trades = survivors
            event["open_trade_records"] = per_trade_records
            # Back-compat top-level fields when exactly one trade was active
            if len(per_trade_records) == 1:
                r0 = per_trade_records[0]
                event["v12_exit_decision"] = r0["exit_decision"]
                event["trade_open_ts"] = r0["entry_ts"]
                event["trade_side"] = r0["side"]
                event["trade_bars"] = r0["bars_in_trade"]
                event["trade_pnl_bps"] = r0["pnl_bps"]
                event["trade_peak_bps"] = r0["peak_bps"]
                event["trade_mae_bps"] = r0["mae_bps"]
                event["order_status"] = r0["order_status"]

            # An unavailable Exit contract is neither HOLD nor permission to
            # evaluate/open a new trade. Persist evidence and end this M1 step.
            if exit_decision_unavailable:
                event["order_status"] = "EXIT_MODEL_DECISION_UNAVAILABLE"
                event["exit_model_decision_unavailable"] = True
                log_journal_event(daily_journal_path(args.journal_suffix), event)
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue

            # ── ENTRY branch: evaluate the admitted model-native XAU model ──
            if recovery_only:
                event["order_status"] = (
                    "IMMUTABLE_EXIT_RECOVERY_ONLY_NO_NEW_ENTRY"
                )
                log_journal_event(
                    daily_journal_path(args.journal_suffix),
                    event,
                )
                if not open_trades:
                    LOG.warning(
                        "Exit recovery completed; restart is required "
                        "before any new Entry authority can be evaluated"
                    )
                    return 0
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue
            if len(open_trades) >= args.max_trades:
                event.setdefault("order_status", "AT_MAX_TRADES")
                log_journal_event(daily_journal_path(args.journal_suffix), event)
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue

            try:
                require_runtime_entry_launch_lease(
                    expected_lease=startup_launch_lease
                )
            except RuntimeError as exc:
                event["order_status"] = "LAUNCH_AUTHORITY_UNAVAILABLE_NO_ORDER"
                event["launch_authority_evidence"] = str(exc)
                log_journal_event(
                    daily_journal_path(args.journal_suffix),
                    event,
                )
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue

            try:
                entry_live_tail = require_runtime_new_entry_live_tail()
            except RuntimeError as exc:
                event["order_status"] = (
                    "LIVE_TAIL_ADMISSION_UNAVAILABLE_NO_ORDER"
                )
                event["live_tail_admission_evidence"] = str(exc)
                log_journal_event(
                    daily_journal_path(args.journal_suffix),
                    event,
                )
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue

            try:
                decision = make_v12_decision(
                    pipeline,
                    datetime.now(timezone.utc),
                    bid,
                    ask,
                )
            except EntryDecisionUnavailable as exc:
                event["order_status"] = "MODEL_DECISION_UNAVAILABLE"
                event["model_decision_unavailable_reason"] = exc.reason
                event["model_decision_unavailable_evidence"] = exc.evidence
                log_journal_event(daily_journal_path(args.journal_suffix), event)
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue
            try:
                decision_live_tail = require_runtime_new_entry_live_tail(
                    expected_pair_generation_id=str(
                        decision.get("entry_source_pair_generation_id") or ""
                    ),
                    expected_pair_manifest_sha256=str(
                        decision.get(
                            "entry_source_pair_manifest_sha256"
                        )
                        or ""
                    ),
                )
            except RuntimeError as exc:
                event["order_status"] = (
                    "LIVE_TAIL_ADMISSION_UNAVAILABLE_NO_ORDER"
                )
                event["live_tail_admission_evidence"] = str(exc)
                event["pre_inference_live_tail_authority"] = entry_live_tail
                log_journal_event(
                    daily_journal_path(args.journal_suffix),
                    event,
                )
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue
            decision["_v10_snapshot"] = require_executable_model_native_entry_decision(
                decision,
                current_minute,
            )
            event["live_tail_runtime_authority"] = decision_live_tail
            event["v12_decision"] = decision
            direction_to_action = {
                direction: (
                    index,
                    MODEL_DIRECTION_ACTION_BY_INDEX[index],
                )
                for direction, index in MODEL_DIRECTION_INDEX_BY_NAME.items()
            }
            model_direction = str(decision.get("model_direction") or "")
            if model_direction not in direction_to_action:
                raise RuntimeError(
                    f"model decision lacks exact LONG/SHORT/FLAT direction: {model_direction!r}"
                )
            expected_index, expected_action = direction_to_action[model_direction]
            if decision.get("model_direction_index") != expected_index:
                raise RuntimeError(
                    "model direction index disagrees with model direction: "
                    f"{decision.get('model_direction_index')!r} != {expected_index}"
                )
            if decision.get("action") != expected_action:
                raise RuntimeError(
                    "runner action disagrees with model direction argmax: "
                    f"{decision.get('action')!r} != {expected_action!r}"
                )
            if args.shadow_only:
                event["order_status"] = "SHADOW_ONLY_NO_ORDER"
                event["shadow_only"] = True
                log_journal_event(daily_journal_path(args.journal_suffix), event)
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue

            if expected_action in ("TAKE_LONG_NOW", "TAKE_SHORT_NOW"):
                side = MODEL_DIRECTION_EXECUTION_SIDE_BY_INDEX.get(expected_index)
                if side not in {"long", "short"}:
                    raise RuntimeError(
                        "model direction contract has no executable broker side"
                    )
                # NETTING-guard removed 2026-05-20: account migrated to HEDGING
                # (101-004-31061417-002). Each market order now opens a separate
                # trade regardless of existing positions on the same instrument,
                # so opposite-side entries no longer net against existing trades.
                # If you reinstate a NETTING account, restore the guard from git.

                try:
                    sizing_constraints = learned_sizing_runtime_constraints(
                        client,
                        bid=bid,
                        ask=ask,
                        validated_authority=validated_sizing_authority,
                    )
                    decision_sizing_authority = require_model_native_sizing_authority_contract(
                        decision["_v10_snapshot"]["sizing_authority_contract"],
                        context="V12_PAPER_RUNNER_DECISION_SIZING_AUTHORITY",
                        required_mode=MODEL_NATIVE_SIZING_MODE_LEARNED,
                    )
                    if decision_sizing_authority != runtime_sizing_authority:
                        raise ModelNativeSizingUnavailable(
                            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] decision/startup adoption mismatch"
                        )
                    sizing_application = apply_model_native_sizing(
                        validated_authority=validated_sizing_authority,
                        position_size_logit=decision["_v10_snapshot"][
                            "position_size_logit"
                        ],
                        model_direction=model_direction,
                        runtime_constraints=sizing_constraints,
                        context="V12_PAPER_RUNNER_ENTRY",
                    )
                except (ModelNativeSizingUnavailable, RuntimeError) as exc:
                    LOG.error("[SIZING_UNAVAILABLE] no order: %s", exc)
                    event["order_status"] = "SIZING_UNAVAILABLE_NO_ORDER"
                    event["sizing_unavailable_evidence"] = str(exc)
                    log_journal_event(daily_journal_path(args.journal_suffix), event)
                    last_decision_minute = current_minute
                    consecutive_errors = 0
                    time.sleep(args.poll_seconds)
                    continue
                event["sizing_application"] = sizing_application
                if not sizing_application["authorized_order"]:
                    event["order_status"] = "MODEL_NATIVE_SIZING_NO_ORDER"
                    event["sizing_no_order_reason"] = sizing_application[
                        "no_order_reason"
                    ]
                    log_journal_event(daily_journal_path(args.journal_suffix), event)
                    last_decision_minute = current_minute
                    consecutive_errors = 0
                    time.sleep(args.poll_seconds)
                    continue
                try:
                    require_runtime_entry_launch_lease(
                        expected_lease=startup_launch_lease
                    )
                except RuntimeError as exc:
                    event["order_status"] = (
                        "LAUNCH_AUTHORITY_UNAVAILABLE_NO_ORDER"
                    )
                    event["launch_authority_evidence"] = str(exc)
                    log_journal_event(
                        daily_journal_path(args.journal_suffix),
                        event,
                    )
                    last_decision_minute = current_minute
                    consecutive_errors = 0
                    time.sleep(args.poll_seconds)
                    continue
                try:
                    require_runtime_new_entry_live_tail(
                        expected_pair_generation_id=decision[
                            "entry_source_pair_generation_id"
                        ],
                        expected_pair_manifest_sha256=decision[
                            "entry_source_pair_manifest_sha256"
                        ],
                    )
                except RuntimeError as exc:
                    event["order_status"] = (
                        "LIVE_TAIL_ADMISSION_UNAVAILABLE_NO_ORDER"
                    )
                    event["live_tail_admission_evidence"] = str(exc)
                    log_journal_event(
                        daily_journal_path(args.journal_suffix),
                        event,
                    )
                    last_decision_minute = current_minute
                    consecutive_errors = 0
                    time.sleep(args.poll_seconds)
                    continue
                trade_units = int(sizing_application["units"])
                event["units_mode"] = MODEL_NATIVE_SIZING_MODE_LEARNED
                event["units"] = trade_units
                event["sizing_authority_contract"] = sizing_application[
                    "sizing_authority_contract"
                ]
                (
                    model_bundle_binding,
                    entry_source_pair_binding,
                ) = runtime_trade_immutable_bindings(
                    pipeline,
                    decision,
                )
                if args.dry_run:
                    virtual_fill_ts = pd.Timestamp(datetime.now(timezone.utc))
                    event["order_status"] = "DRY_RUN"
                    virtual_id = f"virtual_{current_minute.strftime('%Y%m%dT%H%M%S')}"
                    new_trade = TradeState.open(
                        entry_ts=virtual_fill_ts,
                        side=side, entry_bid=bid, entry_ask=ask,
                        v10_snapshot=decision["_v10_snapshot"],
                        trade_id=virtual_id,
                        units=trade_units,
                        sizing_application=sizing_application,
                        fill_transaction_id=f"virtual:{virtual_id}",
                        execution_mode="learned_virtual_dry_run",
                        model_bundle_binding=model_bundle_binding,
                        entry_source_pair_binding=(
                            entry_source_pair_binding
                        ),
                    )
                    journal.log_entry_snapshot(
                        trade_id=virtual_id,
                        entry_time=virtual_fill_ts.isoformat(),
                        instrument=INSTRUMENT,
                        side=side,
                        entry_price=ask if side == "long" else bid,
                        model_evidence=dict(decision["_v10_snapshot"]),
                        entry_bid=bid,
                        entry_ask=ask,
                        entry_spread_bps=new_trade.entry_spread_bps,
                        session=str(decision["session"]),
                        model_policy=str(decision["policy"]),
                        execution_checks=[
                            "fresh_quote",
                            "literal_spread_supplied_to_model",
                            "exposure_safety_admitted",
                            "virtual_dry_run",
                            "learned_sizing_proof_bound",
                        ],
                        capacity_units=sizing_application["capacity_units"],
                        reference_pre_round_units=sizing_application[
                            "reference_pre_round_units"
                        ],
                        pre_round_units=sizing_application["pre_round_units"],
                        units=trade_units,
                        applied_size_multiplier=sizing_application[
                            "applied_size_multiplier"
                        ],
                        sizing_application=sizing_application,
                        atr_bps=decision["_v10_snapshot"]["atr_bps"],
                        model_bundle_binding=model_bundle_binding,
                        entry_source_pair_binding=(
                            entry_source_pair_binding
                        ),
                    )
                    new_trade.save(TRADE_STATE_DIR)
                    open_trades.append(new_trade)
                    LOG.info(f"[DRY] virtual trade opened  side={side}  id={virtual_id}")
                else:
                    try:
                        require_broker_xau_trade_reconciliation(
                            client,
                            local_open_trades=open_trades,
                            max_trades=args.max_trades,
                            expected_exposure_transaction_id=sizing_constraints[
                                "exposure_last_transaction_id"
                            ],
                        )
                    except ModelNativeSizingUnavailable as exc:
                        event["order_status"] = "BROKER_TRADE_COUNT_UNAVAILABLE_NO_ORDER"
                        event["broker_trade_count_evidence"] = str(exc)
                        log_journal_event(
                            daily_journal_path(args.journal_suffix),
                            event,
                        )
                        last_decision_minute = current_minute
                        consecutive_errors = 0
                        time.sleep(args.poll_seconds)
                        continue
                    try:
                        require_runtime_entry_launch_lease(
                            expected_lease=startup_launch_lease
                        )
                    except RuntimeError as exc:
                        event["order_status"] = (
                            "LAUNCH_AUTHORITY_UNAVAILABLE_NO_ORDER"
                        )
                        event["launch_authority_evidence"] = str(exc)
                        log_journal_event(
                            daily_journal_path(args.journal_suffix),
                            event,
                        )
                        last_decision_minute = current_minute
                        consecutive_errors = 0
                        time.sleep(args.poll_seconds)
                        continue
                    try:
                        require_runtime_new_entry_live_tail(
                            expected_pair_generation_id=decision[
                                "entry_source_pair_generation_id"
                            ],
                            expected_pair_manifest_sha256=decision[
                                "entry_source_pair_manifest_sha256"
                            ],
                        )
                    except RuntimeError as exc:
                        event["order_status"] = (
                            "LIVE_TAIL_ADMISSION_UNAVAILABLE_NO_ORDER"
                        )
                        event["live_tail_admission_evidence"] = str(exc)
                        log_journal_event(
                            daily_journal_path(args.journal_suffix),
                            event,
                        )
                        last_decision_minute = current_minute
                        consecutive_errors = 0
                        time.sleep(args.poll_seconds)
                        continue
                    if not isinstance(startup_launch_lease, dict):
                        raise RuntimeError(
                            "BROKER_ENTRY_LAUNCH_LEASE_MISSING"
                        )
                    (
                        entry_intent,
                        entry_intent_path,
                        order_result,
                    ) = submit_market_entry_under_authority_lease(
                        client,
                        side=side,
                        units=trade_units,
                        decision_snapshot=decision["_v10_snapshot"],
                        sizing_application=sizing_application,
                        model_bundle_binding=model_bundle_binding,
                        entry_source_pair_binding=(
                            entry_source_pair_binding
                        ),
                        broker_account_binding=broker_account_binding,
                        launch_lease=startup_launch_lease,
                    )
                    event["broker_entry_intent"] = {
                        "path": str(entry_intent_path),
                        "sha256": hashlib.sha256(
                            entry_intent_path.read_bytes()
                        ).hexdigest(),
                        "client_order_id": entry_intent[
                            "client_order_id"
                        ],
                    }
                    event["order_status"] = order_result["status"]
                    event["order_details"] = order_result
                    if order_result.get("status") == "unknown_outcome":
                        log_journal_event(
                            daily_journal_path(args.journal_suffix),
                            event,
                        )
                        raise SystemExit(
                            "[BROKER_ENTRY_OUTCOME_UNKNOWN] durable intent "
                            f"{entry_intent_path} requires reconciliation"
                        )
                    if order_result.get("status") in {
                        "filled_units_mismatch",
                        "filled_structure_mismatch",
                    }:
                        mismatch_trade_id = str(
                            order_result.get("trade_id") or ""
                        ).strip()
                        observed_signed_units = order_result.get(
                            "filled_signed_units"
                        )
                        event["order_status"] = (
                            "FILL_CONTRACT_MISMATCH_RECONCILIATION_REQUIRED"
                        )
                        write_trade_alert(
                            "FILL CONTRACT MISMATCH "
                            f"requested={order_result.get('requested_signed_units')} "
                            f"observed={observed_signed_units!r} "
                            f"trade_id={mismatch_trade_id or '(missing)'}; "
                            "no inferred close mutation was sent"
                        )
                        journal.log(
                            event_type="BROKER_RECONCILIATION_REQUIRED",
                            trade_id=mismatch_trade_id or None,
                            payload={
                                "ts_utc": current_minute.isoformat(),
                                "reason": (
                                    "Entry fill units/structure did not "
                                    "prove one exact new trade"
                                ),
                                "order_result": order_result,
                                "automatic_close_forbidden": True,
                            },
                        )
                        log_journal_event(
                            daily_journal_path(args.journal_suffix), event
                        )
                        raise SystemExit(
                            "[BROKER_RECONCILIATION_REQUIRED] anomalous "
                            "Entry fill did not prove an exact new exposure; "
                            "the durable Entry intent remains unresolved and "
                            "an automatic close by inferred trade identity is "
                            "forbidden"
                        )
                    if order_result.get("status") == "filled":
                        filled_trade_units = abs(
                            int(order_result["filled_signed_units"])
                        )
                        if filled_trade_units != trade_units:
                            raise SystemExit(
                                "[BROKER_FILL_UNITS_CONTRACT_BROKEN] filled status without exact units"
                            )
                        event["filled_units"] = filled_trade_units
                        raw_fill = order_result.get("raw", {}).get("orderFillTransaction", {})
                        if (
                            not raw_fill.get("tradeOpened", {}).get("tradeID")
                            or raw_fill.get("tradesClosed")
                            or raw_fill.get("tradeReduced")
                        ):
                            raise SystemExit(
                                "[BROKER_FILL_STRUCTURE_CONTRACT_BROKEN] filled status "
                                "contained netting/reduction legs"
                            )
                        fill_price = _float_or_none(order_result.get("fill_price"))
                        fill_trade_id = str(order_result.get("trade_id") or "").strip()
                        fill_transaction_id = str(
                            order_result.get("oanda_transaction_id") or ""
                        ).strip()
                        raw_fill_time = order_result.get("fill_time")
                        try:
                            fill_ts = pd.Timestamp(raw_fill_time)
                        except (TypeError, ValueError):
                            fill_ts = None
                        if (
                            fill_ts is not None
                            and (
                                pd.isna(fill_ts)
                                or fill_ts.tzinfo is None
                                or fill_ts.utcoffset() is None
                                or fill_ts.utcoffset().total_seconds() != 0.0
                            )
                        ):
                            fill_ts = None
                        if fill_ts is not None:
                            fill_ts = fill_ts.tz_convert("UTC")
                        observed_bid = _float_or_none(bid)
                        observed_ask = _float_or_none(ask)
                        state_entry_bid = _float_or_none(
                            order_result.get("fill_bid")
                        )
                        state_entry_ask = _float_or_none(
                            order_result.get("fill_ask")
                        )
                        paired_entry_state_valid = bool(
                            order_result.get("fill_price_pair_exact") is True
                            and
                            state_entry_bid is not None
                            and state_entry_ask is not None
                            and state_entry_bid > 0.0
                            and state_entry_ask > state_entry_bid
                        )
                        if (
                            fill_price is None
                            or fill_price <= 0.0
                            or not fill_trade_id
                            or not fill_transaction_id
                            or fill_ts is None
                            or not paired_entry_state_valid
                        ):
                            # The broker has accepted/filled an order, but the
                            # authoritative entry state needed by TradeState is
                            # incomplete. Never substitute the polling quote and
                            # then let Exit decisioning run on a fabricated entry.
                            # A trade ID without the exact fill-side price/time
                            # contract is not sufficient authority to close
                            # exposure automatically.
                            event["order_status"] = (
                                "FILLED_STATE_UNAVAILABLE_RECONCILIATION_REQUIRED"
                            )
                            event["filled_state_unavailable"] = {
                                "fill_price": fill_price,
                                "trade_id": fill_trade_id or None,
                                "fill_transaction_id": fill_transaction_id or None,
                                "fill_time": raw_fill_time,
                                "observed_bid": observed_bid,
                                "observed_ask": observed_ask,
                                "fill_bid": state_entry_bid,
                                "fill_ask": state_entry_ask,
                                "fill_price_pair_exact": order_result.get(
                                    "fill_price_pair_exact"
                                ),
                                "paired_entry_state_valid": paired_entry_state_valid,
                            }
                            write_trade_alert(
                                f"FILL STATE UNAVAILABLE side={side} units={trade_units} "
                                f"trade_id={fill_trade_id or '(missing)'}; "
                                "no inferred close mutation was sent"
                            )
                            journal.log(
                                event_type=(
                                    "FILLED_STATE_UNAVAILABLE_"
                                    "RECONCILIATION_REQUIRED"
                                ),
                                trade_id=fill_trade_id or None,
                                payload={
                                    "ts_utc": current_minute.isoformat(),
                                    "side": side,
                                    "units": trade_units,
                                    "fill_price": fill_price,
                                    "observed_bid": observed_bid,
                                    "observed_ask": observed_ask,
                                    "paired_entry_state_valid": paired_entry_state_valid,
                                    "automatic_close_forbidden": True,
                                },
                            )
                            log_journal_event(daily_journal_path(args.journal_suffix), event)
                            raise SystemExit(
                                "[BROKER_RECONCILIATION_REQUIRED] filled "
                                "order lacks authoritative TradeState inputs; "
                                "the durable Entry intent remains unresolved "
                                "and automatic close is forbidden"
                            )
                        new_trade = TradeState.open(
                            entry_ts=fill_ts,
                            side=side, entry_bid=state_entry_bid, entry_ask=state_entry_ask,
                            v10_snapshot=decision["_v10_snapshot"],
                            trade_id=fill_trade_id,
                            units=filled_trade_units,
                            sizing_application=sizing_application,
                            fill_transaction_id=fill_transaction_id,
                            execution_mode="learned_broker_fill",
                            model_bundle_binding=model_bundle_binding,
                            entry_source_pair_binding=(
                                entry_source_pair_binding
                            ),
                            broker_account_binding=broker_account_binding,
                        )
                        new_trade.save(TRADE_STATE_DIR)
                        open_trades.append(new_trade)

                        # ── TradeJournal: log entry lifecycle ──
                        if new_trade.trade_id:
                            v10 = dict(decision["_v10_snapshot"])
                            journal.log_order_submitted(
                                trade_id=new_trade.trade_id,
                                instrument=INSTRUMENT, side=side,
                                units=trade_units, order_type="MARKET",
                                client_order_id=order_result.get("client_order_id"),
                                oanda_env=creds.env,
                            )
                            journal.log_order_filled(
                                trade_id=new_trade.trade_id,
                                oanda_trade_id=new_trade.trade_id,
                                oanda_order_id=order_result.get("oanda_order_id"),
                                oanda_transaction_id=order_result.get("oanda_transaction_id"),
                                fill_price=fill_price,
                                fill_units=filled_trade_units,
                                ts_oanda=order_result.get("fill_time"),
                            )
                            journal.log_entry_snapshot(
                                trade_id=new_trade.trade_id,
                                entry_time=fill_ts.isoformat(),
                                instrument=INSTRUMENT,
                                side=side,
                                entry_price=fill_price,
                                model_evidence=v10,
                                entry_bid=state_entry_bid,
                                entry_ask=state_entry_ask,
                                entry_spread_bps=new_trade.entry_spread_bps,
                                session=str(decision["session"]),
                                model_policy=str(decision["policy"]),
                                execution_checks=[
                                    "fresh_quote",
                                    "literal_spread_supplied_to_model",
                                    "broker_state_reconciled",
                                    "exposure_safety_admitted",
                                    "learned_sizing_proof_bound",
                                ],
                                capacity_units=sizing_application["capacity_units"],
                                reference_pre_round_units=sizing_application[
                                    "reference_pre_round_units"
                                ],
                                pre_round_units=sizing_application["pre_round_units"],
                                units=filled_trade_units,
                                applied_size_multiplier=sizing_application[
                                    "applied_size_multiplier"
                                ],
                                sizing_application=sizing_application,
                                atr_bps=v10["atr_bps"],
                                model_bundle_binding=(
                                    model_bundle_binding
                                ),
                                entry_source_pair_binding=(
                                    entry_source_pair_binding
                                ),
                            )
                            journal.log_oanda_trade_update(
                                trade_id=new_trade.trade_id,
                                event_type="TRADE_OPENED_OANDA",
                                oanda_trade_id=new_trade.trade_id,
                                price=float(order_result.get("fill_price") or ask),
                                units=filled_trade_units,
                            )
                        resolve_broker_entry_intent(
                            entry_intent_path
                        )

                        write_trade_alert(
                            f"OPEN  trade_id={new_trade.trade_id}  side={side}  "
                            f"entry={ask if side=='long' else bid:.2f}  "
                            f"spread={spread_bps:.1f}bps  units={trade_units}  "
                            f"open_count={len(open_trades)}/{args.max_trades}  "
                            f"direction={model_direction}  "
                            f"q_margin={_as_float(decision.get('entry_action_q_margin_bps')):+.3f}bps  "
                            f"mode={decision.get('selection_score_mode','')}  "
                            f"score={_fmt_optional_float(decision.get('selection_score'), '+.2f')}  "
                            f"lat={_fmt_optional_float(decision.get('entry_signal_latency_sec'), '.0f')}s"
                        )
                        LOG.info(f"opened trade  id={new_trade.trade_id}  side={side}  "
                                  f"entry={ask if side=='long' else bid}  "
                                  f"open_count={len(open_trades)}/{args.max_trades}  "
                                  f"direction={model_direction}  "
                                  f"q_margin={_as_float(decision.get('entry_action_q_margin_bps')):+.3f}bps  "
                                  f"mode={decision.get('selection_score_mode','')}  "
                                  f"score={_fmt_optional_float(decision.get('selection_score'), '+.2f')}  "
                                  f"entry_latency={_fmt_optional_float(decision.get('entry_signal_latency_sec'), '.0f')}s")
                    elif order_result.get("status") == "rejected":
                        # Reject without a trade_id — log via run-level JSONL so reject stream
                        # is preserved for triage even if no per-trade journal exists.
                        journal.log(
                            event_type="ORDER_REJECTED",
                            payload={
                                "ts_utc": current_minute.isoformat(),
                                "side": side,
                                "units": trade_units,
                                "reason": order_result.get("reason"),
                                "client_order_id": order_result.get("client_order_id"),
                                "model_action": expected_action,
                                "model_direction": model_direction,
                                "model_direction_index": expected_index,
                                "entry_action_q_bps": decision.get("entry_action_q_bps"),
                                "entry_action_q_margin_bps": decision.get(
                                    "entry_action_q_margin_bps"
                                ),
                                "bid": bid, "ask": ask,
                                "n_open_trades": len(open_trades),
                            },
                        )
                        resolve_broker_entry_intent(
                            entry_intent_path
                        )
            else:
                event["order_status"] = "MODEL_DIRECTION_FLAT"
                event["units_mode"] = MODEL_NATIVE_SIZING_MODE_LEARNED
                event["units"] = 0

            log_journal_event(daily_journal_path(args.journal_suffix), event)
            last_decision_minute = current_minute
            consecutive_errors = 0

        except Exception as exc:
            consecutive_errors += 1
            import traceback as _tb
            LOG.error(f"loop error (consec={consecutive_errors}): {exc}\n{_tb.format_exc()}")
            backoff = min(args.poll_seconds * (2 ** min(consecutive_errors, 5)), 300)
            time.sleep(backoff)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
