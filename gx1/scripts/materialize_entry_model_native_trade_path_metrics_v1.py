#!/usr/bin/env python3
"""Publish research-only MAE/MFE evidence from an exact unified Exit replay.

The input rows must already have been produced by the canonical full-TEST
Entry/Exit replay.  This command never trains, predicts, chooses an order,
or supplies a missing execution cost.  It writes an immutable, hash-bound
research report and per-trade parquet that explicitly remain unusable for
net-PnL, candidate admission, demo, or live trading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from gx1.contracts.entry_model_native_sizing_execution_v1 import (
    read_bound_parquet_exact,
)
from gx1.contracts.entry_model_native_trade_path_metrics_v1 import (
    TRADE_PATH_METRICS_DECISION,
    derive_unified_exit_trade_path_metrics,
)
from gx1.contracts.immutable_event_authority_v1 import (
    next_immutable_event_created_utc,
    write_immutable_json_event,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    atomic_write_parquet_immutable,
)


TRADE_PATH_METRICS_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_TRADE_PATH_METRICS"
TRADE_PATH_METRICS_EVENT_SCHEMA_VERSION = (
    "gx1_unified_exit_trade_path_metrics_event_v1"
)


class TradePathMetricsMaterializationError(RuntimeError):
    """An immutable research-only trade-path report cannot be published."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_regular_file(path: Path, *, context: str) -> dict[str, str]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise TradePathMetricsMaterializationError(
            f"{context} must be a regular input file: {candidate}"
        )
    resolved = candidate.resolve()
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _exact_sha256(value: str, *, context: str) -> str:
    observed = str(value).strip().lower()
    if len(observed) != 64 or any(
        char not in "0123456789abcdef" for char in observed
    ):
        raise TradePathMetricsMaterializationError(
            f"{context} must be a lowercase SHA-256 value"
        )
    return observed


def _source_inventory() -> tuple[dict[str, str], str]:
    root = Path(__file__).resolve().parents[2]
    files = (
        root / "gx1/contracts/entry_model_native_trade_path_metrics_v1.py",
        Path(__file__).resolve(),
    )
    inventory: dict[str, str] = {}
    for path in files:
        resolved = path.resolve()
        if not resolved.is_file():
            raise TradePathMetricsMaterializationError(
                f"reporter source file missing: {resolved}"
            )
        inventory[str(resolved.relative_to(root))] = _sha256_file(resolved)
    encoded = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return inventory, hashlib.sha256(encoded).hexdigest()


def _require_unchanged(
    observed: dict[str, str],
    *,
    context: str,
) -> None:
    current = _bound_regular_file(Path(observed["path"]), context=context)
    if current != observed:
        raise TradePathMetricsMaterializationError(
            f"{context} bytes changed while report was being materialized"
        )


def materialize_unified_exit_trade_path_metrics(
    *,
    replay_rows_path: Path,
    exit_trace_rows_path: Path,
    candidate_bundle_sha256: str,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Write one immutable, research-only path-metrics event and trade table."""

    candidate_sha = _exact_sha256(
        candidate_bundle_sha256,
        context="candidate_bundle_sha256",
    )
    replay_binding = _bound_regular_file(replay_rows_path, context="replay_rows")
    trace_binding = _bound_regular_file(
        exit_trace_rows_path,
        context="exit_trace_rows",
    )
    replay_rows = read_bound_parquet_exact(
        replay_binding,
        context="TRADE_PATH_METRICS_REPLAY_ROWS",
    )
    exit_trace_rows = read_bound_parquet_exact(
        trace_binding,
        context="TRADE_PATH_METRICS_EXIT_TRACE_ROWS",
    )
    report, trades = derive_unified_exit_trade_path_metrics(
        replay_rows=replay_rows,
        exit_trace_rows=exit_trace_rows,
        candidate_bundle_sha256=candidate_sha,
    )
    if report["decision"] != TRADE_PATH_METRICS_DECISION:
        raise TradePathMetricsMaterializationError(
            "trade-path report unexpectedly changed its research-only decision"
        )
    _require_unchanged(replay_binding, context="replay_rows")
    _require_unchanged(trace_binding, context="exit_trace_rows")

    root = Path(output_dir).expanduser().resolve()
    created = next_immutable_event_created_utc(
        root,
        TRADE_PATH_METRICS_EVENT_PREFIX,
    )
    stamp = created.strftime("%Y%m%dT%H%M%S%fZ")
    trades_path = root / f"unified_exit_trade_path_trades_{stamp}.parquet"
    atomic_write_parquet_immutable(trades, trades_path)
    trades_binding = _bound_regular_file(trades_path, context="published_trades")
    source_files, source_inventory_sha256 = _source_inventory()
    payload = {
        "schema_version": TRADE_PATH_METRICS_EVENT_SCHEMA_VERSION,
        "created_utc": created.isoformat(),
        "decision": TRADE_PATH_METRICS_DECISION,
        "failures": report["failures"],
        "candidate_bundle_sha256": candidate_sha,
        "replay_rows": replay_binding,
        "exit_trace_rows": trace_binding,
        "trades": trades_binding,
        "producer_source_files": source_files,
        "producer_source_inventory_sha256": source_inventory_sha256,
        "report": report,
        "production_authority_ready": False,
        "edge_claim_allowed": False,
    }
    event_path, event = write_immutable_json_event(
        root,
        TRADE_PATH_METRICS_EVENT_PREFIX,
        payload,
    )
    return event_path, event


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-rows", type=Path, required=True)
    parser.add_argument("--exit-trace-rows", type=Path, required=True)
    parser.add_argument("--candidate-bundle-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    event_path, event = materialize_unified_exit_trade_path_metrics(
        replay_rows_path=args.replay_rows,
        exit_trace_rows_path=args.exit_trace_rows,
        candidate_bundle_sha256=args.candidate_bundle_sha256,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "event_path": str(event_path),
                "event_sha256": _sha256_file(event_path),
                "decision": event["decision"],
                "production_authority_ready": False,
                "edge_claim_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
