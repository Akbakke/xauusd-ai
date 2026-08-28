#!/usr/bin/env python3
"""V12 OANDA practice data collector.

Polls XAU_USD M1 candles every 60s via OANDA practice API and appends to
daily parquet files. Designed to run continuously for 1-2+ weeks to build
a real live-data corpus for V12 shadow-replay.

Output:
    /home/andre2/GX1_DATA/reports/v12_live_data_strict_m1_v1/
        xauusd_m1_YYYYMMDD.parquet

Each file: time, open/high/low/close (mid), bid_open/.../close, ask_open/.../close, volume.
Companion script v12_shadow_replay.py runs V12 stack on collected data.

Run:
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 \\
        gx1/execution/v12_oanda_data_collector.py

Reads OANDA credentials from .env in repo root or env vars.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENV_FILE = REPO_ROOT / ".env"

from gx1.contracts.xau_tape_provenance_v1 import (  # noqa: E402
    CANONICAL_NATIVE_REQUIRED_COLUMNS,
    canonical_native_rows_sha256,
    validate_canonical_native_frame,
)
from gx1.contracts.gx1_scope_v1 import require_offline_scope  # noqa: E402
from gx1.execution.oanda_client import (  # noqa: E402
    OandaAPIError,
    OandaClient,
    OandaClientConfig,
    OandaDataContractError,
)
from gx1.execution.oanda_credentials import load_oanda_credentials  # noqa: E402

LOG = logging.getLogger("v12_collector")
INSTRUMENT = "XAU_USD"
# Poll cadence is env-overridable (no magic constant) — live SLA tuning. Default 60s
# (OANDA-friendly); set GX1_COLLECTOR_POLL_SECONDS=15 in the systemd unit to pick up a
# newly-closed M1 bar within ~15s instead of ~60s (tightens the M1 exit price reaction).
# Note: a *closed* M1 bar is inherently up to 60s old; sub-1-min reaction needs tick
# streaming (OANDA pricing/stream), not a faster poll — this only removes poll latency.
DEFAULT_POLL_SECONDS = 60
POLL_SECONDS = DEFAULT_POLL_SECONDS
FETCH_LOOKBACK = 30                    # poll the last 30 M1 candles each iteration
OUT_DIR = Path(
    "/home/andre2/GX1_DATA/reports/v12_live_data_strict_m1_v1"
)
COLLECTOR_LOCK_FILENAME = ".xauusd_m1_collector.lock"
COLLECTOR_FAILURE_LATCH_FILENAME = "COLLECTOR_CONTRACT_FAILURE_LATCH.json"


class CollectorDataContractError(RuntimeError):
    """A fetched or persisted candle violates the immutable M1 contract."""

    def __init__(
        self,
        message: str,
        *,
        evidence: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.evidence = dict(evidence or {})


def _validate_collector_frame(
    frame: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    try:
        return validate_canonical_native_frame(
            frame,
            timeframe="M1",
            label=label,
        )
    except RuntimeError as exc:
        raise CollectorDataContractError(str(exc)) from exc


def _candle_date_path(timestamp: pd.Timestamp) -> Path:
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        raise RuntimeError("COLLECTOR_CANDLE_TIME_TZ_MISSING")
    return OUT_DIR / f"xauusd_m1_{value.tz_convert('UTC').strftime('%Y%m%d')}.parquet"


def _merge_and_dedupe(existing: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge exact completed bars and reject any historical correction.

    A completed OANDA candle is immutable evidence.  ``keep="last"`` used to
    hide a conflicting value for an already persisted timestamp.  Exact
    lookback overlap is expected; any non-identical overlap now fails closed.
    """
    frames = [
        frame
        for frame in (existing, new_df)
        if frame is not None and len(frame) > 0
    ]
    if not frames:
        raise CollectorDataContractError("COLLECTOR_MERGE_EMPTY")
    combined = pd.concat(frames, ignore_index=True)
    if "time" not in combined.columns:
        raise CollectorDataContractError("COLLECTOR_TIME_COLUMN_MISSING")
    duplicate_times = combined.loc[
        combined.duplicated(subset=["time"], keep=False),
        "time",
    ].drop_duplicates()
    conflicts: list[dict[str, object]] = []

    def _rows_evidence(rows: pd.DataFrame) -> list[dict[str, object]]:
        evidence: list[dict[str, object]] = []
        for position in range(len(rows)):
            row = rows.iloc[[position]].loc[
                :, list(CANONICAL_NATIVE_REQUIRED_COLUMNS)
            ]
            payload = json.loads(
                row.to_json(
                    orient="records",
                    date_format="iso",
                    date_unit="ns",
                )
            )[0]
            evidence.append(
                {
                    "row_sha256": canonical_native_rows_sha256(
                        row,
                        timeframe="M1",
                    ),
                    "row": payload,
                }
            )
        return evidence

    for timestamp in duplicate_times:
        rows = combined.loc[combined["time"] == timestamp]
        combined_evidence = _rows_evidence(rows)
        row_hashes = {item["row_sha256"] for item in combined_evidence}
        if len(row_hashes) != 1:
            old_rows = existing.loc[existing["time"] == timestamp]
            incoming_rows = new_df.loc[new_df["time"] == timestamp]
            conflicts.append(
                {
                    "time_utc": pd.Timestamp(timestamp).isoformat(),
                    "existing": _rows_evidence(old_rows),
                    "incoming": _rows_evidence(incoming_rows),
                }
            )
    if conflicts:
        conflict_evidence: dict[str, object] = {
            "completed_bar_conflicts": conflicts[:10],
        }
        response_sha = new_df.attrs.get("source_response_sha256")
        if isinstance(response_sha, str):
            conflict_evidence["source_response_sha256"] = response_sha
        raise CollectorDataContractError(
            "COLLECTOR_COMPLETED_BAR_CONFLICT: "
            + ",".join(item["time_utc"] for item in conflicts[:10]),
            evidence=conflict_evidence,
        )
    combined = combined.drop_duplicates(subset=["time"], keep="first")
    combined = combined.sort_values("time").reset_index(drop=True)
    return combined


def _require_partition_date(frame: pd.DataFrame, out_path: Path) -> None:
    expected = out_path.stem.rsplit("_", 1)[-1]
    observed = frame["time"].dt.strftime("%Y%m%d")
    if not observed.eq(expected).all():
        raise CollectorDataContractError(
            "COLLECTOR_PARTITION_CANDLE_DATE_MISMATCH: "
            f"path={out_path.name} observed={sorted(observed.unique().tolist())}"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_contract_failure_latch(
    exc: BaseException,
    *,
    frame: pd.DataFrame | None = None,
) -> Path:
    """Persist the first contract failure and require explicit operator review."""

    latch_path = OUT_DIR / COLLECTOR_FAILURE_LATCH_FILENAME
    frame_payload: dict[str, object] | None = None
    if isinstance(frame, pd.DataFrame):
        try:
            encoded_frame = frame.to_json(
                orient="split",
                date_format="iso",
                date_unit="ns",
            ).encode("utf-8")
        except Exception:
            encoded_frame = repr(
                (list(frame.columns), frame.shape)
            ).encode("utf-8")
        frame_payload = {
            "rows": int(len(frame)),
            "columns": [str(name) for name in frame.columns],
            "diagnostic_sha256": hashlib.sha256(encoded_frame).hexdigest(),
        }
        if "time" in frame.columns and len(frame):
            parsed = pd.to_datetime(frame["time"], utc=True, errors="coerce")
            valid = parsed.dropna()
            frame_payload["time_min_utc"] = (
                valid.min().isoformat() if len(valid) else None
            )
            frame_payload["time_max_utc"] = (
                valid.max().isoformat() if len(valid) else None
            )
    exception_evidence = getattr(exc, "evidence", None)
    if not isinstance(exception_evidence, dict):
        exception_evidence = {}
    response_sha = (
        frame.attrs.get("source_response_sha256")
        if isinstance(frame, pd.DataFrame)
        else None
    )
    if isinstance(response_sha, str):
        exception_evidence = {
            **exception_evidence,
            "source_response_sha256": response_sha,
        }
    payload = {
        "schema_version": "gx1_collector_contract_failure_latch_v2",
        "created_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "instrument": INSTRUMENT,
        "granularity": "M1",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "contract_evidence": exception_evidence,
        "frame": frame_payload,
        "resolution": (
            "inspect source/partition evidence, repair explicitly, then archive "
            "this latch before restarting; automatic retry is forbidden"
        ),
    }
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(latch_path, flags, 0o600)
    except FileExistsError:
        return latch_path
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(OUT_DIR)
    return latch_path


def _write_parquet_atomic(frame: pd.DataFrame, out_path: Path) -> None:
    """Durably replace one mutable collector snapshot without torn readers."""

    frame = _validate_collector_frame(frame, label="COLLECTOR_WRITE")
    out_path = Path(out_path)
    parent = out_path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError(f"COLLECTOR_OUTPUT_ROOT_INVALID: {parent}")
    if out_path.is_symlink():
        raise RuntimeError(f"COLLECTOR_OUTPUT_SYMLINK_FORBIDDEN: {out_path}")
    temporary = parent / f".{out_path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            frame.to_parquet(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        observed = _validate_collector_frame(
            pd.read_parquet(temporary),
            label="COLLECTOR_WRITE_READBACK",
        )
        if canonical_native_rows_sha256(
            observed,
            timeframe="M1",
        ) != canonical_native_rows_sha256(frame, timeframe="M1"):
            raise CollectorDataContractError(
                "COLLECTOR_WRITE_READBACK_ROW_IDENTITY_MISMATCH"
            )
        os.replace(temporary, out_path)
        _fsync_directory(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            if temporary.is_symlink() or not temporary.is_file():
                raise RuntimeError(
                    f"COLLECTOR_TEMPORARY_PATH_INVALID: {temporary}"
                )
            temporary.unlink()
            _fsync_directory(parent)


@contextmanager
def _collector_process_lock():
    if OUT_DIR.is_symlink() or not OUT_DIR.is_dir():
        raise RuntimeError(f"COLLECTOR_OUTPUT_ROOT_INVALID: {OUT_DIR}")
    lock_path = OUT_DIR / COLLECTOR_LOCK_FILENAME
    if lock_path.is_symlink():
        raise RuntimeError(f"COLLECTOR_LOCK_SYMLINK_FORBIDDEN: {lock_path}")
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("COLLECTOR_PROCESS_ALREADY_RUNNING") from exc
        yield


def _persist_collected_batch(frame: pd.DataFrame) -> list[Path]:
    work = _validate_collector_frame(frame, label="COLLECTOR_FETCH")
    written: list[Path] = []
    for _date, daily in work.groupby(work["time"].dt.strftime("%Y%m%d"), sort=True):
        daily = daily.sort_values("time").reset_index(drop=True)
        out_path = _candle_date_path(pd.Timestamp(daily["time"].iloc[0]))
        _require_partition_date(daily, out_path)
        if out_path.exists():
            if out_path.is_symlink():
                raise RuntimeError(
                    f"COLLECTOR_OUTPUT_SYMLINK_FORBIDDEN: {out_path}"
                )
            existing = _validate_collector_frame(
                pd.read_parquet(out_path),
                label="COLLECTOR_EXISTING_PARTITION",
            )
            _require_partition_date(existing, out_path)
            daily = _merge_and_dedupe(existing, daily)
            daily = _validate_collector_frame(
                daily,
                label="COLLECTOR_MERGED_PARTITION",
            )
            _require_partition_date(daily, out_path)
        _write_parquet_atomic(daily, out_path)
        written.append(out_path)
    return written


def collect_loop(client: OandaClient) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_DIR.is_symlink() or not OUT_DIR.is_dir():
        raise RuntimeError(f"COLLECTOR_OUTPUT_ROOT_INVALID: {OUT_DIR}")
    LOG.info(f"V12 collector starting → {OUT_DIR}")
    LOG.info(f"  instrument={INSTRUMENT}  poll={POLL_SECONDS}s  fetch_lookback={FETCH_LOOKBACK}")

    consecutive_errors = 0
    while True:
        loop_start = time.time()
        try:
            df = client.get_candles(INSTRUMENT, granularity="M1", count=FETCH_LOOKBACK)
        except OandaDataContractError as exc:
            latch = _write_contract_failure_latch(exc)
            LOG.critical(
                "OANDA source contract failed; collector latched at %s",
                latch,
            )
            raise CollectorDataContractError(str(exc)) from exc
        except OandaAPIError as exc:
            consecutive_errors += 1
            LOG.error(f"poll failed (consecutive={consecutive_errors}): {exc}")
            # Exponential backoff up to 5 min
            backoff = min(POLL_SECONDS * (2 ** min(consecutive_errors, 6)), 300)
            time.sleep(backoff)
            continue
        # OANDA client returns time as index; normalize to the one persisted schema.
        if df.index.name == "time" or "time" not in df.columns:
            df = df.reset_index()
        if len(df) == 0:
            LOG.warning("OANDA returned 0 completed candles — sleeping")
        else:
            try:
                written = _persist_collected_batch(df)
            except Exception as exc:
                latch = _write_contract_failure_latch(exc, frame=df)
                LOG.critical(
                    "collector persistence contract failed; latched at %s",
                    latch,
                )
                raise
            latest_time = pd.to_datetime(df["time"].max())
            LOG.info(
                "persisted completed lookback latest=%s -> %s",
                latest_time.isoformat(),
                ",".join(path.name for path in written),
            )
        consecutive_errors = 0

        elapsed = time.time() - loop_start
        sleep_for = max(0, POLL_SECONDS - elapsed)
        time.sleep(sleep_for)


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Continuously collect XAU_USD M1 candles. This command accepts "
            "no operational arguments; unknown arguments fail before any "
            "credentials, network calls, or file writes."
        )
    )


def main(argv: list[str] | None = None) -> int:
    global POLL_SECONDS
    build_parser().parse_args(argv)
    # Parse first: --help and malformed invocations cannot load credentials or
    # touch the collector output. This closes the historical side effect where
    # even a harmless introspection call started the infinite collection loop.
    require_offline_scope("oanda_collector")
    if ENV_FILE.is_file():
        with ENV_FILE.open() as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key, value)
    POLL_SECONDS = int(
        os.environ.get(
            "GX1_COLLECTOR_POLL_SECONDS",
            str(DEFAULT_POLL_SECONDS),
        )
    )
    if POLL_SECONDS <= 0:
        raise RuntimeError("GX1_COLLECTOR_POLL_SECONDS must be a positive integer")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with _collector_process_lock():
        failure_latch = OUT_DIR / COLLECTOR_FAILURE_LATCH_FILENAME
        if failure_latch.exists() or failure_latch.is_symlink():
            raise RuntimeError(
                "COLLECTOR_CONTRACT_FAILURE_LATCHED: "
                f"{failure_latch}"
            )
        creds = load_oanda_credentials()
        LOG.info(
            "OANDA env=%s api_url=%s",
            creds.env,
            creds.api_url,
        )

        cfg = OandaClientConfig(
            api_key=creds.api_token,
            account_id=creds.account_id,
            env=creds.env,
        )
        client = OandaClient(cfg)

        # Sanity-test: pull 5 candles to verify connectivity
        try:
            df_test = client.get_candles(
                INSTRUMENT,
                granularity="M1",
                count=5,
            )
            LOG.info(
                "connectivity test OK — fetched %d candles, latest=%s",
                len(df_test),
                (
                    df_test["time"].max()
                    if "time" in df_test.columns
                    else df_test.index.max()
                ),
            )
        except OandaDataContractError as exc:
            latch = _write_contract_failure_latch(exc)
            LOG.critical(
                "connectivity response violated the candle contract; "
                "collector latched at %s",
                latch,
            )
            return 1
        except OandaAPIError as exc:
            LOG.error(f"connectivity test FAILED: {exc}")
            return 1

        try:
            collect_loop(client)
        except KeyboardInterrupt:
            LOG.info("interrupted — exiting")
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
