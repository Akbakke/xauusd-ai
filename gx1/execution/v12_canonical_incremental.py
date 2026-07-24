#!/usr/bin/env python3
"""V12 canonical/BASE28 pair updater with full-history feature equivalence.

Strategy
--------
For each new native, completed canonical M5 bar:
  1. Verify the one-owner canonical M5 manifest and every partition hash.
  2. Run build_canonical_v2 on the complete causal M5 history. Bounded
     warmup is forbidden because EWM, RSI, regime-age and structure state do
     not have proven finite-window equivalence.
  3. Keep only rows later than the admitted pair cutoff.
  3. Apply canonical_v3 augment to the new rows (drops 12 + adds 6).
  4. Apply add_ctx_cont logic to compute the 32 BASE34-style features
     for the new rows, using the existing BASE34 prebuilt's distribution
     for percentile-based features (vol_regime_id, atr_bucket, etc.).
  5. Write both complete candidates into an unpublished staging directory.
  6. Publish one immutable pair generation and atomically replace the single
     canonical-v3/BASE28 pair pointer. No individual artifact is ever activated.

This correctness-first implementation is intentionally not claimed to be
cheap. A future recursive-state accelerator must prove bit-equivalence against
this complete-history owner before it can replace the computation.

To run continuously:
    nohup python3 -u gx1/execution/v12_canonical_incremental.py --loop > log 2>&1 &
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import re
import shutil
import sys
import time as _time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.contracts.xau_tape_provenance_v1 import (  # noqa: E402
    CANONICAL_M5_REQUIRED_COLUMNS,
    canonical_xau_source_descriptor_v1,
)
from gx1.execution.v12_state_from_prebuilt import (  # noqa: E402
    PREBUILT_PAIR_MANIFEST_PATH,
    PREBUILT_PAIR_ROOT,
    PREBUILT_PAIR_SCHEMA_VERSION,
    inspect_prebuilt_artifact,
    pair_generation_id_for_artifacts,
    read_prebuilt_pair_manifest,
    verify_prebuilt_pair,
)
from gx1.execution.v12_m1_to_m5_downsample import (  # noqa: E402
    closed_m5_start_for_m1_bar_labels,
)
from gx1.features.htf_features import (  # noqa: E402
    REGIME_V4_V2_MTF_PER_TF,
    REGIME_V4_V2_MTF_SKIP,
    REGIME_V4_V2_MTF_TFS,
)
from gx1.features.basic_v1 import (  # noqa: E402
    PLUS5_FEATURES,
    compute_plus5_features,
)
from gx1.features.micro_structure_v1 import MICRO_FEATURE_NAMES_V1  # noqa: E402
from gx1.features.regime_v4_features import REGIME_V4_DERIVED_COLS  # noqa: E402
from gx1.features.swing_structure_v1 import SWING_FEATURE_NAMES_V1  # noqa: E402
from gx1.features.volume_features import VOLUME_FEATURE_NAMES  # noqa: E402
from gx1.scripts.materialize_build_canonical_features_v2 import (  # noqa: E402
    build_canonical_v2,
)
from gx1.scripts.materialize_canonical_v3_augment import (  # noqa: E402
    DROP_COLUMNS,
    add_cyclic_time_features,
    add_smc_premium_state_interaction,
    add_cross_tf_momentum,
)

LOG = logging.getLogger("v12_incr")

CANONICAL_M1_DIR = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL")
CANONICAL_M5_DIR = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL")
COLLECTOR_DIR = Path("/home/andre2/GX1_DATA/reports/v12_live_data")
PAIR_CANONICAL_FILENAME = "canonical_v3.parquet"
PAIR_BASE28_FILENAME = "base28.parquet"
PAIR_PUBLISH_LOCK_FILENAME = ".canonical_v3_base28_pair_publish.lock"
_PAIR_STAGING_NAME = re.compile(r"\.staging-[0-9a-f]{32}\Z")

# PLUS5: 5 features re-added on 2026-05-21 because the PLUS5 Entry-IQL ensemble
# was trained on real values.  This function is the retained computation source.
M1_MARKET_IDENTITY_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
)
# BASE28 is the M1-cadence lane.  Every observable M1 market field is therefore
# owned by native M1, never by a broadcast closed-M5 row with the same name.
BASE34_RAW_M1_OWNED_COLUMNS = M1_MARKET_IDENTITY_COLUMNS
_BASE34_V2_MTF_OWNED_COLUMNS = tuple(
    f"{timeframe}_{live_fragment}_v2"
    for timeframe in REGIME_V4_V2_MTF_TFS
    for live_fragment, _source_column in REGIME_V4_V2_MTF_PER_TF
    if (timeframe, live_fragment) not in REGIME_V4_V2_MTF_SKIP
)
BASE34_AUGMENT_OWNED_COLUMNS = frozenset(
    (
        "atr_bps",
        "spread_bps",
        "session_id",
        "is_ASIA",
        "_v1_is_EU",
        "_v1_is_US",
        "minutes_since_session_open",
        "minutes_to_next_session_boundary",
        "session_change_flag",
        "session_tradable",
        "_v1_int_ema_us",
        "_v1_int_range_us",
        "_v1_int_slope_h1_us",
        "trend_regime_id",
        "vol_regime_id",
        "atr_bucket",
        "spread_bucket",
        "D1_dist_from_ema200_atr",
        "D1_atr_percentile_252",
        "H1_range_compression_ratio",
        "M15_range_compression_ratio",
        "H4_trend_sign_cat",
        *MICRO_FEATURE_NAMES_V1,
        *SWING_FEATURE_NAMES_V1,
        *VOLUME_FEATURE_NAMES,
        *REGIME_V4_DERIVED_COLS,
        *_BASE34_V2_MTF_OWNED_COLUMNS,
    )
)


def _align_exact_canonical_schema(
    existing: pd.DataFrame,
    incremental: pd.DataFrame,
) -> pd.DataFrame:
    """Return incremental columns in canonical order or reject any schema drift."""

    existing_columns = list(existing.columns)
    incremental_columns = list(incremental.columns)
    if len(existing_columns) != len(set(existing_columns)):
        raise RuntimeError("canonical_v3 existing schema contains duplicate columns")
    if len(incremental_columns) != len(set(incremental_columns)):
        raise RuntimeError("canonical_v3 incremental schema contains duplicate columns")
    existing_set = set(existing_columns)
    incremental_set = set(incremental_columns)
    missing_columns = sorted(existing_set - incremental_set)
    extra_columns = sorted(incremental_set - existing_set)
    if missing_columns or extra_columns:
        raise RuntimeError(
            "canonical_v3 incremental schema mismatch: "
            f"missing={missing_columns} extra={extra_columns}"
        )
    return incremental.loc[:, existing_columns]


def _build_base34_owned_frame(
    *,
    output_columns: list[str],
    cv3: pd.DataFrame,
    cv3_aug: pd.DataFrame,
    m1: pd.DataFrame,
) -> pd.DataFrame:
    """Build an M1-cadence BASE28 frame from one exact owner per column.

    The mapping is vectorized because a full-history bootstrap contains
    millions of M1 rows.  It retains the same fail-closed ownership semantics
    as the former per-row loop while avoiding Python work per row and field.
    """

    if len(output_columns) != len(set(output_columns)):
        raise RuntimeError("BASE34 output schema contains duplicate columns")
    for label, frame in (("M1", m1), ("canonical_v3", cv3), ("augmented", cv3_aug)):
        if (
            not isinstance(frame.index, pd.DatetimeIndex)
            or frame.index.hasnans
            or not frame.index.is_unique
            or not frame.index.is_monotonic_increasing
        ):
            raise RuntimeError(f"BASE34 {label} index is not exact chronological UTC")
    closed_keys = closed_m5_start_for_m1_bar_labels(m1.index)
    missing_canonical = closed_keys[~closed_keys.isin(cv3.index)]
    if len(missing_canonical):
        raise RuntimeError(
            "BASE34 append lacks exact closed M5 state at "
            f"{missing_canonical[0]}"
        )
    missing_augmented = closed_keys[~closed_keys.isin(cv3_aug.index)]
    if len(missing_augmented):
        raise RuntimeError(
            "BASE34 augmented source lacks exact closed M5 state at "
            f"{missing_augmented[0]}"
        )

    canonical_aligned = cv3.loc[closed_keys].copy(deep=False)
    canonical_aligned.index = m1.index
    augmented_aligned = cv3_aug.loc[closed_keys].copy(deep=False)
    augmented_aligned.index = m1.index
    output: dict[str, np.ndarray] = {}
    for column in output_columns:
        if column in BASE34_RAW_M1_OWNED_COLUMNS:
            if column not in m1.columns:
                raise RuntimeError(f"BASE34 exact M1 source lacks {column}")
            source = m1[column]
            context = f"BASE34 M1.{column}"
        elif column == "is_model_bar":
            # An M1 row labelled xx:04/09/... closes at xx:05/10/... and is
            # the first row that can observe the just-completed M5 bucket.
            output[column] = (m1.index.minute.to_numpy() % 5 == 4).astype(
                np.float64
            )
            continue
        elif column in BASE34_AUGMENT_OWNED_COLUMNS:
            if column not in augmented_aligned.columns:
                raise RuntimeError(
                    f"BASE34 augment-owned column {column!r} lacks its exact producer"
                )
            source = augmented_aligned[column]
            context = f"BASE34 augmented.{column}"
        elif column in canonical_aligned.columns:
            source = canonical_aligned[column]
            context = f"BASE34 canonical_v3.{column}"
        else:
            raise RuntimeError(
                f"BASE34 column {column!r} has no exact current-bar producer"
            )
        if pd.api.types.is_bool_dtype(source.dtype):
            raise RuntimeError(f"{context}: boolean is not numeric feature evidence")
        try:
            values = pd.to_numeric(source, errors="raise").to_numpy(
                dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{context}: feature value is not numeric") from exc
        if values.shape != (len(m1),) or not np.isfinite(values).all():
            raise RuntimeError(f"{context}: feature value is non-finite")
        output[column] = values
    return pd.DataFrame(output, index=m1.index)


def _compute_plus5_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility call-site delegating to the basic_v1 PLUS5 owner."""
    return compute_plus5_features(df)


def _fsync_file(path: Path) -> None:
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _candidate_staging_path(generation_root: Path) -> Path:
    """Reserve a unique unpublished path without creating an empty cycle artifact."""
    generation_root = Path(generation_root)
    generation_root.mkdir(parents=True, exist_ok=True)
    if generation_root.is_symlink() or not generation_root.is_dir():
        raise RuntimeError(f"pair generation root is invalid: {generation_root}")
    generation_root = generation_root.resolve(strict=True)
    return generation_root / f".staging-{uuid.uuid4().hex}"


def _discard_pair_staging_dir(staging_dir: Path, *, generation_root: Path) -> None:
    """Delete only this process's exact unpublished staging directory."""
    generation_root = Path(generation_root).resolve(strict=True)
    staging_dir = Path(staging_dir)
    candidate = staging_dir.absolute()
    if (
        candidate.parent != generation_root
        or _PAIR_STAGING_NAME.fullmatch(candidate.name) is None
    ):
        raise RuntimeError(f"refusing unsafe pair staging cleanup: {staging_dir}")
    if not candidate.exists():
        if candidate.is_symlink():
            raise RuntimeError(f"refusing symlink pair staging cleanup: {candidate}")
        return
    if candidate.is_symlink() or not candidate.is_dir():
        raise RuntimeError(f"refusing non-directory pair staging cleanup: {candidate}")
    allowed = {PAIR_CANONICAL_FILENAME, PAIR_BASE28_FILENAME}
    entries = list(candidate.iterdir())
    if any(
        item.name not in allowed or item.is_symlink() or not item.is_file()
        for item in entries
    ):
        raise RuntimeError(
            f"refusing pair staging cleanup with unexpected contents: {candidate}"
        )
    shutil.rmtree(candidate)
    _fsync_directory(generation_root)


def _discard_unpublished_generation_dir(
    generation_dir: Path,
    *,
    generation_root: Path,
    pair_manifest_path: Path,
    pair_generation_id: str,
) -> None:
    """Delete only a generation created by this failed, pre-pointer publication."""
    generation_root = Path(generation_root).resolve(strict=True)
    generation_dir = Path(generation_dir).absolute()
    if (
        generation_dir.parent != generation_root
        or generation_dir.name != pair_generation_id
        or len(pair_generation_id) != 64
        or any(char not in "0123456789abcdef" for char in pair_generation_id)
    ):
        raise RuntimeError(
            f"refusing unsafe unpublished generation cleanup: {generation_dir}"
        )
    if generation_dir.is_symlink() or not generation_dir.is_dir():
        raise RuntimeError(
            f"refusing invalid unpublished generation cleanup: {generation_dir}"
        )
    if set(item.name for item in generation_dir.iterdir()) != {
        PAIR_CANONICAL_FILENAME,
        PAIR_BASE28_FILENAME,
    }:
        raise RuntimeError(
            f"refusing unpublished generation cleanup with unexpected contents: "
            f"{generation_dir}"
        )
    artifacts = {
        "canonical_v3": inspect_prebuilt_artifact(
            generation_dir / PAIR_CANONICAL_FILENAME,
            label="canonical_v3",
        ),
        "base28": inspect_prebuilt_artifact(
            generation_dir / PAIR_BASE28_FILENAME,
            label="base28",
        ),
    }
    if pair_generation_id_for_artifacts(artifacts) != pair_generation_id:
        raise RuntimeError(
            f"refusing unpublished generation cleanup after identity mismatch: "
            f"{generation_dir}"
        )
    pair_manifest_path = Path(pair_manifest_path)
    if pair_manifest_path.exists() or pair_manifest_path.is_symlink():
        current = read_prebuilt_pair_manifest(
            pair_manifest_path,
            generation_root=generation_root,
        )
        if current.pair_generation_id == pair_generation_id:
            raise RuntimeError(
                f"refusing cleanup of published pair generation: {generation_dir}"
            )
    shutil.rmtree(generation_dir)
    _fsync_directory(generation_root)


def _write_candidate_parquet(
    frame: pd.DataFrame,
    path: Path,
    *,
    index: bool,
) -> None:
    """Write one unpublished candidate; partial staging is never served."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite pair candidate: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=index)
    _fsync_file(path)
    _fsync_directory(path.parent)


def _copy_candidate_parquet(source: Path, path: Path) -> None:
    """Copy an unchanged pair member into staging without rewriting its schema."""
    source = Path(source)
    path = Path(path)
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"pair source is not an exact regular file: {source}")
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite pair candidate: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, path)
    _fsync_file(path)
    _fsync_directory(path.parent)


def _publish_prebuilt_pair_generation(
    staging_dir: Path,
    *,
    pair_manifest_path: Path,
    generation_root: Path,
    expected_pair_generation_id: str | None,
    expected_manifest_sha256: str | None,
    created_utc: str | None = None,
) -> str:
    """Publish a complete immutable pair through one atomic pointer replacement.

    Artifact computation is complete before this function begins. Under the
    publisher lock it verifies the previously admitted pointer, renames the
    complete staging directory to its content-derived generation id, fsyncs it,
    and only then replaces the one serving pointer. A failure before the pointer
    replacement leaves the previous generation active.
    """
    staging_dir = Path(staging_dir)
    pair_manifest_path = Path(pair_manifest_path)
    generation_root = Path(generation_root)
    if generation_root.is_symlink() or not generation_root.is_dir():
        raise RuntimeError(f"pair generation root is invalid: {generation_root}")
    generation_root = generation_root.resolve(strict=True)
    try:
        staging_resolved = staging_dir.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"pair staging directory is missing: {staging_dir}") from exc
    if (
        staging_dir.is_symlink()
        or not staging_resolved.is_dir()
        or staging_resolved.parent != generation_root
        or _PAIR_STAGING_NAME.fullmatch(staging_resolved.name) is None
    ):
        raise RuntimeError(f"pair staging directory is not exact: {staging_dir}")

    canonical_candidate = staging_resolved / PAIR_CANONICAL_FILENAME
    base28_candidate = staging_resolved / PAIR_BASE28_FILENAME
    if set(item.name for item in staging_resolved.iterdir()) != {
        PAIR_CANONICAL_FILENAME,
        PAIR_BASE28_FILENAME,
    }:
        raise RuntimeError("pair staging directory must contain exactly two artifacts")
    artifacts = {
        "canonical_v3": inspect_prebuilt_artifact(
            canonical_candidate,
            label="canonical_v3",
        ),
        "base28": inspect_prebuilt_artifact(base28_candidate, label="base28"),
    }
    pair_generation_id = pair_generation_id_for_artifacts(artifacts)
    final_dir = generation_root / pair_generation_id

    pair_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_parent = pair_manifest_path.parent.resolve(strict=True)
    if pair_manifest_path.is_symlink() or manifest_parent != pair_manifest_path.parent:
        raise RuntimeError(f"pair manifest path is not exact: {pair_manifest_path}")
    lock_path = manifest_parent / PAIR_PUBLISH_LOCK_FILENAME
    if lock_path.is_symlink():
        raise RuntimeError(f"pair publish lock path is not exact: {lock_path}")
    with lock_path.open("a+b") as lock_handle:
        if lock_path.is_symlink() or not lock_path.is_file():
            raise RuntimeError(f"pair publish lock path is not exact: {lock_path}")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if expected_pair_generation_id is None or expected_manifest_sha256 is None:
            if pair_manifest_path.exists() or pair_manifest_path.is_symlink():
                raise RuntimeError(
                    "pair bootstrap refused because an active pointer already exists"
                )
        else:
            current = read_prebuilt_pair_manifest(
                pair_manifest_path,
                generation_root=generation_root,
            )
            if (
                current.pair_generation_id != expected_pair_generation_id
                or current.manifest_sha256 != expected_manifest_sha256
            ):
                raise RuntimeError(
                    "active pair changed during candidate computation; "
                    "refusing stale publication"
                )

        created_final_dir = False
        if final_dir.exists() or final_dir.is_symlink():
            if final_dir.is_symlink() or not final_dir.is_dir():
                raise RuntimeError(
                    f"immutable pair generation path is invalid: {final_dir}"
                )
            if set(item.name for item in final_dir.iterdir()) != {
                PAIR_CANONICAL_FILENAME,
                PAIR_BASE28_FILENAME,
            }:
                raise RuntimeError(
                    f"immutable pair generation contents are invalid: {final_dir}"
                )
            existing_artifacts = {
                "canonical_v3": inspect_prebuilt_artifact(
                    final_dir / PAIR_CANONICAL_FILENAME,
                    label="canonical_v3",
                ),
                "base28": inspect_prebuilt_artifact(
                    final_dir / PAIR_BASE28_FILENAME,
                    label="base28",
                ),
            }
            if pair_generation_id_for_artifacts(existing_artifacts) != pair_generation_id:
                raise RuntimeError(
                    f"immutable pair generation identity collision: {final_dir}"
                )
            _discard_pair_staging_dir(
                staging_resolved,
                generation_root=generation_root,
            )
        else:
            os.rename(staging_resolved, final_dir)
            created_final_dir = True
            _fsync_directory(generation_root)

        pointer_replaced = False
        try:
            published_artifacts: dict[str, dict[str, object]] = {}
            for label, filename in (
                ("canonical_v3", PAIR_CANONICAL_FILENAME),
                ("base28", PAIR_BASE28_FILENAME),
            ):
                contract = dict(artifacts[label])
                contract["parquet_path"] = str(
                    (final_dir / filename).resolve(strict=True)
                )
                published_artifacts[label] = contract
            manifest = {
                "schema_version": PREBUILT_PAIR_SCHEMA_VERSION,
                "pair_generation_id": pair_generation_id,
                "created_utc": created_utc
                or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "artifacts": published_artifacts,
            }
            pointer_tmp = manifest_parent / (
                f".{pair_manifest_path.name}.{uuid.uuid4().hex}.tmp"
            )
            encoded = (
                json.dumps(manifest, sort_keys=True, indent=2) + "\n"
            ).encode("utf-8")
            try:
                with pointer_tmp.open("xb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(pointer_tmp, pair_manifest_path)
                pointer_replaced = True
            finally:
                if pointer_tmp.exists():
                    if pointer_tmp.is_symlink() or not pointer_tmp.is_file():
                        raise RuntimeError(
                            "refusing unsafe pair pointer temp cleanup: "
                            f"{pointer_tmp}"
                        )
                    pointer_tmp.unlink()
                    _fsync_directory(manifest_parent)
            _fsync_directory(manifest_parent)

            admitted = read_prebuilt_pair_manifest(
                pair_manifest_path,
                generation_root=generation_root,
            )
            if admitted.pair_generation_id != pair_generation_id:
                raise RuntimeError(
                    "published pair pointer failed identity re-admission"
                )
            verify_prebuilt_pair(admitted)
        except Exception:
            if created_final_dir and not pointer_replaced:
                _discard_unpublished_generation_dir(
                    final_dir,
                    generation_root=generation_root,
                    pair_manifest_path=pair_manifest_path,
                    pair_generation_id=pair_generation_id,
                )
            raise
    return pair_generation_id


def bootstrap_prebuilt_pair(
    *,
    canonical_v3_path: Path,
    base28_path: Path,
    pair_manifest_path: Path = PREBUILT_PAIR_MANIFEST_PATH,
    generation_root: Path = PREBUILT_PAIR_ROOT,
) -> str:
    """Create the first pair pointer from two explicit artifacts.

    This is the sole bootstrap control route and is never called implicitly by
    ``run_one_cycle``. It succeeds only while no pair pointer exists; once a
    pointer exists every producer advance must use its exact generation id and
    manifest hash as the compare-before-publish contract.
    """
    staging_dir = _candidate_staging_path(generation_root)
    try:
        _copy_candidate_parquet(
            canonical_v3_path,
            staging_dir / PAIR_CANONICAL_FILENAME,
        )
        _copy_candidate_parquet(
            base28_path,
            staging_dir / PAIR_BASE28_FILENAME,
        )
        return _publish_prebuilt_pair_generation(
            staging_dir,
            pair_manifest_path=pair_manifest_path,
            generation_root=generation_root,
            expected_pair_generation_id=None,
            expected_manifest_sha256=None,
        )
    finally:
        _discard_pair_staging_dir(
            staging_dir,
            generation_root=generation_root,
        )


def _coerce_time_col(df: pd.DataFrame) -> pd.DataFrame | None:
    """Return df with a usable 'time' COLUMN, or None for a torn/malformed read.
    Guards the 2026-06-17 KeyError:'time' race: a collector/canonical M1 parquet read
    mid-write (or with time stored as the index) lacks a 'time' column → returning None
    makes the caller skip that file THIS cycle; the next 15s cycle re-reads the completed
    file. Non-blocking, fail-safe (never fabricates data)."""
    if "time" in df.columns:
        return df
    if df.index.name == "time" or isinstance(df.index, pd.DatetimeIndex):
        return df.reset_index().rename(columns={df.index.name or "index": "time"})
    return None


def _load_m1_collector_for_window(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Union of canonical M1 tape + live collector parquets covering [start, end]."""
    parts: list[pd.DataFrame] = []
    for fp in sorted(COLLECTOR_DIR.glob("xauusd_m1_*.parquet")):
        try:
            df = pd.read_parquet(fp)
        except Exception as exc:
            raise RuntimeError(f"live M1 source is unreadable: {fp}") from exc
        df = _coerce_time_col(df)
        if df is None:
            raise RuntimeError(f"live M1 source lacks exact time: {fp}")
        df["time"] = pd.to_datetime(df["time"], utc=True)
        sub = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)]
        if len(sub) > 0:
            parts.append(sub)
    for yr in range(start_ts.year, end_ts.year + 1):
        fp = CANONICAL_M1_DIR / f"year={yr}" / "part-000.parquet"
        if not fp.exists():
            continue
        try:
            df = pd.read_parquet(fp)
        except Exception as exc:
            raise RuntimeError(f"canonical M1 source is unreadable: {fp}") from exc
        df = _coerce_time_col(df)
        if df is None:
            raise RuntimeError(f"canonical M1 source lacks exact time: {fp}")
        df["time"] = pd.to_datetime(df["time"], utc=True)
        sub = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)]
        if len(sub) > 0:
            parts.append(sub)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    duplicate_rows = combined[combined.duplicated(subset=["time"], keep=False)]
    identity_columns = [
        column
        for column in M1_MARKET_IDENTITY_COLUMNS
        if column in combined.columns
    ]
    if len(duplicate_rows) and not identity_columns:
        raise RuntimeError("overlapping M1 sources lack market identity columns")
    if len(duplicate_rows):
        numeric_identity = duplicate_rows[identity_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if not np.isfinite(
            numeric_identity.to_numpy(dtype=np.float64)
        ).all():
            raise RuntimeError("overlapping M1 source has non-finite market values")
        distinct = (
            pd.concat(
                [duplicate_rows[["time"]].reset_index(drop=True), numeric_identity.reset_index(drop=True)],
                axis=1,
            )
            .groupby("time", sort=False)[identity_columns]
            .nunique(dropna=False)
        )
        conflicts = distinct.columns[(distinct > 1).any(axis=0)].tolist()
        if conflicts:
            first_conflict = distinct.index[(distinct[conflicts] > 1).any(axis=1)][0]
            raise RuntimeError(
                "canonical/live M1 source conflict at "
                f"{first_conflict}: columns={conflicts}"
            )
    return (
        combined.drop_duplicates(subset=["time"], keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )


def _load_full_canonical_m5(end_ts: pd.Timestamp) -> pd.DataFrame:
    """Load only hash-bound native M5 rows known complete by ``end_ts``."""

    canonical_xau_source_descriptor_v1(CANONICAL_M5_DIR, timeframe="M5")
    parts = sorted(CANONICAL_M5_DIR.glob("year=*/part-000.parquet"))
    if not parts:
        raise RuntimeError("canonical native M5 source has no year partitions")
    frames: list[pd.DataFrame] = []
    required = list(CANONICAL_M5_REQUIRED_COLUMNS)
    for path in parts:
        try:
            frame = pd.read_parquet(path, columns=required)
        except Exception as exc:
            raise RuntimeError(
                f"canonical native M5 partition is unreadable: {path}"
            ) from exc
        frames.append(frame)
    m5 = pd.concat(frames, ignore_index=True)
    m5["time"] = pd.to_datetime(m5["time"], utc=True, errors="coerce")
    if m5["time"].isna().any():
        raise RuntimeError("canonical native M5 contains invalid timestamps")
    if m5["time"].duplicated().any():
        duplicate = m5.loc[m5["time"].duplicated(keep=False), "time"].iloc[0]
        raise RuntimeError(
            f"canonical native M5 contains duplicate timestamp: {duplicate}"
        )
    m5 = m5.sort_values("time", kind="mergesort").set_index("time")
    end = pd.Timestamp(end_ts)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    latest_complete_start = end.floor("5min") - pd.Timedelta(minutes=5)
    m5 = m5.loc[m5.index <= latest_complete_start]
    if m5.empty:
        raise RuntimeError(
            "canonical native M5 has no completed rows at decision time"
        )
    numeric_columns = required[1:]
    numeric = m5[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("canonical native M5 contains non-finite market data")
    return numeric


def _require_existing_cv3_market_identity(
    cv3: pd.DataFrame,
    m5: pd.DataFrame,
) -> None:
    """Reject append onto a pair built from different historical market bytes."""

    identity_columns = ("open", "high", "low", "close", "volume")
    missing_cv3 = [name for name in identity_columns if name not in cv3.columns]
    missing_m5 = [name for name in identity_columns if name not in m5.columns]
    if missing_cv3 or missing_m5:
        raise RuntimeError(
            "canonical-v3/native-M5 identity columns missing: "
            f"cv3={missing_cv3} m5={missing_m5}"
        )
    common = cv3.index.intersection(m5.index)
    if common.empty or len(common) != len(cv3.index):
        raise RuntimeError(
            "canonical-v3 history is not completely covered by native M5: "
            f"cv3_rows={len(cv3)} common_rows={len(common)}"
        )
    cv3_values = cv3.loc[common, list(identity_columns)].to_numpy(
        dtype=np.float64
    )
    m5_values = m5.loc[common, list(identity_columns)].to_numpy(
        dtype=np.float64
    )
    equal = np.isclose(cv3_values, m5_values, rtol=0.0, atol=0.0)
    if not bool(equal.all()):
        row_index, column_index = np.argwhere(~equal)[0]
        raise RuntimeError(
            "canonical-v3/native-M5 market identity mismatch: "
            f"time={common[int(row_index)]} "
            f"column={identity_columns[int(column_index)]}"
        )


def _apply_canonical_v3_augment(v2: pd.DataFrame) -> pd.DataFrame:
    v3 = v2.copy()
    if "time" in v3.columns and not isinstance(v3.index, pd.DatetimeIndex):
        v3["time"] = pd.to_datetime(v3["time"], utc=True)
        v3 = v3.set_index("time")
    to_drop = [c for c in DROP_COLUMNS if c in v3.columns]
    v3 = v3.drop(columns=to_drop)
    v3 = add_cyclic_time_features(v3)
    v3 = add_smc_premium_state_interaction(v3)
    v3 = add_cross_tf_momentum(v3)
    return v3


def update_canonical_v3_incremental(
    *,
    source_path: Path,
    output_path: Path,
) -> tuple[int, pd.Timestamp | None]:
    """Compute an extended canonical-v3 candidate at an unpublished path.

    Returns (n_appended, new_cutoff_ts). n_appended=0 means nothing new.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    if source_path.is_symlink() or not source_path.is_file():
        raise RuntimeError(f"canonical_v3 pair source missing: {source_path}")
    if output_path.resolve(strict=False) == source_path.resolve(strict=True):
        raise RuntimeError("canonical_v3 candidate cannot overwrite its source")
    # Load existing prebuilt (full file — ~200 MB, ~1 sec).
    t0 = _time.perf_counter()
    cv3 = pd.read_parquet(source_path)
    if "time" in cv3.columns:
        cv3["time"] = pd.to_datetime(cv3["time"], utc=True)
        cv3 = cv3.set_index("time")
    cv3 = cv3.sort_index()
    last_in_prebuilt = cv3.index[-1]

    now_ts = pd.Timestamp.now(tz="UTC").floor("min")
    m5 = _load_full_canonical_m5(now_ts)
    _require_existing_cv3_market_identity(cv3, m5)

    # Identify NEW M5 bars (post-prebuilt-cutoff)
    new_m5 = m5[m5.index > last_in_prebuilt]
    if new_m5.empty:
        return 0, last_in_prebuilt

    LOG.info(f"new M5 bars to append: {len(new_m5)}  (range {new_m5.index[0]} → {new_m5.index[-1]})")

    # Recompute from complete native M5 history. No finite warmup window is
    # assumed equivalent to training for recursive/age-dependent families.
    v2 = build_canonical_v2(m5.rename_axis("time").reset_index())
    # Apply v3 augment
    v3_new = _apply_canonical_v3_augment(v2)
    # PLUS5 uses the same complete native M5 history.
    plus5_df = _compute_plus5_features(m5[["open", "high", "low", "close", "volume"]])
    for c in PLUS5_FEATURES:
        v3_new[c] = plus5_df[c].reindex(v3_new.index).astype(np.float32)
    # Take only the new bars
    v3_new = v3_new[v3_new.index > last_in_prebuilt]
    if v3_new.empty:
        LOG.warning("v3 augment produced no new rows")
        return 0, last_in_prebuilt

    # Align columns with the existing immutable schema. Missing values cannot
    # be manufactured as zeros because every field can influence the models.
    # PLUS5 cols are added to v3_new above; they will survive the alignment if
    # cv3 has them (after one-shot backfill).
    v3_new = _align_exact_canonical_schema(cv3, v3_new)
    numeric = v3_new.apply(pd.to_numeric, errors="coerce")
    invalid_columns = [
        column
        for column in numeric.columns
        if not np.isfinite(numeric[column].to_numpy(dtype=np.float64)).all()
    ]
    if invalid_columns:
        raise RuntimeError(
            "canonical_v3 incremental output contains non-finite features: "
            f"{invalid_columns}"
        )

    # Concat + write into unpublished pair staging. Publication is a separate
    # operation and is the only place active serving identity can change.
    cv3_extended = pd.concat([cv3, v3_new])
    _write_candidate_parquet(
        cv3_extended.reset_index(),
        output_path,
        index=False,
    )

    new_cutoff = v3_new.index[-1]
    elapsed = _time.perf_counter() - t0
    LOG.info(f"canonical_v3 extended +{len(v3_new)} bars in {elapsed*1000:.0f} ms  "
              f"new cutoff: {new_cutoff}")
    return len(v3_new), new_cutoff


def update_base34_incremental(
    new_cutoff: pd.Timestamp,
    *,
    source_base28_path: Path,
    canonical_v3_path: Path,
    output_path: Path,
) -> int:
    """Compute a BASE28/BASE34 candidate from the matching canonical candidate."""
    source_base28_path = Path(source_base28_path)
    canonical_v3_path = Path(canonical_v3_path)
    output_path = Path(output_path)
    for label, source in (
        ("BASE28", source_base28_path),
        ("canonical_v3", canonical_v3_path),
    ):
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"{label} pair source missing: {source}")
    if output_path.resolve(strict=False) in {
        source_base28_path.resolve(strict=True),
        canonical_v3_path.resolve(strict=True),
    }:
        raise RuntimeError("BASE28 candidate cannot overwrite a pair source")

    t0 = _time.perf_counter()
    base34 = pd.read_parquet(source_base28_path)
    if not isinstance(base34.index, pd.DatetimeIndex):
        if "time" in base34.columns:
            base34["time"] = pd.to_datetime(base34["time"], utc=True)
            base34 = base34.set_index("time")
    base34 = base34.sort_index()
    last_in_base34 = base34.index[-1]

    if new_cutoff <= last_in_base34:
        _copy_candidate_parquet(source_base28_path, output_path)
        return 0

    # Load M1 bars from [last_in_base34 + 1min, new_cutoff + 5min]
    # The +5min on new_cutoff is to include the M1 bars within the closing M5 bucket
    start_ts = last_in_base34 + pd.Timedelta(minutes=1)
    end_ts = new_cutoff + pd.Timedelta(minutes=5)
    m1 = _load_m1_collector_for_window(start_ts, end_ts)
    if m1.empty:
        raise RuntimeError("BASE34 append lacks exact M1 source rows")
    m1["time"] = pd.to_datetime(m1["time"], utc=True)
    m1 = m1.set_index("time").sort_index()
    new_m1 = m1[(m1.index > last_in_base34) & (m1.index <= end_ts)]
    if new_m1.empty:
        raise RuntimeError("BASE34 append has no new M1 rows for advanced cv3 cutoff")

    # Use the LATEST M5-aligned feature values from canonical_v3 as the
    # ffill-source for each new M1 bar (no lookahead — uses just-closed
    # M5 bar's features for the next 5 M1 bars).
    cv3 = pd.read_parquet(canonical_v3_path)
    if "time" in cv3.columns:
        cv3["time"] = pd.to_datetime(cv3["time"], utc=True)
        cv3 = cv3.set_index("time")
    cv3 = cv3.sort_index()

    # 2026-06-11 FREEZE FIX: the 37 BASE34-only columns (32 ctx-augment features + session/regime/
    # swing/micro flags + is_model_bar) used to be COPY-FORWARDED from the last base34 row on every
    # append → ALL of them froze from 2026-05-25 18:25 (session pinned US, atr_bps 5.348, vol MEDIUM,
    # trend NEUTRAL — journal-confirmed all the way into the live entry/exit state vectors). They are
    # now RECOMPUTED per cycle via the ONE-TRUTH live augmenter
    # (v12_ctx_augment_live.augment_canonical_v3). Full canonical history is
    # mandatory: bounded windows reset EWM, swing/BOS/CHOCH, regime-age and D1
    # trend-age state and are not state-equivalent to training.
    from gx1.execution.v12_ctx_augment_live import augment_canonical_v3
    cv3_win = cv3.copy()
    # REGIME_V4 (2026-06-13 cutover): attach the per-TF V2 multi-TF scalars REGIME_V4 needs as
    # inputs (R1/R2/R3) BEFORE augment, via the ONE-TRUTH helper shared with serve + build. Without
    # this, augment_canonical_v3's REGIME_V4 block (GX1_REGIME_V4=1) is fail-closed-missing and the
    # 52 regime cols carry-forward FROZEN on append (the 2026-05-25 freeze class). This recompute is
    # the per-append cost driver (~17s @ 420d) but runs only when a new M5 closes (run_one_cycle gates
    # update_base34_incremental on n_cv3>0 — every 5 min), and the write is atomic.
    from gx1.features.htf_features import attach_default_regime_v4_v2_scalars
    attach_default_regime_v4_v2_scalars(cv3_win)
    _m5_cols = [c for c in ("open", "high", "low", "close", "volume") if c in cv3_win.columns]
    cv3_aug = augment_canonical_v3(cv3_win, cv3_win[_m5_cols].copy())
    base34_cols = list(base34.columns)
    output_columns = list(dict.fromkeys([*base34_cols, *BASE34_RAW_M1_OWNED_COLUMNS]))
    # Every field has exactly one owner. Native M1 owns all 13 observable M1
    # market values, recomputed context/HTF/regime fields come from the
    # augmenter even when a stale column with the same name exists in cv3, and
    # all remaining fields come from canonical_v3.
    new_df = _build_base34_owned_frame(
        output_columns=output_columns,
        cv3=cv3,
        cv3_aug=cv3_aug,
        m1=new_m1,
    )
    new_df.index.name = "time"
    extended = pd.concat([base34, new_df])

    _write_candidate_parquet(extended, output_path, index=True)

    elapsed = _time.perf_counter() - t0
    LOG.info(f"BASE34 extended +{len(new_df)} M1 rows in {elapsed*1000:.0f} ms  "
              f"new cutoff: {extended.index[-1]}")
    return len(new_df)


def run_one_cycle(
    *,
    pair_manifest_path: Path = PREBUILT_PAIR_MANIFEST_PATH,
    generation_root: Path = PREBUILT_PAIR_ROOT,
) -> dict:
    """Compute from one admitted pair and publish the next pair atomically."""
    t0 = _time.perf_counter()
    current = read_prebuilt_pair_manifest(
        pair_manifest_path,
        generation_root=generation_root,
    )
    verify_prebuilt_pair(current)
    staging_dir = _candidate_staging_path(generation_root)
    canonical_candidate = staging_dir / PAIR_CANONICAL_FILENAME
    base28_candidate = staging_dir / PAIR_BASE28_FILENAME
    try:
        n_cv3, new_cv3_cutoff = update_canonical_v3_incremental(
            source_path=current.canonical_v3.parquet_path,
            output_path=canonical_candidate,
        )
        n_base34 = 0
        published_generation_id = current.pair_generation_id
        if n_cv3 > 0:
            if new_cv3_cutoff is None or not canonical_candidate.is_file():
                raise RuntimeError(
                    "canonical updater reported progress without a complete candidate"
                )
            n_base34 = update_base34_incremental(
                new_cv3_cutoff,
                source_base28_path=current.base28.parquet_path,
                canonical_v3_path=canonical_candidate,
                output_path=base28_candidate,
            )
            if not base28_candidate.is_file():
                raise RuntimeError(
                    "BASE28 updater returned without a complete pair candidate"
                )
            published_generation_id = _publish_prebuilt_pair_generation(
                staging_dir,
                pair_manifest_path=pair_manifest_path,
                generation_root=generation_root,
                expected_pair_generation_id=current.pair_generation_id,
                expected_manifest_sha256=current.manifest_sha256,
            )
        elapsed = _time.perf_counter() - t0
        return {
            "cv3_appended": n_cv3,
            "base34_appended": n_base34,
            "new_cutoff": (
                str(new_cv3_cutoff) if new_cv3_cutoff is not None else None
            ),
            "pair_published": n_cv3 > 0,
            "pair_generation_id": published_generation_id,
            "elapsed_sec": round(elapsed, 2),
        }
    finally:
        _discard_pair_staging_dir(
            staging_dir,
            generation_root=generation_root,
        )


def backfill_base34_ctx(
    since_ts: pd.Timestamp,
    *,
    pair_manifest_path: Path = PREBUILT_PAIR_MANIFEST_PATH,
    generation_root: Path = PREBUILT_PAIR_ROOT,
) -> dict:
    """One-shot repair of the 2026-05-25 FREEZE: recompute the 37 BASE34-only ctx columns for all
    rows after `since_ts` via the same ONE-TRUTH augmenter the incremental path uses."""
    from gx1.execution.v12_ctx_augment_live import augment_canonical_v3
    current = read_prebuilt_pair_manifest(
        pair_manifest_path,
        generation_root=generation_root,
    )
    verify_prebuilt_pair(current)
    base34 = pd.read_parquet(current.base28.parquet_path)
    base34.index = pd.to_datetime(base34.index, utc=True)
    cv3 = pd.read_parquet(current.canonical_v3.parquet_path)
    if "time" in cv3.columns:
        cv3["time"] = pd.to_datetime(cv3["time"], utc=True)
        cv3 = cv3.set_index("time")
    cv3 = cv3.sort_index()
    # Backfill must use the same full-history state as normal append.
    win = cv3.copy()
    # REGIME_V4: same ONE-TRUTH per-TF V2 scalar attach as the live append path, so the backfill
    # recomputes the 52 regime cols (not just the legacy ctx) when GX1_REGIME_V4=1.
    from gx1.features.htf_features import attach_default_regime_v4_v2_scalars
    attach_default_regime_v4_v2_scalars(win)
    _m5_cols = [c for c in ("open", "high", "low", "close", "volume") if c in win.columns]
    cv3_aug = augment_canonical_v3(win, win[_m5_cols].copy())
    target = [c for c in base34.columns if c not in cv3.columns and c != "is_model_bar"
              and c in cv3_aug.columns]
    mask = base34.index > since_ts
    n_rows = int(mask.sum())
    if n_rows == 0:
        return {
            "rows_backfilled": 0,
            "cols": len(target),
            "nunique_before_sample": {},
            "nunique_after_sample": {},
            "pair_published": False,
            "pair_generation_id": current.pair_generation_id,
        }
    # map each M1 row to its last CLOSED M5 bar (same semantics as the append path);
    # int64-ns on both sides (tz-aware vs naive .values would raise in searchsorted)
    closed_ns = closed_m5_start_for_m1_bar_labels(
        pd.DatetimeIndex(base34.index[mask])
    ).asi8
    aug_idx = np.searchsorted(cv3_aug.index.asi8, closed_ns, side="right") - 1
    valid = aug_idx >= 0
    if not bool(valid.all()):
        raise RuntimeError(
            "BASE34 backfill lacks exact prior augmented M5 state for target rows"
        )
    before = {c: int(base34.loc[mask, c].nunique()) for c in target[:6]}
    for c in target:
        vals = cv3_aug[c].to_numpy()[aug_idx]
        try:
            numeric = np.asarray(vals, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"BASE34 backfill column {c!r} is not numeric"
            ) from exc
        if not np.isfinite(numeric).all():
            raise RuntimeError(
                f"BASE34 backfill column {c!r} contains non-finite evidence"
            )
        base34.loc[mask, c] = pd.Series(
            numeric,
            index=base34.index[mask],
        ).astype(np.float32)
    if "is_model_bar" in base34.columns:
        base34.loc[mask, "is_model_bar"] = base34.index[mask].isin(cv3.index)
    after = {c: int(base34.loc[mask, c].nunique()) for c in target[:6]}
    staging_dir = _candidate_staging_path(generation_root)
    try:
        _copy_candidate_parquet(
            current.canonical_v3.parquet_path,
            staging_dir / PAIR_CANONICAL_FILENAME,
        )
        _write_candidate_parquet(
            base34,
            staging_dir / PAIR_BASE28_FILENAME,
            index=True,
        )
        published_generation_id = _publish_prebuilt_pair_generation(
            staging_dir,
            pair_manifest_path=pair_manifest_path,
            generation_root=generation_root,
            expected_pair_generation_id=current.pair_generation_id,
            expected_manifest_sha256=current.manifest_sha256,
        )
        return {"rows_backfilled": n_rows, "cols": len(target),
                "nunique_before_sample": before, "nunique_after_sample": after,
                "pair_published": True,
                "pair_generation_id": published_generation_id}
    finally:
        _discard_pair_staging_dir(
            staging_dir,
            generation_root=generation_root,
        )


def main() -> int:
    p = argparse.ArgumentParser(description="V12 incremental canonical/BASE34 updater")
    p.add_argument("--loop", action="store_true", help="Loop continuously (default: one-shot)")
    p.add_argument("--interval", type=int, default=60, help="Loop interval in seconds (default 60)")
    p.add_argument("--backfill-base34-since", type=str, default=None,
                   help="One-shot: recompute the frozen BASE34 ctx cols for rows after this UTC ts "
                        "and publish a complete atomic pair generation.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")

    if args.backfill_base34_since:
        stats = backfill_base34_ctx(pd.Timestamp(args.backfill_base34_since, tz="UTC"))
        print(json.dumps(stats, indent=2))
        return 0

    if not args.loop:
        stats = run_one_cycle()
        print(json.dumps(stats, indent=2))
        return 0

    # Hang forensics (2026-06-12, hang #3 — NOT the self-check; 60 threads in
    # futex_wait + main in do_select): SIGUSR1 dumps ALL thread stacks to stderr
    # (lands in the daemon log). The watchdog sends it before restarting, so the
    # next hang self-documents its root cause. py-spy needs ptrace/sudo — this doesn't.
    import faulthandler
    import signal as _signal
    faulthandler.register(_signal.SIGUSR1, all_threads=True)
    LOG.info(f"starting incremental updater loop (interval={args.interval}s)")
    # Rule-9 self-check MOVED OUT of this loop (2026-06-12, standing decision after
    # hang #2): the hourly in-process check (reading BOTH prebuilts inside the
    # appender) was the last sign of life before BOTH daemon hangs at the 21-22Z
    # pause boundary (2026-06-12 00:05 and 22:00). It now runs as its own systemd
    # --user timer (gx1-rule9-selfcheck.timer → feature_liveness --live-tail) so a
    # stuck check can never stall data collection. The launch preflight remains
    # the hard gate; this daemon only appends.
    while True:
        try:
            stats = run_one_cycle()
            if stats["cv3_appended"] > 0:
                LOG.info(f"cycle stats: {stats}")
        except Exception as exc:
            LOG.exception(f"cycle failed: {exc}")
        _time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
